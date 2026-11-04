#!/usr/bin/env python3
"""
yt_channel_scraper.py — scrape, download, and transcribe every video on one or more YouTube channels.

For each video it captures:
    Name | Length | Date Posted | Likes | Views | Comments | Description | Hashtags | Video File | Video Transcript
(plus URL, Video ID, Channel, Tags, Transcript Source for traceability)

Transcripts: uses YouTube captions first (creator-uploaded preferred, then auto-generated).
If a video has no captions, it falls back to local transcription with Whisper (faster-whisper).

Outputs (in --out folder):
    results.csv        written row-by-row, so an interrupted run can be resumed
    results.xlsx       spreadsheet version (written at the end; long cells truncated to Excel's limit)
    videos/            downloaded .mp4 files
    transcripts/       full transcript per video as .txt (never truncated)
    subtitles/         raw caption files from YouTube

Setup:
    pip install -U yt-dlp faster-whisper openpyxl
    Install ffmpeg (Mac: brew install ffmpeg | Windows: winget install ffmpeg | Linux: apt install ffmpeg)

Examples:
    python yt_channel_scraper.py https://www.youtube.com/@SomeChannel
    python yt_channel_scraper.py @ChannelA @ChannelB --limit 50 --after 20250101
    python yt_channel_scraper.py @SomeChannel --no-video          # metadata + transcripts only
    python yt_channel_scraper.py @SomeChannel --shorts            # include Shorts tab
"""

import argparse
import csv
import json
import re
import sys
import tempfile
import time
from pathlib import Path

try:
    import yt_dlp
except ImportError:
    sys.exit("yt-dlp is not installed. Run: pip install -U yt-dlp")

COLUMNS = [
    "Name", "Length", "Date Posted", "Likes", "Views", "Comments",
    "Description", "Hashtags", "Video File", "Video Transcript",
    "URL", "Video ID", "Channel", "Tags", "Transcript Source",
]
EXCEL_CELL_LIMIT = 32000
HASHTAG_RE = re.compile(r"(?<!\w)#([\w\u00C0-\uFFFF]+)")


# ---------------------------------------------------------------- helpers

def normalize_url(target: str) -> str:
    """Accept full URLs, @handles, or bare channel IDs."""
    target = target.strip()
    if target.startswith("@"):
        return f"https://www.youtube.com/{target}"
    if target.startswith("UC") and len(target) == 24:
        return f"https://www.youtube.com/channel/{target}"
    if not target.startswith("http"):
        return f"https://www.youtube.com/{target}"
    return target


def format_duration(seconds) -> str:
    if seconds is None:
        return ""
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def format_date(yyyymmdd) -> str:
    if not yyyymmdd or len(str(yyyymmdd)) != 8:
        return ""
    d = str(yyyymmdd)
    return f"{d[:4]}-{d[4:6]}-{d[6:]}"


def extract_hashtags(*texts) -> str:
    seen, out = set(), []
    for text in texts:
        for tag in HASHTAG_RE.findall(text or ""):
            key = tag.lower()
            if key not in seen:
                seen.add(key)
                out.append("#" + tag)
    return " ".join(out)


def json3_to_text(path: Path) -> str:
    """Convert a YouTube json3 caption file to plain text."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    parts = []
    for ev in data.get("events", []):
        segs = ev.get("segs")
        if not segs:
            continue
        t = "".join(s.get("utf8", "") for s in segs).replace("\n", " ").strip()
        if t:
            parts.append(t)
    return re.sub(r"\s+", " ", " ".join(parts)).strip()


def log(msg: str):
    print(time.strftime("[%H:%M:%S] ") + msg, flush=True)


# ---------------------------------------------------------------- listing

def list_video_ids(target: str, include_shorts: bool, limit, base_opts: dict) -> list:
    """Return video IDs for a channel, playlist, or single video URL."""
    url = normalize_url(target)

    if "watch?v=" in url or "youtu.be/" in url or "/shorts/" in url:
        with yt_dlp.YoutubeDL({**base_opts, "extract_flat": True}) as ydl:
            info = ydl.extract_info(url, download=False)
            return [info["id"]] if info else []

    if "playlist?list=" in url:
        urls = [url]
    else:
        base = re.sub(r"/(videos|shorts|streams|featured|playlists|about)/?$", "", url.rstrip("/"))
        urls = [f"{base}/videos"] + ([f"{base}/shorts"] if include_shorts else [])

    opts = {**base_opts, "extract_flat": "in_playlist", "skip_download": True}
    if limit:
        opts["playlistend"] = limit

    ids = []
    with yt_dlp.YoutubeDL(opts) as ydl:
        for u in urls:
            try:
                info = ydl.extract_info(u, download=False)
            except yt_dlp.utils.DownloadError as e:
                log(f"  Could not list {u}: {e}")
                continue
            for entry in (info or {}).get("entries") or []:
                vid = entry and entry.get("id")
                if vid and vid not in ids:
                    ids.append(vid)
    return ids[:limit] if limit else ids


# ---------------------------------------------------------------- transcription

class Transcriber:
    """Lazily loads Whisper only if a video has no captions."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None

    @property
    def enabled(self):
        return self.model_name.lower() != "none"

    def transcribe(self, media_path: Path) -> str:
        if self.model is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError:
                log("  faster-whisper not installed; skipping Whisper fallback "
                    "(pip install faster-whisper)")
                self.model_name = "none"
                return ""
            log(f"  Loading Whisper model '{self.model_name}' (first time only)...")
            self.model = WhisperModel(self.model_name, device="auto", compute_type="auto")
        segments, _ = self.model.transcribe(str(media_path), vad_filter=True)
        return re.sub(r"\s+", " ", " ".join(s.text.strip() for s in segments)).strip()


def caption_transcript(info: dict, langs: list):
    """Pick the best downloaded caption file: manual before auto, in language-preference order."""
    requested = info.get("requested_subtitles") or {}
    manual = set((info.get("subtitles") or {}).keys())
    ordered = sorted(
        requested.items(),
        key=lambda kv: (kv[0] not in manual,
                        langs.index(kv[0]) if kv[0] in langs else len(langs)),
    )
    for lang, sub in ordered:
        path = sub.get("filepath")
        if path and Path(path).exists():
            try:
                text = json3_to_text(Path(path))
            except (json.JSONDecodeError, OSError):
                continue
            if text:
                kind = "YouTube captions (creator)" if lang in manual else "YouTube captions (auto)"
                return text, f"{kind} [{lang}]"
    return "", ""


def download_audio_only(url: str, base_opts: dict, tmpdir: str):
    opts = {
        **base_opts,
        "format": "bestaudio/best",
        "outtmpl": str(Path(tmpdir) / "%(id)s.%(ext)s"),
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        dl = (info.get("requested_downloads") or [{}])[0]
        return Path(dl["filepath"]) if dl.get("filepath") else None


# ---------------------------------------------------------------- per video

def process_video(vid: str, args, out: Path, base_opts: dict, transcriber: Transcriber):
    url = f"https://www.youtube.com/watch?v={vid}"
    opts = {
        **base_opts,
        "format": args.format,
        "merge_output_format": "mp4",
        "outtmpl": {
            "default": str(out / "videos" / "%(upload_date)s_%(id)s_%(title).80B.%(ext)s"),
            "subtitle": str(out / "subtitles" / "%(id)s.%(ext)s"),
        },
        "restrictfilenames": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": args.langs,
        "subtitlesformat": "json3",
        "skip_download": args.no_video,
    }

    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
        if info is None:
            raise RuntimeError("no metadata returned")
        if args.after and (info.get("upload_date") or "") < args.after:
            return None  # filtered out by date
        info = ydl.process_ie_result(info, download=True)

    video_file = ""
    if not args.no_video:
        dl = (info.get("requested_downloads") or [{}])[0]
        video_file = dl.get("filepath") or ""

    transcript, source = ("", "") if args.force_whisper else caption_transcript(info, args.langs)

    if not transcript and transcriber.enabled:
        log("  No captions found; transcribing with Whisper...")
        if video_file and Path(video_file).exists():
            transcript = transcriber.transcribe(Path(video_file))
        else:
            with tempfile.TemporaryDirectory() as tmp:
                audio = download_audio_only(url, base_opts, tmp)
                if audio:
                    transcript = transcriber.transcribe(audio)
        source = f"Whisper ({transcriber.model_name})" if transcript else ""

    if transcript:
        (out / "transcripts" / f"{vid}.txt").write_text(transcript, encoding="utf-8")

    title = info.get("title") or ""
    description = info.get("description") or ""
    return {
        "Name": title,
        "Length": format_duration(info.get("duration")),
        "Date Posted": format_date(info.get("upload_date")),
        "Likes": info.get("like_count") if info.get("like_count") is not None else "",
        "Views": info.get("view_count") if info.get("view_count") is not None else "",
        "Comments": info.get("comment_count") if info.get("comment_count") is not None else "",
        "Description": description,
        "Hashtags": extract_hashtags(title, description),
        "Video File": video_file,
        "Video Transcript": transcript,
        "URL": info.get("webpage_url") or url,
        "Video ID": vid,
        "Channel": info.get("channel") or info.get("uploader") or "",
        "Tags": ", ".join(info.get("tags") or []),
        "Transcript Source": source or "none available",
    }


# ---------------------------------------------------------------- output

def load_done_ids(csv_path: Path) -> set:
    if not csv_path.exists():
        return set()
    with csv_path.open(newline="", encoding="utf-8") as f:
        return {row["Video ID"] for row in csv.DictReader(f) if row.get("Video ID")}


def append_row(csv_path: Path, row: dict):
    new = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if new:
            w.writeheader()
        w.writerow(row)


def write_xlsx(csv_path: Path, xlsx_path: Path):
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Alignment, Font
    except ImportError:
        log("openpyxl not installed; skipping .xlsx (CSV is complete).")
        return
    wb = Workbook()
    ws = wb.active
    ws.title = "Videos"
    ws.append(COLUMNS)
    for c in ws[1]:
        c.font = Font(bold=True)
    numeric = {"Likes", "Views", "Comments"}
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            vals = []
            for col in COLUMNS:
                v = row.get(col, "")
                if col in numeric and str(v).isdigit():
                    v = int(v)
                elif isinstance(v, str) and len(v) > EXCEL_CELL_LIMIT:
                    v = v[:EXCEL_CELL_LIMIT] + " …[truncated — see transcripts folder]"
                vals.append(v)
            ws.append(vals)
    widths = {"Name": 45, "Description": 60, "Video Transcript": 80, "Video File": 40, "Hashtags": 30}
    for i, col in enumerate(COLUMNS, start=1):
        ws.column_dimensions[ws.cell(1, i).column_letter].width = widths.get(col, 14)
    for row in ws.iter_rows(min_row=2):
        for c in row:
            c.alignment = Alignment(wrap_text=False, vertical="top")
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions
    wb.save(xlsx_path)


# ---------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser(description="Scrape, download, and transcribe YouTube channel videos.")
    p.add_argument("channels", nargs="+", help="Channel URLs, @handles, playlist URLs, or video URLs")
    p.add_argument("-o", "--out", default="yt_output", help="Output folder (default: yt_output)")
    p.add_argument("--limit", type=int, help="Max videos per channel (newest first)")
    p.add_argument("--after", help="Only videos posted on/after this date, YYYYMMDD")
    p.add_argument("--shorts", action="store_true", help="Also include the channel's Shorts")
    p.add_argument("--no-video", action="store_true", help="Skip downloading video files")
    p.add_argument("--format", default="bv*[height<=720][ext=mp4]+ba[ext=m4a]/b[height<=720]/b",
                   help="yt-dlp format string (default: up to 720p mp4)")
    p.add_argument("--langs", default="en,en-US,en-GB,en-orig",
                   help="Caption languages in preference order (comma-separated)")
    p.add_argument("--whisper-model", default="small",
                   help="Whisper fallback model: tiny/base/small/medium/large-v3, or 'none' to disable")
    p.add_argument("--force-whisper", action="store_true",
                   help="Ignore YouTube captions and transcribe everything with Whisper")
    p.add_argument("--sleep", type=float, default=2.0, help="Seconds to wait between videos")
    p.add_argument("--cookies-from-browser", help="e.g. chrome or firefox (for age-restricted videos)")
    args = p.parse_args()
    args.langs = [l.strip() for l in args.langs.split(",") if l.strip()]

    out = Path(args.out)
    for sub in ("videos", "transcripts", "subtitles"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    csv_path, xlsx_path = out / "results.csv", out / "results.xlsx"

    base_opts = {
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "ignoreerrors": False,
        "retries": 5,
        "sleep_interval_requests": 0.5,
    }
    if args.cookies_from_browser:
        base_opts["cookiesfrombrowser"] = (args.cookies_from_browser,)

    done = load_done_ids(csv_path)
    if done:
        log(f"Resuming: {len(done)} videos already in {csv_path}")
    transcriber = Transcriber(args.whisper_model)
    saved = failed = skipped = 0

    for target in args.channels:
        log(f"Listing videos for {target} ...")
        ids = list_video_ids(target, args.shorts, args.limit, base_opts)
        todo = [v for v in ids if v not in done]
        log(f"  {len(ids)} found, {len(todo)} to process")

        for i, vid in enumerate(todo, 1):
            log(f"[{i}/{len(todo)}] {vid}")
            try:
                row = process_video(vid, args, out, base_opts, transcriber)
            except KeyboardInterrupt:
                log("Interrupted — progress saved. Re-run the same command to resume.")
                write_xlsx(csv_path, xlsx_path)
                sys.exit(1)
            except Exception as e:  # keep going on any single-video failure
                log(f"  FAILED: {e}")
                failed += 1
                continue
            if row is None:
                skipped += 1
                continue
            append_row(csv_path, row)
            done.add(vid)
            saved += 1
            log(f"  ✓ {row['Name'][:70]}  ({row['Transcript Source']})")
            time.sleep(args.sleep)

    if csv_path.exists():
        write_xlsx(csv_path, xlsx_path)
    log(f"Done. Saved {saved}, skipped {skipped} (date filter), failed {failed}.")
    log(f"Results: {csv_path.resolve()}")


if __name__ == "__main__":
    main()
