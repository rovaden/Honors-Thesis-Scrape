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
    python yt_channel_scraper.py --status                         # summary of results so far
    python yt_channel_scraper.py --status @SomeChannel            # ...plus how many aren't saved yet

On Windows PowerShell, put @handles in quotes: "@SomeChannel"
"""

import argparse
import copy
import csv
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path

# Silence harmless Hugging Face notices shown when Whisper downloads its model on Windows.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")

try:
    import yt_dlp
except ImportError:
    sys.exit("yt-dlp is not installed. Run: pip install -U yt-dlp")

COLUMNS = [
    "Name", "Length", "Date Posted", "Likes", "Views", "Comments",
    "Description", "Hashtags", "Video File", "Video Transcript",
    "Type", "URL", "Video ID", "Channel", "Tags", "Transcript Source",
]
EXCEL_CELL_LIMIT = 32000
RATE_LIMIT_WAITS = (30, 120)  # seconds to wait before each retry after a 429 on captions
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

def list_video_ids(target: str, include_shorts: bool, limit, base_opts: dict) -> dict:
    """Return {video_id: type} for a channel, playlist, or single video URL.
    Type is "Video" or "Short" (or "" when it can't be told, e.g. playlists).
    With --limit, the limit applies separately to the Videos and Shorts tabs."""
    url = normalize_url(target)

    if "watch?v=" in url or "youtu.be/" in url or "/shorts/" in url:
        with yt_dlp.YoutubeDL({**base_opts, "extract_flat": True}) as ydl:
            info = ydl.extract_info(url, download=False)
        if not info:
            return {}
        return {info["id"]: "Short" if "/shorts/" in url else "Video"}

    if "playlist?list=" in url:
        tabs = [(url, "")]
    else:
        base = re.sub(r"/(videos|shorts|streams|featured|playlists|about)/?$", "", url.rstrip("/"))
        tabs = [(f"{base}/videos", "Video")] + ([(f"{base}/shorts", "Short")] if include_shorts else [])

    opts = {**base_opts, "extract_flat": "in_playlist", "skip_download": True}
    if limit:
        opts["playlistend"] = limit

    found = {}
    with yt_dlp.YoutubeDL(opts) as ydl:
        for u, kind in tabs:
            try:
                info = ydl.extract_info(u, download=False)
            except yt_dlp.utils.DownloadError as e:
                log(f"  Could not list {u}: {e}")
                continue
            for entry in (info or {}).get("entries") or []:
                vid = entry and entry.get("id")
                if vid and vid not in found:
                    found[vid] = kind
    return found


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
        segments, _ = self.model.transcribe(load_audio(media_path), vad_filter=True)
        return re.sub(r"\s+", " ", " ".join(s.text.strip() for s in segments)).strip()


def load_audio(media_path: Path, sr: int = 16000):
    """Decode audio with ffmpeg into the 16 kHz mono float array Whisper expects.
    (Avoids faster-whisper's own decoder, which breaks with newer PyAV versions.)"""
    import subprocess
    import numpy as np
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-i", str(media_path),
           "-f", "s16le", "-ac", "1", "-ar", str(sr), "-"]
    try:
        raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    except FileNotFoundError:
        raise RuntimeError("ffmpeg not found — install it (winget install ffmpeg) and restart the terminal")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg could not read the audio: {e.stderr.decode(errors='ignore').strip()[:200]}")
    return np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0


def caption_transcript(info: dict, langs: list, subs_dir: Path):
    """Pick the best downloaded caption file: manual before auto, in language-preference order."""
    requested = info.get("requested_subtitles") or {}
    manual = set((info.get("subtitles") or {}).keys())
    ordered = sorted(
        requested.items(),
        key=lambda kv: (kv[0] not in manual,
                        langs.index(kv[0]) if kv[0] in langs else len(langs)),
    )
    for lang, sub in ordered:
        # yt-dlp's reported filepath can be wrong when a separate subtitle template is used,
        # so look for the file where the subtitle template actually puts it.
        candidates = [subs_dir / f"{info.get('id')}.{lang}.json3", Path(sub.get("filepath") or "")]
        path = next((c for c in candidates if c.name and c.is_file()), None)
        if path:
            try:
                text = json3_to_text(path)
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

def process_video(vid: str, args, out: Path, base_opts: dict, transcriber: Transcriber,
                  need_video: bool = True, need_transcript: bool = True, kind: str = ""):
    """Fetch metadata, plus the video file and/or transcript only if they're needed."""
    url = f"https://www.youtube.com/watch?v={vid}"
    want_video = need_video and not args.no_video
    opts = {
        **base_opts,
        "format": args.format,
        "merge_output_format": "mp4",
        "outtmpl": {
            "default": str(out / "videos" / "%(upload_date)s_%(id)s_%(title).80B.%(ext)s"),
            "subtitle": str(out / "subtitles" / "%(id)s.%(ext)s"),
        },
        "restrictfilenames": True,
        "writesubtitles": need_transcript,
        "writeautomaticsub": need_transcript,
        "subtitleslangs": args.langs,
        "subtitlesformat": "json3",
        "skip_download": not want_video,
    }

    with yt_dlp.YoutubeDL(opts) as ydl:
        raw = ydl.extract_info(url, download=False)
    if raw is None:
        raise RuntimeError("no metadata returned")
    if args.after and (raw.get("upload_date") or "") < args.after:
        return None  # filtered out by date

    # YouTube sometimes answers caption downloads with "429 Too Many Requests".
    # Wait and retry; if it keeps happening, save everything else and leave the
    # transcript blank so a later run fills it in.
    captions_blocked = False
    waits = list(RATE_LIMIT_WAITS)
    while True:
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.process_ie_result(copy.deepcopy(raw), download=True)
            break
        except yt_dlp.utils.DownloadError as e:
            msg = str(e)
            if not ("subtitle" in msg.lower() and ("429" in msg or "Too Many Requests" in msg)):
                raise
            if waits:
                wait = waits.pop(0)
                log(f"  YouTube is rate-limiting captions (429). Waiting {wait}s, then retrying...")
                time.sleep(wait)
                continue
            log("  Captions still rate-limited; saving without a transcript (next run will retry it).")
            opts = {**opts, "writesubtitles": False, "writeautomaticsub": False}
            captions_blocked = True
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.process_ie_result(copy.deepcopy(raw), download=True)
            break

    video_file = ""
    if want_video:
        dl = (info.get("requested_downloads") or [{}])[0]
        video_file = dl.get("filepath") or ""

    transcript, source = "", ""
    if need_transcript and not args.force_whisper:
        transcript, source = caption_transcript(info, args.langs, out / "subtitles")

    if need_transcript and not transcript and transcriber.enabled and not captions_blocked:
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
        "Type": kind,
        "URL": info.get("webpage_url") or url,
        "Video ID": vid,
        "Channel": info.get("channel") or info.get("uploader") or "",
        "Tags": ", ".join(info.get("tags") or []),
        "Transcript Source": source or "none available",
    }


# ---------------------------------------------------------------- output

def is_blank(col: str, val) -> bool:
    """True if a cell counts as 'not pulled yet' and may be filled in by a later run."""
    if val is None or str(val).strip() == "":
        return True
    if col == "Transcript Source" and val == "none available":
        return True
    if col == "Video File" and not Path(val).is_file():
        return True
    return False


def missing_parts(row: dict, args):
    """Which pieces of an already-saved video still need to be pulled: (video, transcript)."""
    need_video = not args.no_video and is_blank("Video File", row.get("Video File"))
    need_transcript = is_blank("Video Transcript", row.get("Video Transcript"))
    return need_video, need_transcript


def merge_rows(old: dict, new: dict) -> dict:
    """Keep every existing value; only fill cells that were empty."""
    return {c: (old.get(c, "") if not is_blank(c, old.get(c)) else new.get(c, "")) for c in COLUMNS}


def load_rows(csv_path: Path) -> dict:
    """Existing results keyed by Video ID, in file order."""
    if not csv_path.exists():
        return {}
    with csv_path.open(newline="", encoding="utf-8") as f:
        return {row["Video ID"]: row for row in csv.DictReader(f) if row.get("Video ID")}


def write_rows(csv_path: Path, rows):
    """Rewrite the whole CSV safely (write a temp file, then swap it in)."""
    tmp = csv_path.with_suffix(".csv.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, csv_path)


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


# ---------------------------------------------------------------- status

def show_status(csv_path: Path, args, base_opts: dict):
    """Print a summary of what's been collected so far. Downloads nothing."""
    rows = load_rows(csv_path)
    print(f"\nResults file: {csv_path.resolve()}")
    if not rows:
        print("  No results saved yet.")
    else:
        rows_list = list(rows.values())
        by_channel = {}
        for r in rows_list:
            by_channel[r.get("Channel") or "(unknown)"] = by_channel.get(r.get("Channel") or "(unknown)", 0) + 1
        with_tx = [r for r in rows_list if not is_blank("Video Transcript", r.get("Video Transcript"))]
        captions = sum(1 for r in with_tx if r.get("Transcript Source", "").startswith("YouTube"))
        whisper = sum(1 for r in with_tx if r.get("Transcript Source", "").startswith("Whisper"))
        with_vid = sum(1 for r in rows_list if not is_blank("Video File", r.get("Video File")))
        n_short = sum(1 for r in rows_list if r.get("Type") == "Short")
        n_long = sum(1 for r in rows_list if r.get("Type") == "Video")
        n_unknown = len(rows_list) - n_short - n_long
        print(f"  Videos saved:  {len(rows_list)}  ({n_long} regular videos, {n_short} Shorts"
              + (f", {n_unknown} not yet labeled" if n_unknown else "") + ")")
        for ch, n in sorted(by_channel.items(), key=lambda kv: -kv[1]):
            print(f"      {ch}: {n}")
        print(f"  Transcripts:   {len(with_tx)} have one ({captions} YouTube captions, {whisper} Whisper), "
              f"{len(rows_list) - len(with_tx)} missing")
        print(f"  Video files:   {with_vid} downloaded, {len(rows_list) - with_vid} missing")

        incomplete = []
        for r in rows_list:
            missing = []
            if is_blank("Video Transcript", r.get("Video Transcript")):
                missing.append("transcript")
            if is_blank("Video File", r.get("Video File")):
                missing.append("video")
            if missing:
                incomplete.append((r, missing))
        if incomplete:
            print(f"\n  Incomplete videos ({len(incomplete)}) — a normal run will try to fill these in"
                  " (video files are skipped if you use --no-video):")
            for r, missing in incomplete[:25]:
                print(f"      {r['Video ID']}  missing {' + '.join(missing):<20} {r.get('Name', '')[:55]}")
            if len(incomplete) > 25:
                print(f"      ... and {len(incomplete) - 25} more")

    for target in args.channels:
        print(f"\nChecking {target} on YouTube...")
        kinds = list_video_ids(target, args.shorts, args.limit, base_opts)
        for label in ("Video", "Short") if args.shorts else ("Video",):
            ids = [v for v, k in kinds.items() if k == label]
            not_saved = [v for v in ids if v not in rows]
            print(f"  {label}s: {len(ids)} on channel, {len(ids) - len(not_saved)} saved, "
                  f"{len(not_saved)} not saved yet (new or previously failed)")
        if not args.shorts:
            print("  (add --shorts to include the channel's Shorts)")
    print()


# ---------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser(description="Scrape, download, and transcribe YouTube channel videos.")
    p.add_argument("channels", nargs="*", help="Channel URLs, @handles, playlist URLs, or video URLs")
    p.add_argument("--status", action="store_true",
                   help="Just report what's in the results so far (no downloading). "
                        "Add channels to also see how many of their videos aren't saved yet.")
    p.add_argument("--sub-sleep", type=float, default=5.0,
                   help="Seconds to wait before each caption download, to avoid 429 errors (default: 5)")
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
    if not args.channels and not args.status:
        p.error("give at least one channel (e.g. \"@healthbyro\"), or use --status")
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
        "sleep_interval_subtitles": args.sub_sleep,
    }
    if args.cookies_from_browser:
        base_opts["cookiesfrombrowser"] = (args.cookies_from_browser,)

    if args.status:
        show_status(csv_path, args, base_opts)
        return

    rows = load_rows(csv_path)
    if rows:
        log(f"Resuming: {len(rows)} videos already in {csv_path}")
        # Upgrade CSVs from older versions of this script that lack newer columns (e.g. "Type").
        with csv_path.open(newline="", encoding="utf-8") as f:
            header = next(csv.reader(f), [])
        if header != COLUMNS:
            write_rows(csv_path, rows.values())
    transcriber = Transcriber(args.whisper_model)
    saved = updated = failed = skipped = 0

    for target in args.channels:
        log(f"Listing videos for {target} ...")
        kinds = list_video_ids(target, args.shorts, args.limit, base_opts)
        ids = list(kinds)
        n_shorts = sum(1 for k in kinds.values() if k == "Short")
        if n_shorts:
            log(f"  (includes {n_shorts} Shorts)")

        # Label saved rows whose Type is still blank (saved before the Type column existed).
        labeled = False
        for vid, kind in kinds.items():
            if vid in rows and kind and not rows[vid].get("Type"):
                rows[vid]["Type"] = kind
                labeled = True
        if labeled:
            write_rows(csv_path, rows.values())

        # New videos get the full treatment; saved videos only get their missing pieces re-pulled.
        todo = []
        for vid in ids:
            if vid in rows:
                need_video, need_transcript = missing_parts(rows[vid], args)
                if need_video or need_transcript:
                    todo.append((vid, need_video, need_transcript))
            else:
                todo.append((vid, True, True))
        n_new = sum(1 for v, _, _ in todo if v not in rows)
        log(f"  {len(ids)} found: {n_new} new, {len(todo) - n_new} incomplete, "
            f"{len(ids) - len(todo)} already complete")

        for i, (vid, need_video, need_transcript) in enumerate(todo, 1):
            old = rows.get(vid)
            if old is None:
                log(f"[{i}/{len(todo)}] {vid}")
            else:
                missing = [n for n, m in (("video", need_video), ("transcript", need_transcript)) if m]
                log(f"[{i}/{len(todo)}] {vid}  (filling in missing {' + '.join(missing)})")
            try:
                new = process_video(vid, args, out, base_opts, transcriber, need_video, need_transcript,
                                    kinds.get(vid, ""))
            except KeyboardInterrupt:
                log("Interrupted — progress saved. Re-run the same command to resume.")
                if csv_path.exists():
                    write_xlsx(csv_path, xlsx_path)
                sys.exit(1)
            except Exception as e:  # keep going on any single-video failure
                log(f"  FAILED: {e}")
                failed += 1
                continue
            if new is None:
                skipped += 1
                continue
            if old is None:
                append_row(csv_path, new)
                rows[vid] = new
                saved += 1
            else:
                rows[vid] = merge_rows(old, new)
                write_rows(csv_path, rows.values())
                updated += 1
            log(f"  ✓ {rows[vid]['Name'][:70]}  ({rows[vid]['Transcript Source']})")
            time.sleep(args.sleep)

    if csv_path.exists():
        write_xlsx(csv_path, xlsx_path)
    log(f"Done. New {saved}, filled in {updated}, skipped {skipped} (date filter), failed {failed}.")
    log(f"Results: {csv_path.resolve()}")


if __name__ == "__main__":
    main()
