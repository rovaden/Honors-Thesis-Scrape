#!/usr/bin/env python3
"""
yt_comment_scraper.py — companion to yt_channel_scraper.py.

Reads the results.csv produced by yt_channel_scraper.py, scrapes every comment
(and reply) on each video listed there, and writes one row per comment:

    ID | Video ID | Commenter Name | Text | Likes | Replies | Is Reply? | Reply ID | Date Posted

    ID            YouTube's comment ID
    Replies       number of replies to this comment (0 for replies themselves)
    Is Reply?     "Yes" if the comment is a reply to another comment, else "No"
    Reply ID      for replies: the ID of the comment being replied to; blank for top-level comments
    Date Posted   approximate — YouTube only exposes relative times ("3 months ago")

Outputs (written next to the input results.csv unless -o is given):
    comments.csv          written video-by-video, so an interrupted run can be resumed
    comments.xlsx         spreadsheet version, built at the end
    comments_done.txt     video IDs already finished (used for resuming)

Setup: same as yt_channel_scraper.py  (pip install -U yt-dlp openpyxl)

Examples:
    python yt_comment_scraper.py                                   # auto-finds results.csv
    python yt_comment_scraper.py -i ro_output/results.csv
    python yt_comment_scraper.py -i ro_output/results.csv --max-per-video 500 --sort top
"""

import argparse
import csv
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    import yt_dlp
except ImportError:
    sys.exit("yt-dlp is not installed. Run: pip install -U yt-dlp")

COLUMNS = ["ID", "Video ID", "Commenter Name", "Text", "Likes", "Replies",
           "Is Reply?", "Reply ID", "Date Posted"]
EXCEL_CELL_LIMIT = 32000


def log(msg: str):
    print(time.strftime("[%H:%M:%S] ") + msg, flush=True)


# ---------------------------------------------------------------- input

def find_input(explicit):
    if explicit:
        p = Path(explicit)
        if not p.exists():
            sys.exit(f"Input file not found: {p}")
        return p
    default = Path("yt_output/results.csv")
    if default.exists():
        return default
    candidates = sorted(Path(".").glob("*/results.csv"))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        sys.exit("Couldn't find a results.csv. Run yt_channel_scraper.py first, "
                 "or pass one with -i path/to/results.csv")
    sys.exit("Found several results.csv files; pick one with -i:\n  "
             + "\n  ".join(str(c) for c in candidates))


def read_video_ids(path: Path) -> list:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "Video ID" not in (reader.fieldnames or []):
            sys.exit(f"{path} has no 'Video ID' column — is it the output of yt_channel_scraper.py?")
        ids = []
        for row in reader:
            vid = (row.get("Video ID") or "").strip()
            if vid and vid not in ids:
                ids.append(vid)
    return ids


# ---------------------------------------------------------------- scraping

def fetch_comments(vid: str, base_opts: dict, max_per_video, sort: str) -> list:
    limit = str(max_per_video) if max_per_video else "all"
    opts = {
        **base_opts,
        "skip_download": True,
        "getcomments": True,
        "extractor_args": {"youtube": {
            # total comments, top-level threads, total replies, replies per thread
            "max_comments": [limit, "all", "all", "all"],
            "comment_sort": [sort],
        }},
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(f"https://www.youtube.com/watch?v={vid}", download=False)
    if info is None:
        raise RuntimeError("no data returned")
    return info.get("comments") or []


def to_rows(vid: str, comments: list) -> list:
    """Convert yt-dlp comment dicts into spreadsheet rows."""
    reply_counts = {}
    for c in comments:
        parent = c.get("parent") or "root"
        if parent != "root":
            reply_counts[parent] = reply_counts.get(parent, 0) + 1

    rows = []
    for c in comments:
        cid = c.get("id") or ""
        parent = c.get("parent") or "root"
        is_reply = parent != "root"
        ts = c.get("timestamp")
        date = (datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
                if isinstance(ts, (int, float)) else "")
        likes = c.get("like_count")
        rows.append({
            "ID": cid,
            "Video ID": vid,
            "Commenter Name": c.get("author") or "",
            "Text": c.get("text") or "",
            "Likes": likes if likes is not None else 0,
            "Replies": 0 if is_reply else reply_counts.get(cid, 0),
            "Is Reply?": "Yes" if is_reply else "No",
            "Reply ID": parent if is_reply else "",
            "Date Posted": date,
        })
    return rows


# ---------------------------------------------------------------- output

def append_rows(csv_path: Path, rows: list):
    new = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if new:
            w.writeheader()
        w.writerows(rows)


def write_xlsx(csv_path: Path, xlsx_path: Path):
    try:
        from openpyxl import Workbook
        from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE
        from openpyxl.styles import Font
    except ImportError:
        log("openpyxl not installed; skipping .xlsx (CSV is complete).")
        return
    wb = Workbook(write_only=False)
    ws = wb.active
    ws.title = "Comments"
    ws.append(COLUMNS)
    for c in ws[1]:
        c.font = Font(bold=True)
    numeric = {"Likes", "Replies"}
    with csv_path.open(newline="", encoding="utf-8") as f:
        for r_idx, row in enumerate(csv.DictReader(f), start=2):
            for c_idx, col in enumerate(COLUMNS, start=1):
                v = row.get(col, "")
                if col in numeric and str(v).isdigit():
                    v = int(v)
                elif isinstance(v, str):
                    v = ILLEGAL_CHARACTERS_RE.sub("", v)
                    if len(v) > EXCEL_CELL_LIMIT:
                        v = v[:EXCEL_CELL_LIMIT] + " …[truncated — see CSV]"
                cell = ws.cell(row=r_idx, column=c_idx, value=v)
                if isinstance(v, str) and v.startswith("="):
                    cell.data_type = "s"  # keep comments like "=)" as text, not formulas
    widths = {"ID": 28, "Video ID": 14, "Commenter Name": 24, "Text": 80, "Reply ID": 28, "Date Posted": 12}
    for i, col in enumerate(COLUMNS, start=1):
        ws.column_dimensions[ws.cell(1, i).column_letter].width = widths.get(col, 10)
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions
    wb.save(xlsx_path)


# ---------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser(description="Scrape all comments for videos in a yt_channel_scraper results.csv")
    p.add_argument("-i", "--input", help="Path to results.csv (default: auto-detect)")
    p.add_argument("-o", "--out", help="Output folder (default: same folder as the input)")
    p.add_argument("--max-per-video", type=int, help="Cap comments per video (default: all)")
    p.add_argument("--sort", choices=["new", "top"], default="new",
                   help="Order YouTube returns comments in (matters only with --max-per-video)")
    p.add_argument("--sleep", type=float, default=2.0, help="Seconds to wait between videos")
    p.add_argument("--cookies-from-browser", help="e.g. chrome or firefox (for age-restricted videos)")
    args = p.parse_args()

    in_path = find_input(args.input)
    out = Path(args.out) if args.out else in_path.parent
    out.mkdir(parents=True, exist_ok=True)
    csv_path, xlsx_path, done_path = out / "comments.csv", out / "comments.xlsx", out / "comments_done.txt"

    base_opts = {"quiet": True, "no_warnings": True, "noprogress": True,
                 "retries": 5, "sleep_interval_requests": 0.5}
    if args.cookies_from_browser:
        base_opts["cookiesfrombrowser"] = (args.cookies_from_browser,)

    video_ids = read_video_ids(in_path)
    done = set(done_path.read_text().split()) if done_path.exists() else set()
    todo = [v for v in video_ids if v not in done]
    log(f"Input: {in_path}  —  {len(video_ids)} videos, {len(todo)} still to scrape")

    total = failed = 0
    for i, vid in enumerate(todo, 1):
        log(f"[{i}/{len(todo)}] {vid}")
        try:
            comments = fetch_comments(vid, base_opts, args.max_per_video, args.sort)
        except KeyboardInterrupt:
            log("Interrupted — progress saved. Re-run the same command to resume.")
            break
        except Exception as e:
            log(f"  FAILED: {e}")
            failed += 1
            continue
        rows = to_rows(vid, comments)
        if rows:
            append_rows(csv_path, rows)
        with done_path.open("a") as f:
            f.write(vid + "\n")
        n_replies = sum(r["Is Reply?"] == "Yes" for r in rows)
        log(f"  ✓ {len(rows)} comments ({len(rows) - n_replies} top-level, {n_replies} replies)")
        total += len(rows)
        time.sleep(args.sleep)

    if csv_path.exists():
        write_xlsx(csv_path, xlsx_path)
        log(f"Done. {total} comments added this run, {failed} videos failed.")
        log(f"Results: {csv_path.resolve()}")
    else:
        log("No comments were collected.")


if __name__ == "__main__":
    main()
