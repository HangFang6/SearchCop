"""
Weekly report generator.

Reads `fh_gait/工作记录.md` (or any --src markdown file), extracts the most
recent N days' H2 sections (`## YYYY-MM-DD ...`), and produces a clean
`weekly_report.md` you can copy-paste into a slack message / boss email.

Conventions in 工作记录.md (must hold for parser):
  - Each day's entry starts with `## YYYY-MM-DD ...`
  - Sub-bullets follow until the next `## ` line or `---` separator
  - Each entry typically contains `### 涉及文件`, `### 备注`, etc.

Usage:
  python scripts/weekly_report.py
  python scripts/weekly_report.py --days 7 --out fh_gait/weekly_2026-W21.md
  python scripts/weekly_report.py --src /path/to/工作记录.md --days 14
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import sys
from typing import List, Tuple


DEFAULT_SRC = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "fh_gait", "工作记录.md")
)
DEFAULT_OUT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "fh_gait", "weekly_report.md")
)


_DATE_HEADING = re.compile(r"^##\s+(\d{4})-(\d{2})-(\d{2})\b(.*)$")


def _parse_entries(text: str) -> List[Tuple[dt.date, str, str]]:
    """Return list of (date, heading_line, body) tuples, ordered as in file."""
    lines = text.splitlines()
    entries: List[Tuple[dt.date, str, List[str]]] = []
    cur_date = None
    cur_head = ""
    buf: List[str] = []

    for line in lines:
        m = _DATE_HEADING.match(line)
        if m:
            # flush previous
            if cur_date is not None:
                entries.append((cur_date, cur_head, "\n".join(buf).strip()))
                buf = []
            try:
                cur_date = dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            except ValueError:
                cur_date = None
                continue
            cur_head = line
            continue
        if cur_date is not None:
            buf.append(line)

    if cur_date is not None:
        entries.append((cur_date, cur_head, "\n".join(buf).strip()))

    return entries


def _extract_section(body: str, header_name: str) -> str:
    """Pull out the contents of a `### header_name` section if present."""
    pattern = re.compile(
        rf"###\s+{re.escape(header_name)}\s*\n(.*?)(?=\n###\s|\n---\s*$|\Z)",
        re.DOTALL,
    )
    m = pattern.search(body)
    return m.group(1).strip() if m else ""


def _summarize_entry(date: dt.date, heading: str, body: str) -> str:
    """Compose a compact bullet-style summary of a single day entry."""
    title = heading.lstrip("#").strip()
    out = [f"## {title}"]

    # Try to surface the most useful sub-sections in a stable order.
    keys = [
        "今日产出",
        "今晚产出",
        "今晚增量产出",
        "今晚产出（5 个文件新增/重写）",
        "改动清单",
        "关键决策",
        "关键技术决策",
        "下一步",
        "涉及文件",
        "备注",
    ]
    used = []
    for k in keys:
        s = _extract_section(body, k)
        if s and k not in used:
            out.append(f"\n**{k}**\n\n{s}")
            used.append(k)

    if len(out) == 1:
        # No sub-sections — just dump body shortened to ~30 lines
        body_lines = body.splitlines()
        snippet = "\n".join(body_lines[:30])
        if len(body_lines) > 30:
            snippet += f"\n... ({len(body_lines) - 30} more lines truncated)"
        out.append("\n" + snippet)

    return "\n".join(out)


def build_report(src: str, days: int, today: dt.date) -> str:
    if not os.path.isfile(src):
        raise FileNotFoundError(f"source not found: {src}")
    with open(src, "r", encoding="utf-8") as f:
        text = f.read()

    entries = _parse_entries(text)
    cutoff = today - dt.timedelta(days=days - 1)
    recent = [(d, h, b) for (d, h, b) in entries if d >= cutoff]
    recent.sort(key=lambda x: x[0])    # oldest -> newest

    if not recent:
        return f"# Weekly Report ({cutoff} → {today})\n\n_No entries in the last {days} days._\n"

    header = (
        f"# Weekly Report\n\n"
        f"- **Window**: {cutoff} → {today} (last {days} days)\n"
        f"- **Source**: `{os.path.basename(src)}`\n"
        f"- **Entries**: {len(recent)} day(s)\n"
        f"- **Generated**: {dt.datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
        "---\n\n"
    )

    # Top-level summary table
    table = ["## 速览\n",
             "| 日期 | 标题 |",
             "|---|---|"]
    for d, h, _ in recent:
        title = h.lstrip("#").strip()
        # take portion after the date
        m = _DATE_HEADING.match(h)
        if m and m.group(4).strip().lstrip("·•-").strip():
            title = m.group(4).strip().lstrip("·•-").strip()
        table.append(f"| {d.isoformat()} | {title} |")
    table.append("")

    # Per-day details
    body_parts = [_summarize_entry(d, h, b) for d, h, b in recent]

    return header + "\n".join(table) + "\n---\n\n" + "\n\n---\n\n".join(body_parts) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src",  default=DEFAULT_SRC, help="path to 工作记录.md")
    ap.add_argument("--out",  default=DEFAULT_OUT, help="output markdown file")
    ap.add_argument("--days", type=int, default=7, help="window size in days")
    ap.add_argument("--today", default=None,
                    help="override 'today' for testing (YYYY-MM-DD)")
    ap.add_argument("--stdout", action="store_true",
                    help="also print the report to stdout")
    args = ap.parse_args()

    today = dt.date.fromisoformat(args.today) if args.today else dt.date.today()
    report = build_report(args.src, days=args.days, today=today)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"[OK] report -> {args.out}  ({len(report)} chars)")
    if args.stdout:
        print("\n" + "=" * 60 + "\n")
        print(report)


if __name__ == "__main__":
    main()
