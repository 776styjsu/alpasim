#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Generate a searchable case-to-reasoning-signature report for benchmark.json."""

from __future__ import annotations

import argparse
import html
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "tools"))

from reasoning_coverage.table_program import (  # noqa: E402
    compile_table_program,
    primary_reasoning_items,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=Path("benchmark.json"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/reasoning_coverage/benchmark150_reasoning_signatures.html"),
    )
    return parser.parse_args()


def _entries(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if isinstance(payload, list):
        entries = payload
    elif isinstance(payload, dict):
        entries = payload.get("results", payload.get("entries"))
    else:
        entries = None
    if not isinstance(entries, list) or not all(
        isinstance(entry, dict) for entry in entries
    ):
        raise ValueError(f"Expected a benchmark entry list in {path}")
    return entries


def _signature_label(items: tuple[str, ...]) -> str:
    paths = [
        item.removeprefix("table:path=")
        for item in items
        if item.startswith("table:path=")
    ]
    if paths:
        return " & ".join(paths)
    decision = next(
        (
            item.removeprefix("table:z3_pair=")
            for item in items
            if item.startswith("table:z3_pair=")
        ),
        "none|none",
    )
    return f"unobserved>{decision}"


def _case(entry: dict[str, Any]) -> dict[str, Any]:
    clip_id = str(entry["clip_id"])
    program = compile_table_program(entry.get("chain_of_thought", ""))
    signature = tuple(sorted(primary_reasoning_items(program)))
    reliability = entry.get("cot_reliability")
    reliable = reliability.get("reliable") if isinstance(reliability, dict) else None
    inconsistent = reliable is True and entry.get("cot_action_consistency") is False
    unsafe = entry.get("planned_trajectory_safe") is False
    return {
        "clip_id": clip_id,
        "reasoning": program.text,
        "signature": signature,
        "signature_label": _signature_label(signature),
        "inconsistent": inconsistent,
        "unsafe": unsafe,
        "faulty": inconsistent or unsafe,
        "unreliable": reliable is False,
    }


def _badge(case: dict[str, Any]) -> str:
    badges = []
    if case["inconsistent"]:
        badges.append('<span class="badge fault">inconsistent reliable CoT</span>')
    if case["unsafe"]:
        badges.append('<span class="badge fault">unsafe trajectory</span>')
    if case["unreliable"]:
        badges.append('<span class="badge muted">unreliable CoT</span>')
    if not badges:
        badges.append('<span class="badge clean">no curve fault</span>')
    return " ".join(badges)


def _signature_section(
    rank: int,
    signature: tuple[str, ...],
    cases: list[dict[str, Any]],
    visible_label_count: int,
) -> str:
    signature_id = f"S{rank:03d}"
    label = cases[0]["signature_label"]
    faulty = sum(case["faulty"] for case in cases)
    duplicate_note = (
        f'<span class="badge muted" title="The readable path label is shared by '
        f'{visible_label_count} distinct full coverage-item sets">'
        f"label shared by {visible_label_count} full signatures</span>"
        if visible_label_count > 1
        else ""
    )
    rows = []
    for case in sorted(cases, key=lambda item: item["clip_id"]):
        search_blob = " ".join(
            (
                case["clip_id"],
                case["reasoning"],
                label,
                " ".join(signature),
            )
        ).lower()
        status = "faulty" if case["faulty"] else "clean"
        rows.append(
            f"""
            <tr class="case" data-status="{status}"
                data-search="{html.escape(search_blob, quote=True)}">
              <td><code>{html.escape(case['clip_id'])}</code></td>
              <td class="reasoning">{html.escape(case['reasoning'])}</td>
              <td class="status">{_badge(case)}</td>
            </tr>"""
        )
    coverage_items = "".join(
        f"<li><code>{html.escape(item)}</code></li>" for item in signature
    )
    return f"""
    <section class="signature" data-signature-id="{signature_id}">
      <details{'' if rank > 10 else ' open'}>
        <summary>
          <span class="signature-id">{signature_id}</span>
          <span class="summary-count">{len(cases)} case{'s' if len(cases) != 1 else ''}</span>
          <span class="summary-faults">{faulty} faulty</span>
          {duplicate_note}
          <code class="path-label">{html.escape(label)}</code>
        </summary>
        <div class="signature-body">
          <details class="coverage-items">
            <summary>Exact full signature · {len(signature)} coverage items</summary>
            <ul>{coverage_items}</ul>
          </details>
          <div class="table-wrap">
            <table>
              <thead><tr><th>Clip ID</th><th>Reasoning</th><th>Human-label status</th></tr></thead>
              <tbody>{''.join(rows)}</tbody>
            </table>
          </div>
        </div>
      </details>
    </section>"""


def _document(cases: list[dict[str, Any]], benchmark: Path) -> str:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        grouped[case["signature"]].append(case)
    ranked = sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0]))
    label_counts = Counter(group[0]["signature_label"] for _, group in ranked)
    unique_reasoning = len({case["reasoning"] for case in cases})
    faulty = sum(case["faulty"] for case in cases)
    sections = "".join(
        _signature_section(
            rank,
            signature,
            signature_cases,
            label_counts[signature_cases[0]["signature_label"]],
        )
        for rank, (signature, signature_cases) in enumerate(ranked, start=1)
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Benchmark-150 reasoning signatures</title>
  <style>
    :root {{ color-scheme: light; --ink:#17202a; --subtle:#5d6874; --line:#dfe4e8;
      --paper:#fff; --wash:#f4f6f8; --blue:#155eef; --blue-wash:#eef4ff;
      --red:#b42318; --red-wash:#fef3f2; --green:#067647; --green-wash:#ecfdf3; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:var(--wash); color:var(--ink); font:14px/1.5 ui-sans-serif,
      system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
    header {{ background:var(--paper); border-bottom:1px solid var(--line);
      padding:32px max(24px,calc((100% - 1320px)/2)); }}
    h1 {{ margin:0 0 6px; font-size:28px; letter-spacing:-.02em; }}
    .subtitle {{ color:var(--subtle); max-width:920px; margin:0; }}
    .stats {{ display:flex; flex-wrap:wrap; gap:10px; margin-top:20px; }}
    .stat {{ background:var(--wash); border:1px solid var(--line); border-radius:9px; padding:8px 12px; }}
    .stat strong {{ font-size:18px; margin-right:5px; }}
    .controls {{ position:sticky; top:0; z-index:5; display:flex; gap:12px; align-items:center;
      padding:12px max(24px,calc((100% - 1320px)/2)); background:rgba(244,246,248,.96);
      border-bottom:1px solid var(--line); backdrop-filter:blur(8px); }}
    input,select {{ border:1px solid #b8c1ca; border-radius:8px; background:white; padding:9px 11px;
      color:var(--ink); font:inherit; }}
    input {{ flex:1; min-width:220px; }}
    #visible {{ color:var(--subtle); white-space:nowrap; }}
    main {{ max-width:1320px; margin:20px auto 60px; padding:0 24px; }}
    .signature {{ background:var(--paper); border:1px solid var(--line); border-radius:11px;
      margin-bottom:12px; overflow:hidden; box-shadow:0 1px 2px rgba(16,24,40,.03); }}
    .signature > details > summary {{ display:grid; grid-template-columns:auto auto auto 1fr; gap:9px;
      align-items:center; padding:14px 16px; cursor:pointer; list-style:none; }}
    .signature > details > summary::-webkit-details-marker {{ display:none; }}
    .signature > details > summary::before {{ content:"›"; font-size:20px; color:var(--subtle);
      transition:transform .15s; }}
    .signature > details[open] > summary::before {{ transform:rotate(90deg); }}
    .signature-id {{ color:var(--blue); font-weight:750; }}
    .summary-count,.summary-faults {{ border-left:1px solid var(--line); padding-left:9px; white-space:nowrap; }}
    .summary-faults {{ color:var(--red); }}
    .path-label {{ grid-column:2/-1; display:block; color:#344054; white-space:normal; overflow-wrap:anywhere; }}
    .signature-body {{ border-top:1px solid var(--line); }}
    .coverage-items {{ padding:10px 16px; background:var(--blue-wash); border-bottom:1px solid #d6e4ff; }}
    .coverage-items summary {{ cursor:pointer; color:#1849a9; font-weight:600; }}
    .coverage-items ul {{ columns:2; padding-left:22px; }}
    .coverage-items li {{ break-inside:avoid; margin:4px 0; overflow-wrap:anywhere; }}
    .table-wrap {{ overflow-x:auto; }}
    table {{ width:100%; border-collapse:collapse; }}
    th {{ color:var(--subtle); font-size:12px; text-align:left; text-transform:uppercase; letter-spacing:.04em; }}
    th,td {{ padding:12px 16px; border-bottom:1px solid var(--line); vertical-align:top; }}
    tbody tr:last-child td {{ border-bottom:0; }}
    td:first-child {{ width:31%; }} td.reasoning {{ width:49%; font-size:15px; }} td.status {{ width:20%; }}
    code {{ font-family:ui-monospace,SFMono-Regular,Consolas,"Liberation Mono",monospace; font-size:.9em; }}
    .badge {{ display:inline-block; border-radius:999px; padding:2px 8px; margin:1px 2px 1px 0;
      font-size:11px; font-weight:650; white-space:nowrap; }}
    .badge.fault {{ color:var(--red); background:var(--red-wash); }}
    .badge.clean {{ color:var(--green); background:var(--green-wash); }}
    .badge.muted {{ color:#475467; background:#f2f4f7; }}
    .empty {{ display:none; max-width:1320px; margin:35px auto; text-align:center; color:var(--subtle); }}
    @media (max-width:760px) {{ .controls {{ flex-wrap:wrap; }}
      .signature > details > summary {{ grid-template-columns:auto auto 1fr; }}
      .path-label {{ grid-column:2/-1; }} .coverage-items ul {{ columns:1; }} th,td {{ padding:10px; }} }}
    @media print {{ .controls {{ display:none; }} .signature {{ break-inside:avoid; }} body {{ background:white; }} }}
  </style>
</head>
<body>
  <header>
    <h1>Benchmark-150 reasoning signatures</h1>
    <p class="subtitle">Each case appears once under its exact primary-stage coverage-item set.
      The readable path can repeat when full signatures differ by details such as location,
      modality, or lateral direction.
      Source: <code>{html.escape(str(benchmark))}</code>.</p>
    <div class="stats">
      <span class="stat"><strong>{len(cases)}</strong>cases</span>
      <span class="stat"><strong>{unique_reasoning}</strong>unique reasoning texts</span>
      <span class="stat"><strong>{len(ranked)}</strong>exact signatures</span>
      <span class="stat"><strong>{len(label_counts)}</strong>readable path labels</span>
      <span class="stat"><strong>{faulty}</strong>curve-fault cases</span>
    </div>
  </header>
  <div class="controls">
    <input id="search" type="search" placeholder="Search clip ID, reasoning, path, or coverage item…" autofocus>
    <select id="status" aria-label="Filter by status">
      <option value="all">All cases</option>
      <option value="faulty">Faulty cases</option>
      <option value="clean">No curve fault</option>
    </select>
    <span id="visible"></span>
  </div>
  <main id="report">{sections}</main>
  <p id="empty" class="empty">No cases match the current filters.</p>
  <script>
    const search = document.querySelector('#search');
    const status = document.querySelector('#status');
    const visible = document.querySelector('#visible');
    const empty = document.querySelector('#empty');
    function filter() {{
      const query = search.value.trim().toLowerCase();
      const wanted = status.value;
      let shownCases = 0, shownSignatures = 0;
      document.querySelectorAll('.signature').forEach(section => {{
        let groupCases = 0;
        section.querySelectorAll('.case').forEach(row => {{
          const matchesText = !query || row.dataset.search.includes(query);
          const matchesStatus = wanted === 'all' || row.dataset.status === wanted;
          const show = matchesText && matchesStatus;
          row.hidden = !show;
          if (show) {{ shownCases++; groupCases++; }}
        }});
        section.hidden = groupCases === 0;
        if (groupCases) {{
          shownSignatures++;
          if (query || wanted !== 'all') {{
            section.querySelector('details').open = true;
          }}
        }}
      }});
      visible.textContent = `${{shownCases}} cases · ${{shownSignatures}} signatures`;
      empty.style.display = shownCases ? 'none' : 'block';
    }}
    search.addEventListener('input', filter);
    status.addEventListener('change', filter);
    filter();
  </script>
</body>
</html>
"""


def main() -> int:
    args = parse_args()
    cases = [_case(entry) for entry in _entries(args.benchmark)]
    document = _document(cases, args.benchmark)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(document, encoding="utf-8")
    print(f"Wrote {args.output} ({len(cases)} cases)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
