"""Build a side-by-side HTML comparison report of CoT consistency judges."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

ROOT = Path("/project/AV/alpasim")

# Match tools/check_consistency_accuracy.py: score > 2 => consistent.
LLM_INCONSISTENT_THRESHOLD = 2


def load_json(p: Path):
    with p.open() as f:
        return json.load(f)


def first_existing(*paths: Path) -> Path:
    for p in paths:
        if p.exists():
            return p
    raise FileNotFoundError("None of these paths exist: " + ", ".join(str(p) for p in paths))


def resolve_paths(dataset: str, output: str | None) -> dict[str, Path]:
    return {
        "dataset": dataset,
        "gt": first_existing(ROOT / f"{dataset}.json", ROOT / "experiments" / "archive" / f"{dataset}.json"),
        "llm_plain": first_existing(
            ROOT / f"{dataset}.cot_consistency.json",
            ROOT / "experiments" / "archive" / f"{dataset}.cot_consistency.json",
        ),
        "llm_lane": first_existing(
            ROOT / f"{dataset}.cot_consistency_lane_center.json",
            ROOT / "experiments" / "archive" / f"{dataset}.cot_consistency_lane_center.json",
        ),
        "rule": first_existing(
            ROOT / f"{dataset}.rule_consistency.json",
            ROOT / "experiments" / "archive" / f"{dataset}.rule_consistency.json",
        ),
        "out": Path(output) if output else ROOT / "experiments" / "results" / f"{dataset}.comparison.html",
    }


def llm_label(score):
    if score is None:
        return ("unknown", "n/a")
    if score <= LLM_INCONSISTENT_THRESHOLD:
        return ("inconsistent", f"score={score}")
    return ("consistent", f"score={score}")


def rule_label(report):
    if not report:
        return ("unknown", "n/a")
    label = report.get("label", "unknown")
    is_consistent = label == "consistent"
    return ("consistent" if is_consistent else "inconsistent", label)


def agree_class(judge_label: str, gt_label: str) -> str:
    if judge_label == "unknown":
        return "agree-na"
    return "agree-yes" if judge_label == gt_label else "agree-no"


def fmt_list(xs):
    if not xs:
        return ""
    if isinstance(xs, str):
        return xs
    return ", ".join(str(x) for x in xs)


def esc(s) -> str:
    if s is None:
        return ""
    return html.escape(str(s))


def build_rows(paths: dict[str, Path]):
    gt_list = load_json(paths["gt"])
    llm_plain = {r["clip_id"]: r for r in load_json(paths["llm_plain"])["results"]}
    llm_lane = {r["clip_id"]: r for r in load_json(paths["llm_lane"])["results"]}
    rule = {r["clip_id"]: r for r in load_json(paths["rule"])["results"]}

    rows = []
    for gt in gt_list:
        cid = gt["clip_id"]
        gt_consistent = bool(gt.get("cot_action_consistency"))
        gt_label = "consistent" if gt_consistent else "inconsistent"

        lp = llm_plain.get(cid, {})
        lp_eval = (lp.get("evaluation") or {})
        lp_align = lp_eval.get("cot_output_alignment", {}) or {}
        lp_label, lp_label_detail = llm_label(lp_align.get("score"))

        ll = llm_lane.get(cid, {})
        ll_eval = (ll.get("evaluation") or {})
        ll_align = ll_eval.get("cot_output_alignment", {}) or {}
        ll_label, ll_label_detail = llm_label(ll_align.get("score"))

        rl = rule.get(cid, {})
        rl_report = rl.get("report") or {}
        rl_lon = rl_report.get("longitudinal", {}) or {}
        rl_lat = rl_report.get("lateral", {}) or {}
        rl_label_v, rl_label_detail = rule_label(rl_report)

        rows.append({
            "clip_id": cid,
            "scene": gt.get("scene_folder", ""),
            "cot": gt.get("chain_of_thought", ""),
            "rollout_mp4": gt.get("rollout_mp4", ""),

            "gt_label": gt_label,
            "gt_categories": fmt_list(gt.get("inconsistency_categories")),
            "gt_justification": fmt_list(gt.get("justification")),
            "gt_note": gt.get("note", ""),

            "lp_label": lp_label,
            "lp_label_detail": lp_label_detail,
            "lp_align_score": lp_align.get("score"),
            "lp_align_just": lp_align.get("justification", ""),

            "ll_label": ll_label,
            "ll_label_detail": ll_label_detail,
            "ll_align_score": ll_align.get("score"),
            "ll_align_just": ll_align.get("justification", ""),
            "ll_align_category": ll_align.get("category", ""),

            "rl_label": rl_label_v,
            "rl_label_detail": rl_label_detail,
            "rl_score": rl_report.get("score"),
            "rl_lon_verdict": rl_lon.get("verdict", ""),
            "rl_lon_cot": rl_lon.get("cot_label", ""),
            "rl_lon_traj": rl_lon.get("trajectory_label", ""),
            "rl_lon_traj_seq": fmt_list(rl_lon.get("trajectory_sequence")),
            "rl_lon_cot_seq": fmt_list(rl_lon.get("cot_sequence")),
            "rl_lat_verdict": rl_lat.get("verdict", ""),
            "rl_lat_cot": rl_lat.get("cot_label", ""),
            "rl_lat_traj": rl_lat.get("trajectory_label", ""),
            "rl_lat_traj_seq": fmt_list(rl_lat.get("trajectory_sequence")),
            "rl_lat_cot_seq": fmt_list(rl_lat.get("cot_sequence")),

            "lp_agree": agree_class(lp_label, gt_label),
            "ll_agree": agree_class(ll_label, gt_label),
            "rl_agree": agree_class(rl_label_v, gt_label),
        })
    return rows


def summarize(rows):
    n = len(rows)
    n_gt_consistent = sum(1 for r in rows if r["gt_label"] == "consistent")
    n_gt_inconsistent = n - n_gt_consistent

    def acc(judge_key):
        agree = sum(1 for r in rows if r[f"{judge_key}_agree"] == "agree-yes")
        valid = sum(1 for r in rows if r[f"{judge_key}_agree"] != "agree-na")
        return agree, valid

    out = {
        "n": n,
        "n_gt_consistent": n_gt_consistent,
        "n_gt_inconsistent": n_gt_inconsistent,
    }
    for key, name in [("lp", "LLM (no lane)"), ("ll", "LLM (lane center)"), ("rl", "Rule-based")]:
        agree, valid = acc(key)
        out[key] = {
            "name": name,
            "agree": agree,
            "valid": valid,
            "accuracy": (agree / valid) if valid else 0.0,
        }
    return out


CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       margin: 0; padding: 16px; background: #f5f5f7; color: #1d1d1f; }
h1 { margin: 0 0 8px; font-size: 20px; }
.subtitle { color: #555; margin-bottom: 16px; font-size: 13px; }

.controls { background: #fff; padding: 12px 16px; border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.06); margin-bottom: 12px;
            display: flex; flex-wrap: wrap; gap: 12px; align-items: center; }
.controls label { font-size: 12px; color: #444; margin-right: 4px; }
.controls select, .controls input { padding: 4px 8px; font-size: 12px;
            border: 1px solid #d2d2d7; border-radius: 4px; }
.controls input[type=text] { min-width: 220px; }

.summary { background: #fff; padding: 12px 16px; border-radius: 8px;
           box-shadow: 0 1px 3px rgba(0,0,0,0.06); margin-bottom: 12px;
           display: flex; flex-wrap: wrap; gap: 24px; font-size: 13px; }
.summary .stat { }
.summary .stat strong { display: block; font-size: 18px; }
.summary .stat .label { color: #555; font-size: 11px; text-transform: uppercase; letter-spacing: .5px; }

table { width: 100%; border-collapse: collapse; background: #fff;
        font-size: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        border-radius: 8px; overflow: hidden; table-layout: fixed; }
th, td { padding: 8px 10px; text-align: left; vertical-align: top;
         border-bottom: 1px solid #ececef; word-wrap: break-word; }
th { background: #fafafa; font-weight: 600; font-size: 11px; text-transform: uppercase;
     color: #555; letter-spacing: .3px; cursor: pointer; user-select: none; position: sticky; top: 0; }
th.sort-asc::after { content: " \\25B2"; color: #0070c9; }
th.sort-desc::after { content: " \\25BC"; color: #0070c9; }

tr.detail-row td { background: #fbfbfd; padding: 14px 18px; border-top: 0; }
tr.detail-row { display: none; }
tr.expanded + tr.detail-row { display: table-row; }

.label-pill { display: inline-block; padding: 2px 8px; border-radius: 12px;
              font-size: 11px; font-weight: 600; }
.label-consistent { background: #e3f7e8; color: #176b2c; }
.label-inconsistent { background: #fde2e1; color: #a31712; }
.label-unknown { background: #ececef; color: #555; }

.agree-yes { background: #e3f7e8; }
.agree-no  { background: #fde2e1; }
.agree-na  { background: #fff8e1; }

.cot { font-style: italic; color: #444; }
.muted { color: #777; font-size: 11px; }
.detail-grid { display: grid; grid-template-columns: 1fr 1fr 1fr 1fr; gap: 16px; }
.detail-card { background: #fff; padding: 10px 12px; border-radius: 6px;
               border: 1px solid #ececef; }
.detail-card h4 { margin: 0 0 6px; font-size: 12px; color: #1d1d1f; text-transform: uppercase; letter-spacing: .4px; }
.detail-card .field { margin-bottom: 6px; font-size: 12px; }
.detail-card .field .k { color: #777; font-size: 11px; }
.expander { color: #0070c9; cursor: pointer; user-select: none; font-size: 11px; }
.score { display: inline-block; min-width: 18px; text-align: center; padding: 1px 6px;
         border-radius: 4px; font-weight: 600; font-size: 11px; margin-left: 4px; }
.score-low { background: #fde2e1; color: #a31712; }
.score-mid { background: #fff5d6; color: #8a6300; }
.score-high { background: #e3f7e8; color: #176b2c; }
.cell-clip { font-family: ui-monospace, Menlo, monospace; font-size: 11px; }

.col-clip { width: 220px; }
.col-cot { width: 200px; }
.col-judge { width: 150px; }
.col-gt { width: 170px; }
"""

JS = """
const tbl = document.getElementById('cmp');
const rows = Array.from(tbl.querySelectorAll('tbody > tr.main'));

document.querySelectorAll('th[data-sort]').forEach((th) => {
  th.addEventListener('click', () => {
    const idx = parseInt(th.dataset.sort, 10);
    const asc = !th.classList.contains('sort-asc');
    document.querySelectorAll('th[data-sort]').forEach((x) => x.classList.remove('sort-asc', 'sort-desc'));
    th.classList.toggle('sort-asc', asc);
    th.classList.toggle('sort-desc', !asc);
    const tbody = tbl.querySelector('tbody');
    const pairs = [];
    rows.forEach((r) => {
      const detail = r.nextElementSibling;
      pairs.push([r, detail]);
    });
    pairs.sort((a, b) => {
      const ta = a[0].cells[idx]?.innerText.trim() || '';
      const tb = b[0].cells[idx]?.innerText.trim() || '';
      const na = parseFloat(ta), nb = parseFloat(tb);
      let cmp;
      if (!isNaN(na) && !isNaN(nb)) cmp = na - nb;
      else cmp = ta.localeCompare(tb);
      return asc ? cmp : -cmp;
    });
    pairs.forEach(([r, d]) => { tbody.appendChild(r); tbody.appendChild(d); });
  });
});

function applyFilters() {
  const q = document.getElementById('search').value.toLowerCase();
  const gt = document.getElementById('flt-gt').value;
  const ag = document.getElementById('flt-agree').value;
  rows.forEach((r) => {
    let show = true;
    const detail = r.nextElementSibling;
    if (q) {
      const text = (r.innerText + ' ' + (detail?.innerText || '')).toLowerCase();
      if (!text.includes(q)) show = false;
    }
    if (gt && r.dataset.gt !== gt) show = false;
    if (ag === 'any-disagree') {
      if (r.dataset.lp === 'agree-yes' && r.dataset.ll === 'agree-yes' && r.dataset.rl === 'agree-yes') show = false;
    } else if (ag === 'all-agree') {
      if (!(r.dataset.lp === 'agree-yes' && r.dataset.ll === 'agree-yes' && r.dataset.rl === 'agree-yes')) show = false;
    } else if (ag === 'lp-wrong') {
      if (r.dataset.lp !== 'agree-no') show = false;
    } else if (ag === 'll-wrong') {
      if (r.dataset.ll !== 'agree-no') show = false;
    } else if (ag === 'rl-wrong') {
      if (r.dataset.rl !== 'agree-no') show = false;
    }
    r.style.display = show ? '' : 'none';
    if (detail) detail.style.display = show && r.classList.contains('expanded') ? 'table-row' : 'none';
  });
}
['search', 'flt-gt', 'flt-agree'].forEach((id) => {
  document.getElementById(id).addEventListener('input', applyFilters);
  document.getElementById(id).addEventListener('change', applyFilters);
});

rows.forEach((r) => {
  r.querySelector('.expander')?.addEventListener('click', (e) => {
    e.stopPropagation();
    r.classList.toggle('expanded');
    const detail = r.nextElementSibling;
    if (detail) detail.style.display = r.classList.contains('expanded') ? 'table-row' : 'none';
  });
});

document.getElementById('expand-all').addEventListener('click', () => {
  rows.forEach((r) => {
    if (r.style.display === 'none') return;
    r.classList.add('expanded');
    if (r.nextElementSibling) r.nextElementSibling.style.display = 'table-row';
  });
});
document.getElementById('collapse-all').addEventListener('click', () => {
  rows.forEach((r) => {
    r.classList.remove('expanded');
    if (r.nextElementSibling) r.nextElementSibling.style.display = 'none';
  });
});
"""


def score_class(score):
    if score is None:
        return ""
    s = float(score)
    if s <= 2:
        return "score-low"
    if s <= 3:
        return "score-mid"
    return "score-high"


def score_pill(score):
    if score is None:
        return '<span class="muted">n/a</span>'
    return f'<span class="score {score_class(score)}">{score}</span>'


def label_pill(label, detail=""):
    cls = f"label-{label}" if label in ("consistent", "inconsistent") else "label-unknown"
    text = label
    if detail:
        text = f"{label} <span class='muted'>({esc(detail)})</span>"
    else:
        text = esc(label)
    return f'<span class="label-pill {cls}">{text}</span>'


def render(rows, summary, dataset: str):
    head_cells = [
        ("Clip", 0),
        ("Chain of Thought", 1),
        ("Ground Truth", 2),
        ("LLM (no lane)", 3),
        ("LLM (lane center)", 4),
        ("Rule-based", 5),
        ("Details", None),
    ]
    th_html = "".join(
        f'<th class="col-{ "clip" if i==0 else "cot" if i==1 else "gt" if i==2 else "judge"}"'
        + (f' data-sort="{i}"' if i is not None else "")
        + f'>{esc(name)}</th>'
        for name, i in head_cells
    )

    body_parts = []
    for idx, r in enumerate(rows):
        cot_short = r["cot"]
        if len(cot_short) > 140:
            cot_short = cot_short[:137] + "..."

        gt_cell = (
            f'{label_pill(r["gt_label"])}'
            + (f'<div class="muted">{esc(r["gt_categories"])}</div>' if r["gt_categories"] else "")
        )

        lp_cell = (
            f'{label_pill(r["lp_label"])} '
            f'<span class="muted">align{score_pill(r["lp_align_score"])}</span>'
        )
        ll_cell = (
            f'{label_pill(r["ll_label"])} '
            f'<span class="muted">align{score_pill(r["ll_align_score"])}</span>'
            + (f'<div class="muted">cat: {esc(r["ll_align_category"])}</div>' if r["ll_align_category"] else "")
        )
        rl_cell = (
            f'{label_pill(r["rl_label"], r["rl_label_detail"])} '
            + (f'<div class="muted">lon: {esc(r["rl_lon_verdict"])} ({esc(r["rl_lon_cot"])} vs {esc(r["rl_lon_traj"])})</div>')
            + (f'<div class="muted">lat: {esc(r["rl_lat_verdict"])} ({esc(r["rl_lat_cot"])} vs {esc(r["rl_lat_traj"])})</div>')
        )

        body_parts.append(
            f'<tr class="main" data-gt="{r["gt_label"]}" data-lp="{r["lp_agree"]}" data-ll="{r["ll_agree"]}" data-rl="{r["rl_agree"]}">'
            f'<td class="cell-clip">{esc(r["clip_id"])}<div class="muted">{esc(r["scene"])}</div></td>'
            f'<td class="cot">{esc(cot_short)}</td>'
            f'<td>{gt_cell}</td>'
            f'<td class="{r["lp_agree"]}">{lp_cell}</td>'
            f'<td class="{r["ll_agree"]}">{ll_cell}</td>'
            f'<td class="{r["rl_agree"]}">{rl_cell}</td>'
            f'<td><span class="expander">[expand]</span></td>'
            f'</tr>'
        )

        # Detail row
        detail_grid = f"""
<div class="detail-grid">
  <div class="detail-card">
    <h4>Ground truth</h4>
    <div class="field"><span class="k">label:</span> {label_pill(r['gt_label'])}</div>
    <div class="field"><span class="k">categories:</span> {esc(r['gt_categories']) or '<span class="muted">none</span>'}</div>
    <div class="field"><span class="k">justification:</span><br>{esc(r['gt_justification'])}</div>
    {f'<div class="field"><span class="k">note:</span><br>{esc(r["gt_note"])}</div>' if r['gt_note'] else ''}
    <div class="field"><span class="k">CoT:</span><br><span class="cot">{esc(r['cot'])}</span></div>
  </div>
  <div class="detail-card">
    <h4>LLM judge (no lane reference)</h4>
    <div class="field"><span class="k">verdict:</span> {label_pill(r['lp_label'], r['lp_label_detail'])}</div>
    <div class="field"><span class="k">alignment {score_pill(r['lp_align_score'])}:</span><br>{esc(r['lp_align_just'])}</div>
  </div>
  <div class="detail-card">
    <h4>LLM judge (with lane center)</h4>
    <div class="field"><span class="k">verdict:</span> {label_pill(r['ll_label'], r['ll_label_detail'])}</div>
    <div class="field"><span class="k">alignment {score_pill(r['ll_align_score'])}:</span><br>{esc(r['ll_align_just'])}</div>
    {f'<div class="field"><span class="k">category:</span> {esc(r["ll_align_category"])}</div>' if r['ll_align_category'] else ''}
  </div>
  <div class="detail-card">
    <h4>Rule-based judge</h4>
    <div class="field"><span class="k">verdict:</span> {label_pill(r['rl_label'], r['rl_label_detail'])} (score={esc(r['rl_score'])})</div>
    <div class="field"><span class="k">longitudinal:</span> {esc(r['rl_lon_verdict'])} &mdash; cot=<b>{esc(r['rl_lon_cot'])}</b>, traj=<b>{esc(r['rl_lon_traj'])}</b></div>
    <div class="field muted">cot seq: {esc(r['rl_lon_cot_seq']) or '(none)'}<br>traj seq: {esc(r['rl_lon_traj_seq']) or '(none)'}</div>
    <div class="field"><span class="k">lateral:</span> {esc(r['rl_lat_verdict'])} &mdash; cot=<b>{esc(r['rl_lat_cot'])}</b>, traj=<b>{esc(r['rl_lat_traj'])}</b></div>
    <div class="field muted">cot seq: {esc(r['rl_lat_cot_seq']) or '(none)'}<br>traj seq: {esc(r['rl_lat_traj_seq']) or '(none)'}</div>
    {f'<div class="field"><span class="k">rollout video:</span><br><span class="muted">{esc(r["rollout_mp4"])}</span></div>' if r['rollout_mp4'] else ''}
  </div>
</div>
"""
        body_parts.append(f'<tr class="detail-row"><td colspan="7">{detail_grid}</td></tr>')

    summary_html = f"""
<div class="summary">
  <div class="stat"><span class="label">Total clips</span><strong>{summary['n']}</strong></div>
  <div class="stat"><span class="label">GT consistent</span><strong>{summary['n_gt_consistent']}</strong></div>
  <div class="stat"><span class="label">GT inconsistent</span><strong>{summary['n_gt_inconsistent']}</strong></div>
  <div class="stat"><span class="label">LLM (no lane) acc</span><strong>{summary['lp']['accuracy']*100:.1f}%</strong>
    <span class="muted">{summary['lp']['agree']}/{summary['lp']['valid']}</span></div>
  <div class="stat"><span class="label">LLM (lane) acc</span><strong>{summary['ll']['accuracy']*100:.1f}%</strong>
    <span class="muted">{summary['ll']['agree']}/{summary['ll']['valid']}</span></div>
  <div class="stat"><span class="label">Rule-based acc</span><strong>{summary['rl']['accuracy']*100:.1f}%</strong>
    <span class="muted">{summary['rl']['agree']}/{summary['rl']['valid']}</span></div>
</div>
"""

    controls = """
<div class="controls">
  <label>Search:</label><input id="search" type="text" placeholder="clip id, CoT, justification..."/>
  <label>Ground truth:</label>
  <select id="flt-gt">
    <option value="">all</option>
    <option value="consistent">consistent</option>
    <option value="inconsistent">inconsistent</option>
  </select>
  <label>Agreement:</label>
  <select id="flt-agree">
    <option value="">all</option>
    <option value="all-agree">all judges agree with GT</option>
    <option value="any-disagree">at least one judge disagrees with GT</option>
    <option value="lp-wrong">LLM (no lane) disagrees with GT</option>
    <option value="ll-wrong">LLM (lane center) disagrees with GT</option>
    <option value="rl-wrong">Rule-based disagrees with GT</option>
  </select>
  <button id="expand-all">expand all</button>
  <button id="collapse-all">collapse all</button>
</div>
"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>CoT Consistency Judge Comparison &mdash; {esc(dataset)}</title>
  <style>{CSS}</style>
</head>
<body>
  <h1>CoT Consistency Judge Comparison</h1>
  <div class="subtitle">{esc(dataset)} &mdash; ground truth vs LLM (no lane), LLM (lane center), and rule-based judges.
    LLM verdicts derived from cot_output_alignment score (&le;{LLM_INCONSISTENT_THRESHOLD} =&gt; inconsistent).
    Cell background: <span class="agree-yes">&nbsp;green&nbsp;</span> agrees with GT,
    <span class="agree-no">&nbsp;red&nbsp;</span> disagrees,
    <span class="agree-na">&nbsp;yellow&nbsp;</span> missing/n.a.</div>
  {summary_html}
  {controls}
  <table id="cmp">
    <thead><tr>{th_html}</tr></thead>
    <tbody>{''.join(body_parts)}</tbody>
  </table>
  <script>{JS}</script>
</body>
</html>
"""


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset",
        nargs="?",
        default="benchmark_expanded_50",
        help="Benchmark basename, for example benchmark_expanded_75.",
    )
    parser.add_argument(
        "--output",
        help="Output HTML path. Defaults to experiments/results/<dataset>.comparison.html.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    paths = resolve_paths(args.dataset, args.output)
    rows = build_rows(paths)
    summary = summarize(rows)
    html_text = render(rows, summary, args.dataset)
    paths["out"].write_text(html_text)
    print(f"wrote {paths['out']} ({paths['out'].stat().st_size:,} bytes, {len(rows)} rows)")
    print(f"  GT: {summary['n_gt_consistent']} consistent, {summary['n_gt_inconsistent']} inconsistent")
    for k in ("lp", "ll", "rl"):
        s = summary[k]
        print(f"  {s['name']}: accuracy = {s['accuracy']*100:.1f}% ({s['agree']}/{s['valid']})")


if __name__ == "__main__":
    main()
