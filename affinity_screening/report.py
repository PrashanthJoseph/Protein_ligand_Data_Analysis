"""Self-contained HTML report.

Figures are embedded as base64 data URIs so the report is a single portable
file that survives being emailed or copied off the cluster.
"""

from __future__ import annotations

import base64
import html as html_lib
from datetime import datetime
from pathlib import Path

import pandas as pd

from .config import LOWER_IS_BETTER

STYLESHEET = """
  :root { --navy:#16324f; --blue:#224b7a; --green:#238b57;
           --red:#b33a3a; --ink:#1f2933; --muted:#5f6b76; --line:#dfe5ea;
           --paper:#ffffff; --wash:#f4f7f9; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--wash); color:var(--ink);
          font-family:Inter,Segoe UI,Arial,sans-serif; line-height:1.55; }
  .page { max-width:1180px; margin:32px auto; background:var(--paper);
           padding:46px 54px; box-shadow:0 6px 30px rgba(20,40,60,.09); }
  h1 { margin:0 0 8px; color:var(--navy); font-size:2.15rem; line-height:1.15; }
  h2 { margin:42px 0 14px; padding-bottom:7px; border-bottom:2px solid var(--line);
        color:var(--navy); font-size:1.45rem; }
  h3 { color:var(--navy); margin-top:26px; }
  .subtitle { color:var(--muted); margin-bottom:28px; }
  .cards { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:14px; }
  .card { border:1px solid var(--line); border-radius:10px; padding:16px;
           background:#fbfcfd; }
  .card .value { font-size:1.65rem; font-weight:700; color:var(--blue); }
  .card .label { color:var(--muted); font-size:.9rem; }
  .callout { padding:15px 18px; border-left:5px solid var(--blue);
              background:#eef5fb; margin:18px 0; }
  .warning { border-left-color:#b87818; background:#fff7e8; }
  .data-table { width:100%; border-collapse:collapse; margin:12px 0 26px;
                 font-size:.9rem; }
  .data-table th { text-align:left; background:var(--navy); color:white;
                    padding:9px 10px; position:sticky; top:0; }
  .data-table td { padding:8px 10px; border-bottom:1px solid var(--line);
                    vertical-align:top; }
  .data-table tr:nth-child(even) td { background:#f7f9fb; }
  .table-wrap { overflow-x:auto; }
  .plots { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:24px; }
  figure { margin:0; border:1px solid var(--line); border-radius:10px;
            overflow:hidden; background:white; }
  figure img { display:block; width:100%; height:auto; }
  figcaption { padding:9px 13px; color:var(--muted); font-size:.88rem;
                border-top:1px solid var(--line); }
  code { background:#edf1f4; padding:2px 5px; border-radius:4px; }
  .foot { margin-top:42px; padding-top:16px; border-top:1px solid var(--line);
           color:var(--muted); font-size:.84rem; }
  @media (max-width:850px) { .page { margin:0; padding:28px 20px; }
    .cards,.plots { grid-template-columns:1fr; } }
  @media print { body { background:white; } .page { margin:0; max-width:none;
    box-shadow:none; padding:20px; } figure { break-inside:avoid; }
    h2 { break-after:avoid; } }
"""

FIGURE_CAPTIONS = {
    "scatter": "Experimental affinity versus predicted score",
    "tradeoff": "Screening trade-off across the full cutoff range",
    "distributions": "Predicted outputs by experimental affinity class",
    "roc": "Strong-versus-poor ROC analysis",
    "cutoff_comparison": "Direct comparison of selected cutoffs",
}


# ------------------------------------------------------------
# formatting helpers
# ------------------------------------------------------------

def format_number(value, decimals=3, fallback="Not available"):
    if value is None or pd.isna(value):
        return fallback
    return f"{value:.{decimals}f}"


def format_p_value(value):
    if value is None or pd.isna(value):
        return "Not available"
    if value < 0.001:
        return f"{value:.2e}"
    return f"{value:.4f}"


def correlation_interpretation(value, direction):
    """Cautious, direction-aware reading of a correlation coefficient."""
    if value is None or pd.isna(value):
        return "Could not be estimated from the available data."

    magnitude = abs(value)

    if magnitude < 0.20:
        strength = "Very weak"
    elif magnitude < 0.40:
        strength = "Weak"
    elif magnitude < 0.60:
        strength = "Moderate"
    elif magnitude < 0.80:
        strength = "Strong"
    else:
        strength = "Very strong"

    if direction == LOWER_IS_BETTER:
        as_expected = value < 0
        expected_text = (
            "in the expected direction: more-negative predicted scores tend to "
            "accompany higher experimental affinity"
            if as_expected
            else "opposite to the expected direction for a score where lower is better"
        )
    else:
        as_expected = value > 0
        expected_text = (
            "in the expected direction: higher predicted values tend to accompany "
            "higher experimental affinity"
            if as_expected
            else "opposite to the expected direction for a score where higher is better"
        )

    return f"{strength} association, {expected_text}."


def auc_interpretation(value):
    if value is None or pd.isna(value):
        return "ROC AUC could not be calculated."

    if value < 0.50:
        label = "worse than random in the tested direction"
    elif value < 0.60:
        label = "little discrimination"
    elif value < 0.70:
        label = "limited discrimination"
    elif value < 0.80:
        label = "acceptable discrimination"
    elif value < 0.90:
        label = "good discrimination"
    else:
        label = "excellent discrimination"

    return f"AUC {value:.3f} indicates {label} between strong and poor binders."


def image_data_uri(path):
    if path is None or not Path(path).exists():
        return None

    encoded = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def dataframe_html(data, formats=None, max_rows=None):
    shown = data.copy()

    if max_rows is not None:
        shown = shown.head(max_rows)

    if formats:
        for column, formatter in formats.items():
            if column in shown.columns:
                shown[column] = shown[column].map(
                    lambda value: formatter(value) if pd.notna(value) else "—"
                )

    return shown.to_html(index=False, border=0, classes="data-table", escape=True)


# ------------------------------------------------------------
# report sections
# ------------------------------------------------------------

def _correlation_section(results, config):
    table = results["correlations"].copy()

    table["Interpretation"] = [
        correlation_interpretation(row["value"], row["direction"])
        for _, row in table.iterrows()
    ]

    display = table[["output", "statistic", "value", "p_value", "n", "Interpretation"]]
    display = display.rename(
        columns={
            "output": "Output",
            "statistic": "Statistic",
            "value": "Value",
            "p_value": "p-value",
            "n": "n",
        }
    )

    return dataframe_html(
        display,
        formats={"Value": lambda v: f"{v:.3f}", "p-value": format_p_value},
    )


def _roc_section(results, config):
    rows = []

    labels = {
        "affinity": f"{config.tool_name} affinity value",
        "probability": f"{config.tool_name} binary probability",
    }

    for key, roc in results["roc"].items():
        if roc is None and key == "probability":
            continue

        rows.append(
            {
                "Output": labels[key],
                "ROC AUC": roc["auc"] if roc else None,
                "Strong / poor pairs": (
                    f"{roc['n_strong']} × {roc['n_poor']} = {roc['n_pairs']}"
                    if roc else "—"
                ),
                "Youden-optimal cutoff": (
                    f"{roc['youden_cutoff']:.3g}" if roc else "—"
                ),
                "Interpretation": auc_interpretation(roc["auc"] if roc else None),
            }
        )

    return dataframe_html(
        pd.DataFrame(rows), formats={"ROC AUC": lambda v: f"{v:.3f}"}
    )


def _figure_section(figures):
    blocks = []

    for key, caption in FIGURE_CAPTIONS.items():
        uri = image_data_uri(figures.get(key))

        if uri is None:
            continue

        safe = html_lib.escape(caption)
        blocks.append(
            f'<figure><img src="{uri}" alt="{safe}">'
            f"<figcaption>{safe}</figcaption></figure>"
        )

    return "".join(blocks)


# ------------------------------------------------------------
# assembly
# ------------------------------------------------------------

def build_html(analysis, results, figures, config):
    """Render the whole report as one HTML string."""
    counts = results["counts"]
    proposed = results["proposed"]

    tool = html_lib.escape(str(config.tool_name))
    generated = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")

    probability_note = (
        "A valid optional probability column was included."
        if analysis.has_probability
        else "No valid probability column was supplied; probability analyses were skipped."
    )

    strong_loss_warning = (
        f"The proposed cutoff rejects {proposed['strong_rejected_n']} "
        "experimentally strong binder(s), so it should not be treated as a "
        "stand-alone pass/fail rule."
        if proposed["strong_rejected_n"] > 0
        else "No experimentally strong binder in this dataset is rejected by the "
             "proposed cutoff."
    )

    if config.rescue_cutoff is None:
        rescue_note = "No rescue band was configured."
    else:
        rescue_note = (
            f"Scores between {config.proposed_cutoff:g} and "
            f"{config.rescue_cutoff:g} are assigned to the rescue group."
        )

    comparison = "&lt;" if config.lower_is_better else "&gt;"

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{tool} affinity-screening report</title>
<style>{STYLESHEET}</style>
</head>
<body><main class="page">
  <h1>{tool} affinity-screening report</h1>
  <div class="subtitle">Generated {html_lib.escape(generated)} ·
  Experimental affinity is reported as {html_lib.escape(str(config.ka_col))}.</div>

  <section>
    <div class="cards">
      <div class="card"><div class="value">{counts['total']}</div>
        <div class="label">Variants analyzed</div></div>
      <div class="card"><div class="value">{counts['poor']}</div>
        <div class="label">Poor (Ka &lt; {config.poor_ka_threshold:g})</div></div>
      <div class="card"><div class="value">{counts['intermediate']}</div>
        <div class="label">Intermediate ({config.poor_ka_threshold:g}–{config.strong_ka_threshold:g})</div></div>
      <div class="card"><div class="value">{counts['strong']}</div>
        <div class="label">Strong (Ka &gt; {config.strong_ka_threshold:g})</div></div>
    </div>

    <div class="callout"><strong>Proposed operating rule:</strong> retain candidates with
      <code>{html_lib.escape(str(config.affinity_col))} {comparison} {config.proposed_cutoff:g}</code>.
      This eliminates <strong>{proposed['rejected_n']}/{proposed['total_n']}
      ({proposed['total_eliminated_pct']:.1f}%)</strong> of all candidates,
      eliminates <strong>{proposed['poor_rejected_n']}/{proposed['poor_total']}
      ({proposed['poor_eliminated_pct']:.1f}%)</strong> of poor binders, and retains
      <strong>{proposed['strong_retained_n']}/{proposed['strong_total']}
      ({proposed['strong_retained_pct']:.1f}%)</strong> of strong binders.</div>

    <div class="callout warning"><strong>Interpretation:</strong> {strong_loss_warning}
      {rescue_note} The cutoff performance describes this calibration dataset and
      should be validated on held-out antibodies before prospective use.</div>
  </section>

  <h2>Dataset and method</h2>
  <p>{analysis.n_before} input rows were provided; {analysis.n_after} rows with valid
  predicted affinity and experimental Ka were analyzed.
  {html_lib.escape(probability_note)}</p>
  <p>Experimental classes were defined as poor when Ka &lt; {config.poor_ka_threshold:g},
  intermediate when {config.poor_ka_threshold:g} ≤ Ka ≤ {config.strong_ka_threshold:g},
  and strong when Ka &gt; {config.strong_ka_threshold:g}. ROC analysis excludes the
  intermediate class and asks how well each output ranks strong binders above poor
  binders. The screening rule is <code>{html_lib.escape(config.retention_rule)}</code>
  ({html_lib.escape(config.score_hint)}).</p>

  <h2>Association with experimental affinity</h2>
  <div class="table-wrap">{_correlation_section(results, config)}</div>
  <p>Correlation measures trend agreement, not absolute prediction accuracy. Pearson r
  measures linear association with log10(Ka), while Spearman rho measures rank-order
  agreement with Ka.</p>

  <h2>Strong-versus-poor discrimination</h2>
  <div class="table-wrap">{_roc_section(results, config)}</div>
  <p>An AUC is the probability that a randomly selected strong binder receives a better
  ranking score than a randomly selected poor binder. It does not select the operating
  cutoff or quantify calibration; the Youden-optimal cutoff shown is the point that
  maximises sensitivity + specificity − 1 on this dataset, which is optimistic when
  read as a prospective threshold.</p>

  <h2>Cutoff comparison</h2>
  <div class="table-wrap">{dataframe_html(
      results["cutoff_summary"],
      formats={
          "Cutoff": lambda v: f"{v:g}",
          "Total eliminated (%)": lambda v: f"{v:.1f}",
          "Poor eliminated (%)": lambda v: f"{v:.1f}",
          "Strong retained (%)": lambda v: f"{v:.1f}",
          "Rejected pool that is poor (%)": lambda v: f"{v:.1f}",
          "Retained pool that is strong (%)": lambda v: f"{v:.1f}",
      },
  )}</div>

  <h2>Figures</h2>
  <div class="plots">{_figure_section(figures)}</div>

  <h2>Experimentally strong binders</h2>
  <div class="table-wrap">{dataframe_html(results["strong_binders"])}</div>

  <h2>Candidates failing the proposed cutoff</h2>
  <div class="table-wrap">{dataframe_html(results["rejected_candidates"])}</div>

  <h2>Complete classified dataset</h2>
  <div class="table-wrap">{dataframe_html(analysis.frame)}</div>

  <div class="foot">This report is descriptive and dataset-specific. Statistical
  significance and AUC estimates can be unstable in small samples; use external or
  cross-validated evaluation before treating the threshold as a production rule.</div>
</main></body></html>"""


def write_report(analysis, results, figures, config, path=None):
    """Write the HTML report and return its path."""
    path = Path(
        path or config.output_path(f"{config.tool_name}_affinity_screening_report.html")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_html(analysis, results, figures, config), encoding="utf-8")

    return path
