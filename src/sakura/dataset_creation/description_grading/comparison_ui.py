# flake8: noqa: E501

import json
from typing import Any

from sakura.dataset_creation.description_grading.grading_criteria import (
    FIDELITY_OPTIONS,
    PERCEIVED_OPTIONS,
)

# Self-contained static report: the comparison payload is embedded as JSON and
# rendered entirely client-side, so the file can be archived or shared without
# a running server. The rubric labels come from grading_criteria.py and the
# data from comparison.build_comparison_payload, spliced in via unique
# __TOKEN__ placeholders (str.replace, because the HTML/JS is full of braces).
HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Sakura Grade Comparison</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #e7eaf0;
      --panel: #f6f7f9;
      --panel-2: #edf0f5;
      --line: #d8dde7;
      --line-strong: #c3cbd8;
      --text: #1f2733;
      --body: #333e4e;
      --muted: #55647a;
      --accent: #c9366b;
      --accent-deep: #a92857;
      --good: #2f7d46;
      --ok: #2c6e63;
      --warn: #a8611f;
      --bad: #b3261e;
      --shadow: 0 1px 2px rgba(31, 39, 51, .04), 0 10px 32px rgba(31, 39, 51, .07);
      --serif: Charter, "Iowan Old Style", "Palatino Linotype", Georgia, serif;
      --mono: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background:
        radial-gradient(circle at 12% -4%, rgba(201, 54, 107, .05), transparent 26rem),
        radial-gradient(circle at 96% 104%, rgba(63, 125, 111, .05), transparent 30rem),
        var(--bg);
      color: var(--text);
      font: 15px/1.55 Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      -webkit-font-smoothing: antialiased;
    }
    button, input { font: inherit; }
    button { cursor: pointer; }
    header {
      padding: 16px 24px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      flex-wrap: wrap;
      border-bottom: 1px solid var(--line);
      background: rgba(246, 247, 249, .88);
      backdrop-filter: blur(14px);
      position: sticky;
      top: 0;
      z-index: 10;
    }
    .brand { display: flex; gap: 12px; align-items: center; }
    .mark {
      width: 38px; height: 38px; display: grid; place-items: center;
      border-radius: 12px; color: #fff; background: var(--accent);
      font-size: 18px; box-shadow: 0 6px 16px rgba(201, 54, 107, .28);
    }
    h1 { font-size: 17px; margin: 0; letter-spacing: .01em; font-weight: 650; }
    .subtle { color: var(--muted); font-size: 13.5px; }
    .raters { display: flex; gap: 8px; flex-wrap: wrap; }
    .rater-chip { display: inline-flex; gap: 7px; align-items: center; padding: 6px 11px; border: 1px solid var(--line-strong); border-radius: 999px; background: #fbfcfe; font-size: 13px; font-weight: 650; }
    .rater-chip .kind { font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: .07em; padding: 2px 7px; border-radius: 999px; color: #fff; }
    .kind-human { background: var(--accent); }
    .kind-agent { background: #275f8f; }
    main { max-width: 1180px; margin: 0 auto; padding: 22px 20px 60px; display: grid; gap: 20px; }
    .panel { border: 1px solid var(--line); border-radius: 16px; background: var(--panel); box-shadow: var(--shadow); overflow: hidden; }
    .panel-head { padding: 15px 20px; border-bottom: 1px solid var(--line); background: var(--panel-2); display: flex; justify-content: space-between; align-items: baseline; gap: 12px; flex-wrap: wrap; }
    .eyebrow { color: var(--accent-deep); text-transform: uppercase; letter-spacing: .1em; font-size: 12.5px; font-weight: 700; }
    .panel-body { padding: 18px 20px; }
    h2 { margin: 2px 0 0; font: 600 19px/1.3 var(--serif); color: var(--text); }
    h3 { margin: 22px 0 8px; font-size: 15px; font-weight: 650; }
    h3:first-child { margin-top: 0; }
    .hint { margin: 2px 0 10px; color: var(--muted); font-size: 13px; }
    .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 12px; }
    .card { padding: 14px 16px; border: 1px solid var(--line); border-radius: 13px; background: var(--panel-2); }
    .card .label { color: var(--muted); font-size: 12.5px; font-weight: 650; text-transform: uppercase; letter-spacing: .06em; }
    .card .value { margin-top: 4px; font-size: 24px; font-weight: 700; }
    .card .note { margin-top: 2px; color: var(--muted); font-size: 12.5px; }
    .table-wrap { overflow-x: auto; }
    table { border-collapse: collapse; width: 100%; font-size: 13.5px; }
    th, td { padding: 7px 11px; text-align: right; border-bottom: 1px solid var(--line); white-space: nowrap; }
    th { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .05em; font-weight: 700; }
    th:first-child, td:first-child { text-align: left; }
    td.name { font-weight: 650; }
    tr:last-child td { border-bottom: none; }
    .metric-good { color: var(--good); font-weight: 650; }
    .metric-ok { color: var(--ok); font-weight: 650; }
    .metric-warn { color: var(--warn); font-weight: 650; }
    .metric-bad { color: var(--bad); font-weight: 650; }
    .metric-na { color: var(--muted); }
    tr.truth-row td { border-top: 2px solid var(--line-strong); color: var(--accent-deep); font-weight: 650; background: rgba(201, 54, 107, .04); }
    .matrices { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 14px; }
    .matrix { border: 1px solid var(--line); border-radius: 13px; background: var(--panel-2); padding: 13px 15px; }
    .matrix .title { font-weight: 650; font-size: 13.5px; margin-bottom: 8px; }
    .matrix table { font-size: 12.5px; }
    .matrix td { text-align: center; }
    .matrix td.count { border-radius: 6px; }
    .filters { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
    .filters button { padding: 7px 13px; border: 2px solid var(--line-strong); border-radius: 999px; background: #fbfcfe; color: var(--muted); font-size: 13px; font-weight: 650; transition: .15s ease; }
    .filters button:hover { border-color: var(--accent); color: var(--accent-deep); }
    .filters button.active { background: var(--accent); border-color: var(--accent); color: #fff; }
    .filters input { padding: 7px 12px; border: 2px solid var(--line-strong); border-radius: 999px; background: #fbfcfe; min-width: 210px; }
    .filters input:focus-visible { outline: 3px solid rgba(201, 54, 107, .35); outline-offset: 1px; }
    .entries { display: grid; gap: 14px; }
    .entry { border: 1px solid var(--line); border-radius: 14px; background: var(--panel-2); overflow: hidden; }
    .entry-head { padding: 12px 16px; display: flex; justify-content: space-between; gap: 12px; align-items: baseline; flex-wrap: wrap; border-bottom: 1px solid var(--line); }
    .entry-id { font-weight: 700; font-size: 14px; }
    .entry-method { font: 12px/1.5 var(--mono); color: var(--muted); overflow-wrap: anywhere; }
    .truth-badge { padding: 4px 11px; border-radius: 999px; font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: .06em; color: #fff; background: var(--accent-deep); }
    .entry-body { padding: 13px 16px; }
    .entry-desc { margin: 0 0 12px; white-space: pre-wrap; font: 14.5px/1.65 var(--serif); color: #26303e; }
    .delta-0 { color: var(--good); font-weight: 650; }
    .delta-1 { color: var(--warn); font-weight: 650; }
    .delta-2 { color: var(--bad); font-weight: 650; }
    details.rationale, details.code { margin-top: 9px; border: 1px solid var(--line); border-radius: 10px; background: var(--panel); }
    details.rationale summary, details.code summary { cursor: pointer; padding: 8px 12px; font-size: 13px; font-weight: 650; color: var(--muted); }
    details.rationale[open] summary, details.code[open] summary { border-bottom: 1px solid var(--line); }
    details.code pre { margin: 0; padding: 14px 16px; max-height: 480px; overflow: auto; tab-size: 4; color: #2b3648; font: 12.5px/1.6 var(--mono); }
    .tok-comment { color: #8a94a6; font-style: italic; }
    .tok-string { color: #2f7d46; }
    .tok-annotation { color: #b3591f; }
    .tok-number { color: #0f766e; }
    .tok-keyword { color: #6042a6; }
    .tok-type { color: #275f8f; }
    .rationale-body { padding: 10px 13px; display: grid; gap: 9px; }
    .rationale-body .crit { font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: .06em; color: var(--accent-deep); }
    .rationale-body p { margin: 2px 0 0; font-size: 13.5px; line-height: 1.55; color: var(--body); }
    .empty { padding: 26px; text-align: center; color: var(--muted); }
    footer { max-width: 1180px; margin: 0 auto; padding: 0 20px 40px; color: var(--muted); font-size: 12.5px; }
    @media (max-width: 700px) {
      main { padding: 12px 10px 40px; }
      .panel-body { padding: 14px 12px; }
    }
  </style>
</head>
<body>
  <header>
    <div class="brand"><div class="mark">✿</div><div><h1>Grade Comparison</h1><div class="subtle" id="headline"></div></div></div>
    <div class="raters" id="raters"></div>
  </header>
  <main>
    <section class="panel">
      <div class="panel-head"><div><div class="eyebrow">Summary</div><h2>Agreement &amp; scoring</h2></div><div class="subtle" id="summaryNote">Metrics computed over the commonly graded entries only.</div></div>
      <div class="panel-body">
        <div class="cards" id="cards"></div>
        <div id="groupAgreement"></div>
        <h3>Perceived level vs ground truth</h3>
        <p class="hint">The perceived scale is coded 1&ndash;5; the true level sits at 1/3/5, so one step is half an abstraction level. Exact requires the precise level (straddles never match); &plusmn;&frac12; level also accepts the adjacent straddle. Bias is the mean signed error in steps (positive = read as more abstract than generated). AC&#8322; and &alpha; are the group coefficients computed with ground truth as a second rater.</p>
        <div class="table-wrap" id="truthTable"></div>
        <h3>Error distribution (perceived &minus; truth)</h3>
        <p class="hint">How far off each rater's perceived level is, in abstraction levels. 0 is an exact match; &plusmn;&frac12; is the adjacent straddle.</p>
        <div class="table-wrap" id="errorDist"></div>
        <h3>Confusion matrices (truth &times; perceived)</h3>
        <div class="matrices" id="matrices"></div>
        <h3>Score distributions</h3>
        <div class="table-wrap" id="fidelityDist"></div>
        <div class="table-wrap" style="margin-top:12px" id="perceivedDist"></div>
      </div>
    </section>
    <section class="panel">
      <div class="panel-head">
        <div><div class="eyebrow">Entries</div><h2>Side-by-side grades</h2></div>
        <div class="filters">
          <button data-filter="all" class="active">All</button>
          <button data-filter="perceived">Perceived disagreement</button>
          <button data-filter="fidelity">Fidelity disagreement</button>
          <button data-filter="exact">Exact match</button>
          <button data-filter="half">Off by &frac12; level</button>
          <button data-filter="truth">Missed truth (&gt; &frac12; level)</button>
          <input type="search" id="search" placeholder="Filter by project, class, ID&hellip;">
        </div>
      </div>
      <div class="panel-body">
        <p class="hint" id="shownCount"></p>
        <div class="entries" id="entries"></div>
      </div>
    </section>
  </main>
  <footer>Fidelity: 4-point ordinal scale (1&ndash;4). Perceived abstraction: 5-point ordinal scale including straddles. Group agreement: Gwet's AC&#8322; (ordinal weights, paradox-resistant), Fleiss' &kappa; (nominal), Krippendorff's &alpha; (ordinal). Vs truth: &kappa; lin/quad = Cohen's kappa (linearly/quadratically weighted), &rho; = Spearman's rho, AC&#8322;/&alpha; = the group coefficients with ground truth as a second rater. MAE and bias are in scale steps (one step = &frac12; abstraction level; positive bias = read as more abstract than generated).</footer>
  <script>
    const DATA = __PAYLOAD__;
    const FIDELITY_OPTIONS = __FIDELITY_OPTIONS__;
    const PERCEIVED_OPTIONS = __PERCEIVED_OPTIONS__;
    const PERCEIVED_LABEL = Object.fromEntries(PERCEIVED_OPTIONS.map(o => [o.value, o.label]));
    const PERCEIVED_CODE = Object.fromEntries(PERCEIVED_OPTIONS.map((o, i) => [o.value, i + 1]));
    const TRUTH_CODE = { low: 1, medium: 3, high: 5 };
    const el = id => document.getElementById(id);
    const esc = text => String(text).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

    const JAVA_TOKENS = new RegExp([
      /(\/\*[\s\S]*?\*\/|\/\/[^\n]*)/.source,
      /("(?:[^"\\\n]|\\.)*"|'(?:[^'\\\n]|\\.)*')/.source,
      /(@[A-Za-z_]\w*)/.source,
      /(\b\d[\d_]*(?:\.\d[\d_]*)?[fFdDlL]?\b)/.source,
      /(\b(?:abstract|assert|boolean|break|byte|case|catch|char|class|const|continue|default|do|double|else|enum|extends|final|finally|float|for|goto|if|implements|import|instanceof|int|interface|long|native|new|package|private|protected|public|record|return|short|static|strictfp|super|switch|synchronized|this|throw|throws|transient|try|var|void|volatile|while|true|false|null)\b)/.source,
      /(\b[A-Z][\w$]*\b)/.source
    ].join('|'), 'g');
    const TOKEN_CLASSES = ['tok-comment', 'tok-string', 'tok-annotation', 'tok-number', 'tok-keyword', 'tok-type'];
    function highlightJava(code) {
      let html = '', last = 0, match;
      JAVA_TOKENS.lastIndex = 0;
      while ((match = JAVA_TOKENS.exec(code))) {
        html += esc(code.slice(last, match.index));
        const cls = TOKEN_CLASSES[match.slice(1).findIndex(group => group !== undefined)];
        html += `<span class="${cls}">${esc(match[0])}</span>`;
        last = JAVA_TOKENS.lastIndex;
      }
      return html + esc(code.slice(last));
    }

    const pct = v => v == null ? '&mdash;' : (v * 100).toFixed(1) + '%';
    const num = (v, digits = 3) => v == null ? '&mdash;' : v.toFixed(digits);
    function agreementClass(v) {
      if (v == null) return 'metric-na';
      if (v >= 0.8) return 'metric-good';
      if (v >= 0.6) return 'metric-ok';
      if (v >= 0.4) return 'metric-warn';
      return 'metric-bad';
    }
    const agreeCell = v => `<td class="${agreementClass(v)}">${num(v)}</td>`;
    const rateClass = v => v == null ? 'metric-na' : v >= 0.9 ? 'metric-good' : v >= 0.75 ? 'metric-ok' : v >= 0.5 ? 'metric-warn' : 'metric-bad';
    const rateCell = v => `<td class="${rateClass(v)}">${pct(v)}</td>`;

    const SINGLE = DATA.users.length === 1;

    function renderHeader() {
      el('headline').textContent = SINGLE
        ? `${DATA.summary.n_common} graded entries by ${DATA.users[0]}, scored against ground truth`
        : `${DATA.summary.n_common} entries graded by all ${DATA.users.length} raters`;
      el('summaryNote').textContent = SINGLE
        ? 'Metrics computed over every entry this rater graded, against the generated (ground-truth) abstraction level.'
        : 'Metrics computed over the commonly graded entries only.';
      el('raters').innerHTML = DATA.users.map(user => {
        const meta = DATA.user_meta[user];
        return `<span class="rater-chip" title="${esc(meta.path)} — ${meta.graded_count} graded"><span class="kind kind-${meta.kind}">${meta.kind}</span>${esc(user)}</span>`;
      }).join('');
    }

    function renderCards() {
      const s = DATA.summary;
      let cards;
      if (SINGLE) {
        const truth = s.perceived.vs_truth[DATA.users[0]];
        const biasLevels = truth.bias == null ? '—' : `${truth.bias > 0 ? '+' : ''}${(truth.bias / 2).toFixed(2)}`;
        cards = [
          { label: 'Graded entries', value: String(s.n_common), note: 'the whole graded set for this rater' },
          { label: 'Exact match', value: pct(truth.exact), note: 'perceived = true level', cls: rateClass(truth.exact) },
          { label: 'Within ½ level', value: pct(truth.within_half_level), note: 'exact or adjacent straddle', cls: rateClass(truth.within_half_level) },
          { label: 'MAE · bias', value: `${num(truth.mae, 2)} · ${num(truth.bias, 2)}`, note: `in steps · bias ${biasLevels} levels (+ = more abstract)` },
          { label: 'AC₂ vs truth', value: num(truth.agreement.gwet_ac2), note: "Gwet's AC₂, ordinal, rater + truth", cls: agreementClass(truth.agreement.gwet_ac2) },
        ];
      } else {
        cards = [
          { label: 'Common entries', value: String(s.n_common), note: 'graded by every selected rater' },
          { label: 'AC₂ · fidelity', value: num(s.fidelity.agreement.gwet_ac2), note: "Gwet's AC₂, ordinal, all raters", cls: agreementClass(s.fidelity.agreement.gwet_ac2) },
          { label: 'AC₂ · perceived', value: num(s.perceived.agreement.gwet_ac2), note: "Gwet's AC₂, ordinal, all raters", cls: agreementClass(s.perceived.agreement.gwet_ac2) },
        ];
        for (const user of DATA.users) {
          const truth = s.perceived.vs_truth[user];
          cards.push({ label: `${user} vs truth`, value: pct(truth.within_half_level), note: `within ½ level · exact ${(truth.exact * 100).toFixed(1)}%`, cls: rateClass(truth.within_half_level) });
        }
      }
      el('cards').innerHTML = cards.map(card =>
        `<div class="card"><div class="label">${esc(card.label)}</div><div class="value ${card.cls || ''}">${card.value}</div><div class="note">${card.note}</div></div>`
      ).join('');
    }

    function renderGroupAgreement() {
      if (SINGLE) { el('groupAgreement').innerHTML = ''; return; }
      const row = (label, a) =>
        `<tr><td class="name">${label}</td>${agreeCell(a.gwet_ac2)}${agreeCell(a.fleiss_kappa)}${agreeCell(a.krippendorff_alpha)}</tr>`;
      el('groupAgreement').innerHTML = `
        <h3>Group agreement (all raters)</h3>
        <p class="hint">Gwet's AC&#8322; is the primary coefficient: its chance model stays stable when ratings concentrate on few categories, where the marginal-based Fleiss &kappa; and Krippendorff &alpha; are deflated by the kappa paradox. A large AC&#8322;&ndash;&kappa; gap signals skewed score usage, not rater disagreement.</p>
        <div class="table-wrap"><table>
          <thead><tr><th>Scale</th><th>Gwet AC&#8322; (ordinal)</th><th>Fleiss &kappa; (nominal)</th><th>Krippendorff &alpha; (ordinal)</th></tr></thead>
          <tbody>${row('Fidelity (1&ndash;4)', DATA.summary.fidelity.agreement)}${row('Perceived abstraction (5-point)', DATA.summary.perceived.agreement)}</tbody>
        </table></div>`;
    }

    function renderTruthTable() {
      const rows = DATA.users.map(user => {
        const m = DATA.summary.perceived.vs_truth[user];
        const bias = m.bias == null ? '&mdash;' : `${m.bias > 0 ? '+' : ''}${m.bias.toFixed(2)}`;
        return `<tr><td class="name">${esc(user)}</td><td>${m.n}</td>${rateCell(m.exact)}${rateCell(m.within_half_level)}<td>${num(m.mae, 2)}</td><td>${bias}</td>${agreeCell(m.spearman)}${agreeCell(m.kappa_linear)}${agreeCell(m.kappa_quadratic)}${agreeCell(m.agreement.gwet_ac2)}${agreeCell(m.agreement.krippendorff_alpha)}</tr>`;
      }).join('');
      el('truthTable').innerHTML = `<table><thead><tr><th>Rater</th><th>N</th><th>Exact</th><th>&plusmn;&frac12; level</th><th>MAE</th><th>Bias</th><th>&rho;</th><th>&kappa; lin</th><th>&kappa; quad</th><th>AC&#8322;</th><th>&alpha;</th></tr></thead><tbody>${rows}</tbody></table>`;
    }

    function renderErrorDist() {
      const steps = [-4, -3, -2, -1, 0, 1, 2, 3, 4];
      const levelLabel = step => {
        if (step === 0) return '0';
        const magnitude = Math.abs(step) / 2;
        return `${step > 0 ? '+' : '−'}${magnitude % 1 ? (magnitude > 1 ? `${Math.floor(magnitude)}½` : '½') : magnitude}`;
      };
      const header = steps.map(step => `<th>${levelLabel(step)}</th>`).join('');
      const max = Math.max(1, ...DATA.users.flatMap(user =>
        steps.map(step => DATA.summary.perceived.vs_truth[user].signed_counts[String(step)] || 0)));
      const rows = DATA.users.map(user => {
        const counts = DATA.summary.perceived.vs_truth[user].signed_counts;
        return `<tr><td class="name">${esc(user)}</td>` + steps.map(step => {
          const count = counts[String(step)] || 0;
          const background = count ? `rgba(${step === 0 ? '47, 125, 70' : Math.abs(step) === 1 ? '168, 97, 31' : '201, 54, 107'}, ${(0.12 + 0.5 * count / max).toFixed(2)})` : 'transparent';
          return `<td style="text-align:center;background:${background}">${count || ''}</td>`;
        }).join('') + '</tr>';
      }).join('');
      el('errorDist').innerHTML = `<table><thead><tr><th title="perceived minus truth, in abstraction levels">Rater</th>${header}</tr></thead><tbody>${rows}</tbody></table>`;
    }

    function renderMatrices() {
      el('matrices').innerHTML = DATA.users.map(user => {
        const confusion = DATA.summary.perceived.vs_truth[user].confusion;
        const max = Math.max(1, ...confusion.flat());
        const header = PERCEIVED_OPTIONS.map(o => `<th>${esc(o.label)}</th>`).join('');
        const rows = ['low', 'medium', 'high'].map((level, i) =>
          `<tr><td class="name">${level}</td>` + confusion[i].map((count, j) => {
            const onDiagonal = TRUTH_CODE[level] === j + 1;
            const background = count ? `rgba(${onDiagonal ? '47, 125, 70' : '201, 54, 107'}, ${(0.12 + 0.5 * count / max).toFixed(2)})` : 'transparent';
            return `<td class="count" style="background:${background}">${count || ''}</td>`;
          }).join('') + '</tr>'
        ).join('');
        return `<div class="matrix"><div class="title">${esc(user)}</div><div class="table-wrap"><table><thead><tr><th title="truth level (rows) by perceived level (columns)">truth</th>${header}</tr></thead><tbody>${rows}</tbody></table></div></div>`;
      }).join('');
    }

    function renderDistributions() {
      const fidelityRows = DATA.users.map(user => {
        const stats = DATA.summary.fidelity.per_user[user];
        return `<tr><td class="name">${esc(user)}</td>` +
          FIDELITY_OPTIONS.map(o => `<td>${stats.counts[String(o.value)] || 0}</td>`).join('') +
          `<td>${num(stats.mean, 2)}</td></tr>`;
      }).join('');
      el('fidelityDist').innerHTML = `<table><thead><tr><th>Fidelity</th>${FIDELITY_OPTIONS.map(o => `<th title="${esc(o.label)}">${o.value}</th>`).join('')}<th>Mean</th></tr></thead><tbody>${fidelityRows}</tbody></table>`;
      const perceivedRows = DATA.users.map(user => {
        const stats = DATA.summary.perceived.per_user[user];
        return `<tr><td class="name">${esc(user)}</td>` + PERCEIVED_OPTIONS.map(o => `<td>${stats.counts[o.value] || 0}</td>`).join('') + '</tr>';
      }).join('');
      const truthCounts = {};
      for (const entry of DATA.entries) truthCounts[entry.true_level] = (truthCounts[entry.true_level] || 0) + 1;
      const truthRow = '<tr class="truth-row"><td class="name">ground truth</td>' +
        PERCEIVED_OPTIONS.map(o => `<td>${o.value in TRUTH_CODE ? truthCounts[o.value] || 0 : '·'}</td>`).join('') + '</tr>';
      el('perceivedDist').innerHTML = `<table><thead><tr><th>Perceived</th>${PERCEIVED_OPTIONS.map(o => `<th>${esc(o.label)}</th>`).join('')}</tr></thead><tbody>${perceivedRows}${truthRow}</tbody></table>`;
    }

    function deltaMarkup(perceived, truth) {
      const steps = PERCEIVED_CODE[perceived] - TRUTH_CODE[truth];
      const magnitude = Math.min(Math.abs(steps), 2);
      const label = steps === 0 ? 'match' : `${steps > 0 ? '+' : '−'}${Math.abs(steps) === 1 ? '½' : Math.abs(steps) / 2} level${Math.abs(steps) > 2 ? 's' : ''}`;
      return `<span class="delta-${magnitude}">${label}</span>`;
    }

    function entryFlags(entry) {
      const perceivedValues = new Set(DATA.users.map(u => entry.grades[u].perceived_level));
      const fidelityValues = new Set(DATA.users.map(u => entry.grades[u].fidelity));
      const deltas = DATA.users.map(u => Math.abs(PERCEIVED_CODE[entry.grades[u].perceived_level] - TRUTH_CODE[entry.true_level]));
      return {
        perceived: perceivedValues.size > 1,
        fidelity: fidelityValues.size > 1,
        exact: deltas.every(d => d === 0),
        half: deltas.some(d => d === 1),
        truth: deltas.some(d => d >= 2),
      };
    }

    function entryMarkup(entry) {
      const rows = DATA.users.map(user => {
        const grade = entry.grades[user];
        return `<tr><td class="name">${esc(user)}</td><td>${grade.fidelity}</td><td style="text-align:left">${esc(PERCEIVED_LABEL[grade.perceived_level])}</td><td>${deltaMarkup(grade.perceived_level, entry.true_level)}</td></tr>`;
      }).join('');
      const code = entry.code_context
        ? `<details class="code"><summary>Test code context</summary><pre><code>${highlightJava(entry.code_context)}</code></pre></details>`
        : '<p class="hint">Code context unavailable for this entry.</p>';
      const rationales = Object.entries(entry.rationales || {}).map(([user, byCriterion]) => {
        const blocks = Object.entries(byCriterion).map(([criterion, text]) =>
          `<div><div class="crit">${esc(criterion.replace('_', ' '))}</div><p>${esc(text)}</p></div>`
        ).join('');
        return `<details class="rationale"><summary>Rationale &middot; ${esc(user)}</summary><div class="rationale-body">${blocks}</div></details>`;
      }).join('');
      return `<article class="entry">
        <div class="entry-head">
          <div><span class="entry-id">#${entry.position + 1} &middot; ID ${entry.id}</span><div class="entry-method">${esc(entry.project_name)} &middot; ${esc(entry.qualified_class_name)} &middot; ${esc(entry.method_signature)}</div></div>
          <span class="truth-badge">true: ${esc(entry.true_level)}</span>
        </div>
        <div class="entry-body">
          <p class="entry-desc">${esc(entry.description)}</p>
          <div class="table-wrap"><table><thead><tr><th>Rater</th><th>Fidelity</th><th style="text-align:left">Perceived</th><th>&Delta; vs truth</th></tr></thead><tbody>${rows}</tbody></table></div>
          ${code}
          ${rationales}
        </div>
      </article>`;
    }

    let activeFilter = 'all';
    function renderEntries() {
      const query = el('search').value.trim().toLowerCase();
      const visible = DATA.entries.filter(entry => {
        const flags = entryFlags(entry);
        if (activeFilter !== 'all' && !flags[activeFilter]) return false;
        if (!query) return true;
        return [entry.project_name, entry.qualified_class_name, entry.method_signature, String(entry.id)]
          .some(field => field.toLowerCase().includes(query));
      });
      el('shownCount').textContent = `Showing ${visible.length} of ${DATA.entries.length} common entries.`;
      el('entries').innerHTML = visible.length
        ? visible.map(entryMarkup).join('')
        : '<div class="empty">No entries match the current filter.</div>';
    }

    // Rater-disagreement filters need at least two raters; the truth-offset
    // filters replace them when a single rater is compared to ground truth.
    const hiddenFilters = SINGLE ? ['perceived', 'fidelity'] : ['exact', 'half'];
    hiddenFilters.forEach(name => document.querySelector(`.filters button[data-filter="${name}"]`).remove());
    document.querySelectorAll('.filters button').forEach(button => {
      button.addEventListener('click', () => {
        document.querySelectorAll('.filters button').forEach(b => b.classList.remove('active'));
        button.classList.add('active');
        activeFilter = button.dataset.filter;
        renderEntries();
      });
    });
    el('search').addEventListener('input', renderEntries);

    renderHeader();
    renderCards();
    renderGroupAgreement();
    renderTruthTable();
    renderErrorDist();
    renderMatrices();
    renderDistributions();
    renderEntries();
  </script>
</body>
</html>
"""


def _embed_json(value: Any) -> str:
    # "</" must not appear literally inside an inline <script> block.
    return json.dumps(value, ensure_ascii=False).replace("</", "<\\/")


def render_report(payload: dict[str, Any]) -> str:
    """The comparison payload rendered as a self-contained HTML document."""
    return (
        HTML_TEMPLATE.replace("__PAYLOAD__", _embed_json(payload))
        .replace("__FIDELITY_OPTIONS__", _embed_json(FIDELITY_OPTIONS))
        .replace("__PERCEIVED_OPTIONS__", _embed_json(PERCEIVED_OPTIONS))
    )
