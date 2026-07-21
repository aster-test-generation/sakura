# flake8: noqa: E501

HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Sakura Description Grader</title>
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
      --accent-soft: #f5e7ed;
      --accent-line: #e6c5d3;
      --flag: #a8611f;
      --danger: #b3261e;
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
    .shell { height: 100vh; display: grid; grid-template-rows: auto 1fr auto; }
    header {
      min-height: 66px;
      padding: 13px 24px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      border-bottom: 1px solid var(--line);
      background: rgba(246, 247, 249, .88);
      backdrop-filter: blur(14px);
    }
    .brand { display: flex; gap: 12px; align-items: center; }
    .mark {
      width: 38px; height: 38px; display: grid; place-items: center;
      border-radius: 12px; color: #fff; background: var(--accent);
      font-size: 18px; box-shadow: 0 6px 16px rgba(201, 54, 107, .28);
    }
    h1 { font-size: 17px; margin: 0; letter-spacing: .01em; font-weight: 650; }
    .subtle { color: var(--muted); font-size: 13.5px; }
    .progress-wrap { min-width: 230px; text-align: right; }
    .progress-line { display: flex; justify-content: flex-end; gap: 10px; align-items: center; }
    .progress-line strong { font-size: 14px; }
    .progress { height: 6px; margin-top: 7px; border-radius: 9px; overflow: hidden; background: #d9dee8; }
    .progress > div { height: 100%; width: 0; border-radius: 9px; background: var(--accent); transition: width .2s; }
    main { min-height: 0; padding: 18px 20px; display: grid; grid-template-columns: minmax(0, 1.15fr) minmax(400px, 1fr); gap: 18px; }
    .panel { min-height: 0; border: 1px solid var(--line); border-radius: 16px; background: var(--panel); box-shadow: var(--shadow); overflow: hidden; }
    .code-panel { display: grid; grid-template-rows: auto 1fr; }
    .panel-head { padding: 15px 19px; border-bottom: 1px solid var(--line); background: var(--panel-2); }
    .eyebrow { color: var(--accent-deep); text-transform: uppercase; letter-spacing: .1em; font-size: 12.5px; font-weight: 700; }
    .method { margin-top: 4px; font: 12.5px/1.5 var(--mono); color: var(--muted); overflow-wrap: anywhere; }
    pre { margin: 0; padding: 22px; min-height: 0; overflow: auto; tab-size: 4; background: var(--panel); color: #2b3648; font: 13px/1.65 var(--mono); }
    .tok-comment { color: #8a94a6; font-style: italic; }
    .tok-string { color: #2f7d46; }
    .tok-annotation { color: #b3591f; }
    .tok-number { color: #0f766e; }
    .tok-keyword { color: #6042a6; }
    .tok-type { color: #275f8f; }
    .review-panel { overflow-y: auto; padding: 20px; }
    .section + .section { margin-top: 22px; padding-top: 22px; border-top: 1px solid var(--line); }
    h2 { margin: 6px 0 10px; font: 600 19px/1.3 var(--serif); letter-spacing: .002em; color: var(--text); }
    .project-text { margin: 0; color: var(--body); font-size: 14.5px; line-height: 1.65; }
    .description { margin: 0; white-space: pre-wrap; font: 16px/1.7 var(--serif); color: #26303e; }
    .level-tabs { display: grid; grid-template-columns: repeat(3, 1fr); gap: 7px; max-width: 440px; margin: 8px 0 11px; }
    .level-tabs button { display: grid; gap: 2px; justify-items: center; align-content: center; padding: 8px 6px; border: 1px solid var(--line-strong); border-radius: 10px; background: #fbfcfe; color: var(--muted); font-size: 14px; font-weight: 650; transition: .15s ease; }
    .section-hint { margin: 6px 0 0; color: var(--muted); font-size: 13.5px; }
    .level-tabs button:hover { border-color: var(--accent); color: var(--accent-deep); }
    .level-tabs button:focus-visible { outline: 3px solid rgba(201, 54, 107, .35); outline-offset: 2px; }
    .level-tabs button.viewing { background: var(--accent); border-color: var(--accent); color: #fff; }
    .criterion { margin-top: 16px; padding: 16px; border: 1px solid var(--line); border-radius: 14px; background: var(--panel-2); }
    .criterion-title { display: flex; justify-content: space-between; gap: 10px; align-items: baseline; }
    .criterion h3 { margin: 0; font-size: 16.5px; font-weight: 650; }
    .criterion p { margin: 6px 0 13px; color: var(--muted); font-size: 14.5px; line-height: 1.55; }
    .scale { position: relative; display: grid; max-width: 640px; gap: 7px; }
    .scale input { position: absolute; opacity: 0; pointer-events: none; }
    .scale label { min-height: 44px; display: grid; place-items: center; padding: 6px 5px; border: 1px solid var(--line-strong); border-radius: 10px; color: var(--muted); background: #fbfcfe; text-align: center; font-size: 13.5px; font-weight: 650; line-height: 1.25; transition: .15s ease; }
    .scale label:hover { border-color: var(--accent); color: var(--accent-deep); transform: translateY(-1px); }
    .scale label.straddle { border-style: dashed; background: var(--panel-2); }
    .scale label.straddle:hover { border-color: var(--flag); color: var(--flag); }
    .scale input:focus-visible + label { outline: 3px solid rgba(201, 54, 107, .35); outline-offset: 2px; }
    .scale input:checked + label { color: #fff; background: var(--accent); border-color: var(--accent); }
    .scale input:checked + label.straddle { background: var(--flag); border-color: var(--flag); border-style: solid; }
    .anchor { margin-top: 11px; color: var(--muted); font-size: 14.5px; line-height: 1.5; font-style: italic; }
    .anchor.selected { color: var(--body); font-style: normal; }
    footer { padding: 13px 22px; display: flex; justify-content: space-between; gap: 12px; align-items: center; border-top: 1px solid var(--line); background: rgba(246, 247, 249, .92); }
    .actions { display: flex; gap: 10px; }
    .button { min-width: 110px; border: 1px solid var(--line-strong); border-radius: 11px; padding: 10px 15px; color: var(--text); background: #fbfcfe; font-weight: 650; }
    .button:hover:not(:disabled) { border-color: var(--accent); color: var(--accent-deep); }
    .button.primary { min-width: 150px; border-color: var(--accent); color: #fff; background: var(--accent); box-shadow: 0 5px 14px rgba(201, 54, 107, .25); }
    .button.primary:hover:not(:disabled) { color: #fff; background: var(--accent-deep); }
    .button:disabled { cursor: not-allowed; opacity: .45; }
    .save-state { color: var(--muted); font-size: 13px; }
    .overlay, .complete { position: fixed; inset: 0; z-index: 20; display: grid; place-items: center; padding: 24px; background: rgba(231, 234, 240, .85); backdrop-filter: blur(8px); }
    .overlay[hidden], .complete[hidden], .app[hidden] { display: none; }
    .dialog { max-width: 570px; padding: 30px; border: 1px solid var(--line); border-radius: 18px; background: var(--panel); box-shadow: var(--shadow); text-align: center; }
    .spinner { width: 34px; height: 34px; margin: 0 auto 15px; border: 3px solid var(--line); border-top-color: var(--accent); border-radius: 50%; animation: spin .8s linear infinite; }
    .error { color: var(--danger); white-space: pre-wrap; }
    @keyframes spin { to { transform: rotate(360deg); } }
    @media (prefers-reduced-motion: reduce) {
      .spinner { animation-duration: 1.6s; }
      .scale label, .progress > div { transition: none; }
    }
    @media (max-width: 920px) {
      .shell { height: auto; min-height: 100vh; }
      main { grid-template-columns: 1fr; }
      .code-panel { min-height: 67vh; }
      .review-panel { overflow: visible; }
      footer { position: sticky; bottom: 0; z-index: 10; }
    }
    @media (max-width: 600px) {
      header { align-items: flex-start; }
      .progress-wrap { min-width: 130px; }
      main { padding: 10px; gap: 10px; }
      footer { align-items: stretch; flex-direction: column; }
      .actions { display: grid; grid-template-columns: 1fr 1.4fr; }
      .button { min-width: 0; }
    }
  </style>
</head>
<body>
  <div class="app shell" id="app" hidden>
    <header>
      <div class="brand"><div class="mark">✿</div><div><h1>Description Grader</h1><div class="subtle" id="reviewer"></div></div></div>
      <div class="progress-wrap"><div class="progress-line"><span class="subtle" id="completed"></span><strong id="position"></strong></div><div class="progress"><div id="progressBar"></div></div></div>
    </header>
    <main>
      <section class="panel code-panel">
        <div class="panel-head"><div class="eyebrow" id="project"></div><div class="method" id="method"></div></div>
        <pre><code id="code"></code></pre>
      </section>
      <aside class="panel review-panel">
        <section class="section"><div class="eyebrow">Project context</div><h2 id="projectTitle"></h2><p class="project-text" id="projectText"></p></section>
        <section class="section">
          <div class="eyebrow">Abstraction level</div>
          <p class="section-hint">Click a level to read its definition.</p>
          <div class="level-tabs" id="levelTabs">
            <button type="button" data-level="low">Low</button>
            <button type="button" data-level="medium">Medium</button>
            <button type="button" data-level="high">High</button>
          </div>
          <p class="project-text" id="levelNote"></p>
          <p class="section-hint">Invariant across levels: each description is one self-contained paragraph, leaves the test name unspecified, and ends by listing the testing framework, assertion library, and mocking library. These traits carry no level signal.</p>
        </section>
        <section class="section"><div class="eyebrow">Generated description</div><h2>What the agent wrote</h2><p class="description" id="description"></p></section>
        <section class="section"><div class="eyebrow">Evaluation</div><h2>Rate this description</h2><div id="criteria"></div></section>
      </aside>
    </main>
    <footer><div class="save-state" id="saveState">All scores are saved when you continue.</div><div class="actions"><button class="button" id="back">Back</button><button class="button primary" id="next" disabled>Save &amp; Next</button></div></footer>
  </div>
  <div class="overlay" id="overlay"><div class="dialog"><div class="spinner" id="spinner"></div><h2 id="overlayTitle">Preparing your review</h2><p class="subtle" id="overlayText">Loading description data and source context.</p><p class="error" id="error"></p><button class="button" id="retry" hidden>Retry</button></div></div>
  <div class="complete" id="complete" hidden><div class="dialog"><div class="mark" style="margin:0 auto 16px">✓</div><div class="eyebrow">Session complete</div><h2>Every selected description is graded</h2><p class="subtle" id="completeText"></p><button class="button primary" id="close">Close Viewer</button></div></div>
  <script>
    const token = new URLSearchParams(location.search).get('token');
    const FIDELITY_OPTIONS = [
      { value: 1, label: 'Strongly Disagree', anchor: 'The description contradicts or entirely misses the tested behavior.' },
      { value: 2, label: 'Disagree', anchor: 'Inaccuracies or omissions would mislead a reader about what the test verifies.' },
      { value: 3, label: 'Agree', anchor: 'The description conveys what the test verifies, with only minor inaccuracies or omissions.' },
      { value: 4, label: 'Strongly Agree', anchor: 'The description fully preserves the essential behavior and outcome, including order where it matters.' }
    ];
    const PERCEIVED_OPTIONS = [
      { value: 'low', label: 'Low', anchor: 'Reads clearly as a low-level, implementation-complete description.' },
      { value: 'low_medium', label: 'Low / Medium', straddle: true, anchor: 'Straddles low and medium. The description mixes exact implementation detail with architectural phrasing and does not settle at either level.' },
      { value: 'medium', label: 'Medium', anchor: 'Reads clearly as a medium-level, architectural description.' },
      { value: 'medium_high', label: 'Medium / High', straddle: true, anchor: 'Straddles medium and high. The description mixes architectural phrasing with business-level intent and does not settle at either level.' },
      { value: 'high', label: 'High', anchor: 'Reads clearly as a high-level, business-focused description.' }
    ];
    const CRITERIA = [
      {
        key: 'fidelity',
        title: 'Fidelity',
        range: '1 to 4',
        help: 'How strongly do you agree that the description accurately preserves the tested behavior and outcome?',
        options: FIDELITY_OPTIONS,
        defaultAnchor: 'Select a rating to view its meaning.'
      },
      {
        key: 'perceived_level',
        title: 'Perceived Abstraction',
        range: 'Low to High',
        help: 'Judged from the description alone, which abstraction level does it read as? The dashed options mark a description that straddles two levels without fitting either.',
        options: PERCEIVED_OPTIONS,
        defaultAnchor: 'Select the level this description reads as.'
      }
    ];
    const levelInfo = {
      low: 'An implementation-complete specification. It uses the exact class, method, and variable names from the code and exact literal values for every input. Helper logic is fully unwrapped and inlined step by step, application code is specified as exact method invocations with their arguments while its internal behavior is omitted, each chained call is enumerated in order, and every assertion is listed with its exact API and expected value.',
      medium: 'Architectural guidance that leaves room for implementation choices. It refers to classes and methods by semantic descriptors, using variable names only where needed for disambiguation. Helpers are described by their intent without implementation details, application code is a black box described by its invocation and observable effects, call chains are collapsed into intent-based logical operations, inputs are characterized by type, constraints, or characteristics, and each assertion is described by its intent with references to the relevant variables.',
      high: 'A business-level requirement. It uses business-level entities and concepts only, with no technical identifiers apart from the closing framework and library listing. Helpers and method calls are invisible: only resulting states or preconditions appear, and call chains are reduced to their final state. Inputs are high-level domain archetypes and scenarios, and verification is framed as observable business outcomes without technical detail.'
    };
    let session, entry, position = 0, selected = {}, retryAction = null;
    const el = id => document.getElementById(id);

    const escapeHtml = text => text.replace(/[&<>]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));
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
        html += escapeHtml(code.slice(last, match.index));
        const cls = TOKEN_CLASSES[match.slice(1).findIndex(group => group !== undefined)];
        html += `<span class="${cls}">${escapeHtml(match[0])}</span>`;
        last = JAVA_TOKENS.lastIndex;
      }
      return html + escapeHtml(code.slice(last));
    }

    async function api(path, options = {}) {
      const response = await fetch(path, { ...options, headers: { 'Content-Type': 'application/json', 'X-Session-Token': token, ...(options.headers || {}) } });
      const body = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(body.error || `Request failed with status ${response.status}`);
      return body;
    }
    function busy(title, text) {
      el('overlayTitle').textContent = title;
      el('overlayText').textContent = text;
      el('error').textContent = '';
      el('retry').hidden = true;
      el('spinner').hidden = false;
      el('overlay').hidden = false;
    }
    function fail(error, retry) {
      retryAction = retry;
      el('overlayTitle').textContent = 'Could not load this entry';
      el('overlayText').textContent = 'The session is still intact. Fix the reported issue or retry the request.';
      el('error').textContent = error.message || String(error);
      el('spinner').hidden = true;
      el('retry').hidden = false;
      el('overlay').hidden = false;
    }
    function showLevel(level) {
      for (const tab of el('levelTabs').children) {
        tab.classList.toggle('viewing', tab.dataset.level === level);
      }
      el('levelNote').textContent = levelInfo[level];
    }
    function renderLevel() {
      for (const tab of el('levelTabs').children) tab.classList.remove('viewing');
      el('levelNote').textContent = 'Levels range from implementation-complete detail (low) to business-level intent (high).';
    }
    function buildCriterion({ key, title, range, help, options, defaultAnchor }) {
      const box = document.createElement('div'); box.className = 'criterion';
      box.innerHTML = `<div class="criterion-title"><h3>${title}</h3><span class="subtle">${range}</span></div><p>${help}</p>`;
      const scale = document.createElement('div'); scale.className = 'scale';
      scale.style.gridTemplateColumns = `repeat(${options.length}, 1fr)`;
      const anchor = document.createElement('div'); anchor.className = 'anchor';
      for (const option of options) {
        const input = document.createElement('input'); input.type = 'radio'; input.name = key; input.id = `${key}-${option.value}`; input.value = option.value;
        input.checked = selected[key] === option.value;
        const label = document.createElement('label'); label.htmlFor = input.id; label.textContent = option.label; label.title = option.anchor;
        if (option.straddle) label.classList.add('straddle');
        input.addEventListener('change', () => { selected[key] = option.value; anchor.textContent = option.anchor; anchor.classList.add('selected'); updateNext(); });
        scale.append(input, label);
      }
      const current = options.find(option => option.value === selected[key]);
      if (current) { anchor.textContent = current.anchor; anchor.classList.add('selected'); }
      else anchor.textContent = defaultAnchor;
      box.append(scale, anchor);
      return box;
    }
    function renderCriteria() {
      el('criteria').replaceChildren(...CRITERIA.map(buildCriterion));
    }
    function updateNext() {
      const ready = Number.isInteger(selected.fidelity) && typeof selected.perceived_level === 'string';
      el('next').disabled = !ready;
    }
    function renderEntry(data) {
      entry = data; position = data.position; selected = data.grades ? { ...data.grades } : {};
      el('project').textContent = data.project_name;
      el('projectTitle').textContent = data.project_name;
      el('projectText').textContent = data.project_description;
      el('method').textContent = `${data.qualified_class_name} · ${data.method_signature} · ID ${data.id}`;
      renderLevel();
      el('code').innerHTML = highlightJava(data.code_context);
      el('description').textContent = data.description;
      el('position').textContent = `${position + 1} / ${data.total}`;
      el('completed').textContent = `${session.completed} graded`;
      el('progressBar').style.width = `${(session.completed / data.total) * 100}%`;
      el('back').disabled = position === 0;
      el('next').textContent = position === data.total - 1 ? 'Save & Finish' : 'Save & Next';
      renderCriteria(); updateNext();
      el('overlay').hidden = true; el('app').hidden = false;
      document.querySelector('.review-panel').scrollTop = 0;
      document.querySelector('pre').scrollTop = 0;
      if (position + 1 < data.total) api(`/api/entries/${position + 1}`).catch(() => {});
    }
    async function loadEntry(index) {
      busy('Preparing code context', 'A missing project analysis may be built and cached on first use.');
      try { renderEntry(await api(`/api/entries/${index}`)); }
      catch (error) { fail(error, () => loadEntry(index)); }
    }
    async function initialize() {
      try {
        session = await api('/api/session');
        el('reviewer').textContent = `Reviewer ${session.user}`;
        if (session.resume_position >= session.total) return showComplete();
        await loadEntry(session.resume_position);
      } catch (error) { fail(error, initialize); }
    }
    async function saveAndNext() {
      el('next').disabled = true; el('saveState').textContent = 'Saving scores…';
      try {
        session = await api(`/api/entries/${entry.id}/grades`, { method: 'PUT', body: JSON.stringify(selected) });
        el('saveState').textContent = 'Scores saved.';
        if (position === entry.total - 1) showComplete(); else await loadEntry(position + 1);
      } catch (error) { el('saveState').textContent = 'Scores were not saved.'; fail(error, saveAndNext); }
    }
    function showComplete() {
      el('app').hidden = true; el('overlay').hidden = true; el('complete').hidden = false;
      el('completeText').textContent = `${session.total} entries saved to ${session.output_path}`;
    }
    el('levelTabs').addEventListener('click', event => {
      const tab = event.target.closest('button');
      if (tab) showLevel(tab.dataset.level);
    });
    el('retry').addEventListener('click', () => retryAction && retryAction());
    el('back').addEventListener('click', () => loadEntry(position - 1));
    el('next').addEventListener('click', saveAndNext);
    el('close').addEventListener('click', async () => {
      el('close').disabled = true; el('close').textContent = 'Closing…';
      try { await api('/api/shutdown', { method: 'POST', body: '{}' }); window.close(); el('completeText').textContent = 'Viewer stopped. You can close this tab.'; }
      catch (error) { el('completeText').textContent = error.message; }
    });
    initialize();
  </script>
</body>
</html>
"""
