(() => {
  "use strict";
  const $ = (s) => document.querySelector(s);
  const state = { sessionId: null, responses: [], selected: -1, busy: false };
  const apiKey = (() => {
    try {
      const fromUrl = new URLSearchParams(location.search).get("key");
      if (fromUrl) localStorage.setItem("agentApiKey", fromUrl);
      return localStorage.getItem("agentApiKey") || "";
    } catch { return ""; }
  })();

  async function api(path, opts = {}) {
    const headers = { "Content-Type": "application/json", ...(apiKey ? { "X-API-Key": apiKey } : {}) };
    const res = await fetch(path, { ...opts, headers });
    const body = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(body.detail || `HTTP ${res.status}`);
    return body;
  }

  // ------------------------------------------------------------ markdown (escape first, then format)
  const esc = (s) => s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  function inline(s) {
    return s
      .replace(/`([^`]+)`/g, "<code>$1</code>")
      .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
      .replace(/(^|\W)_([^_]+)_(?=\W|$)/g, "$1<em>$2</em>")
      .replace(/\[(E\d+)\]/g, '<span class="cite" data-e="$1">$1</span>');
  }
  function md(text) {
    const lines = esc(text).split("\n");
    let html = "", inList = false, inCode = false, code = [];
    const closeList = () => { if (inList) { html += "</ul>"; inList = false; } };
    for (const raw of lines) {
      const line = raw.replace(/\s+$/, "");
      if (line.trim().startsWith("```")) {
        if (inCode) { html += `<pre><code>${code.join("\n")}</code></pre>`; code = []; inCode = false; }
        else { closeList(); inCode = true; }
        continue;
      }
      if (inCode) { code.push(line.replace(/^\s{2}/, "")); continue; }
      if (/^\s*[-*] /.test(line)) {
        if (!inList) { html += "<ul>"; inList = true; }
        html += `<li>${inline(line.replace(/^\s*[-*] /, ""))}</li>`;
      } else if (/^&gt; ?/.test(line)) {
        closeList(); html += `<blockquote>${inline(line.replace(/^&gt; ?/, ""))}</blockquote>`;
      } else if (/^#{1,4} /.test(line)) {
        closeList(); html += `<p><strong>${inline(line.replace(/^#{1,4} /, ""))}</strong></p>`;
      } else if (line.trim() === "") {
        closeList();
      } else {
        closeList(); html += `<p>${inline(line)}</p>`;
      }
    }
    closeList();
    if (inCode) html += `<pre><code>${code.join("\n")}</code></pre>`;
    return html;
  }

  // ------------------------------------------------------------ integrations
  async function loadIntegrations() {
    document.querySelectorAll(".pill").forEach((p) => { p.className = "pill"; p.title = "checking…"; });
    try {
      const info = await api("/api/integrations");
      $("#mode").textContent = info.mode === "llm" ? `LLM · ${info.model}` : "offline planner";
      for (const [sys, v] of Object.entries(info.systems)) {
        const pill = document.querySelector(`.pill[data-sys="${sys}"]`);
        if (!pill) continue;
        pill.classList.add(v.status === "ok" ? "ok" : v.status === "unavailable" || v.status === "timeout" ? "bad" : "warn");
        pill.title = `${v.status}: ${v.detail}`;
      }
    } catch (e) { $("#mode").textContent = "API error"; }
  }

  // ------------------------------------------------------------ chat
  function addMessage(role, html, extra = "") {
    $("#welcome")?.remove();
    const el = document.createElement("div");
    el.className = `msg ${role}`;
    el.innerHTML = `<div class="bubble">${html}</div>${extra}`;
    $("#messages").appendChild(el);
    el.scrollIntoView({ behavior: "smooth", block: "end" });
    return el;
  }

  async function send(text) {
    if (state.busy || !text.trim()) return;
    state.busy = true; $("#send").disabled = true;
    addMessage("user", esc(text));
    const pending = addMessage("agent", '<span class="thinking">Gathering evidence</span>');
    try {
      const resp = await api("/api/chat", { method: "POST", body: JSON.stringify({ message: text, session_id: state.sessionId }) });
      state.sessionId = resp.session_id;
      const idx = state.responses.push(resp) - 1;
      const systems = resp.systems_consulted.length ? resp.systems_consulted.join(", ") : "none";
      const g = resp.grounding;
      pending.querySelector(".bubble").innerHTML = md(resp.answer);
      pending.insertAdjacentHTML("beforeend",
        `<div class="meta"><span>${resp.tool_calls.length} tool call(s)</span><span>systems: ${esc(systems)}</span>` +
        `<span>${(resp.duration_ms / 1000).toFixed(1)}s</span>` +
        `<span>${g ? (g.grounded ? "✓ grounded" : "⚠ unverified values") : ""}</span>` +
        `<span class="mono">${resp.request_id}</span></div>`);
      pending.dataset.idx = idx;
      pending.addEventListener("click", (ev) => {
        const cite = ev.target.closest(".cite");
        select(idx, cite ? cite.dataset.e : null);
      });
      select(idx);
    } catch (e) {
      pending.querySelector(".bubble").innerHTML = `<p><strong>Request failed:</strong> ${esc(e.message)}</p>`;
    } finally {
      state.busy = false; $("#send").disabled = false; $("#input").focus();
    }
  }

  // ------------------------------------------------------------ evidence panel
  const systemOf = (tool) => tool.startsWith("dvc_") ? "dvc" : tool.startsWith("mlflow_") ? "mlflow"
    : tool.startsWith("airflow_") ? "airflow" : tool.startsWith("k8s_") ? "kubernetes" : "insight";

  function select(idx, flash = null) {
    state.selected = idx;
    document.querySelectorAll(".msg.agent").forEach((m) => m.classList.toggle("selected", Number(m.dataset.idx) === idx));
    const r = state.responses[idx];
    $("#ev-meta").textContent = `${r.mode} · ${r.request_id}`;
    renderApprovals(r);
    const g = r.grounding;
    $("#grounding").innerHTML = g ? `<div class="ground ${g.grounded ? "ok" : "bad"}"><strong>${g.grounded ? "Grounding verified" : "Unverified values"}</strong> · ${g.checked_values} value(s) checked against tool output${g.note ? `<br><span class="muted">${esc(g.note)}</span>` : ""}</div>` : "";
    const trace = $("#trace");
    trace.innerHTML = r.tool_calls.length ? "" : '<li class="empty">No tools were called.</li>';
    for (const t of r.tool_calls) {
      const li = document.createElement("li");
      li.className = "step"; li.id = `ev-${t.evidence_id}`;
      li.style.setProperty("--sys", `var(--${systemOf(t.tool)})`);
      li.innerHTML = `<header><span class="eid">${t.evidence_id}</span><span class="tool">${esc(t.tool)}</span>` +
        `<span class="status ${t.status}">${t.status.replace("_", " ")}</span><span class="ms">${Math.round(t.duration_ms)} ms</span></header>` +
        (t.summary ? `<p class="sum">${esc(t.summary)}</p>` : "") + (t.error ? `<p class="err">${esc(t.error)}</p>` : "") +
        `<details><summary>arguments</summary><pre>${esc(JSON.stringify(t.arguments, null, 2))}</pre></details>`;
      trace.appendChild(li);
    }
    if (flash) {
      const el = document.getElementById(`ev-${flash}`);
      if (el) { el.scrollIntoView({ behavior: "smooth", block: "center" }); el.classList.add("flash"); setTimeout(() => el.classList.remove("flash"), 1400); }
    }
  }

  function renderApprovals(r) {
    const box = $("#approvals"); box.innerHTML = "";
    for (const p of r.pending_confirmations || []) {
      const card = document.createElement("div");
      card.className = "card"; card.id = `card-${p.confirmation_id}`;
      const decided = p._decision;
      card.innerHTML = `<h3>Approval required · ${esc(p.permission)}</h3><p>${esc(p.description)}</p>` +
        `<p class="muted mono">${p.confirmation_id} · expires ${new Date(p.expires_at).toLocaleTimeString()}</p>` +
        (decided ? `<p><strong>${esc(decided)}</strong></p>` :
          `<div class="actions"><button class="btn-approve">Approve</button><button class="btn-decline">Decline</button></div>`);
      if (decided) card.classList.add("done");
      card.querySelector(".btn-approve")?.addEventListener("click", () => decide(p, true));
      card.querySelector(".btn-decline")?.addEventListener("click", () => decide(p, false));
      box.appendChild(card);
    }
  }

  async function decide(p, approve) {
    try {
      const out = await api(`/api/confirmations/${p.confirmation_id}`, { method: "POST", body: JSON.stringify({ approve, session_id: state.sessionId }) });
      const res = out.result;
      p._decision = approve ? `Approved → ${res ? `${res.status}: ${res.summary || res.error}` : "executed"}` : "Declined - nothing was changed.";
      addMessage("agent", md(`**${approve ? "Action approved" : "Action declined"}** - ${p.description}` +
        (res ? `\n\n- Result: **${res.status}** - ${res.summary || res.error}` : "")));
    } catch (e) {
      p._decision = `Could not apply decision: ${e.message}`;
    }
    select(state.selected);
  }

  // ------------------------------------------------------------ wiring
  $("#composer").addEventListener("submit", (e) => { e.preventDefault(); const v = $("#input").value; $("#input").value = ""; send(v); });
  $("#input").addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("#composer").requestSubmit(); } });
  $("#input").addEventListener("input", (e) => { e.target.style.height = "auto"; e.target.style.height = `${e.target.scrollHeight}px`; });
  $("#chips").addEventListener("click", (e) => { const q = e.target.closest("button")?.dataset.q; if (q) send(q); });
  $("#refresh").addEventListener("click", loadIntegrations);
  loadIntegrations();
})();
