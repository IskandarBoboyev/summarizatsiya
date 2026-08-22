/**
 * Asosiy SPA mantiqi (Vanilla JS, CDN yo'q).
 *
 * - Chap panel: LLM / ASR sozlamalari va papka tanlash
 * - Papkalar qo'shish / o'chirish
 * - Hujjatlar kartasi: chap — asl matn, o'ng — xulosa / tarjima / transkripsiya
 * - Infinite scroll + WebSocket orqali real-time qo'shilish
 */
(() => {
  "use strict";

  const PAGE = 12;
  const HOME_PAGE = 20;
  const PERIOD_LABELS = {
    day: "Bugun",
    week: "Shu hafta",
    month: "Shu oy",
    year: "Shu yil",
  };
  const VIEWS = ["home", "pipeline", "flow", "chat"];
  const FLOW_STAGES = [
    { id: "queued", label: "Fayl keldi", hint: "Navbat" },
    { id: "extract", label: "O‘qish", hint: "OCR / ASR" },
    { id: "summarize", label: "Xulosa", hint: "Gemma" },
    { id: "translate", label: "Tarjima", hint: "NLLB / TranslateGemma" },
    { id: "done", label: "Natija", hint: "Tayyor" },
  ];
  const COUNTRIES = [
    { key: "kz", label: "Qozogʻiston" },
    { key: "uz", label: "Oʻzbekiston" },
    { key: "tj", label: "Tojikiston" },
    { key: "kg", label: "Qirgʻiziston" },
    { key: "af", label: "Afgʻoniston" },
    { key: "tm", label: "Turkmaniston" },
  ];
  const state = {
    view: "home",
    country: "uz",
    period: "day",
    chartOpen: false,
    offset: 0,
    total: 0,
    loading: false,
    hasMore: true,
    docs: new Map(),
    openSummaries: new Set(),
    settings: {},
    countries: COUNTRIES,
    folders: [],
    asrModels: [],
    workers: [],
    modelPanelOpen: false,
    modelPanelId: "",
    flow: { items: [], active: null, columns: {}, selectedId: 0 },
    homeCountry: "",
    home: {
      offset: 0,
      loading: false,
      hasMore: false,
      ids: new Set(),
      groups: [],
    },
    rag: { documents: 0, chunks: 0, by_country: {} },
  };

  const $ = (id) => document.getElementById(id);
  const on = (el, ev, fn) => {
    if (el) el.addEventListener(ev, fn);
  };
  const feed = $("feed");
  const empty = $("empty-state");
  const sentinel = $("sentinel");
  const asrLangSelect = $("asr-language");
  const pillDevice = $("pill-device");
  const pillQueue = $("pill-queue");
  const sidebarBtn = $("btn-sidebar");
  const sidebar = $("sidebar");
  const SIDEBAR_KEY = "sidebar_open";
  const THEME_KEY = "ui_theme";
  const DROPDOWNS = ["dd-llm", "dd-ocr", "dd-asr", "dd-files"];
  const pickerState = {
    path: "",
    parent: null,
    country: "",
    watched: new Set(),
    entries: [],
  };

  /* ---------- API ---------- */

  async function api(path, options) {
    const res = await fetch(path, {
      headers: { "Content-Type": "application/json" },
      ...options,
    });
    if (!res.ok) {
      let detail = res.statusText;
      try {
        const body = await res.json();
        detail = body.detail || JSON.stringify(body);
      } catch (_) {
        /* ignore */
      }
      throw new Error(detail);
    }
    if (res.status === 204) return null;
    return res.json();
  }

  /* ---------- Holat paneli ---------- */

  function formatBytes(n) {
    if (!n && n !== 0) return "—";
    const units = ["B", "KB", "MB", "GB"];
    let i = 0;
    let v = Number(n);
    while (v >= 1024 && i < units.length - 1) {
      v /= 1024;
      i += 1;
    }
    return `${v.toFixed(i === 0 ? 0 : 1)} ${units[i]}`;
  }

  function formatWhen(iso) {
    if (!iso) return "";
    try {
      const d = new Date(iso);
      return d.toLocaleString("uz-UZ", {
        hour: "2-digit",
        minute: "2-digit",
        day: "2-digit",
        month: "2-digit",
      });
    } catch (_) {
      return iso;
    }
  }

  function renderStatus(data) {
    const dev = data.device || {};
    const q = data.queue || {};
    if (dev.device || dev.name) {
      pillDevice.textContent = `${(dev.device || "cpu").toUpperCase()} · ${dev.name || ""} · ${dev.memory_gb || "?"} GB`;
    }
    if (q && (q.total != null || q.pending != null)) {
      const pending = (q.pending || 0) + (q.processing || 0);
      pillQueue.textContent = pending
        ? `Navbat: ${q.pending || 0} kutmoqda / ${q.processing || 0} ishlanmoqda`
        : `Jami: ${q.total || 0}`;
      pillQueue.classList.toggle("pill-busy", pending > 0);
    }
    if (data.workers) renderWorkers(data.workers);
  }

  function setModelPanelOpen(open, workerId) {
    const root = $("model-health");
    const panel = $("model-health-panel");
    if (!root || !panel) return;
    if (workerId) state.modelPanelId = workerId;
    state.modelPanelOpen = !!open;
    panel.hidden = !state.modelPanelOpen;
    root.classList.toggle("is-open", state.modelPanelOpen);
    root.querySelectorAll(".model-chip").forEach((chip) => {
      const match = state.modelPanelOpen && chip.getAttribute("data-worker") === state.modelPanelId;
      chip.classList.toggle("is-open", match);
    });
    if (state.modelPanelOpen) renderWorkerPanel();
  }

  function toggleModelPanel(workerId) {
    if (state.modelPanelOpen && state.modelPanelId === workerId) {
      setModelPanelOpen(false);
      return;
    }
    setModelPanelOpen(true, workerId);
  }

  function selectedTranslateWorker() {
    const key = String(state.settings.llm_translate_model || "nllb-200").toLowerCase();
    return key.startsWith("translategemma") ? "translategemma" : "nllb";
  }

  const CHIP_TITLES = {
    gemma: "Xulosa",
    gigaam: "Transkripsiya",
    seamless: "Transkripsiya",
    nllb: "Tarjima",
    translategemma: "Tarjima",
    surya: "OCR",
  };

  function chipTitle(w) {
    return CHIP_TITLES[w && w.id] || "Transkripsiya";
  }

  function paintChip(chip, w) {
    if (!chip || !w) return;
    chip.setAttribute("data-worker", w.id);
    chip.classList.remove("is-live", "is-listening", "is-down", "is-weights_missing", "is-unknown");
    chip.classList.add(`is-${w.state || "unknown"}`);
    const shown = chipTitle(w);
    const text = chip.querySelector(".model-chip-text");
    if (text) text.textContent = shown;
    chip.title = `${shown}: ${w.state_label || "noma’lum"}`;
  }

  function renderWorkers(workers) {
    const root = $("model-health");
    if (!root) return;
    state.workers = workers;
    const translateId = selectedTranslateWorker();
    const translateChip = $("chip-translate");
    for (const w of workers) {
      if (w.id === "nllb" || w.id === "translategemma" || w.id === "seamless") continue;
      paintChip(root.querySelector(`[data-worker="${w.id}"]`), w);
    }
    const translateWorker = workers.find((w) => w.id === translateId);
    if (translateChip && translateWorker) paintChip(translateChip, translateWorker);
    paintOptionStates();
    if (
      state.modelPanelOpen &&
      (state.modelPanelId === "nllb" || state.modelPanelId === "translategemma") &&
      state.modelPanelId !== translateId
    ) {
      setModelPanelOpen(false);
      return;
    }
    if (state.modelPanelOpen) renderWorkerPanel();
  }

  function renderWorkerPanel() {
    const panel = $("model-health-panel");
    if (!panel) return;
    const w = (state.workers || []).find((x) => x.id === state.modelPanelId);
    if (!w) {
      panel.innerHTML = "<p class=\"hint\">Model holati topilmadi.</p>";
      return;
    }
    const running = !!(w.ready || w.listening);
    const meta = [
      `Port ${w.port}`,
      w.ready ? "xotirada" : w.listening ? "server ochiq" : "ulanish yo‘q",
      w.disk_ready ? "diskda bor" : "diskda yo‘q",
      w.backend || "",
      w.device || "",
    ]
      .filter(Boolean)
      .join(" · ");
    const shown = chipTitle(w);
    panel.innerHTML = `
      <div class="model-health-head">
        <strong>${escapeHtml(shown)}</strong>
        <button type="button" class="btn btn-tiny" id="btn-model-refresh">Tekshirish</button>
      </div>
      <article class="model-health-card is-${escapeHtml(w.state || "unknown")}">
        <h3><span class="model-dot"></span>${escapeHtml(shown)} — ${escapeHtml(w.state_label || "")}</h3>
        <p>${escapeHtml(w.role || "")}</p>
        <div class="model-health-meta">${escapeHtml(meta)}</div>
        <p>${escapeHtml(w.hint || "")}</p>
        ${w.error && w.state !== "live" ? `<p>${escapeHtml(w.error)}</p>` : ""}
      </article>
      <div class="model-health-actions">
        <button type="button" class="btn btn-primary btn-tiny" data-worker-start="${escapeHtml(w.id)}"${
          running ? " disabled" : ""
        }>Ishga tushirish</button>
        <button type="button" class="btn btn-tiny btn-stop" data-worker-stop="${escapeHtml(w.id)}"${
          running ? "" : " disabled"
        }>To‘xtatish</button>
      </div>
    `;
  }

  async function controlWorker(id, action) {
    const sel = action === "start" ? `[data-worker-start="${id}"]` : `[data-worker-stop="${id}"]`;
    const btn = document.querySelector(sel);
    if (btn) {
      btn.disabled = true;
      btn.textContent = action === "start" ? "Ishga tushmoqda…" : "To‘xtatilmoqda…";
    }
    try {
      const data = await api(`/api/models/${encodeURIComponent(id)}/${action}`, { method: "POST" });
      if (data.worker) {
        const next = (state.workers || []).slice();
        const idx = next.findIndex((x) => x.id === id);
        if (idx >= 0) next[idx] = data.worker;
        else next.push(data.worker);
        renderWorkers(next);
      }
    } catch (err) {
      alert(err.message || "Amal bajarilmadi.");
      renderWorkers(state.workers || []);
    }
  }

  function countryLabel(key) {
    const found = (state.countries || COUNTRIES).find((c) => c.key === key);
    return found ? found.label : key || "—";
  }

  function renderCountryTabs() {
    const countries = state.countries || COUNTRIES;
    const html = countries
      .map(
        (c) =>
          `<button type="button" class="nav-btn${
            c.key === state.country ? " is-active" : ""
          }" data-country="${c.key}">${escapeHtml(c.label)}</button>`
      )
      .join("");
    const chat = $("chat-countries");
    const pipe = $("pipeline-countries");
    if (chat) chat.innerHTML = html;
    if (pipe) pipe.innerHTML = html;
  }

  function asrModelOptions(selected) {
    const models = (state.asrModels || []).length
      ? state.asrModels
      : [
          { key: "gigaam-multilingual", label: "GigaAM Multilingual" },
          { key: "seamless-m4t-v2", label: "SeamlessM4T v2" },
        ];
    const inherit = selected ? "" : " selected";
    const opts = [`<option value=""${inherit}>Umumiy sozlama</option>`];
    models.forEach((m) => {
      const sel = selected === m.key ? " selected" : "";
      const mark = m.ready === false ? " (yo‘q)" : "";
      opts.push(
        `<option value="${escapeHtml(m.key)}"${sel}>${escapeHtml(m.label)}${mark}</option>`
      );
    });
    return opts.join("");
  }

  function renderCountryFiles() {
    const list = $("country-file-list");
    if (!list) return;
    const byCountry = {};
    (state.folders || []).forEach((f) => {
      if (f.country) byCountry[f.country] = f;
    });
    list.innerHTML = (state.countries || COUNTRIES)
      .map((c) => {
        const f = byCountry[c.key];
        const path = f ? f.path : "Tanlanmagan";
        const asr = f
          ? `<label class="asr-row">
              <span>Transkripsiya</span>
              <select data-folder-asr data-folder-id="${f.id}">${asrModelOptions(f.asr_model || "")}</select>
            </label>`
          : `<p class="hint">Avval papka tanlang — keyin ASR modelini belgilaysiz.</p>`;
        return `<li>
          <div class="row">
            <strong>${escapeHtml(c.label)}</strong>
            <button type="button" class="btn btn-tiny" data-pick-country="${c.key}">Tanlash</button>
          </div>
          <code title="${escapeHtml(path)}">${escapeHtml(path)}</code>
          ${asr}
        </li>`;
      })
      .join("");
    const n = Object.keys(byCountry).length;
    const filesValue = $("files-value");
    if (filesValue) filesValue.textContent = n ? `${n} / 6 davlat` : "Davlat bo‘yicha tanlang";
    const current = byCountry[state.country];
    const bar = $("pipeline-file-path");
    if (bar) {
      bar.textContent = current
        ? current.path
        : `${countryLabel(state.country)} uchun fayl tanlanmagan`;
      bar.title = current ? current.path : "";
    }
  }

  function renderHomeCountryTabs() {
    const el = $("home-countries");
    if (!el) return;
    const allBtn = `<button type="button" class="nav-btn${
      !state.homeCountry ? " is-active" : ""
    }" data-home-country="">Barchasi</button>`;
    const rest = (state.countries || COUNTRIES)
      .map(
        (c) =>
          `<button type="button" class="nav-btn${
            c.key === state.homeCountry ? " is-active" : ""
          }" data-home-country="${c.key}">${escapeHtml(c.label)}</button>`
      )
      .join("");
    el.innerHTML = allBtn + rest;
  }

  function renderHomeGroups(groups) {
    const grid = $("home-grid");
    if (!grid) return;
    const title = document.querySelector("#view-home .home-main .view-title");
    if (title) {
      title.textContent = `Davlatlar bo‘yicha xulosalar · ${
        PERIOD_LABELS[state.period] || PERIOD_LABELS.day
      }`;
    }
    const filtered = state.homeCountry
      ? (groups || []).filter((g) => g.key === state.homeCountry)
      : groups || [];
    state.home.ids = new Set();
    const parts = [];
    filtered.forEach((g) => {
      const items = g.items || [];
      items.forEach((doc) => {
        if (doc && doc.id) state.home.ids.add(doc.id);
      });
      if (!items.length && !state.homeCountry) return;
      const tiles = items.length
        ? `<div class="home-grid-inner">${items.map((doc) => homeTileHtml(doc)).join("")}</div>`
        : `<p class="home-group-empty">Bu davrda hali xulosa yo‘q.</p>`;
      parts.push(`
        <section class="home-group" data-home-group="${escapeHtml(g.key)}">
          <div class="home-group-head">
            <h3>${escapeHtml(g.label)}</h3>
            <span>${g.total || 0} ta fayl</span>
          </div>
          ${tiles}
        </section>`);
    });
    grid.innerHTML = parts.join("");
    const sk = $("home-skeleton");
    if (sk) sk.hidden = true;
    $("home-empty").hidden = state.home.ids.size > 0;
  }

  function renderCountryStats(data) {
    const items = data.countries || [];
    const list = $("stat-list");
    if (list) {
      list.innerHTML = items
        .map(
          (c) => `<button type="button" class="stat-card${
            c.key === state.homeCountry ? " is-active" : ""
          }" data-home-country="${c.key}"><span>${escapeHtml(c.label)}</span><strong>${c.count}</strong></button>`
        )
        .join("");
    }
    const chart = $("stat-chart");
    if (!chart) return;
    const max = Math.max(1, ...items.map((c) => c.count));
    chart.innerHTML = items
      .map((c) => {
        const h = Math.max(4, Math.round((c.count / max) * 110));
        return `<div class="stat-bar"><div class="stat-bar-fill" style="height:${h}px"></div><span>${escapeHtml(
          c.label.slice(0, 4)
        )}<br>${c.count}</span></div>`;
      })
      .join("");
    chart.hidden = !state.chartOpen;
    const chartBtn = $("btn-stats-chart");
    if (chartBtn) chartBtn.classList.toggle("is-active", state.chartOpen);
  }

  async function loadCountryStats() {
    try {
      const data = await api(`/api/stats/countries?period=${state.period}`);
      renderCountryStats(data);
    } catch (_) {
      /* ignore */
    }
  }

  function updateCountryChrome() {
    renderCountryTabs();
    renderCountryFiles();
    const label = countryLabel(state.country);
    const title = $("pipeline-country-title");
    if (title) title.textContent = label;
    const emptyP = empty && empty.querySelector("p");
    if (emptyP) {
      emptyP.textContent = `${label} uchun hali qayta ishlangan fayl yo‘q. Chapdan shu davlat uchun fayl qo‘shing.`;
    }
    const hint = document.querySelector("#view-chat .hint");
    if (hint) {
      const rag = (state.rag && state.rag.by_country && state.rag.by_country[state.country]) || {};
      const n = Number(rag.documents || 0);
      hint.textContent = n
        ? `Javob ${label} bazasidan olinadi (${n} ta hujjat).`
        : `Javob ${label} bazasidan olinadi. Bu davlatda hali «Bazaga yozish» qilinmagan.`;
    }
  }

  function resetChatWelcome() {
    const box = $("chat-messages");
    if (!box) return;
    const label = countryLabel(state.country);
    box.innerHTML = `<div class="chat-bubble bot">${escapeHtml(
      label
    )} bo‘yicha savol bering. Pipeline da «Bazaga yozish» orqali shu davlat bazasiga yozilgan matnlar ishlatiladi.</div>`;
  }

  function setCountry(key, reload) {
    if (!key) return;
    const changed = key !== state.country;
    state.country = key;
    updateCountryChrome();
    if (changed) resetChatWelcome();
    if (reload !== false) resetPipelineFeed();
  }

  function resetPipelineFeed() {
    state.docs.clear();
    if (feed) feed.innerHTML = "";
    state.offset = 0;
    state.hasMore = true;
    state.loading = false;
    loadPage();
  }

  function fillSelect(select, entries, current) {
    select.innerHTML = "";
    for (const [value, label] of entries) {
      const opt = document.createElement("option");
      opt.value = value;
      opt.textContent = label;
      if (value === current) opt.selected = true;
      select.appendChild(opt);
    }
  }

  function setDropdownOpen(id, open) {
    const root = $(id);
    if (!root) return;
    const trigger = root.querySelector(".dropdown-trigger");
    const panel = root.querySelector(".dropdown-panel");
    root.classList.toggle("is-open", open);
    if (panel) panel.hidden = !open;
    if (trigger) trigger.setAttribute("aria-expanded", open ? "true" : "false");
  }

  function toggleDropdown(id) {
    const willOpen = !$(id).classList.contains("is-open");
    DROPDOWNS.forEach((other) => setDropdownOpen(other, other === id && willOpen));
    if (willOpen) setModelPanelOpen(false);
    if (willOpen && id === "dd-files") {
      refreshFolders().catch(() => {});
    }
  }

  function workerIdForModel(key, kind) {
    const k = String(key || "").toLowerCase();
    if (kind === "translate") return k.startsWith("translategemma") ? "translategemma" : "nllb";
    if (kind === "asr") return k.startsWith("seamless") ? "seamless" : "gigaam";
    if (kind === "ocr") return k === "surya" ? "surya" : "gemma";
    return "gemma";
  }

  function workerById(id) {
    return (state.workers || []).find((w) => w.id === id) || null;
  }

  function paintOptionStates() {
    document.querySelectorAll(".option-btn[data-worker-ref]").forEach((btn) => {
      const w = workerById(btn.getAttribute("data-worker-ref"));
      const ready = !btn.disabled;
      const st = (w && w.state) || (ready ? "unknown" : "weights_missing");
      btn.classList.remove("is-live", "is-listening", "is-down", "is-weights_missing", "is-unknown");
      btn.classList.add(`is-${st}`);
      const hint = btn.querySelector("small");
      if (hint && w && w.state_label) {
        const extra = ready ? btn.getAttribute("data-backend") || "" : "diskda yo‘q";
        hint.textContent = extra ? `${w.state_label} · ${extra}` : w.state_label;
      }
    });
  }

  function fillOptionList(list, models, current, attr, kind) {
    list.innerHTML = "";
    if (!models.length) {
      list.innerHTML = `<div class="picker-empty">Tizimda model topilmadi</div>`;
      return;
    }
    for (const m of models) {
      const btn = document.createElement("button");
      btn.type = "button";
      const workerId = workerIdForModel(m.key, kind);
      btn.className = "option-btn" + (m.key === current ? " is-current" : "");
      btn.disabled = !m.ready;
      btn.setAttribute(attr, m.key);
      btn.setAttribute("data-worker-ref", workerId);
      btn.setAttribute("data-backend", m.ready ? m.backend || "lokal" : "");
      btn.innerHTML = `<span class="option-head"><span class="model-dot" aria-hidden="true"></span><span>${escapeHtml(
        m.label
      )}</span></span><small>${
        m.ready ? escapeHtml(m.backend || "lokal") : "diskda yo‘q"
      }</small>`;
      list.appendChild(btn);
    }
    paintOptionStates();
  }

  function currentLabel(models, key, fallback) {
    const found = (models || []).find((m) => m.key === key);
    if (found) return found.ready ? found.label : `${found.label} (yo‘q)`;
    return fallback;
  }

  async function loadStatus() {
    const data = await api("/api/status");
    state.settings = data.settings || {};
    state.rag = data.rag || { documents: 0, chunks: 0, by_country: {} };
    if (data.countries && data.countries.length) state.countries = data.countries;
    renderStatus(data);
    renderCountryTabs();
    updateCountryChrome();
    loadCountryStats().catch(() => {});

    const llmModels = data.models || [];
    const trModels = data.translation_models || [];
    const summaryKey = state.settings.llm_summary_model || state.settings.llm_model;
    const translateKey = state.settings.llm_translate_model || "nllb-200";
    fillOptionList($("llm-summary-list"), llmModels, summaryKey, "data-llm-summary", "llm");
    fillOptionList($("llm-translate-list"), trModels, translateKey, "data-llm-translate", "translate");
    const sumLabel = currentLabel(llmModels, summaryKey, "Tanlang");
    const trLabel = currentLabel(trModels, translateKey, "NLLB-200");
    $("llm-value").textContent = `Xulosa: ${sumLabel} · Tarjima: ${trLabel}`;

    const asrModels = data.asr_models || [];
    state.asrModels = asrModels;
    fillOptionList($("asr-list"), asrModels, state.settings.asr_model, "data-asr", "asr");
    $("asr-value").textContent = currentLabel(asrModels, state.settings.asr_model, "Tanlang");

    const ocrModels = data.ocr_models || [];
    const ocrKey = state.settings.ocr_model || "surya";
    fillOptionList($("ocr-list"), ocrModels, ocrKey, "data-ocr", "ocr");
    if ($("ocr-value")) $("ocr-value").textContent = currentLabel(ocrModels, ocrKey, "Surya OCR");

    const langs = Object.entries(data.asr_languages || {});
    fillSelect(asrLangSelect, langs, state.settings.asr_language);
    renderCountryFiles();
    return data;
  }

  async function saveSettings(patch) {
    state.settings = await api("/api/settings", {
      method: "PUT",
      body: JSON.stringify(patch),
    });
  }

  /* ---------- Kartalar ---------- */

  function fileKindLabel(type) {
    const map = {
      pdf: "PDF",
      docx: "DOCX",
      text: "MATN",
      image: "RASM",
      audio: "AUDIO",
      video: "VIDEO",
    };
    return map[type] || (type || "FAYL").toUpperCase();
  }

  function statusLabel(status) {
    const map = {
      pending: "navbatda",
      processing: "qayta ishlanmoqda",
      done: "tayyor",
      error: "xato",
    };
    return map[status] || status;
  }

  function textOrPlaceholder(text, placeholder) {
    if (text && String(text).trim()) return escapeHtml(text);
    return `<span class="prose muted">${escapeHtml(placeholder)}</span>`;
  }

  function escapeHtml(str) {
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function clipText(text, n) {
    const s = String(text || "").replace(/\s+/g, " ").trim();
    if (!s) return "";
    return s.length > n ? `${s.slice(0, n).trim()}…` : s;
  }

  function homeTileHtml(doc) {
    const summary = clipText(doc.summary || doc.translation_uz || "", 280);
    const kind = fileKindLabel(doc.file_type);
    return `
      <button type="button" class="home-tile" data-open-doc="${doc.id}" data-doc-country="${escapeHtml(doc.country || "")}" id="home-${doc.id}">
        <h3 title="${escapeHtml(doc.filename || "")}">${escapeHtml(doc.filename || "")}</h3>
        <p class="meta">${escapeHtml(countryLabel(doc.country))} · ${kind} · ${formatWhen(doc.updated_at || doc.created_at)}</p>
        <pre class="prose">${
          summary
            ? escapeHtml(summary)
            : `<span class="prose muted">${
                doc.status === "done" ? "Xulosa yo‘q." : statusLabel(doc.status)
              }</span>`
        }</pre>
      </button>`;
  }

  function upsertHomeTile(doc) {
    if (!doc || !doc.id) return;
    loadHomePage().catch(() => {});
  }

  async function loadHomePage() {
    const h = state.home;
    const period = state.period || "day";
    h.seq = (h.seq || 0) + 1;
    const seq = h.seq;
    h.loading = true;
    const sk = $("home-skeleton");
    const emptyHome = $("home-empty");
    if (sk && !h.ids.size) {
      sk.hidden = false;
      if (emptyHome) emptyHome.hidden = true;
    }
    try {
      const data = await api(
        `/api/documents/home?limit=${HOME_PAGE}&period=${encodeURIComponent(period)}`
      );
      if (seq !== h.seq) return;
      h.groups = data.countries || [];
      renderHomeCountryTabs();
      renderHomeGroups(h.groups);
    } catch (err) {
      console.error(err);
      if (seq !== h.seq) return;
      if (sk) sk.hidden = true;
      if (emptyHome && !h.ids.size) emptyHome.hidden = false;
    } finally {
      if (seq !== h.seq) return;
      h.loading = false;
      if (sk && h.ids.size) sk.hidden = true;
    }
  }

  function setHomeCountry(key) {
    state.homeCountry = key || "";
    renderHomeCountryTabs();
    renderHomeGroups(state.home.groups || []);
    loadCountryStats();
  }

  function setView(name) {
    const view = VIEWS.includes(name) ? name : "home";
    state.view = view;
    document.querySelector(".app").setAttribute("data-view", view);
    VIEWS.forEach((v) => {
      const el = $(`view-${v}`);
      if (el) el.hidden = v !== view;
    });
    document.querySelectorAll(".app-nav [data-view]").forEach((btn) => {
      btn.classList.toggle("is-active", btn.getAttribute("data-view") === view);
    });
    if (view !== "pipeline") setSidebarOpen(false);
    if (view === "flow") refreshFlow().catch(() => {});
    if (location.hash.replace("#", "") !== view) {
      history.replaceState(null, "", `#${view}`);
    }
  }

  function flowStageOf(doc) {
    return doc && doc.pipeline_stage ? doc.pipeline_stage : "queued";
  }

  function flowStageIndex(id) {
    const i = FLOW_STAGES.findIndex((s) => s.id === id);
    return i < 0 ? -1 : i;
  }

  function renderFlow(data) {
    const stages = data.stages || FLOW_STAGES;
    const columns = data.columns || {};
    const items = data.items || [];
    const active = data.active || null;
    const prevId = Number(state.flow.selectedId || 0);
    const selected =
      items.find((d) => Number(d.id) === prevId) ||
      active ||
      items.find((d) => d.status === "processing") ||
      items[0] ||
      null;
    state.flow = {
      items,
      active,
      columns,
      selectedId: selected ? Number(selected.id) : 0,
    };

    const filesEl = $("flow-files");
    if (filesEl) {
      filesEl.innerHTML = items.length
        ? items
            .slice(0, 24)
            .map((doc) => {
              const sel = selected && Number(selected.id) === Number(doc.id);
              const err = doc.status === "error";
              const stage = stages.find((s) => s.id === flowStageOf(doc));
              return `<button type="button" class="flow-file${sel ? " is-current" : ""}${
                err ? " is-error" : ""
              }" data-flow-select="${doc.id}">
                <strong title="${escapeHtml(doc.filename || "")}">${escapeHtml(doc.filename || "")}</strong>
                <span>${escapeHtml(countryLabel(doc.country))} · ${escapeHtml(
                  stage ? stage.label : statusLabel(doc.status)
                )}</span>
              </button>`;
            })
            .join("")
        : `<p class="hint">Hali fayl yo‘q. Papkaga tushsa, shu yerda yo‘li ochiladi.</p>`;
    }

    const steps = (selected && selected.flow_steps) || [];
    const stepById = Object.fromEntries(steps.map((s) => [s.id, s]));
    const track = $("flow-track");
    if (track) {
      track.innerHTML = stages
        .map((s, idx) => {
          const st = stepById[s.id] || {};
          const count = (columns[s.id] || []).length;
          let cls = `flow-node is-${st.state || "wait"}`;
          return `<li class="${cls}">
            <span class="flow-orb">${idx + 1}</span>
            <strong>${escapeHtml(s.label)}</strong>
            <small>${escapeHtml(s.hint || "")}</small>
            <span class="flow-count">${count}</span>
          </li>`;
        })
        .join("");
    }

    const banner = $("flow-active");
    if (banner) {
      if (selected) {
        const busy = selected.status === "processing" || selected.status === "pending";
        banner.classList.toggle("is-busy", busy);
        const stageMeta = stages.find((s) => s.id === flowStageOf(selected));
        banner.innerHTML = `<h3>${escapeHtml(selected.filename || "Fayl")}</h3>
          <p>${escapeHtml(countryLabel(selected.country))} · ${fileKindLabel(
            selected.file_type
          )} · ${formatBytes(selected.file_size)} · ${escapeHtml(
            stageMeta ? stageMeta.label : statusLabel(selected.status)
          )} · ${escapeHtml(statusLabel(selected.status))}</p>`;
      } else {
        banner.classList.remove("is-busy");
        banner.innerHTML = `<p>Hozir ish ketmayapti. Papkaga fayl tushsa, barcha bosqichlar shu yerda yuradi.</p>`;
      }
    }

    const journey = $("flow-journey");
    if (journey) {
      if (!selected) {
        journey.innerHTML = "";
      } else {
        journey.innerHTML = stages
          .map((s, idx) => {
            const st = stepById[s.id] || { state: "wait", preview: "", empty: "" };
            const body = st.preview
              ? escapeHtml(st.preview)
              : `<span class="prose muted">${escapeHtml(st.empty || "Kutilmoqda")}</span>`;
            return `<section class="flow-step is-${escapeHtml(st.state || "wait")}">
              <header>
                <span class="flow-step-n">${idx + 1}</span>
                <div>
                  <h3>${escapeHtml(s.label)}</h3>
                  <small>${escapeHtml(s.hint || "")} · ${
                    st.state === "done"
                      ? "o‘tdi"
                      : st.state === "current"
                        ? "hozir"
                        : st.state === "error"
                          ? "xato"
                          : "kutilmoqda"
                  }</small>
                </div>
              </header>
              <pre class="prose">${body}</pre>
            </section>`;
          })
          .join("");
      }
    }
  }

  async function refreshFlow() {
    const data = await api("/api/pipeline/live");
    renderFlow(data);
    return data;
  }

  function ragFooter(doc) {
    const hasText = Boolean(String(doc.original_text || doc.transcription || "").trim());
    if (!hasText) return "";
    const indexed = Boolean(doc.in_rag);
    return `
      <div class="card-footer">
        <button
          type="button"
          class="btn btn-primary btn-tiny"
          data-index-rag="${doc.id}"
          title="${indexed ? "RAG bazasini yangilash" : "Yaxshi ma’lumotni RAG bazasiga yozish"}"
        >${indexed ? "Bazada ✓" : "Bazaga yozish"}</button>
      </div>`;
  }

  function cardHtml(doc) {
    const kind = fileKindLabel(doc.file_type);
    const iconClass =
      doc.file_type === "audio" || doc.file_type === "video"
        ? "is-audio"
        : doc.file_type === "image"
          ? "is-image"
          : doc.file_type === "pdf"
            ? "is-pdf"
            : "";

    const processing =
      doc.status === "processing"
        ? `<div class="processing-row"><span class="spinner"></span> Model ishlamoqda…</div>`
        : "";

    const err = doc.status === "error" && doc.error_message
      ? `<div class="err-box">${escapeHtml(doc.error_message)}</div>`
      : "";

    const isMedia = doc.file_type === "audio" || doc.file_type === "video";
    const leftPending = isMedia
      ? "Navbatda. Audio eshitilishi bilan matn yoziladi."
      : "Navbatda. Fayl tushishi bilan matn ajratiladi.";
    const leftEmpty = isMedia
      ? "Transkripsiya hozircha yo‘q."
      : "Asl matn hozircha yo‘q.";

    const leftBody =
      doc.status === "pending"
        ? `<p class="prose muted">${leftPending}</p>`
        : `<pre class="prose">${textOrPlaceholder(doc.original_text, leftEmpty)}</pre>`;

    const rightBody =
      doc.status === "done" || doc.translation_uz
        ? `<pre class="prose">${textOrPlaceholder(doc.translation_uz, "Tarjima hozircha yo‘q.")}</pre>`
        : `<p class="prose muted">O‘zbekcha tarjima shu yerda, asl tuzilmada chiqadi.</p>`;

    const summaryOpen = state.openSummaries.has(Number(doc.id));
    const summaryBlock =
      doc.status === "pending"
        ? ""
        : `
      <details class="summary-fold"${summaryOpen ? " open" : ""}>
        <summary>
          <span class="summary-arrow" aria-hidden="true"></span>
          Xulosa
        </summary>
        <pre class="prose">${textOrPlaceholder(doc.summary, "Xulosa hozircha yo‘q.")}</pre>
      </details>`;

    return `
      <article class="card" data-id="${doc.id}" id="doc-${doc.id}">
        <div class="card-head">
          <div class="card-title">
            <span class="file-icon ${iconClass}">${kind.slice(0, 4)}</span>
            <div>
              <h3 title="${escapeHtml(doc.filepath || "")}">${escapeHtml(doc.filename || "")}</h3>
              <p class="meta">${kind} · ${formatBytes(doc.file_size)} · ${formatWhen(doc.updated_at || doc.created_at)}${
                doc.model_used ? ` · ${escapeHtml(doc.model_used)}` : ""
              }</p>
            </div>
          </div>
          <div class="card-actions">
            <span class="badge ${escapeHtml(doc.status || "")}">${statusLabel(doc.status)}</span>
            <button type="button" class="btn btn-tiny" data-reprocess="${doc.id}">Qayta</button>
          </div>
        </div>
        ${err}
        ${processing}
        <div class="card-body">
          <section class="col">
            <h4>Asl matn</h4>
            ${leftBody}
          </section>
          <section class="col">
            <h4>O‘zbekcha tarjima</h4>
            ${rightBody}
          </section>
        </div>
        ${summaryBlock}
        ${ragFooter(doc)}
      </article>
    `;
  }

  function rememberSummaryFold(card) {
    if (!card) return;
    const fold = card.querySelector(".summary-fold");
    const id = Number(card.dataset.id);
    if (!id) return;
    if (fold && fold.open) state.openSummaries.add(id);
    else state.openSummaries.delete(id);
  }

  function cardFingerprint(doc) {
    return [
      doc.status,
      doc.error_message || "",
      doc.original_text || "",
      doc.translation_uz || "",
      doc.summary || "",
      doc.in_rag ? "1" : "0",
      doc.filename || "",
      doc.file_size || "",
      doc.model_used || "",
      doc.updated_at || "",
    ].join("\x1e");
  }

  function upsertCard(doc) {
    if (!doc || !doc.id) return;
    if (state.country && (doc.country || "") !== state.country) {
      const stale = document.getElementById(`doc-${doc.id}`);
      if (stale) stale.remove();
      state.docs.delete(doc.id);
      state.openSummaries.delete(Number(doc.id));
      syncEmpty();
      return;
    }
    const existing = document.getElementById(`doc-${doc.id}`);
    rememberSummaryFold(existing);
    const prev = state.docs.get(doc.id);
    state.docs.set(doc.id, doc);
    if (existing && prev && cardFingerprint(prev) === cardFingerprint(doc)) {
      return;
    }
    const wrap = document.createElement("div");
    wrap.innerHTML = cardHtml(doc).trim();
    const node = wrap.firstElementChild;
    if (existing) {
      existing.replaceWith(node);
      syncEmpty();
      return;
    }
    // ID o'sish tartibida (yuqorida eski, pastda yangi)
    const cards = [...feed.querySelectorAll(".card")];
    const next = cards.find((c) => Number(c.dataset.id) > Number(doc.id));
    if (next) feed.insertBefore(node, next);
    else feed.appendChild(node);
    syncEmpty();
  }

  function syncEmpty() {
    const has = feed.children.length > 0;
    empty.hidden = has;
    const sk = $("feed-skeleton");
    if (sk && has) sk.hidden = true;
  }

  async function loadPage() {
    if (state.loading || !state.hasMore) return;
    state.loading = true;
    const firstPaint = state.offset === 0 && state.docs.size === 0;
    const sk = $("feed-skeleton");
    if (sk && firstPaint) {
      sk.hidden = false;
      empty.hidden = true;
      if (sentinel) sentinel.hidden = true;
    } else if (sentinel) {
      sentinel.hidden = false;
    }
    try {
      const data = await api(
        `/api/documents?offset=${state.offset}&limit=${PAGE}&country=${encodeURIComponent(state.country)}`
      );
      const items = data.items || [];
      items.forEach((doc) => upsertCard(doc));
      state.offset += items.length;
      state.total = data.total || 0;
      state.hasMore = Boolean(data.has_more);
    } catch (err) {
      console.error(err);
    } finally {
      state.loading = false;
      if (sk) sk.hidden = true;
      if (sentinel) sentinel.hidden = !state.hasMore;
      syncEmpty();
    }
  }

  /* ---------- Papkalar (shu qurilma diskini ochish) ---------- */

  async function refreshFolders() {
    const folders = await api("/api/folders");
    state.folders = folders || [];
    pickerState.watched = new Set((folders || []).map((f) => f.path));
    renderCountryFiles();
    return folders;
  }

  function setPickerOpen(open) {
    const modal = $("picker-modal");
    if (!modal) return;
    modal.hidden = !open;
    const title = $("picker-title");
    if (title) {
      title.textContent = open
        ? `${countryLabel(pickerState.country)} — papka yoki fayl`
        : "Papka yoki fayl tanlash";
    }
  }

  function renderBrowser(data) {
    pickerState.path = data.path || "";
    pickerState.parent = data.parent || null;
    $("picker-path").textContent = pickerState.path;
    $("picker-path").title = pickerState.path;
    $("btn-browse-up").disabled = !pickerState.parent;

    const shortcuts = $("picker-shortcuts");
    shortcuts.innerHTML = (data.shortcuts || [])
      .map((s) => {
        const active = s.path === pickerState.path ? " is-active" : "";
        return `<button type="button" class="${active.trim()}" data-goto="${escapeHtml(s.path)}">${escapeHtml(s.name)}</button>`;
      })
      .join("");

    const list = $("picker-list");
    const entries = data.entries || [];
    pickerState.entries = entries;
    if (!entries.length) {
      list.innerHTML = `<div class="picker-empty">Bu yerda ochiladigan papka yo‘q</div>`;
      return;
    }
    list.innerHTML = entries
      .map((e, idx) => {
        const watched = pickerState.watched.has(e.path) ? " is-watched" : "";
        const isDir = e.is_dir !== false;
        const mark = pickerState.watched.has(e.path) ? " · tanlangan" : "";
        const action = isDir
          ? `<span class="picker-actions">
              <button type="button" class="btn btn-tiny btn-primary" data-act="select" data-idx="${idx}">Tanlash</button>
              <button type="button" class="btn btn-tiny" data-act="open" data-idx="${idx}">Ochish</button>
            </span>`
          : `<button type="button" class="btn btn-tiny btn-primary" data-act="select" data-idx="${idx}">Tanlash</button>`;
        return `
          <div class="picker-row${watched}" role="listitem">
            <button type="button" class="picker-name-btn${isDir ? "" : " is-file"}" data-act="select" data-idx="${idx}" title="${escapeHtml(e.path)}">${escapeHtml(e.name)}${mark}</button>
            ${action}
          </div>`;
      })
      .join("");
  }

  async function browseTo(path) {
    const q = path ? `?path=${encodeURIComponent(path)}` : "";
    const data = await api(`/api/browse${q}`);
    renderBrowser(data);
  }

  async function addWatchPath(raw) {
    const path = (raw || "").trim();
    const country = pickerState.country || state.country;
    if (!path) {
      alert("Fayl yoki papkani tanlang.");
      return;
    }
    if (!country) {
      alert("Avval davlatni tanlang.");
      return;
    }
    try {
      await api("/api/folders", { method: "POST", body: JSON.stringify({ path, country }) });
      setPickerOpen(false);
      await refreshFolders();
      if (country === state.country) resetPipelineFeed();
      loadHomePage().catch(() => {});
      loadCountryStats().catch(() => {});
    } catch (err) {
      alert(err.message);
    }
  }

  function openNativePicker(country) {
    pickerState.country = country || state.country;
    setDropdownOpen("dd-files", false);
    const input = $("file-input");
    if (!input) {
      alert("Fayl tanlash maydoni topilmadi.");
      return;
    }
    input.value = "";
    input.click();
  }

  async function uploadPickedFile(file) {
    const country = pickerState.country || state.country;
    if (!file) return;
    if (!country) {
      alert("Avval davlatni tanlang.");
      return;
    }
    const body = new FormData();
    body.append("country", country);
    body.append("file", file, file.name);
    try {
      const res = await fetch("/api/upload", { method: "POST", body });
      if (!res.ok) {
        let detail = res.statusText;
        try {
          const data = await res.json();
          detail = data.detail || JSON.stringify(data);
        } catch (_) {
          /* ignore */
        }
        throw new Error(detail);
      }
      setCountry(country, false);
      await refreshFolders();
      resetPipelineFeed();
      loadHomePage().catch(() => {});
      loadCountryStats().catch(() => {});
    } catch (err) {
      alert(err.message || "Fayl yuklanmadi.");
    }
  }

  async function openPicker(country) {
    pickerState.country = country || state.country;
    setDropdownOpen("dd-files", false);
    setPickerOpen(true);
    const assigned = (state.folders || []).find((f) => f.country === pickerState.country);
    const start = assigned && assigned.path ? assigned.path : "";
    try {
      await browseTo(start || null);
    } catch (_) {
      await browseTo(null);
    }
  }

  /* ---------- WebSocket ---------- */

  function connectWs() {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${proto}://${location.host}/ws`);

    ws.addEventListener("message", (ev) => {
      let msg;
      try {
        msg = JSON.parse(ev.data);
      } catch (_) {
        return;
      }
      if (msg.type === "document_updated" && msg.payload) {
        const doc = msg.payload;
        const known = state.docs.has(doc.id);
        upsertCard(doc);
        upsertHomeTile(doc);
        loadCountryStats().catch(() => {});
        if (!known) state.total += 1;
        if (state.view === "flow") refreshFlow().catch(() => {});
      }
      if (msg.type === "queue_status" && msg.payload) {
        renderStatus({ device: {}, queue: msg.payload });
        loadStatus().catch(() => {});
      }
      if (msg.type === "folders") {
        refreshFolders().catch(() => {});
      }
      if (msg.type === "hello" && msg.payload && msg.payload.queue) {
        renderStatus({ device: {}, queue: msg.payload.queue });
      }
    });

    ws.addEventListener("close", () => {
      setTimeout(connectWs, 4000);
    });
    ws.addEventListener("error", () => {
      ws.close();
    });
  }

  async function pollUpdates() {
    try {
      await loadStatus();
      const data = await api(
        `/api/documents?offset=0&limit=${PAGE}&country=${encodeURIComponent(state.country)}`
      );
      (data.items || []).forEach((doc) => {
        upsertCard(doc);
      });
      loadHomePage().catch(() => {});
      loadCountryStats().catch(() => {});
      if (state.view === "flow") refreshFlow().catch(() => {});
      state.total = data.total || state.total;
    } catch (_) {
      /* server hali ochilmagan bo'lishi mumkin */
    }
  }

  function setSidebarOpen(open) {
    document.querySelector(".app").classList.toggle("is-sidebar-open", open);
    sidebar.setAttribute("aria-hidden", open ? "false" : "true");
    sidebarBtn.classList.toggle("is-active", open);
    sidebarBtn.setAttribute("aria-expanded", open ? "true" : "false");
    sidebarBtn.setAttribute("aria-label", open ? "Sozlamalarni yopish" : "Sozlamalarni ochish");
    sidebarBtn.title = open ? "Sozlamalarni yopish" : "Sozlamalar";
    try {
      localStorage.setItem(SIDEBAR_KEY, open ? "1" : "0");
    } catch (_) {
      /* ignore */
    }
  }

  function currentTheme() {
    return document.documentElement.getAttribute("data-theme") === "light" ? "light" : "dark";
  }

  function setTheme(theme) {
    const next = theme === "light" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", next);
    const btn = $("btn-theme");
    const isLight = next === "light";
    if (btn) {
      btn.title = isLight ? "Tun rejimi" : "Kun rejimi";
      btn.setAttribute("aria-label", isLight ? "Tun rejimiga o‘tish" : "Kun rejimiga o‘tish");
    }
    try {
      localStorage.setItem(THEME_KEY, next);
    } catch (_) {
      /* ignore */
    }
  }

  /* ---------- Hodisalar ---------- */

  function bind(id, ev, fn) {
    const el = typeof id === "string" ? $(id) : id;
    if (el) el.addEventListener(ev, fn);
  }

  bind("btn-theme", "click", () => {
    setTheme(currentTheme() === "dark" ? "light" : "dark");
  });

  on(document.querySelector(".app-nav"), "click", (e) => {
    const btn = e.target.closest("[data-view]");
    if (btn) setView(btn.getAttribute("data-view"));
  });

  bind("btn-flow-refresh", "click", () => {
    const btn = $("btn-flow-refresh");
    if (btn) {
      btn.disabled = true;
      btn.classList.add("is-busy");
    }
    refreshFlow()
      .catch((err) => alert(err.message))
      .finally(() => {
        if (btn) {
          btn.disabled = false;
          btn.classList.remove("is-busy");
        }
      });
  });

  bind("flow-files", "click", (e) => {
    const chip = e.target.closest("[data-flow-select]");
    if (!chip) return;
    state.flow.selectedId = Number(chip.getAttribute("data-flow-select") || 0);
    renderFlow({
      stages: FLOW_STAGES,
      columns: state.flow.columns,
      items: state.flow.items,
      active: state.flow.active,
    });
  });

  bind("home-grid", "click", (e) => {
    const tile = e.target.closest("[data-open-doc]");
    if (!tile) return;
    const country = tile.getAttribute("data-doc-country");
    if (country) setCountry(country, false);
    setView("pipeline");
    resetPipelineFeed();
  });

  on($("chat-form"), "submit", async (e) => {
    e.preventDefault();
    const input = $("chat-input");
    const text = (input.value || "").trim();
    if (!text) return;
    const box = $("chat-messages");
    box.insertAdjacentHTML(
      "beforeend",
      `<div class="chat-bubble user">${escapeHtml(text)}</div>`
    );
    input.value = "";
    box.scrollTop = box.scrollHeight;
    const sendBtn = $("btn-chat-send");
    sendBtn.disabled = true;
    sendBtn.classList.add("is-busy");
    const pending = document.createElement("div");
    pending.className = "chat-bubble bot is-pending";
    pending.id = "chat-pending";
    pending.innerHTML =
      `<span class="typing-dots" aria-hidden="true"><i></i><i></i><i></i></span>Javob tayyorlanmoqda…`;
    box.appendChild(pending);
    box.scrollTop = box.scrollHeight;
    try {
      const res = await api("/api/chat", {
        method: "POST",
        body: JSON.stringify({ message: text, country: state.country }),
      });
      const names = [...new Set((res.sources || []).map((s) => s.filename).filter(Boolean))];
      const src = names.length
        ? `<p class="chat-sources">Manba: ${escapeHtml(names.join(", "))}</p>`
        : "";
      pending.remove();
      box.insertAdjacentHTML(
        "beforeend",
        `<div class="chat-bubble bot">${escapeHtml(res.reply || "")}${src}</div>`
      );
    } catch (err) {
      pending.remove();
      box.insertAdjacentHTML(
        "beforeend",
        `<div class="chat-bubble bot">${escapeHtml(err.message)}</div>`
      );
    } finally {
      sendBtn.disabled = false;
      sendBtn.classList.remove("is-busy");
      box.scrollTop = box.scrollHeight;
    }
  });

  bind("chat-input", "keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      $("chat-form").requestSubmit();
    }
  });

  on(sidebarBtn, "click", () => {
    setSidebarOpen(!document.querySelector(".app").classList.contains("is-sidebar-open"));
  });

  bind("btn-llm", "click", () => toggleDropdown("dd-llm"));
  bind("btn-ocr", "click", () => toggleDropdown("dd-ocr"));
  bind("btn-asr", "click", () => toggleDropdown("dd-asr"));
  bind("btn-files", "click", () => toggleDropdown("dd-files"));

  bind("model-health", "click", (e) => {
    const startBtn = e.target.closest("[data-worker-start]");
    if (startBtn) {
      e.preventDefault();
      e.stopPropagation();
      controlWorker(startBtn.getAttribute("data-worker-start"), "start");
      return;
    }
    const stopBtn = e.target.closest("[data-worker-stop]");
    if (stopBtn) {
      e.preventDefault();
      e.stopPropagation();
      controlWorker(stopBtn.getAttribute("data-worker-stop"), "stop");
      return;
    }
    if (e.target.closest("#btn-model-refresh")) {
      e.preventDefault();
      e.stopPropagation();
      const refresh = $("btn-model-refresh");
      if (refresh) {
        refresh.disabled = true;
        refresh.textContent = "…";
      }
      api("/api/models/health")
        .then((data) => renderWorkers(data.workers || []))
        .catch(() => {})
        .finally(() => {
          if ($("btn-model-refresh")) {
            $("btn-model-refresh").disabled = false;
            $("btn-model-refresh").textContent = "Tekshirish";
          }
        });
      return;
    }
    const chip = e.target.closest(".model-chip");
    if (!chip) return;
    e.preventDefault();
    e.stopPropagation();
    DROPDOWNS.forEach((id) => setDropdownOpen(id, false));
    toggleModelPanel(chip.getAttribute("data-worker"));
  });

  document.addEventListener("click", (e) => {
    if (!$("model-health") || $("model-health").contains(e.target)) return;
    setModelPanelOpen(false);
  });

  bind("llm-summary-list", "click", (e) => {
    const btn = e.target.closest("[data-llm-summary]");
    if (!btn || btn.disabled) return;
    saveSettings({ llm_summary_model: btn.getAttribute("data-llm-summary") })
      .then(() => loadStatus())
      .catch((err) => alert(err.message));
  });

  bind("llm-translate-list", "click", (e) => {
    const btn = e.target.closest("[data-llm-translate]");
    if (!btn || btn.disabled) return;
    saveSettings({ llm_translate_model: btn.getAttribute("data-llm-translate") })
      .then(() => loadStatus())
      .catch((err) => alert(err.message));
  });

  bind("asr-list", "click", (e) => {
    const btn = e.target.closest("[data-asr]");
    if (!btn || btn.disabled) return;
    saveSettings({ asr_model: btn.getAttribute("data-asr") })
      .then(() => loadStatus())
      .then(() => {})
      .catch((err) => alert(err.message));
  });

  bind("ocr-list", "click", (e) => {
    const btn = e.target.closest("[data-ocr]");
    if (!btn || btn.disabled) return;
    saveSettings({ ocr_model: btn.getAttribute("data-ocr") })
      .then(() => loadStatus())
      .catch((err) => alert(err.message));
  });

  on(asrLangSelect, "change", () => {
    saveSettings({ asr_language: asrLangSelect.value }).catch((e) => alert(e.message));
  });

  bind("country-file-list", "click", (e) => {
    const btn = e.target.closest("button[data-pick-country]");
    if (!btn) return;
    e.preventDefault();
    e.stopPropagation();
    openPicker(btn.getAttribute("data-pick-country")).catch((err) => alert(err.message));
  });

  bind("country-file-list", "change", (e) => {
    const sel = e.target.closest("[data-folder-asr]");
    if (!sel) return;
    e.stopPropagation();
    const id = sel.getAttribute("data-folder-id");
    if (!id) return;
    sel.disabled = true;
    api(`/api/folders/${id}`, {
      method: "PATCH",
      body: JSON.stringify({ asr_model: sel.value || "" }),
    })
      .then(() => refreshFolders())
      .catch((err) => alert(err.message))
      .finally(() => {
        sel.disabled = false;
      });
  });

  bind("btn-upload-file", "click", () => {
    openNativePicker(pickerState.country || state.country);
  });

  bind("file-input", "change", (e) => {
    const file = e.target.files && e.target.files[0];
    if (file) uploadPickedFile(file);
  });

  bind("btn-picker-close", "click", () => setPickerOpen(false));

  on($("picker-modal"), "click", (e) => {
    if (e.target === $("picker-modal")) setPickerOpen(false);
  });

  bind("btn-browse-up", "click", () => {
    if (pickerState.parent) browseTo(pickerState.parent).catch((e) => alert(e.message));
  });

  bind("btn-use-folder", "click", () => {
    if (pickerState.path) addWatchPath(pickerState.path);
  });

  bind("picker-shortcuts", "click", (e) => {
    const btn = e.target.closest("[data-goto]");
    if (!btn) return;
    browseTo(btn.getAttribute("data-goto")).catch((err) => alert(err.message));
  });

  bind("picker-list", "click", (e) => {
    const btn = e.target.closest("[data-act]");
    if (!btn) return;
    e.preventDefault();
    e.stopPropagation();
    const entry = pickerState.entries[Number(btn.getAttribute("data-idx"))];
    if (!entry || !entry.path) return;
    if (btn.getAttribute("data-act") === "open") {
      browseTo(entry.path).catch((err) => alert(err.message));
      return;
    }
    addWatchPath(entry.path);
  });

  function onCountryTabClick(e) {
    const btn = e.target.closest("[data-country]");
    if (btn) setCountry(btn.getAttribute("data-country"));
  }
  bind("pipeline-countries", "click", onCountryTabClick);
  bind("chat-countries", "click", onCountryTabClick);

  bind("home-countries", "click", (e) => {
    const btn = e.target.closest("[data-home-country]");
    if (!btn) return;
    setHomeCountry(btn.getAttribute("data-home-country") || "");
  });

  bind("stat-list", "click", (e) => {
    const btn = e.target.closest("[data-home-country]");
    if (!btn) return;
    setHomeCountry(btn.getAttribute("data-home-country") || "");
  });

  bind("period-tabs", "click", (e) => {
    const btn = e.target.closest("[data-period]");
    if (!btn) return;
    state.period = btn.getAttribute("data-period");
    $("period-tabs").querySelectorAll("[data-period]").forEach((b) => {
      b.classList.toggle("is-active", b === btn);
    });
    loadCountryStats();
    loadHomePage().catch(() => {});
  });

  bind("btn-stats-chart", "click", () => {
    state.chartOpen = !state.chartOpen;
    $("stat-chart").hidden = !state.chartOpen;
    $("btn-stats-chart").classList.toggle("is-active", state.chartOpen);
  });

  on(feed, "toggle", (e) => {
    const fold = e.target.closest && e.target.closest(".summary-fold");
    if (!fold || !feed.contains(fold)) return;
    const card = fold.closest(".card");
    if (!card) return;
    const id = Number(card.dataset.id);
    if (!id) return;
    if (fold.open) state.openSummaries.add(id);
    else state.openSummaries.delete(id);
  });

  on(feed, "click", async (e) => {
    const reprocess = e.target.closest("[data-reprocess]");
    if (reprocess) {
      try {
        await api(`/api/documents/${reprocess.getAttribute("data-reprocess")}/reprocess`, {
          method: "POST",
        });
      } catch (err) {
        alert(err.message);
      }
      return;
    }
    const indexBtn = e.target.closest("[data-index-rag]");
    if (!indexBtn) return;
    const id = indexBtn.getAttribute("data-index-rag");
    indexBtn.disabled = true;
    const prev = indexBtn.textContent;
    indexBtn.textContent = "Yozilmoqda…";
    try {
      const updated = await api(`/api/documents/${id}/index`, { method: "POST" });
      upsertCard({ ...(state.docs.get(Number(id)) || {}), ...updated, in_rag: true });
      if (updated.country) setCountry(updated.country, false);
      loadStatus().catch(() => {});
    } catch (err) {
      alert(err.message);
      indexBtn.disabled = false;
      indexBtn.textContent = prev;
    }
  });

  const io = new IntersectionObserver(
    (entries) => {
      if (entries.some((x) => x.isIntersecting)) loadPage();
    },
    { rootMargin: "200px" }
  );
  if (sentinel) io.observe(sentinel);

  window.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      setPickerOpen(false);
      setModelPanelOpen(false);
    }
  });

  window.addEventListener("hashchange", () => {
    setView(location.hash.replace("#", "") || "home");
  });

  /* ---------- Start ---------- */

  (async function init() {
    setTheme(currentTheme());
    setView(location.hash.replace("#", "") || "home");
    let open = false;
    try {
      open = localStorage.getItem(SIDEBAR_KEY) === "1" && state.view === "pipeline";
    } catch (_) {
      /* ignore */
    }
    setSidebarOpen(open);

    try {
      await loadStatus();
    } catch (err) {
      pillDevice.textContent = "Serverga ulanib bo‘lmadi";
      pillDevice.classList.add("pill-err");
      console.error(err);
    }
    if (sentinel) sentinel.hidden = false;
    updateCountryChrome();
    await Promise.all([loadPage(), loadHomePage(), loadCountryStats()]);
    refreshFolders().catch(() => {});
    connectWs();
    setInterval(pollUpdates, 4000);
  })();
})();
