/* KVMem monitor. Same-origin, no third-party runtime or persisted request content. */
"use strict";
(() => {
  const $ = (s) => document.querySelector(s),
    $$ = (s) => [...document.querySelectorAll(s)];
  const i18n = window.monitorI18n, t = i18n.t;
  const app = $("#app"),
    GiB = 1024 ** 3,
    MiB = 1024 ** 2;
  const number = (n) => typeof n === "number" && Number.isFinite(n);
  const fmt = (n, digits = 0) =>
    number(n)
      ? n.toLocaleString(i18n.language, {
          maximumFractionDigits: digits,
          minimumFractionDigits: digits,
        })
      : "—";
  const ratio = (a, b) =>
    number(a) && number(b) && b > 0 ? (100 * a) / b : null;
  const text = (key, value) => {
    const el = $(`[data-value="${key}"]`);
    if (el) el.textContent = value;
  };
  const storage = (area, key, value) => {
    try {
      if (value === undefined) return area.getItem(key);
      if (value === null) area.removeItem(key);
      else area.setItem(key, value);
    } catch {}
    return null;
  };
  const statuses = new Proxy({
    idle: "空闲",
    queued: "排队",
    reading: "预填充",
    generating: "生成中",
    processing: "处理中",
    media: "处理图像输入",
    completed: "已完成",
    disconnected: "已断开",
    error: "失败",
  }, { get: (map, name) => t(map[name] || "处理中") });
  // A request-level hint, not a claim about the encoder's current stage.
  function processingMedia(row) {
    return row?.has_media === true && ["processing", "reading"].includes(row.status) &&
      !(row.generated_tokens > 0) && !["thinking", "answering"].includes(row.phase);
  }
  function displayState(row) { return processingMedia(row) ? "media" : row.status; }
  function requestStatus(row) {
    if (row.finish_reason === "length") return t(row.finish_phase === "thinking" ? "思考阶段达到输出上限" : "达到输出上限");
    if (["cancel", "cancelled"].includes(row.finish_reason)) return t("已停止");
    if (row.finish_reason === "disconnect") return t("连接中断");
    if (row.finish_reason === "error" || row.status === "error") return t("生成失败");
    if (row.finish_reason === "stop") return t("正常结束");
    if (row.finish_reason === "tool_calls") return t("工具调用");
    if (row.phase === "thinking") return t("思考中");
    if (row.phase === "answering") return t("输出正文");
    return statuses[displayState(row)];
  }
  function meanSpeed(row) { return row?.decode_tps_mean ?? row?.decode_tps; }
  function prefillMetrics(snapshot, row, active) {
    const live = ['ninfer','radiance'].includes(snapshot?.backend) && active && row?.status === 'reading';
    if (!live) return {live: false, rate: row?.prefill_tps, ms: row?.prefill_ms,
      tokens: row?.prompt_new_tokens ?? (number(row?.prompt_tokens) && number(row?.cached_tokens)
        ? Math.max(0, row.prompt_tokens - row.cached_tokens) : null)};
    // NInfer's completed prompt frontier includes the reused prefix. Only newly
    // computed tokens belong in the rate, using the same native progress clock.
    const processed = row.prompt_processed_tokens, cached = row.cached_tokens;
    const valid = number(processed) && number(cached) && cached >= 0 && processed >= cached &&
      (!number(row.prompt_tokens) || processed <= row.prompt_tokens);
    const tokens = valid ? processed - cached : null;
    const ms = number(row.prefill_elapsed_ms) && row.prefill_elapsed_ms >= 0 ? row.prefill_elapsed_ms : null;
    const rate = number(tokens) && ms > 0 ? tokens / ms * 1000 : null;
    return {live: true, rate: number(rate) ? rate : null, ms, tokens};
  }
  function prefillTimes(metrics, row) {
    return metrics.live ? t("预填充中 · 累计平均 · {0}s", fmt(number(metrics.ms) ? metrics.ms / 1000 : null, 2))
      : t("预填充 {0}s · 首 token {1}s", fmt(number(metrics.ms) ? metrics.ms / 1000 : null, 2),
        fmt(number(row?.ttft_ms) ? row.ttft_ms / 1000 : null, 2));
  }
  // NInfer's operational log reports engine throughput for the preceding interval.
  // The request's decode_tps is cumulative and must remain the request mean.
  function parseNinferThroughput(log) {
    if (!log?.running || log.backend !== 'ninfer' || typeof log.text !== 'string') return null;
    let sample = null;
    const clean = log.text.replace(/\x1b\[[0-?]*[ -/]*[@-~]/g, '');
    for (const line of clean.split('\n')) {
      const match = line.match(/^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})\s+INFO\s+throughput \| ([\d.]+)\s*(ms|s) \| (.*)$/);
      if (!match) continue;
      const at = new Date(match[1].replace(' ', 'T')).getTime();
      const seconds = Number(match[2]) / (match[3] === 'ms' ? 1000 : 1);
      const decode = match[4].match(/(?:^| \| )decode ([\d.]+)([kMGT]?) tok\/s \([\d,]+ tok\)(?: \| |$)/);
      if (!/(?:^| \| )running \d+/.test(match[4])) continue;
      if (!decode && /(?:^| \| )decode /.test(match[4])) { sample = null; continue; }
      const rate = decode ? Number(decode[1]) * 1000 ** ['', 'k', 'M', 'G', 'T'].indexOf(decode[2]) : 0;
      if (Number.isFinite(at) && Number.isFinite(rate) && rate >= 0 && seconds > 0 && at >= log.started_at)
        sample = {at, seconds, rate};
    }
    return sample;
  }
  function ninferLiveSpeed(snapshot, row, active, sample) {
    if (!active || row?.status !== 'generating' || !sample) return null;
    const age = snapshot.sampled_at_ms - sample.at;
    // Exclude old sessions/requests and stale reports, including after log failure.
    return age >= -1000 && age <= sample.seconds * 2000 + 2000 && sample.at >= row.started_at_ms
      ? sample.rate : null;
  }
  let throughputLog = null;
  function requestLiveSpeed(snapshot, row, active) {
    if (!active || row?.status !== 'generating' || !number(row.decode_tps_live) ||
        !number(row.decode_tps_sampled_at_ms) || !number(row.decode_tps_window_ms)) return null;
    const age = snapshot.sampled_at_ms - row.decode_tps_sampled_at_ms;
    return row.decode_tps_live >= 0 && age >= -1000 && age <= Math.max(6000, row.decode_tps_window_ms * 2) + 1000
      ? row.decode_tps_live : null;
  }
  let theme = storage(localStorage, "kvmem-monitor-theme") || "dark";
  let key = storage(sessionStorage, "kvmem-monitor-key") || "";
  let selected = "",
    followLatest = true,
    detail = "",
    all = false,
    state = null,
    props = null,
    slots = [],
    stats = null;
  let timer,
    controller,
    busy = false,
    failures = 0,
    authNeeded = false,
    lastGood = 0,
    stopped = false;
  let lastError = null, restartNotice = false, consoleRuntime = null, runtimePropsInstance = null;
  let series = {},
    previousDisk = null,
    speedRequest = null,
    currentInstance = null;
  const media = matchMedia("(prefers-color-scheme:dark)");
  function applyTheme() {
    if (!["dark", "blue", "system"].includes(theme)) theme = "dark";
    app.dataset.theme =
      theme === "system" ? (media.matches ? "dark" : "blue") : theme;
    $("[data-theme-select]").value = theme;
  }
  applyTheme();
  let density = storage(localStorage, "kvmem-monitor-density") === "compact" ? "compact" : "comfortable";
  function applyDensity() {
    app.dataset.density = density;
    $("[data-density-select]").value = density;
  }
  applyDensity();
  $("[data-density-select]").onchange = (e) => {
    density = e.target.value === "compact" ? "compact" : "comfortable";
    storage(localStorage, "kvmem-monitor-density", density);
    applyDensity();
  };
  media.addEventListener("change", applyTheme);
  $("[data-theme-select]").onchange = (e) => {
    theme = e.target.value;
    storage(localStorage, "kvmem-monitor-theme", theme);
    applyTheme();
  };
  i18n.apply();
  $("[data-language-select]").onchange = (e) => {
    i18n.apply(e.target.value);
    i18n.save();
    $("[data-all]").textContent = all ? t("收起") : t("显示全部");
    render(false);
    renderConnection();
    window.dispatchEvent(new Event("kvmem-language-change"));
  };
  function page(name) {
    $$("[data-view]").forEach((el) => (el.hidden = el.dataset.view !== name));
    $$("[data-page]").forEach((el) =>
      el.setAttribute("aria-pressed", String(el.dataset.page === name)),
    );
    window.dispatchEvent(new CustomEvent("kvmem-page-change", {detail: name}));
  }
  $$("[data-page]").forEach((el) => (el.onclick = () => page(el.dataset.page)));
  window.addEventListener("kvmem-navigate", event => page(event.detail));
  function bar(name, value) {
    $(`[data-bar="${name}"]`).style.width =
      `${number(value) ? Math.max(0, Math.min(100, value)) : 0}%`;
  }
  function capacity(name, used, total) {
    // The storage API uses UINT64_MAX for an unbounded host budget. JSON rounds it to 2^64.
    const unbounded = number(total) && total >= 2 ** 64;
    const amount =
      number(used) && used > 0 && used / GiB < 0.1
        ? "<0.1"
        : fmt(number(used) ? used / GiB : null, 1);
    text(
      name,
      number(used)
        ? unbounded
          ? t("{0} GiB / 未设上限", amount)
          : `${amount} / ${fmt(number(total) ? total / GiB : null, 1)} GiB`
        : t("未采集"),
    );
    bar(name, unbounded ? null : ratio(used, total));
  }
  function options(el, items, value) {
    const signature = JSON.stringify(items);
    if (el.dataset.options !== signature) {
      el.replaceChildren(
        ...items.map(([id, label]) => {
          const o = document.createElement("option");
          o.value = id;
          o.textContent = label;
          return o;
        }),
      );
      el.dataset.options = signature;
    }
    el.value = value;
  }
  function records() {
    return state
      ? [...state.active]
          .sort((a, b) => b.started_at_ms - a.started_at_ms)
          .concat(state.recent)
      : [];
  }
  function currentRequest() {
    return (
      (!followLatest && records().find((r) => r.id === selected)) ||
      records()[0] ||
      null
    );
  }
  $("[data-request-select]").onchange = (e) => {
    selected = e.target.value;
    followLatest = !selected;
    series.speed = [];
    render(false);
  };
  $("[data-hardware]").onclick = () => page("about");
  $("[data-all]").onclick = () => {
    all = !all;
    $("[data-all]").textContent = all ? t("收起") : t("显示全部");
    $("[data-all]").setAttribute("aria-expanded", String(all));
    renderTable();
  };
  $("[data-close]").onclick = () => {
    detail = "";
    $("[data-detail]").hidden = true;
  };
  function renderTable() {
    const rows = records();
    const body = $("[data-requests]");
    body.replaceChildren();
    for (const row of all ? rows : rows.slice(0, 5)) {
      const tr = document.createElement("tr"),
        time = document.createElement("button");
      time.textContent = new Date(row.started_at_ms).toLocaleTimeString(
        i18n.language,
        { hour12: false },
      );
      time.onclick = () => {
        detail = row.id;
        renderDetail();
      };
      let td = document.createElement("td");
      td.append(time);
      tr.append(td);
      td = document.createElement("td");
      const badge = document.createElement("span");
      badge.className = "s-badge";
      badge.dataset.kind = displayState(row);
      badge.textContent = requestStatus(row);
      td.append(badge);
      tr.append(td);
      const values = [
        fmt(row.prompt_tokens),
        fmt(row.generated_tokens),
        fmt(meanSpeed(row), 1),
        `${fmt(row.elapsed_ms / 1000, 1)}s`,
      ];
      for (const value of values) {
        td = document.createElement("td");
        td.className = "s-num";
        td.textContent = value;
        tr.append(td);
      }
      body.append(tr);
    }
    if (!rows.length) {
      const tr = document.createElement("tr"),
        td = document.createElement("td");
      td.colSpan = 6;
      td.className = "s-empty";
      td.textContent = t("暂无请求，发送一条消息后会显示统计。");
      tr.append(td);
      body.append(tr);
    }
    renderDetail();
  }
  function renderDetail() {
    const row = records().find((r) => r.id === detail);
    $("[data-detail]").hidden = !row;
    if (!row) return;
    $("[data-detail-title]").textContent =
      t("请求 #{0} · {1}", row.id, requestStatus(row));
    const prefill = prefillMetrics(state, row, state.active.some((r) => r.id === row.id));
    $("[data-detail-body]").textContent =
      t("入口：{0}\n输入 {1} tokens · 复用 {2} · 输出 {3}\n预填充 {4} tok/s · 输入复用率 {5}%\n输出上限 {6} · 耗时 {7} 秒", row.endpoint, fmt(row.prompt_tokens), fmt(row.cached_tokens), fmt(row.generated_tokens), fmt(prefill.rate, 1), fmt(ratio(row.cached_tokens, row.prompt_tokens), 1), fmt(row.limit), fmt(row.elapsed_ms / 1000, 2)) + "\n" +
      prefillTimes(prefill, row);
    const lines = [];
    if(row.image) {
      const v=row.image;
      lines.push((i18n.language==='en'?'Image decode / preprocess / queue / encode ms: ':'图片解码 / 预处理 / 排队 / 编码 ms：')+
        [v.decode_ms,v.preprocess_ms,v.queue_ms,v.encode_ms].map(x=>fmt(x,1)).join(' / '));
      lines.push((i18n.language==='en'?'Image cache hits / misses: ':'图片缓存命中 / 未命中：')+`${fmt(v.cache_hits)} / ${fmt(v.cache_misses)}`);
    }
    const restore = row.cache_semantics_revision >= 2 ? row.restore_path : row.cache_source;
    if (typeof restore === 'string' && restore) lines.push((i18n.language === 'en' ? 'Cache source' : '缓存来源') + `: ${restore}`);
    if (number(row.planned_cached_tokens)) lines.push((i18n.language === 'en' ? 'Planned reuse' : '计划复用') + `: ${fmt(row.planned_cached_tokens)} tokens`);
    if (number(row.actual_cached_tokens)) lines.push((i18n.language === 'en' ? 'Actual reuse' : '实际复用') + `: ${fmt(row.actual_cached_tokens)} tokens`);
    if (number(row.decode_ms)) lines.push(t("解码耗时 {0}s", fmt(row.decode_ms / 1000, 2)));
    if (number(row.drafts_offered) && number(row.drafts_accepted))
      lines.push(t("推测解码：提议 {0} / 接受 {1} · 接受率 {2}%", fmt(row.drafts_offered), fmt(row.drafts_accepted), fmt(ratio(row.drafts_accepted, row.drafts_offered), 1)));
    if (number(row.expert_hit_rate) && row.expert_hit_rate >= 0 && row.expert_hit_rate <= 1)
      lines.push(t("所选请求专家命中率 {0}%", fmt(row.expert_hit_rate * 100, 1)));
    if (lines.length) $("[data-detail-body]").textContent += "\n" + lines.join("\n");
  }
  function gauge(element, percent, label) {
    let arc = element.querySelector("[data-dial-arc]");
    if (!arc) {
      const ns = "http://www.w3.org/2000/svg";
      const svg = document.createElementNS(ns, "svg");
      svg.setAttribute("viewBox", "0 0 120 120");
      svg.setAttribute("aria-hidden", "true");
      for (const name of ["s-dial-track", "s-dial-arc"]) {
        const circle = document.createElementNS(ns, "circle");
        for (const [key, value] of Object.entries({cx: 60, cy: 60, r: 50, pathLength: 100, transform: "rotate(135 60 60)", class: name}))
          circle.setAttribute(key, value);
        circle.setAttribute("stroke-dasharray", "75 100");
        if (name === "s-dial-arc") { circle.dataset.dialArc = ""; arc = circle; }
        svg.append(circle);
      }
      element.prepend(svg);
    }
    const available = number(percent);
    const amount = available ? Math.max(0, Math.min(100, percent)) : 0;
    arc.setAttribute("stroke-dasharray", `${amount * 0.75} 100`);
    // A round line cap would otherwise turn zero or missing data into a colored dot.
    arc.style.visibility = amount > 0 ? "visible" : "hidden";
    element.setAttribute("role", "img");
    element.setAttribute("aria-label", `${label} · ${available ? t("占比 {0}%", fmt(percent, 1)) : t("比例未采集")}`);
  }
  function chart(name, value, append, target = null, ceiling = null) {
    const points = (series[name] ||= []);
    if (append) {
      points.push(number(value) ? value : null);
      if (points.length > 60) points.shift();
    }
    const svg = target || $(`[data-series="${name}"]`);
    if (!svg) return;
    svg.replaceChildren();
    const valid = points.filter(number);
    if (valid.length < 2) {
      svg.setAttribute("aria-label", t("等待更多真实采样"));
      return;
    }
    const max = ceiling || Math.max(1, ...valid) * 1.1;
    let path = "",
      drawing = false;
    points.forEach((p, i) => {
      if (!number(p)) {
        drawing = false;
        return;
      }
      path += `${drawing ? "L" : "M"}${((i * 240) / 59).toFixed(1)},${(30 - (p / max) * 27).toFixed(1)} `;
      drawing = true;
    });
    const line = document.createElementNS("http://www.w3.org/2000/svg", "path");
    line.setAttribute("d", path);
    line.setAttribute("fill", "none");
    line.setAttribute("stroke", "currentColor");
    line.setAttribute("stroke-width", "1.8");
    line.setAttribute("vector-effect", "non-scaling-stroke");
    svg.append(line);
    svg.setAttribute(
      "aria-label",
      t("最近 {0} 次真实采样，当前 {1}", points.length, fmt(value, 1)),
    );
  }
  const pciKey = (value) => typeof value === "string" && /^[0-9a-f]+:[0-9a-f]+:[0-9a-f]+\.[0-7]$/i.test(value)
    ? value.toLowerCase().split(/[:.]/).map((n) => parseInt(n, 16)).join(":") : null;
  function matchesDevice(device, gpu) {
    if (device.uuid && gpu.uuid) return device.uuid.toLowerCase() === gpu.uuid.toLowerCase();
    const a = pciKey(device.pci_bus_id), b = pciKey(gpu.pci_bus_id);
    return a !== null && b !== null ? a === b : null;
  }
  function renderGpus(gpus, fresh, append) {
    const devices = props.execution?.monitor_devices;
    const known = Array.isArray(devices);
    const bound = known ? devices : [];
    const mode = props.execution?.mode;
    const modeLabel = mode === "layer" ? t("层分片") : mode === "tensor" ? t("张量并行") : t("多卡推理");
    text("execution", !known ? t("推理设备待确认") : !bound.length ? t("CPU 推理") : bound.length === 1 ? t("单卡推理") : t("{0} 卡协同 · {1}", bound.length, modeLabel));
    text("gpu-count", known ? t("{0} 张设备参与当前服务", bound.length) : t("设备绑定信息未提供"));
    $("[data-device-note]").textContent = !known ? t("当前服务尚未提供设备绑定信息；不根据负载猜测推理显卡。") : !bound.length ? t("当前模型未使用显卡。") : t("只显示当前服务绑定的显卡 · 整卡读数包含其他进程");
    $("[data-device-note]").hidden = known && bound.length > 0;
    const grid = $("[data-gpus]");
    grid.dataset.count = String(bound.length);
    const signature = JSON.stringify([i18n.language, bound]);
    if (grid.dataset.devices !== signature) {
      grid.replaceChildren(...bound.map((d) => {
        const card = $("#gpu-card").content.firstElementChild.cloneNode(true);
        card.dataset.device = d.uuid || d.pci_bus_id || (Number.isInteger(d.hip_ordinal) ? `hip:${d.hip_ordinal}` : d.name);
        return card;
      }));
      grid.dataset.devices = signature;
    }
    bound.forEach((device, index) => {
      const matched = gpus.find((g) => matchesDevice(device, g));
      const g = fresh ? matched || {} : {};
      const card = grid.children[index];
      const total = number(g.total_bytes) ? g.total_bytes : device.total_bytes;
      const usage = ratio(g.used_bytes, total);
      const logicalId = Number.isInteger(device.hip_ordinal) ? [`HIP ${device.hip_ordinal}`, device.arch].filter(Boolean).join(" · ") : t("设备身份未采集");
      const values = {
        name: (device.name || matched?.name || t("推理设备")).replace(/^NVIDIA GeForce /, ""),
        id: device.uuid ? `UUID · ${device.uuid.slice(-8)}` : matched?.uuid ? `UUID · ${matched.uuid.slice(-8)}` : device.pci_bus_id ? `PCI · ${device.pci_bus_id}` : logicalId,
        status: !matched ? t("读数未采集") : !fresh ? t("读数已过期") : g.unavailable_metrics?.length ? t("部分指标未采集") : t("本服务"),
        load: fmt(g.utilization_percent),
        vram: fmt(number(g.used_bytes) ? g.used_bytes / GiB : null, 1),
        total: number(total) ? t("共 {0} GiB", fmt(total / GiB, 1)) : t("容量未采集"),
        capacity: number(usage) ? t("已用 {0}%", fmt(usage)) : t("比例未采集"),
        temp: fmt(g.temperature_c), power: fmt(g.power_w, 1),
        pcie: number(g.pcie_generation) ? `Gen${g.pcie_generation} ×${fmt(g.pcie_width)}` : "—",
        rx: fmt(number(g.pcie_rx_bytes_per_second) ? g.pcie_rx_bytes_per_second / MiB : null, 1),
      };
      for (const [key, value] of Object.entries(values)) card.querySelector(`[data-gpu="${key}"]`).textContent = value;
      card.querySelector('[data-gpu="power"]').title = fresh && matched && !number(g.power_w) ? t("驱动未提供整卡功耗读数") : "";
      gauge(card.querySelector('[data-dial="load"]'), g.utilization_percent, t("GPU 负载 {0}%", fmt(g.utilization_percent)));
      gauge(card.querySelector('[data-dial="vram"]'), usage, t("整卡显存 {0} / {1} GiB", values.vram, fmt(number(total) ? total / GiB : null, 1)));
      chart(`gpu:${card.dataset.device}`, g.utilization_percent, append, card.querySelector("[data-gpu-chart]"), 100);
    });
    const hardwareList = $('[data-about="devices"]');
    hardwareList.replaceChildren(...gpus.map((g) => {
      const item = document.createElement("p");
      const matches = bound.map((device) => matchesDevice(device, g));
      const role = matches.includes(true) ? t("参与当前服务") : !known || matches.includes(null) ? t("绑定待确认") : t("未参与当前服务");
      item.textContent = `${g.name} · ${role}\n${g.uuid || g.pci_bus_id || t("设备身份未采集")}\n${fresh ? t("{0}% 负载 · {1} / {2} GiB", fmt(g.utilization_percent), fmt(number(g.used_bytes) ? g.used_bytes / GiB : null, 1), fmt(number(g.total_bytes) ? g.total_bytes / GiB : null, 1)) : t("读数已过期")}`;
      return item;
    }));
    if (!gpus.length) hardwareList.textContent = t("硬件读数未采集");
  }
  const sum = (list, field) =>
    list.length && list.every((x) => number(x?.[field]))
      ? list.reduce((n, x) => n + x[field], 0)
      : null;
  function cacheData() {
    if (state.backend === "polystrata" || state.backend === "radiance" ||
        (state.backend === "llamacpp" && Object.hasOwn(state.kvmem || {}, "physical_payload_bytes"))) return state.kvmem || {};
    if (state.backend === "ninfer")
      return stats?.kvmem || props.kvmem?.usage || {};
    const storage = slots.map((s) => s.kvmem?.storage).filter(Boolean),
      conversations = slots.map((s) => s.kvmem?.conversations).filter(Boolean);
    const c = conversations.some((c) => number(c.resident))
      ? conversations.slice(0, 1)
      : conversations;
    const data = {
      retained_sessions: c.reduce((n, item) => n + (number(item.retained_sessions) ? item.retained_sessions : item.count || 0), 0),
      history_hits: sum(c, "extends"),
      history_evictions: sum(c, "evictions"),
    };
    for (const name of [
      "host_payload_bytes",
      "host_payload_budget_bytes",
      "disk_payload_bytes",
      "disk_payload_budget_bytes",
      "disk_read_bytes",
      "disk_written_bytes",
      "disk_reads",
      "disk_writes",
      "disk_errors",
    ])
      data[name] = sum(storage, name);
    return data;
  }
  function renderActualSpec(request) {
    const actual = $('[data-value="speculative-live"]');
    if (actual) {
      const mtp = number(request?.mtp_drafts_offered) && number(request?.mtp_drafts_accepted);
      const offered = mtp ? request.mtp_drafts_offered : request?.drafts_offered;
      const accepted = mtp ? request.mtp_drafts_accepted : request?.drafts_accepted;
      actual.hidden = !number(offered) || !number(accepted);
      actual.textContent = t(mtp ? "所选请求 MTP · 接受 {0} / 提议 {1}" : "所选请求推测 · 接受 {0} / 提议 {1}",fmt(accepted),fmt(offered));
      const ng = request?.speculative;
      if (number(ng?.ngram_drafted_tokens) && ng.ngram_drafted_tokens > 0)
        actual.textContent += t(" · N-gram 接受 {0} / 提议 {1}",fmt(ng.ngram_accepted_tokens),fmt(ng.ngram_drafted_tokens));
    }
  }
  const metricLabels = {
    logical_payload_bytes:['KV 逻辑引用大小','Logical KV references'],
    physical_payload_bytes:['RAM/SSD 物理计费','Physical RAM/SSD charge'],
    sharing_credit_bytes:['KV 共享节省下界','KV sharing savings lower bound'],
    shared_forks:['跨会话缓存分叉','Shared prefix forks'],
    logical_state_bytes:['循环状态逻辑引用','Logical recurrent-state references'],
    physical_state_bytes:['循环状态物理占用','Physical recurrent state'],
    physical_checkpoint_bytes:['检查点及检索元数据物理占用','Physical checkpoint and retrieval metadata'],
    state_sharing_credit_bytes:['循环状态共享节省','Recurrent-state sharing savings'],
    state_references:['循环状态引用数','Recurrent-state references'],
    payload_references:['KV 引用数','KV references'],
    unique_payload_records:['KV 唯一记录数','Unique KV records'],
    active_sessions:['活动会话','Active sessions'], retained_sessions_limit:['保留会话上限','Retained session limit'],
    history_misses:['历史未命中','History misses'], history_swaps:['历史交换','History swaps'],
    history_restored_bytes:['历史恢复量','History restored'], history_spilled_bytes:['历史转存量','History spilled'],
    disk_read_bytes:['磁盘累计读取','Total disk reads'], disk_written_bytes:['磁盘累计写入','Total disk writes'],
    disk_reads:['磁盘读取次数','Disk read operations'], disk_hits:['磁盘命中次数','Disk hits'],
    disk_writes:['磁盘写入次数','Disk write operations'], disk_errors:['磁盘错误','Disk errors'],
    retained_records:['历史记录数','Retained records'], retained_records_limit:['历史记录上限','Retained record limit'],
    retained_recovery_points:['保留恢复点数','Retained recovery points'],
    checkpoint_kv_capture_bytes:['快照累计实际捕获','Checkpoint KV captured'],
    checkpoint_kv_logical_bytes:['快照累计视图大小','Checkpoint KV view bytes'],
    checkpoint_bytes:['检查点容量','Checkpoint size'], checkpoint_tokens:['检查点 tokens','Checkpoint tokens'],
    resident_pages:['驻留页','Resident pages'], mtp_resident_pages:['MTP 驻留页','MTP resident pages'],
    statistics_bytes:['统计数据占用','Statistics memory'], statistics_staging_bytes:['统计暂存占用','Statistics staging'],
    transfer_staging_bytes:['传输暂存占用','Transfer staging'], active_payload_reservation_bytes:['活动数据预留','Active payload reservation'],
    retained_payload_budget_bytes:['保留数据预算','Retained payload budget'], evaluated_tokens:['已计算 tokens','Evaluated tokens'],
    selected_tokens:['驻留 token 预算','Resident token budget'], reserve_tokens:['生成预留 tokens','Generation reserve tokens'],
    host_payload_bytes:['RAM 数据占用','RAM payload'], host_payload_budget_bytes:['RAM 数据预算','RAM payload budget'],
    retained_logical_bytes:['保留缓存逻辑大小','Retained logical bytes'],
    shared_payload_credit_bytes:['共享记录去重节省','Shared payload credit'],
    accounted_cache_bytes:['缓存去重计费大小','Accounted cache bytes'],
    disk_payload_bytes:['SSD 数据占用','SSD payload'], disk_payload_budget_bytes:['SSD 数据预算','SSD payload budget'],
    retained_sessions:['保留会话','Retained sessions'], history_hits:['历史命中','History hits'], history_evictions:['历史淘汰','History evictions'],
    own_history_hits:['自身历史命中','Own history hits'],shared_history_hits:['共享历史命中','Shared history hits'],
    planned_history_hits:['计划历史命中','Planned history hits'],planned_reuse_tokens:['计划复用 tokens','Planned reused tokens'],actual_reuse_tokens:['实际复用 tokens','Actual reused tokens'],
    max_context:['逻辑上下文上限','Context limit'], max_concurrency:['并发上限','Concurrency limit'],
    max_pending_requests:['排队上限','Pending request limit'], max_admitted_requests:['接纳上限','Admission limit'],
    kv_capacity_pages:['KV 容量页数','KV capacity pages'], kv_capacity_tokens:['KV 容量 tokens','KV capacity tokens'], kv_page_tokens:['每页 tokens','Tokens per page'],
    device_state_slots:['设备状态槽','Device state slots'], host_state_slots:['主机状态槽','Host state slots'], host_kv_bytes:['主机 KV','Host KV'],
    depth:['排队数','Queue depth'], entries:['排队请求','Queued requests'], request_id:['请求 ID','Request ID'],
    admitted:['已接纳','Admitted'], decode_ready:['待解码','Decode ready'], materializing:['准备状态','Materializing'],
    prefilling:['预填充中','Prefilling'], running:['执行中','Running'], waiting:['等待中','Waiting'], peak_admitted:['接纳峰值','Peak admitted'],
    average_size:['平均批大小','Average batch size'], rounds:['执行轮数','Rounds'], row_rounds:['请求执行轮数','Row rounds'],
    committed_decode_tokens:['已提交生成 tokens','Committed decode tokens'], computed_prefill_tokens:['新算输入 tokens','Computed prefill tokens'],
    reused_prompt_tokens:['复用输入 tokens','Reused prompt tokens'], decode_rounds:['解码轮数','Decode rounds'], decode_row_rounds:['解码请求轮数','Decode row rounds'],
    committed_decode:['已提交生成 tokens','Committed decode tokens'], computed_prefill:['新算输入 tokens','Computed prefill tokens'],
    decode:['生成','Decode'], prefill:['预填充','Prefill'], bytes:['数据量','Bytes'], pages:['页数','Pages'], count:['次数','Count'], seconds:['耗时','Time'],
    d2d:['设备间','Device to device'], d2h:['设备到主机','Device to host'], h2d:['主机到设备','Host to device'],
    captures:['捕获','Captures'], selections:['选择','Selections'], state_operations:['状态操作','State operations'],
    main_kv_transfers:['主 KV 传输','Main KV transfers'], backend_kv_transfers:['后端 KV 传输','Backend KV transfers'], state_transfers:['状态传输','State transfers'],
    pressure:['压力处理','Pressure'], occupancy:['占用','Occupancy'], completed:['完成','Completed'], skipped:['跳过','Skipped'], aborted:['中止','Aborted'],
    forks:['分叉','Forks'], moves:['移动','Moves'], restores:['恢复','Restores'], spills:['转存','Spills'],
    active:['活动','Active'], resident:['驻留','Resident'], parked:['停放','Parked'], max:['上限','Limit'],
    extends:['会话延续次数','Conversation extensions'], refusals:['拒绝','Refusals'], resets:['重置','Resets'], evictions:['淘汰','Evictions'], switches:['切换','Switches'],
    enabled:['启用','Enabled'], verified:['已验证','Verified'], cuda_residency:['CUDA 驻留','CUDA residency'],
    cuda_free_bytes:['CUDA 可用显存','CUDA free memory'], cuda_total_bytes:['CUDA 总显存','CUDA total memory'], dedicated_bytes:['专用显存','Dedicated memory'],
    shared_bytes:['共享内存','Shared memory'], host_used_bytes:['主机已用','Host used'], host_pool_bytes:['主机池','Host pool'],
    device_allocated_bytes:['设备已分配','Device allocated'], verified_reserve_bytes:['已验证预留','Verified reserve'], shared_baseline_bytes:['共享内存基线','Shared memory baseline'],
    elapsed_seconds:['累计耗时','Cumulative time'], work_class_seconds:['工作分类耗时','Work class time'], device_wait_seconds:['设备等待时间','Device wait time'],
    detail_invocations:['调用次数','Invocations'], detail_subset_seconds:['明细子集耗时','Detail subset time'],
    detail_microseconds_per_invocation:['每次调用耗时','Time per invocation'], units:['工作单元','Work units'],
    decode_device_wait_microseconds_per_round:['每轮解码设备等待','Device wait per decode round'],
    decode_host_microseconds_per_round:['每轮解码主机耗时','Host time per decode round'],
    decode_host_microseconds_per_row_round:['每请求轮解码主机耗时','Host time per decode row round'],
  };
  function metricName(name) {
    return metricLabels[name]?.[i18n.language === 'en' ? 1 : 0] || name.replaceAll('_', ' ');
  }
  function metricValue(path, value) {
    if (value === null) return '—';
    if (typeof value === 'boolean') return t(value ? '是' : '否');
    if (!number(value)) return typeof value === 'string' ? value.slice(0,256) : '—';
    const leaf = path.at(-1);
    if (leaf === 'bytes' || leaf.endsWith('_bytes') || leaf === 'bytes_max') return fmt(value / MiB, 2) + ' MiB';
    if (path.some(p => p.includes('microseconds'))) return fmt(value, 2) + ' μs';
    if (leaf.endsWith('_ms')) return fmt(value, 2) + ' ms';
    if (path.some(p => p === 'seconds' || p.endsWith('_seconds'))) return fmt(value, 2) + ' s';
    if (path.includes('throughput_tokens_per_second')) return fmt(value, 2) + ' tok/s';
    return fmt(value, Number.isInteger(value) ? 0 : 3);
  }
  // Render only the selected native metric groups, never prompts or responses.
  // Keep the details elements stable so polling does not collapse open sections.
  function radianceMetricGroups(stats, engine) {
    const groups = [];
    const pick = keys => Object.fromEntries(keys.filter(k => stats?.[k] !== undefined).map(k => [k,stats[k]]));
    if (stats) {
      groups.push({id:'scheduler',label:'Radiance · 调度器',value:pick(['running','waiting','preempted','in_flight','queue_cap','total_requests','steps','steps_prefill','steps_decode','steps_mixed','preemptions'])});
      groups.push({id:'memory',label:'Radiance · KV 池',value:{
        kv_used_bytes:stats.kv_bytes_used,kv_capacity_bytes:stats.kv_bytes_total,kv_carved_bytes:stats.kv_bytes_carved,
        kv_tokens_used:stats.kv_tokens_used,kv_tokens_total:stats.kv_tokens_total,kv_util:stats.kv_util}});
      groups.push({id:'speculation',label:'Radiance · 草稿验证',value:pick(['draft_accept','draft_positions'])});
      // Native prefix-cache counters are intentionally excluded: KVMem owns reuse.
      // /stats throughput is global and never substitutes for a request's speed.
      groups.push({id:'throughput_tokens_per_second',label:'Radiance · 引擎区间吞吐',value:pick(['prefill_tps','decode_tps'])});
    }
    if (engine) groups.push({id:'kv_allocation',label:'Radiance · KV 分配与放置',value:{kv_groups:engine.kv_groups,
      embedding_placement:engine.embedding_placement,placement:engine.placement,speculation:engine.speculation}});
    return groups;
  }
  function renderNativeMetrics() {
    const root = $('[data-backend-sections]');
    if (!root) return;
    const groups = state.backend === 'radiance' ? radianceMetricGroups(stats,props.radiance_engine) : [];
    if (state.backend === 'ninfer' && stats) {
      for (const [key, label] of [
        ['requests','请求状态'], ['queue','等待队列'], ['capacity','引擎容量'], ['occupancy','引擎驻留'],
        ['memory','内存分配'], ['context_cache','上下文缓存'], ['scheduler','调度器'], ['decode_batch','解码批处理'],
        ['counters','引擎累计计数'], ['tokens','引擎累计 tokens'], ['throughput_tokens_per_second','启动以来平均吞吐'],
        ['host_work','主机与设备耗时'], ['experts','专家缓存'],
      ]) if (stats[key] && typeof stats[key] === 'object') groups.push({id:key,label,value:stats[key]});
      if (number(stats.uptime_seconds)) groups.push({id:'uptime',label:'运行时间',value:{uptime_seconds:stats.uptime_seconds}});
    }
    if (state.backend === 'llamacpp') slots.forEach((slot,index) => {
      if (slot.kvmem?.conversations) groups.push({id:'lane-'+index,label:t('执行槽 {0} · 会话',fmt(slot.id ?? index)),value:slot.kvmem.conversations});
    });
    const keep = new Set();
    for (const group of groups) {
      const id = state.backend + '/' + group.id;
      keep.add(id);
      let section = [...root.children].find(el => el.dataset.metricGroup === id);
      if (!section) {
        section = document.createElement('details');section.dataset.metricGroup=id;
        section.className='s-cache-details';section.append(document.createElement('summary'),document.createElement('dl'));
        section.lastChild.className='s-facts';root.append(section);
      }
      section.firstChild.textContent=t(group.label);
      const rows=[];
      function visit(value,path=[]) {
        if (rows.length >= 512 || path.length > 8) return;
        if (value && typeof value === 'object') {
          for (const [key,item] of Object.entries(value)) {
            if (['prompt','messages','content','response','api_key','authorization'].includes(key)) continue;
            visit(item,[...path,key]);
          }
        } else rows.push([path,metricValue([group.id,...path],value)]);
      }
      visit(group.value);
      const signature=JSON.stringify([i18n.language,rows]);
      if (section.dataset.signature !== signature) {
        section.dataset.signature=signature;
        const fragment=document.createDocumentFragment();
        for (const [path,value] of rows) {
          const dt=document.createElement('dt'),dd=document.createElement('dd');
          dt.textContent=path.map(metricName).join(' / ');dt.title=group.id+'.'+path.join('.');dd.textContent=value;
          fragment.append(dt,dd);
        }
        section.lastChild.replaceChildren(fragment);
      }
    }
    for (const section of [...root.children]) if (!keep.has(section.dataset.metricGroup)) section.remove();
    $('[data-backend-metrics]').hidden=!groups.length;
  }
  function renderCacheMetrics(cache) {
    const root=$('[data-cache-metrics]');if (!root) return;
    const fragment=document.createDocumentFragment();
    for (const name of [
      'disk_read_bytes','disk_written_bytes','disk_reads','disk_hits','disk_writes','disk_errors',
      'logical_payload_bytes','physical_payload_bytes','sharing_credit_bytes','shared_forks',
      'retained_logical_bytes','shared_payload_credit_bytes','accounted_cache_bytes',
      'own_history_hits','shared_history_hits','planned_history_hits','planned_reuse_tokens','actual_reuse_tokens',
      'logical_state_bytes','physical_state_bytes','physical_checkpoint_bytes','state_sharing_credit_bytes','state_references',
      'payload_references','unique_payload_records',
      'history_misses','history_swaps','history_restored_bytes','history_spilled_bytes',
      'active_sessions','retained_sessions_limit','active_payload_reservation_bytes','retained_payload_budget_bytes',
      'retained_records','retained_records_limit','retained_recovery_points',
      'checkpoint_kv_capture_bytes','checkpoint_kv_logical_bytes',
      'checkpoint_bytes','checkpoint_tokens','resident_pages','mtp_resident_pages','evaluated_tokens',
      'selected_tokens','reserve_tokens','statistics_bytes','statistics_staging_bytes','transfer_staging_bytes',
    ]) if (Object.hasOwn(cache,name)) {
      const dt=document.createElement('dt'),dd=document.createElement('dd');
      dt.textContent=metricName(name);dt.title=name;dd.textContent=metricValue([name],cache[name]);
      if (name==='disk_errors' && cache[name]>0) dd.className='s-metric-warning';
      fragment.append(dt,dd);
    }
    root.replaceChildren(fragment);
  }
  function renderRuntimeConfiguration() {
    const root = $('[data-runtime-config]');
    if (!root) return;
    const tr = (zh,en) => i18n.language === 'en' ? en : zh;
    const running = ['starting','ready'].includes(consoleRuntime?.state);
    const config = running ? consoleRuntime.active_config?.parameters || {} : {};
    const backend = config.Backend || state?.backend;
    // A new launch must never inherit the preceding worker's /props, even on the same backend.
    const service = !consoleRuntime || (consoleRuntime.state === 'ready' &&
      runtimePropsInstance === consoleRuntime.instance && state?.backend === backend) ? props || {} : {};
    const plan = running ? consoleRuntime.token_settings || {} : {};
    const scoring = service.kvmem?.scoring || (running ? consoleRuntime.effective_scoring || {} : {});
    const k = service.kvmem || {}, caps = k.capabilities || {}, spec = service.speculative || {};
    const generation = service.default_generation_settings || {}, defaults = service.effective_settings?.effective || {};
    const present = value => value !== undefined && value !== null && value !== '' &&
      (typeof value !== 'number' || Number.isFinite(value));
    const first = (...values) => values.find(present);
    const boolean = value => typeof value === 'boolean' ? value : undefined;
    const numeric = value => number(value) ? value : undefined;
    const same = (a,b) => JSON.stringify(a) === JSON.stringify(b);
    const format = value => typeof value === 'boolean' ? tr(value?'开启':'关闭',value?'On':'Off')
      : number(value) ? value.toLocaleString(i18n.language,{maximumFractionDigits:3}) : String(value);
    const tokens = value => `${fmt(value)} tokens`;
    const memory = value => value === 0 ? tr('关闭','Off') : `${fmt(value / MiB)} MiB`;
    const outputLimit = value => value <= 0 ? tr('随剩余上下文','Remaining context') : tokens(value);
    const outputValue = value => number(value) && value <= 0 ? -1 : value;
    const algorithm = value => ({mtp:'MTP',dflash:'DFlash',dflash2:'DFlash2',none:tr('关闭','Off')})[value] || String(value);
    const policy = value => ({adaptive:tr('自适应','Adaptive'),fixed:tr('固定长度','Fixed length')})[value] || String(value);
    const canonicalKv = value => typeof value === 'string' ? value.toLowerCase().replace(/^fp16$/,'f16') : undefined;
    const kvPair = (a,b) => present(a) || present(b) ? [canonicalKv(a),canonicalKv(b)] : undefined;
    const kvFormat = value => value[0] && value[0] === value[1] ? value[0] : `K ${value[0] || '—'} / V ${value[1] || '—'}`;
    const rows = [];
    function row(group,id,labels,reported,requested,options={}) {
      const planned = options.planned;
      const value = first(reported,planned,requested);
      if (options.optional && !present(value)) return;
      const source = present(reported) ? 'service' : present(planned) ? 'plan' : present(requested) ? 'launch' : 'unknown';
      const display = options.format || format;
      const notes = [];
      const original = first(options.original,requested);
      if (present(planned) && present(original) && !same(planned,original))
        notes.push(tr('填写','Entered') + ` ${display(original)} → ` + tr('启动','Launch') + ` ${display(planned)}`);
      const expected = first(planned,requested);
      const differs = present(reported) && present(expected) && !same(reported,expected);
      if (differs) notes.push(tr('启动值','Launch value') + ` ${display(expected)} · ` + tr('服务报告不同','Service differs'));
      if (options.note) notes.push(options.note);
      rows.push({group,id,label:tr(...labels),source,value:present(value)?display(value):tr('未报告','Not reported'),
        note:notes.join(' · '),differs});
    }
    function token(group,key,labels,reported) {
      row(group,key,labels,numeric(reported),numeric(config[key]),{format:tokens,
        planned:numeric(plan.effective?.[key]),original:numeric(plan.requested?.[key])});
    }
    token('primary','Context',['上下文上限','Context limit'],generation.n_ctx);
    token('primary','Budget',['历史 KV 预算 B','History KV budget B'],first(k.selected_tokens,k.budget_tokens));
    token('primary','Reserve',['生成预留 R','Generation reserve R'],k.reserve_tokens);
    row('primary','KvType',['KV 缓存格式','KV cache format'],kvPair(k.kv_type,k.kv_type),
      kvPair(first(config.KvTypeK,config.KvType),first(config.KvTypeV,config.KvType)),{format:kvFormat});
    token('primary','Prefill',['预填充块大小','Prefill chunk'],k.prefill_tokens);
    row('primary','StateCacheType',['递归状态格式','Recurrent state format'],k.state_cache_type,config.StateCacheType,{optional:true});
    const reportedAlgorithm = first(spec.algorithm,typeof k.mtp === 'boolean' ? k.mtp ? 'mtp' : 'none' : undefined);
    const requestedAlgorithm = first(config.SpeculativeMode,typeof config.MtpEnabled === 'boolean' ? config.MtpEnabled ? 'mtp' : 'none' : undefined);
    row('primary','SpeculativeMode',['推测解码','Speculative decoding'],reportedAlgorithm,requestedAlgorithm,{format:algorithm});
    row('primary','Concurrency',['并发请求数','Concurrent requests'],first(service.total_slots,k.parallel,k.max_active_requests,caps.max_active_requests),config.Concurrency);
    row('primary','MaxTokens',['默认输出上限','Default output limit'],outputValue(first(defaults.max_tokens,generation.params?.n_predict,k.generation_limit)),outputValue(config.MaxTokens),{format:outputLimit});

    row('cache','KvmemEnabled',['KVMem','KVMem'],boolean(k.enabled),undefined,{optional:true});
    row('cache','BlockTokens',['KV 管理块','KV block size'],first(numeric(scoring.block_tokens),numeric(k.block_tokens)),numeric(config.BlockTokens),{format:tokens,planned:numeric(plan.block_tokens)});
    token('cache','SinkTokens',['保留开头','Retained prefix'],k.sink_tokens);
    token('cache','RecentTokens',['保留最近','Retained suffix'],k.recent_tokens);
    row('cache','HostMiB',['RAM 历史预算','RAM history budget'],first(k.host_payload_budget_bytes,number(k.host_mib)?k.host_mib*MiB:undefined),
      number(config.HostMiB)?config.HostMiB*MiB:undefined,{format:memory});
    const diskOff = k.ssd === false || caps.ssd_enabled === false;
    row('cache','DiskMiB',['SSD 历史预算','SSD history budget'],diskOff?0:first(caps.disk_budget_bytes,number(k.disk_mib)?k.disk_mib*MiB:undefined),
      config.DiskEnabled===false?0:number(config.DiskMiB)?config.DiskMiB*MiB:undefined,{format:memory});
    if (config.DiskEnabled !== false) row('cache','DiskPath',['SSD 目录','SSD directory'],undefined,config.DiskPath,{optional:true});
    const retained = backend === 'polystrata' ? number(k.retained_sessions)?Math.max(0,k.retained_sessions-1):undefined
      : first(k.retained_sessions_limit,caps.retained_sessions_limit);
    row('cache','RetainedSessions',['额外保留空闲会话','Idle sessions to retain'],retained,
      backend==='polystrata'&&number(config.RetainedSessions)?Math.max(0,config.RetainedSessions-1):config.RetainedSessions,{optional:true});
    row('cache','RecallStrategy',['召回策略','Recall strategy'],scoring.strategy,config.RecallStrategy||'legacy');
    const retention=k.usage?.checkpoint_retention;
    if(retention||scoring.strategy==='layerwise'||config.RecallStrategy==='layerwise'){
      if(retention){
        row('cache','LayerwiseCheckpointTarget',['热检查点目标','Hot checkpoint target'],retention.target,config.LayerwiseCheckpointTarget);
        row('cache','retained_hot',['实际热检查点','Retained hot checkpoints'],retention.retained_hot);
        row('cache','retained_cold',['仅在 SSD 的检查点','SSD-only checkpoints'],retention.retained_cold);
        row('cache','estimated_hot_capacity',['预算预计可保留','Estimated within budget'],retention.estimated_hot_capacity);
        row('cache','metadata_mode',['元数据预算模式','Metadata budget mode'],retention.metadata_mode,undefined,
          {format:v=>v==='auto'?tr('自动','Auto'):tr('手动','Manual')});
        row('cache','count_evictions',['数量淘汰次数','Count evictions'],retention.count_evictions);
        row('cache','count_demotions',['因数量转入 SSD','Demotions to SSD'],retention.count_demotions);
        row('cache','metadata_evictions',['元数据额度淘汰次数','Metadata evictions'],retention.metadata_evictions);
      }
      row('cache','LayerwiseMtpHistory',['MTP 历史来源','MTP history'],scoring.mtp_history,config.LayerwiseMtpHistory,{optional:true});
      for(const [key,name,label] of [
        ['LayerwiseActivationRamMiB','activation_ram_bytes',['前向暂存 RAM','Forward storage RAM']],
        ['LayerwiseActivationDiskMiB','activation_disk_bytes',['前向暂存 SSD','Forward storage SSD']],
        ['LayerwiseStagingMiB','staging_bytes',['传输暂存','Transfer staging']],
        ['LayerwiseMetadataMiB','metadata_bytes',['历史状态与索引','History state and index']],
        ['LayerwiseActivationMetadataMiB','activation_metadata_bytes',['前向描述信息','Forward descriptors']]
      ])row('cache',key,label,name==='metadata_bytes'?first(retention?.metadata_bytes,scoring[name]):scoring[name],number(config[key])?config[key]*MiB:undefined,{format:memory});
    }else{
      row('cache','QueryScorer',['召回评分','Recall scorer'],first(scoring.scorer,k.query_scorer),config.QueryScorer);
      row('cache','ScoreAggregation',['GQA 聚合','GQA aggregation'],scoring.agg,config.ScoreAggregation,{optional:true});
      row('cache','GraphAlpha',['图传播强度 α','Graph strength α'],first(scoring.graph_alpha,k.graph_alpha),config.GraphAlpha,{optional:true});
      row('cache','Lrfu',['历史累计 LRFU','LRFU history'],boolean(scoring.lrfu),boolean(config.Lrfu),{optional:true});
      row('cache','LrfuLambda',['历史衰减系数','History decay'],scoring.lambda,config.LrfuLambda,{optional:true});
      row('cache','LrfuTopFraction',['正向奖励候选比例','Positive reward fraction'],scoring.top_frac,config.LrfuTopFraction,{optional:true});
      row('cache','QueryMaxTokens',['问题评分上限','Query scoring limit'],first(scoring.query_max_tokens,k.query_max_tokens),config.QueryMaxTokens,{format:tokens});
      row('cache','QueryReplayMaxTokens',['完整回放上限','Full replay limit'],first(scoring.query_replay_max_tokens,k.query_replay_max_tokens),config.QueryReplayMaxTokens,{format:tokens});
    }

    const mode = first(reportedAlgorithm,requestedAlgorithm);
    if (mode && mode !== 'none') {
      row('engine','DraftTokens',['草稿长度上限','Draft token limit'],first(spec.draft_tokens,k.mtp_drafts),first(config.DraftTokens,config.MtpDrafts),{format:tokens});
      row('engine','DraftPolicy',['草稿长度策略','Draft length policy'],typeof spec.adaptive==='boolean'?spec.adaptive?'adaptive':'fixed':undefined,
        first(config.DraftPolicy,typeof config.AdaptiveMtp==='boolean'?config.AdaptiveMtp?'adaptive':'fixed':undefined),{format:policy});
      if (mode.startsWith('dflash')) row('engine','DraftModel',['草稿模型','Draft model'],undefined,config.DraftModel,{optional:true});
    }
    if (backend === 'ninfer') {
      row('engine','NgramDrafts',['N-gram 草稿上限','N-gram draft limit'],first(spec.ngram_draft_tokens,caps.ngram_drafts),config.NgramDrafts,
        {format:value=>value===0?tr('关闭','Off'):tokens(value)});
      row('engine','NgramMinMatch',['N-gram 最短匹配','N-gram minimum match'],caps.ngram_min_match,config.NgramMinMatch,{optional:true,format:tokens});
      row('engine','FastPrefill',['快速预填充','Fast prefill'],undefined,boolean(config.FastPrefill),{optional:true});
      row('engine','DeviceProfile',['设备性能配置','Device profile'],undefined,config.DeviceProfile,{optional:true});
      row('engine','TensorParallel',['张量并行','Tensor parallelism'],service.execution?.mode?service.execution.mode==='tensor':undefined,config.TensorParallel,{optional:true});
      if(service.execution?.mode?service.execution.mode==='tensor':config.TensorParallel){
        row('engine','TensorAttention',['注意力切分','Attention split'],service.execution?.tensor_attention,'heads');
        row('engine','TensorSplit',['TP 权重','TP weights'],service.execution?.tensor_split?.length?service.execution.tensor_split:undefined,config.TensorSplit?.length?config.TensorSplit:undefined,{optional:true,format:value=>Array.isArray(value)?value.join(' / '):String(value)});
        row('engine','TensorShares',['实际 KV heads 分配','Actual KV head allocation'],service.execution?.tensor_shares?.length?service.execution.tensor_shares:undefined,undefined,{optional:true,format:value=>Array.isArray(value)?value.map(s=>`${s.rank+1}: [${s.kv_heads[0]}, ${s.kv_heads[1]})`).join(' / '):String(value)});
      }
    }
    row('engine','StageLayers',['多卡层数分配','Layers per device'],service.execution?.stage_layers?.length?service.execution.stage_layers:undefined,
      config.StageLayers?.length?config.StageLayers:undefined,{optional:true,format:value=>Array.isArray(value)?value.join(' / '):String(value)});
    row('engine','VramReserveMiB',['显存预留','VRAM reserve'],numeric(k.vram_reserve_mib),numeric(config.VramReserveMiB),{optional:true,format:value=>`${fmt(value)} MiB`});
    row('engine','Vision',['图像输入','Image input'],first(boolean(service.modalities?.vision),boolean(k.vision?.enabled),boolean(k.vision)),boolean(config.Vision),{optional:true});
    if (first(service.modalities?.vision,k.vision?.enabled,boolean(k.vision),config.Vision) === true) {
      row('engine','VisionTokens',['图像 token 上限','Image token limit'],undefined,config.VisionTokens,{optional:true,format:tokens});
      row('engine','VisionResidency',['图像编码器位置','Vision encoder placement'],typeof k.vision?.gpu==='boolean'?k.vision.gpu?'resident':'cpu':undefined,
        config.VisionResidency,{optional:true,format:value=>({resident:'GPU',cpu:'CPU'})[value]||String(value)});
    }
    row('engine','ListenHost',['监听地址','Listen address'],undefined,config.ListenHost,{optional:true});
    row('engine','Port',['服务端口','Service port'],undefined,config.Port,{optional:true,format:String});
    row('engine','Worker',['后端程序','Backend executable'],undefined,config.Worker,{optional:true});

    const thinkingMode = first(boolean(k.defaults?.enable_thinking),present(defaults.reasoning_effort)?defaults.reasoning_effort!=='none':undefined);
    row('generation','ThinkingMode',['默认思考模式','Default thinking mode'],thinkingMode,
      config.ThinkingMode==='on'?true:config.ThinkingMode==='off'?false:backend==='radiance'&&typeof thinkingMode==='boolean'?undefined:config.ThinkingMode,{format:value=>value==='default'?tr('跟随模型','Model default'):format(value)});
    row('generation','ReasoningEffort',['默认思考强度','Default reasoning effort'],defaults.reasoning_effort,config.ReasoningEffort,{optional:true});
    // Launcher 0 means no separate cap (llama.cpp translates it to -1); PolyStrata uses automatic answer reserve.
    const thinkingBudget = value => value < 0 || (value===0 && backend!=='polystrata') ? -1 : value;
    row('generation','ThinkingBudget',['默认思考预算','Default reasoning budget'],
      first(defaults.reasoning_budget_tokens,k.defaults?.reasoning_budget_tokens),
      number(config.ThinkingBudget)?thinkingBudget(config.ThinkingBudget):undefined,
      {format:value=>value<0?tr('不单设上限','No separate cap'):value===0?backend==='polystrata'?tr('自动预留答案空间','Automatic answer reserve'):tr('立即结束思考','End reasoning immediately'):tokens(value)});
    row('generation','PreserveThinking',['保留历史思考','Preserve previous reasoning'],boolean(defaults.preserve_thinking),
      backend==='ninfer'&&config.PreserveThinking===false?'default':boolean(config.PreserveThinking),
      {optional:true,format:value=>value==='default'?tr('跟随模型','Model default'):format(value)});
    for (const [key,labels] of [['temperature',['温度','Temperature']],['top_p',['Top-p','Top-p']],['top_k',['Top-k','Top-k']],['min_p',['Min-p','Min-p']]])
      row('generation',key,labels,first(defaults[key],generation.params?.[key]),undefined,{optional:true});

    const compactLabels = {
      StateCacheType:['状态格式','State format'],
      Context:['上下文','Context'],Budget:['历史 KV','History KV'],Reserve:['预留','Reserve'],KvType:['KV','KV'],
      Prefill:['Prefill','Prefill'],SpeculativeMode:['推测','Speculation'],Concurrency:['并发','Concurrency'],MaxTokens:['输出','Output'],
    };
    const toggle = $('[data-runtime-toggle]'), details = $('[data-runtime-config-details]');
    const updateToggle = () => {
      toggle.setAttribute('aria-expanded',String(!details.hidden));
      toggle.textContent = details.hidden ? tr('详情 ▾','Details ▾') : tr('收起 ▴','Collapse ▴');
    };
    toggle.onclick = () => {details.hidden=!details.hidden;updateToggle();};
    updateToggle();
    const signature = JSON.stringify([i18n.language,consoleRuntime?.state,failures>0,rows]);
    if (root.dataset.signature === signature) return;
    root.dataset.signature = signature;
    function fill(container,items,compact=false) {
      const fragment = document.createDocumentFragment();
      for (const item of items) {
        const cell = document.createElement('div');cell.className='s-runtime-item';cell.dataset.runtimeKey=item.id;cell.dataset.source=item.source;
        if (item.differs) cell.dataset.differs='true';
        const dt = document.createElement('dt');
        dt.textContent=compact && compactLabels[item.id] ? tr(...compactLabels[item.id]) : item.label;
        const dd = document.createElement('dd'), value = document.createElement('span');
        value.className='s-runtime-value';value.textContent=compact
          ? item.source==='unknown'?'—':item.value.replace(/ tokens$/,'') : item.value;
        if(compact) cell.title=`${item.label}: ${item.value}`;
        dd.append(value);cell.append(dt,dd);
        if (!compact && item.note) {const note=document.createElement('p');note.className='s-runtime-change';note.textContent=item.note;dd.append(note);}
        fragment.append(cell);
      }
      container.replaceChildren(fragment);
    }
    fill($('[data-runtime-primary]'),rows.filter(row=>row.group==='primary'),true);
    const adjustments=$('[data-runtime-adjustments]');
    adjustments.replaceChildren();
    for(const item of rows.filter(row=>row.group==='primary'&&row.note)) {
      const note=document.createElement('p');note.className='s-runtime-change';note.textContent=`${item.label} · ${item.note}`;
      adjustments.append(note);
    }
    adjustments.hidden=!adjustments.children.length;
    for (const [id,labels] of [
      ['cache',['缓存与召回','Cache & retrieval']],
      ['engine',['推理与服务','Inference & service']],
      ['generation',['默认生成设置','Generation defaults']],
    ]) {
      let section = root.querySelector(`[data-runtime-group="${id}"]`);
      if (!section) {
        section = document.createElement('section');section.dataset.runtimeGroup=id;
        section.append(document.createElement('h3'),document.createElement('dl'));
        section.querySelector('dl').className='s-runtime-grid';
        $('[data-runtime-groups]').append(section);
      }
      section.querySelector('h3').textContent=tr(...labels);
      fill(section.querySelector('dl'),rows.filter(row=>row.group===id));
      section.hidden=!rows.some(row=>row.group===id);
    }
  }
  function render(append) {
    if (!state || !props) return;
    const r = currentRequest(),
      allRecords = records();
    if (speedRequest !== r?.id) {
      series.speed = [];
      speedRequest = r?.id;
    }
    if (!followLatest && !allRecords.some((x) => x.id === selected)) {
      followLatest = true;
      selected = "";
    }
    options(
      $("[data-request-select]"),
      [
        ["", t("自动跟随最新请求")],
        ...allRecords.map((x) => [
          x.id,
          `#${x.id} · ${requestStatus(x)}`,
        ]),
      ],
      followLatest ? "" : selected,
    );
    const model = props.model_name || props.model_alias || t("本地模型");
    text(
      "model",
      `${model} · ${{ninfer: "NInfer", polystrata: "PolyStrata", llamacpp: "llama.cpp"}[state.backend] || state.backend}`,
    );
    const activeStates = new Set(state.active.map(displayState));
    const serviceStatus = !state.active.length ? "idle" :
      ["generating", "media", "reading", "processing", "queued"].find((s) => activeStates.has(s)) || "processing";
    const serviceBadge = $("[data-service-state]");
    serviceBadge.dataset.kind = serviceStatus;
    serviceBadge.textContent = statuses[serviceStatus] + (state.active.length > 1 ? t(" · {0} 个请求", state.active.length) : "");
    const mediaRequests = state.active.filter(processingMedia);
    $("[data-media-hint]").hidden = !mediaRequests.length;
    text("media-hint", t("正在处理含图片的输入，请稍候。") +
      (state.active.length > 1 ? t(" · {0} 个请求", mediaRequests.length) : ""));
    const liveRequest = state.active.some((x) => x.id === r?.id);
    const requestScope = !r ? t("暂无请求") : liveRequest ? t("当前请求") : followLatest ? t("上次请求") : t("历史请求");
    text("request-scope", r ? `${requestScope} #${r.id} · ${requestStatus(r)}` : requestScope);
    $$("[data-request-scope]").forEach((e) => e.textContent = requestScope);
    text(
      "output",
      r
        ? t("已生成 {0} tokens · 输出预算 {1}", fmt(r.generated_tokens), number(r.limit) && r.limit > 0 ? fmt(r.limit) : t("未设上限"))
        : t("等待请求"),
    );
    bar("output", ratio(r?.generated_tokens, r?.limit));
    const meter = $('[role="meter"]');
    if (number(ratio(r?.generated_tokens, r?.limit)))
      meter.setAttribute(
        "aria-valuenow",
        String(Math.min(100, ratio(r.generated_tokens, r.limit))),
      );
    else meter.removeAttribute("aria-valuenow");
    const nativeThroughput = throughputLog?.sample;
    const requestSpeed = requestLiveSpeed(state, r, liveRequest);
    const realtimeSpeed = state.backend === "ninfer"
      ? requestSpeed ?? ninferLiveSpeed(state, r, liveRequest, nativeThroughput)
      : ["llamacpp","radiance"].includes(state.backend) ? requestLiveSpeed(state, r, liveRequest)
      : liveRequest && state.backend === "polystrata" ? r?.decode_tps : null;
    text("decode", fmt(realtimeSpeed, 1));
    text("decode-label", state.backend === "ninfer" && requestSpeed !== null
      ? t("实时 · {0}s", fmt(r.decode_tps_window_ms / 1000, 0))
      : state.backend === "ninfer" && nativeThroughput
      ? t("实时 · {0}s", fmt(nativeThroughput.seconds, nativeThroughput.seconds < 1 ? 2 : 0))
      : ["llamacpp","radiance"].includes(state.backend) && number(r?.decode_tps_window_ms)
        ? t("实时 · {0}s", fmt(r.decode_tps_window_ms / 1000, 1)) : t("实时速度"));
    $("[data-live-speed]").title = state.backend === "ninfer" && requestSpeed === null
      ? t("引擎区间吞吐，包含同期所有请求；本轮平均仅统计所选请求。") : "";
    $("[data-live-speed]").hidden = !liveRequest;
    text("decode-mean", fmt(meanSpeed(r), 1));
    const prefill = prefillMetrics(state, r, liveRequest);
    text("prefill", fmt(prefill.rate, 0));
    text("prefill-tokens", t("新算 {0} · 复用 {1} tokens", fmt(prefill.tokens), fmt(r?.cached_tokens)));
    text("prefill-times", prefillTimes(prefill, r));
    const queued = state.queued_count ?? (state.backend === "ninfer" ? stats?.queue?.depth
      : ["llamacpp", "radiance"].includes(state.backend) && !state.dropped_records ? state.active.filter(x => x.status === "queued").length : null);
    const pending=number(queued)?fmt(queued):"—";
    text("queue", t("活动 {0} · 排队 {1}",fmt(state.active.length),pending));
    renderActualSpec(r);
    renderRuntimeConfiguration();
    const hardware = state.hardware || {},
      gpus = hardware.gpus || [];
    const hardwareFresh =
      number(hardware.sampled_at_ms) &&
      state.sampled_at_ms - hardware.sampled_at_ms < 5000;
    const gpuStamp = Object.hasOwn(hardware, 'gpu_sampled_at_ms') ? hardware.gpu_sampled_at_ms : hardware.sampled_at_ms;
    const gpuFresh = number(gpuStamp) && state.sampled_at_ms - gpuStamp >= -1000 && state.sampled_at_ms - gpuStamp < 5000;
    renderGpus(gpus, gpuFresh, append);
    const cpu = hardwareFresh ? hardware.cpu_percent : null;
    text("cpu", fmt(cpu));
    bar("cpu", cpu);
    const available=hardware.ram_available_bytes ?? (number(hardware.ram_total_bytes)&&number(hardware.ram_used_bytes)?hardware.ram_total_bytes-hardware.ram_used_bytes:null);
    text("ram-available",fmt(hardwareFresh&&number(available)?available/GiB:null,2));
    // The console owns system I/O. Standalone/older servers retain native readings.
    const systemDisk = state.system_hardware ?? hardware;
    const diskAge = Date.now() - systemDisk.sampled_at_ms;
    const diskFresh = number(systemDisk.sampled_at_ms) && diskAge >= -1000 && diskAge < 5000;
    text("system-disk",fmt(diskFresh&&number(systemDisk.system_disk_read_bytes_per_second)?systemDisk.system_disk_read_bytes_per_second/MiB:null,1));
    text("system-disk-write",fmt(diskFresh&&number(systemDisk.system_disk_write_bytes_per_second)?systemDisk.system_disk_write_bytes_per_second/MiB:null,1));
    const context = props.default_generation_settings?.n_ctx;
    const used =
      number(r?.prompt_tokens) && number(r?.generated_tokens)
        ? r.prompt_tokens + r.generated_tokens
        : null;
    const percent = ratio(used, context);
    text("context-percent", number(percent) ? `${fmt(percent, percent > 0 && percent < 1 ? 1 : 0)}%` : "—");
    text("context", `${fmt(used)} / ${fmt(context)}`);
    gauge($('[data-dial="context"]'), percent, t("逻辑上下文 {0} / {1} tokens，非 GPU 驻留 KV", fmt(used), fmt(context)));
    const reuse = ratio(r?.cached_tokens, r?.prompt_tokens);
    text("reuse", number(reuse) ? `${fmt(reuse)}%` : t("未采集"));
    text("reused-tokens", `${fmt(r?.cached_tokens)} / ${fmt(r?.prompt_tokens)} tokens`);
    bar("reuse", reuse);
    capacity(
      "ram",
      hardwareFresh ? hardware.ram_used_bytes : null,
      hardware.ram_total_bytes,
    );
    const cache = cacheData();
    renderCacheMetrics(cache);
    renderNativeMetrics();
    text("history-label", t(state.backend === "polystrata" ? "停放会话恢复 / 淘汰" : "历史命中 / 淘汰"));
    text("cache-age", state.backend === "polystrata" ? number(cache.sampled_at_ms)?t("请求结束时更新 · 读取于 {0}",new Date(cache.sampled_at_ms).toLocaleTimeString(i18n.language)):t("等待首个请求完成后的缓存统计") : "");
    capacity("host", cache.host_payload_bytes, cache.host_payload_budget_bytes);
    capacity("ssd", cache.disk_payload_bytes, cache.disk_payload_budget_bytes);
    text(
      "sessions",
      number(cache.retained_sessions) ? fmt(cache.retained_sessions) : t("未采集"),
    );
    text(
      "history",
      `${fmt(cache.history_hits)} / ${fmt(cache.history_evictions)}`,
    );
    let disk = null;
    if (
      append &&
      previousDisk &&
      state.sampled_at_ms > previousDisk.time &&
      number(cache.disk_read_bytes) &&
      cache.disk_read_bytes >= previousDisk.bytes
    )
      disk =
        ((cache.disk_read_bytes - previousDisk.bytes) * 1000) /
        (state.sampled_at_ms - previousDisk.time) /
        MiB;
    if (append)
      previousDisk = number(cache.disk_read_bytes)
        ? { time: state.sampled_at_ms, bytes: cache.disk_read_bytes }
        : null;
    text("cache-disk-label", t(state.backend === "polystrata" ? "缓存累计读取 · MiB" : "缓存读取 · MiB/s"));
    if (append) text("disk", fmt(state.backend === "polystrata" ? number(cache.disk_read_bytes)?cache.disk_read_bytes/MiB:null : disk, 1));
    const values = {
      speed: realtimeSpeed,
    };
    for (const [name, value] of Object.entries(values))
      chart(name, value, append);
    for (const [name, value] of Object.entries({
      model,
      backend: state.backend,
      context: `${fmt(context)} tokens`,
      origin: location.origin,
    }))
      $(`[data-about="${name}"]`).textContent = value;
    renderTable();
  }
  function message(value, warning = true) {
    $("[data-banner]").hidden = !value;
    $("[data-banner-text]").textContent = value;
    $("[data-connection]").dataset.offline = String(warning);
  }
  function renderConnection() {
    if (consoleRuntime && consoleRuntime.state !== "ready") {
      const labels = {stopped: "推理未启动", starting: "模型加载中", exited: "推理进程已退出"};
      $("[data-connection]").textContent = t(labels[consoleRuntime.state] || "正在连接");
      message(consoleRuntime.state === "exited" ? t("推理进程已退出，请在启动配置中查看日志。") : "", consoleRuntime.state === "exited");
      return;
    }
    if (lastError) {
      $("[data-connection]").textContent = authNeeded ? t("需要密钥") : t("连接中断");
      message(authNeeded ? t("请输入有效的 API Key。")
        : lastError.status === 404 ? t("当前服务未提供监控接口，请使用包含监控功能的新版本。")
        : lastGood ? t("暂时无法更新，显示的是上次数据（已过期）。将自动重试。")
        : t("暂时无法更新。将自动重试。"));
    } else if (state) {
      message(restartNotice ? t("服务已重启，历史与曲线已重新开始。") : "", false);
      $("[data-connection]").textContent = state.active.length
        ? t("{0} 个请求处理中", state.active.length) : t("服务已连接");
    } else {
      $("[data-connection]").textContent = t("正在连接");
    }
    if (state) $("[data-footer]").textContent =
      t("更新于 {0} · {1}{2}", new Date(state.sampled_at_ms).toLocaleTimeString(i18n.language, { hour12: false }), document.hidden ? t("后台 5 秒刷新") : t("每秒刷新"), state.dropped_records ? t(" · {0} 条记录超出容量", state.dropped_records) : "");
  }
  async function get(path, signal) {
    const response = await fetch(new URL(path, location.href), {
      signal,
      cache: "no-store",
      headers: key ? { Authorization: `Bearer ${key}` } : {},
    });
    if (!response.ok) {
      const e = new Error(`HTTP ${response.status}`);
      e.status = response.status;
      throw e;
    }
    return response.json();
  }
  function schedule(delay) {
    clearTimeout(timer);
    if (!stopped && !authNeeded) timer = setTimeout(poll, delay);
  }
  window.addEventListener("kvmem-runtime-status", (event) => {
    const old = consoleRuntime?.state;
    consoleRuntime = event.detail;
    renderRuntimeConfiguration();
    renderConnection();
    if (old !== consoleRuntime?.state && consoleRuntime?.state === "ready" && !busy) schedule(0);
  });
  async function poll() {
    if (busy || stopped) return;
    busy = true;
    controller = new AbortController();
    const launchInstance = consoleRuntime?.instance;
    const abort = setTimeout(() => controller.abort(), 8000);
    try {
      const [snapshot, configuration, lanes, systemHardware] = await Promise.all([
        get("../api/monitor/snapshot", controller.signal),
        get("../props", controller.signal),
        get("../slots", controller.signal).catch(error => {
          if (error.status===404) return null;
          throw error;
        }),
        get("../api/launcher/hardware", controller.signal).catch(error =>
          error.status === 404 ? null : {sampled_at_ms:null}),
      ]);
      if (
        snapshot.schema_version !== 1 ||
        !Array.isArray(snapshot.active) ||
        !Array.isArray(snapshot.recent) ||
        !(Array.isArray(lanes) || (snapshot.backend==="radiance" && lanes===null))
      )
        throw new Error(t("不兼容的监控数据"));
      const runtime =
        ["ninfer","radiance"].includes(snapshot.backend)
          ? await get("../stats", controller.signal)
          : null;
      if (snapshot.backend === "ninfer") {
        if (throughputLog?.instance !== snapshot.server_instance) throughputLog = null;
        // Read the same redacted, complete native log lines as the console tab.
        // A missing log endpoint must not take down other monitor metrics.
        const log = await get("../api/launcher/logs?revision=" + encodeURIComponent(throughputLog?.revision || ""), controller.signal).catch(() => null);
        const sample = log?.running && log.backend === "ninfer"
          ? log.text === null && log.session === throughputLog?.session
            ? throughputLog.sample : parseNinferThroughput(log)
          : null;
        throughputLog = log ? {instance:snapshot.server_instance, session:log.session, revision:log.revision, sample} : null;
      } else throughputLog = null;
      if (consoleRuntime?.instance !== launchInstance) return;
      runtimePropsInstance = launchInstance;
      const restarted =
        currentInstance !== null &&
        currentInstance !== snapshot.server_instance;
      if (restarted) {
        series = {};
        previousDisk = null;
        selected = "";
        followLatest = true;
        detail = "";
      }
      currentInstance = snapshot.server_instance;
      snapshot.system_hardware = systemHardware?.schema_version === 1 && systemHardware.scope === "system"
        ? systemHardware : systemHardware === null ? null : {sampled_at_ms:null};
      for (const row of [...snapshot.active, ...snapshot.recent]) {
        if (number(row.actual_cached_tokens)) row.cached_tokens = row.actual_cached_tokens;
        else if (String(row.reuse_accounting_scope || '').startsWith('admission_quote')) row.cached_tokens = null;
      }
      state = snapshot;
      props = configuration;
      slots = lanes || [];
      stats = runtime;
      failures = 0;
      lastGood = Date.now();
      authNeeded = false;
      $("[data-auth]").hidden = true;
      $$("[data-stale]").forEach((e) => delete e.dataset.stale);
      lastError = null;
      restartNotice = restarted;
      renderConnection();
      render(true);
    } catch (error) {
      controller.abort();
      failures++;
      authNeeded = error.status === 401 || error.status === 403;
      $("[data-auth]").hidden = !authNeeded;
      if (authNeeded && !document.querySelector('[data-view="config"]:not([hidden]), [data-view="logs"]:not([hidden])')) page("monitor");
      lastError = error;
      renderConnection();
      renderRuntimeConfiguration();
      $$(".s-state,.s-metrics,.s-cpu,.s-gpus,.s-bottom,.s-cache")
        .filter((e) => e.closest('[data-view="monitor"]'))
        .forEach((e) => (e.dataset.stale = "true"));
      previousDisk = null;
      for (const values of Object.values(series)) {
        values.push(null);
        if (values.length > 60) values.shift();
      }
    } finally {
      clearTimeout(abort);
      busy = false;
      schedule(
        failures
          ? Math.min(30000, 1000 * 2 ** Math.min(failures, 5))
          : document.hidden
            ? 5000
            : 1000,
      );
    }
  }
  $("[data-retry]").onclick = () => {
    authNeeded = false;
    schedule(0);
  };
  $("[data-auth]").onsubmit = (e) => {
    e.preventDefault();
    key = $("#api-key").value.trim();
    storage(sessionStorage, "kvmem-monitor-key", key);
    $("#api-key").value = "";
    authNeeded = false;
    schedule(0);
  };
  $("[data-forget]").onclick = () => {
    key = "";
    storage(sessionStorage, "kvmem-monitor-key", null);
    state = null;
    props = null;
    selected = "";
    series = {};
    previousDisk = null;
    page("monitor");
    authNeeded = false;
    schedule(0);
  };
  document.addEventListener("visibilitychange", () => {
    if (!busy) schedule(document.hidden ? 5000 : 0);
  });
  window.addEventListener("pagehide", () => {
    stopped = true;
    clearTimeout(timer);
    controller?.abort();
  });
  window.addEventListener("pageshow", () => {
    if (stopped) {
      stopped = false;
      schedule(0);
    }
  });
  poll();
  document.documentElement.dataset.monitorReady = "true";
})();
