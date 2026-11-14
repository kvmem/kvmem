/* Local monitor translations. No network or user-provided text translation. */
"use strict";
(() => {
  const en = {
"所选请求 MTP · 接受 {0} / 提议 {1}":"Selected request MTP · Accepted {0} / proposed {1}",
"所选请求推测 · 接受 {0} / 提议 {1}":"Selected request speculation · Accepted {0} / proposed {1}",
" · N-gram 接受 {0} / 提议 {1}":" · N-gram accepted {0} / proposed {1}",
"解码耗时 {0}s":"Decode time {0}s",
"推测解码：提议 {0} / 接受 {1} · 接受率 {2}%":"Speculative decoding: proposed {0} / accepted {1} · acceptance {2}%",
"所选请求专家命中率 {0}%":"Selected request expert hit rate {0}%",
"是":"Yes",
"否":"No",
"引擎统计":"Engine statistics",
"请求状态":"Request state",
"等待队列":"Waiting queue",
"引擎容量":"Engine capacity",
"引擎驻留":"Engine residency",
"内存分配":"Memory allocation",
"上下文缓存":"Context cache",
"调度器":"Scheduler",
"解码批处理":"Decode batches",
"引擎累计计数":"Cumulative engine counters",
"引擎累计 tokens":"Cumulative engine tokens",
"启动以来平均吞吐":"Mean throughput since startup",
"主机与设备耗时":"Host and device timing",
"专家缓存":"Expert cache",
"运行时间":"Uptime",
"执行槽 {0} · 会话":"Slot {0} · Conversations",
    "外观设置": "Appearance", "配置": "Config",
    "聊天":"Chat",
"未提供":"Unavailable","正常结束":"Finished normally","达到输出上限":"Output limit reached","思考阶段达到输出上限":"Output limit reached during reasoning","已停止":"Stopped","连接中断":"Connection interrupted","生成失败":"Generation failed","工具调用":"Tool call","思考中":"Thinking","输出正文":"Writing answer","新算 {0} · 复用 {1} tokens":"New {0} \u00b7 Reused {1} tokens","预填充 {0}s · 首 token {1}s":"Prefill {0}s \u00b7 First token {1}s","首 token 从进入执行开始计时，不含排队":"First token time excludes queueing","活动 {0} · 排队 {1}":"Active {0} \u00b7 Queued {1}","MTP 状态未提供":"MTP state unavailable","MTP 未配置":"MTP not configured","MTP 已配置 · 本轮提议 {0} / 接受 {1}":"MTP configured \u00b7 Proposed {0} / accepted {1}","MTP 已配置 · 本轮使用待确认":"MTP configured \u00b7 Usage pending","显存预留 {0} MiB":"VRAM reserve {0} MiB","服务默认：输出 {0} · 思考 {1} · 思考预算 {2}":"Service defaults: output {0} \u00b7 reasoning {1} \u00b7 reasoning budget {2}","可用 RAM":"Available RAM","系统磁盘读取":"System disk reads","系统磁盘写入":"System disk writes","系统磁盘包含模型读取、换页和其他进程。":"System disk includes model reads, paging and other processes.","空闲保留会话":"Idle retained sessions","停放会话恢复 / 淘汰":"Parked restores / evictions","请求结束时更新 · 读取于 {0}":"Updated at request completion \u00b7 Read at {0}","等待首个请求完成后的缓存统计":"Waiting for completed-request cache counters","缓存累计读取 · MiB":"Total cache reads \u00b7 MiB","缓存读取 · MiB/s":"Cache reads \u00b7 MiB/s",

    "KVMem · 运行监控": "KVMem · Runtime Monitor",
    "主导航": "Main navigation", "聊天 ↗": "Chat ↗", "监控": "Monitor", "关于": "About",
    "语言": "Language", "正在连接": "Connecting", "颜色主题": "Theme",
    "布局密度": "Layout density", "布局：舒适": "Layout: Comfortable", "布局：紧凑": "Layout: Compact",
    "极夜": "Geek Dark", "蓝图": "Blueprint", "跟随系统": "System",
    "重新连接": "Reconnect", "连接": "Connect", "输入 API Key": "Enter API key",
    "此服务需要 API Key。密钥只保存在当前标签页，不会写入 URL。": "This service requires an API key. It stays in this tab and is never added to the URL.",
    "模型状态": "Model status", "等待服务数据": "Waiting for service data",
    "正在读取模型信息": "Loading model information", "正在读取推理设备": "Loading inference devices",
    "查看请求": "Request", "暂无请求": "No requests", "等待请求": "Waiting for a request",
    "输出 token 预算使用，非完成百分比": "Output token budget used, not completion progress",
    "所选请求统计": "Selected request metrics", "生成速度": "Decode speed",
    "实时速度": "Live speed", "本轮平均": "Request mean",
    "实时 · {0}s": "Live · {0}s",
    "预填充中 · 累计平均 · {0}s": "Prefilling · Cumulative mean · {0}s",
    "引擎区间吞吐，包含同期所有请求；本轮平均仅统计所选请求。": "Engine throughput over the reporting interval, including concurrent requests; request mean covers only the selected request.",
    "平均生成 tok/s": "Mean decode tok/s",
    "平均速度不含预填充，包含思考 token": "Mean excludes prefill and includes reasoning tokens",
    "生成速度，尚无采样": "Decode speed, no samples yet", "预填充速度": "Prefill speed",
    "处理新输入的速度": "New input processing rate", "逻辑上下文": "Logical context",
    "未采集": "Unavailable", "所选请求 · 非 GPU 驻留 KV": "Selected request · not GPU-resident KV",
    "输入 token 复用": "Input token reuse", "推理显卡": "Inference GPUs", "正在读取": "Loading",
    "只显示当前服务实际使用的设备。": "Only devices used by this service are shown.",
    "查看全部硬件 →": "View all hardware →", "最近请求": "Recent requests", "显示全部": "Show all",
    "时间": "Time", "状态": "Status", "输入": "Input", "输出": "Output", "生成 t/s": "Decode t/s", "耗时": "Duration",
    "点击时间查看预填充、复用等详情 · 最近 100 条": "Select a time for prefill and reuse details · Latest 100 requests",
    "收起": "Collapse", "仅统计请求元数据，不记录提示词或回复正文。": "Only request metadata is recorded, never prompts or response content.",
    "系统与缓存": "System & cache", "CPU 负载": "CPU load", "系统内存": "System RAM",
    "CPU 使用率": "CPU usage", "整机读数，包含其他程序": "System-wide usage, including other applications",
    "RAM 历史缓存": "RAM history cache", "SSD 历史缓存": "SSD history cache", "缓存详情": "Cache details",
    "保留会话": "Retained sessions", "历史命中 / 淘汰": "History hits / evictions", "缓存磁盘读取": "Cache disk reads",
    "仅统计 KVMem 历史数据读取。历史 payload 与会话可能共享数据，各项不相加。1 GiB = 1,024³ 字节。": "Reads count only KVMem history data. History payloads and sessions may share data; do not add these figures. 1 GiB = 1,024³ bytes.",
    "连接到当前 KVMem 服务": "Connected to the current KVMem service",
    "模型与引擎": "Model & engine", "模型": "Model", "后端": "Backend", "上下文上限": "Context limit",
    "服务地址": "Service URL", "连接密钥": "API key", "清除当前标签页密钥": "Clear this tab’s key",
    "硬件": "Hardware", "本机显卡及其参与当前服务的情况。": "Local GPUs and their assignment to this service.",
    "说明与致谢": "Notes & credits",
    "页面每秒刷新；显卡指标为整卡读数，可能包含其他程序。": "Refreshes every second. GPU readings cover the whole device, including other applications.",
    "指标说明": "Metric notes",
    "请求统计随所选请求变化。上下文用量为输入与已生成 token 之和；输出预算表示已用额度，不表示完成进度。生成速度包含已接受的思考 token。": "Metrics follow the selected request. Context usage is input plus generated tokens. The output budget shows usage, not completion progress. Decode speed includes accepted reasoning tokens.",
    "上下文用量不等于显存中的 KV 缓存量；输入 token 复用不等于历史缓存命中。": "Context usage differs from GPU-resident KV cache. Input token reuse differs from history cache hits.",
    "曲线从打开页面时开始记录。后台每 5 秒刷新，断线保留最后读数并标记过期；缺失数据显示“未采集”或“—”。": "Charts start when this page opens. Background tabs refresh every 5 seconds. Disconnections retain the last readings and mark them stale. Missing values show “Unavailable” or “—”.",
    "界面参考 Strata 仪表盘，使用 Outfit 字体。感谢开源作者。": "Inspired by the Strata dashboard, with the Outfit font. Thanks to their open-source authors.",
    "开源许可": "Open-source licenses", "本服务": "This service", "GPU 负载": "GPU load",
    "GPU 负载，尚无采样": "GPU load, no samples yet", "整卡显存": "Device VRAM", "容量未采集": "Capacity unavailable",
    "温度": "Temperature", "功耗": "Power", "向 GPU": "To GPU",
    "空闲": "Idle", "排队": "Queued", "预填充": "Prefilling", "生成中": "Generating",
    "处理图像输入": "Processing image input", "正在处理含图片的输入，请稍候。": "Processing input containing images. Please wait.",
    "处理中": "Processing", "已完成": "Completed", "已断开": "Disconnected", "失败": "Failed",
    "未设上限": "Unlimited", "{0} GiB / 未设上限": "{0} GiB / Unlimited", "暂无请求，发送一条消息后会显示统计。": "No requests yet. Send a message to see metrics.",
    "请求 #{0} · {1}": "Request #{0} · {1}",
    "入口：{0}\n输入 {1} tokens · 复用 {2} · 输出 {3}\n预填充 {4} tok/s · 输入复用率 {5}%\n输出上限 {6} · 耗时 {7} 秒": "Endpoint: {0}\nInput {1} tokens · Reused {2} · Output {3}\nPrefill {4} tok/s · Input reuse {5}%\nOutput limit {6} · Duration {7} s",
    "占比 {0}%": "{0}% used", "比例未采集": "Usage unavailable", "等待更多真实采样": "Waiting for more samples",
    "最近 {0} 次真实采样，当前 {1}": "Latest {0} samples, current {1}",
    "层分片": "Layer split", "张量并行": "Tensor parallel", "多卡推理": "Multi-GPU inference",
    "推理设备待确认": "Inference devices unknown", "CPU 推理": "CPU inference", "单卡推理": "Single-GPU inference",
    "{0} 卡协同 · {1}": "{0} GPUs · {1}", "{0} 张设备参与当前服务": "Devices assigned to this service: {0}",
    "设备绑定信息未提供": "Device assignment unavailable",
    "当前服务尚未提供设备绑定信息；不根据负载猜测推理显卡。": "This service has not provided device assignments. Inference GPUs are not inferred from load.",
    "部分指标未采集": "Some metrics unavailable",
    "驱动未提供整卡功耗读数": "The driver did not provide total board power readings",
    "当前模型未使用显卡。": "This model is not using a GPU.",
    "只显示当前服务绑定的显卡 · 整卡读数包含其他进程": "Only GPUs assigned to this service · Readings include other processes",
    "推理设备": "Inference device", "设备身份未采集": "Device identity unavailable", "读数未采集": "Readings unavailable",
    "读数已过期": "Stale readings", "共 {0} GiB": "{0} GiB total", "已用 {0}% · 含其他进程": "{0}% used · Includes other processes",
    "GPU 负载 {0}%": "GPU load {0}%", "整卡显存 {0} / {1} GiB": "Device VRAM {0} / {1} GiB",
    "绑定待确认": "Assignment unknown", "参与当前服务": "Assigned to this service", "未参与当前服务": "Not assigned to this service",
    "{0}% 负载 · {1} / {2} GiB": "{0}% load · {1} / {2} GiB", "硬件读数未采集": "Hardware readings unavailable",
    "自动跟随最新请求": "Follow latest request", "本地模型": "Local model", " · {0} 个请求": " · Requests: {0}",
    "当前请求": "Current request", "上次请求": "Last request", "历史请求": "Past request",
    "已生成 {0} tokens · 输出预算 {1}": "Generated {0} tokens · Output budget {1}",
    "逻辑上下文 {0} / {1} tokens，非 GPU 驻留 KV": "Logical context {0} / {1} tokens, not GPU-resident KV",
    "不兼容的监控数据": "Incompatible monitor data", "服务已重启，历史与曲线已重新开始。": "Service restarted. History and charts have been reset.",
    "{0} 个请求处理中": "Active requests: {0}", "服务已连接": "Connected",
    "更新于 {0} · {1}{2}": "Updated {0} · {1}{2}", "后台 5 秒刷新": "Background refresh: 5 s", "每秒刷新": "Refresh: 1 s",
    " · {0} 条记录超出容量": " · Records beyond capacity: {0}", "需要密钥": "API key required", "连接中断": "Disconnected",
    "请输入有效的 API Key。": "Enter a valid API key.",
    "当前服务未提供监控接口，请使用包含监控功能的新版本。": "This service has no monitor API. Update to a version with monitoring support.",
    "暂时无法更新。将自动重试。": "Unable to refresh. Retrying automatically.",
    "暂时无法更新，显示的是上次数据（已过期）。将自动重试。": "Unable to refresh. Showing stale data; retrying automatically."
  };
  Object.assign(en, {"启动配置": "Launch configuration", "推理未启动": "Inference stopped", "模型加载中": "Loading model", "推理进程已退出": "Inference process exited", "推理进程已退出，请在启动配置中查看日志。": "Inference process exited. Check logs in Launch configuration."});
  Object.assign(en, {"运行参数": "Runtime settings", "关闭说明": "Close notes", "生成速度说明": "Decode speed notes", "预填充说明": "Prefill notes", "上下文说明": "Context notes", "输入复用说明": "Input reuse notes", "显卡指标说明": "GPU metric notes", "请求记录说明": "Request history notes", "系统与缓存说明": "System and cache notes", "输入 token 复用不等于历史缓存命中。": "Input token reuse differs from history cache hits.", "已用 {0}%": "{0}% used"});
  Object.assign(en, {"MTP 设置未提供": "MTP settings unavailable", "MTP 未开启": "MTP disabled", "MTP 已开启": "MTP enabled", "草稿上限 {0} token": "Draft limit: {0} token(s)", "自适应": "Adaptive", "固定长度": "Fixed length", "所选请求：提议 {0} / 接受 {1}": "Selected request: proposed {0} / accepted {1}"});
  Object.assign(en, {"本次运行配置":"Current runtime configuration","详情 ▾":"Details ▾","收起 ▴":"Collapse ▴"});
  let language = "zh-CN";
  try { if (localStorage.getItem("kvmem-monitor-language") === "en") language = "en"; } catch {}
  // A language-specific launcher overrides a saved preference for this opening.
  // Consume the hint so a later selection survives reloads of the same tab.
  const requestedLanguage = new URLSearchParams(location.search).get("lang");
  if (requestedLanguage === "en" || requestedLanguage === "zh-CN") {
    language = requestedLanguage;
    try { localStorage.setItem("kvmem-monitor-language", language); } catch {}
    try {
      const url = new URL(location.href);
      url.searchParams.delete("lang");
      history.replaceState(history.state, "", url.pathname + url.search + url.hash);
    } catch {}
  }
  function t(source, ...values) {
    return (language === "en" ? en[source] || source : source)
      .replace(/\{(\d+)\}/g, (_, i) => String(values[Number(i)] ?? ""));
  }
  // Capture authored UI text before any service data arrives. Never scan model names,
  // device names or other runtime data for translation. Templates are captured separately.
  const bindings = [];
  function capture(root) {
    const walk = document.createTreeWalker(root, NodeFilter.SHOW_TEXT | NodeFilter.SHOW_ELEMENT);
    while (walk.nextNode()) {
      const node = walk.currentNode;
      if (node.nodeType === Node.TEXT_NODE) {
        const source = node.textContent.trim();
        if (Object.hasOwn(en, source)) bindings.push(() => { node.textContent = t(source); });
      } else {
        for (const attr of ["aria-label", "placeholder", "title"]) {
          const source = node.getAttribute(attr);
          if (Object.hasOwn(en, source)) bindings.push(() => node.setAttribute(attr, t(source)));
        }
      }
    }
  }
  capture(document);
  document.querySelectorAll("template").forEach(el => capture(el.content));
  window.monitorI18n = {
    t,
    get language() { return language; },
    apply(value = language) {
      language = value === "en" ? "en" : "zh-CN";
      document.documentElement.lang = language;
      bindings.forEach(update => update());
      document.querySelector("[data-language-select]").value = language;
    },
    save() { try { localStorage.setItem("kvmem-monitor-language", language); } catch {} }
  };
  document.documentElement.dataset.i18nReady = "true";
})();
