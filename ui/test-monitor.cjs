// Model-free browser contract checks. PLAYWRIGHT_MODULE and CHROMIUM may point to local installations.
const fs = require("fs"),
  path = require("path"),
  http = require("http"),
  assert = require("assert/strict");
const { chromium } = process.env.MONITOR_FIXTURE_ONLY ? {} : require(process.env.PLAYWRIGHT_MODULE || "playwright");
const directory = path.join(__dirname, "monitor");
let fixtureScenario = {};
let failure = 0,
  requiredKey = false,
  instance = "browser-fixture",
  requestId = "1",
  generated = 12,
  recent = [];
const record = () => ({
  id: requestId,
  endpoint: "/v1/chat/completions",
  started_at_ms: Date.now() - 1500,
  status: "generating",
  prompt_tokens: 100,
  cached_tokens: 80,
  generated_tokens: generated,
  limit: 128,
  prefill_tps: 50,
  decode_tps: 8,
  elapsed_ms: 1500,
});
const server = http.createServer((req, res) => {
  const fixture = process.env.MONITOR_FIXTURE_FILE && fs.existsSync(process.env.MONITOR_FIXTURE_FILE)
    ? JSON.parse(fs.readFileSync(process.env.MONITOR_FIXTURE_FILE, "utf8").replace(/^\uFEFF/, "")) : fixtureScenario;
  if (fixture.failure) { res.writeHead(fixture.failure); return res.end("{}"); }
  const devices = [{uuid:"GPU-test-a", name:"NVIDIA GeForce RTX 5060 Ti"}, ...(fixture.dual ? [{uuid:"GPU-test-b", name:"NVIDIA GeForce RTX 5060 Ti"}] : [])];
  if (
    req.url.startsWith("/api/") ||
    req.url === "/props" ||
    req.url === "/slots" || req.url === "/stats"
  ) {
    if ((requiredKey || fixture.requiredKey) && req.headers.authorization !== "Bearer test-key") {
      res.writeHead(401);
      return res.end("{}");
    }
    if (failure) {
      res.writeHead(failure);
      return res.end("{}");
    }
    res.setHeader("Content-Type", "application/json");
    res.setHeader("Cache-Control", "no-store");
    if (req.url === '/slots' && fixture.backend === 'radiance') { res.statusCode=404; return res.end('{}'); }
    if (req.url === '/stats') return res.end(JSON.stringify(fixture.nativeStats || {queue:{depth:2}}));
    if (req.url === '/api/launcher/hardware') {
      if (!fixture.sharedDisk || fixture.diskFailure) {
        res.statusCode = fixture.diskFailure ? 503 : 404; return res.end('{}');
      }
      return res.end(JSON.stringify({schema_version:1, scope:'system',
        sampled_at_ms:Date.now()-(fixture.staleDisk ? 10000 : 0),
        system_disk_read_bytes_per_second:fixture.noDisk ? null : 12.5*1048576,
        system_disk_write_bytes_per_second:fixture.noDisk ? null : 6.25*1048576}));
    }
    if (req.url.startsWith('/api/launcher/logs')) {
      if (fixture.logFailure) { res.statusCode=503; return res.end('{}'); }
      const now = new Date(), pad = n => String(n).padStart(2,'0');
      const stamp = `${now.getFullYear()}-${pad(now.getMonth()+1)}-${pad(now.getDate())} ${pad(now.getHours())}:${pad(now.getMinutes())}:${pad(now.getSeconds())}.000`;
      return res.end(JSON.stringify({running:true,backend:'ninfer',session:'test',started_at:Date.now()-10000,revision:String(Date.now()),text:`${stamp}  INFO  throughput | 5.0s | decode 58.4 tok/s (292 tok) | running 1 (decode-ready 1)\n`}));
    }
    const payload =
      req.url === "/props"
        ? {
            model_name: fixture.model || '<img src=x onerror="window.injected=1">',
            default_generation_settings: { n_ctx: 1024 },
            kvmem: fixture.kvmem || {},
            execution: fixture.unknown ? {} : {mode:fixture.dual ? "layer" : "single", monitor_devices: fixture.cpu ? [] : devices},
            ...fixture.props,
          }
        : req.url === "/slots"
          ? [
              {
                kvmem: {
                  storage: {
                    host_payload_bytes: 1024,
                    host_payload_budget_bytes: 2 ** 64,
                    disk_payload_bytes: 0,
                    disk_payload_budget_bytes: 0,
                    disk_read_bytes: 0,
                  },
                  conversations: { count: 1, extends: 1, evictions: 0 },
                },
              },
            ]
          : {
              schema_version: 1,
              server_instance: instance,
              backend: fixture.backend || "llamacpp",
              kvmem: fixture.monitorCache || {},
              sampled_at_ms: Date.now(),
              active: fixture.active ?? (fixture.idle || fixture.empty ? [] : [{...record(), ...(fixture.nativeRequestMetrics || {}), ...(fixture.liveLlama ? {
                decode_tps_live:23.5, decode_tps_mean:8, decode_tps_window_ms:3000,
                decode_tps_sampled_at_ms:Date.now()-(fixture.staleSpeed ? 10000 : 100),
              } : {})}]),
              recent: fixture.empty ? [] : fixture.idle ? [{...record(), status:"completed"}] : recent,
              hardware: {
                sampled_at_ms: Date.now() - (fixture.stale ? 10000 : 0),
                ...(fixture.gpuStale ? {gpu_sampled_at_ms: Date.now() - 10000} : {}),
                gpus: fixture.gpus || [
                  {
                    uuid: "GPU-test-a",
                    name: "NVIDIA GeForce RTX 5060 Ti",
                    utilization_percent: fixture.idle ? 0 : 42,
                    used_bytes: 2 * 1024 ** 3,
                    total_bytes: 8 * 1024 ** 3,
                    temperature_c: 45,
                    power_w: null,
                  },
                  {uuid:"GPU-test-b", name:"NVIDIA GeForce RTX 5060 Ti", utilization_percent:67, used_bytes:7*1024**3, total_bytes:16*1024**3, temperature_c:62, power_w:99},
                  {uuid:"GPU-unused", name:"Other GPU", utilization_percent:98, used_bytes:1024**3, total_bytes:8*1024**3}
                ].filter((g) => !(fixture.missing && g.uuid === "GPU-test-b")),
                cpu_percent: 20,
                ram_used_bytes: 8 * 1024 ** 3,
                ram_total_bytes: 32 * 1024 ** 3,
              },
            };
    return res.end(JSON.stringify(payload));
  }
  const file = path.join(
    directory,
    req.url === "/monitor/" ? "index.html" : req.url.replace("/monitor/", ""),
  );
  if (!file.startsWith(directory) || !fs.existsSync(file)) {
    res.writeHead(404);
    return res.end();
  }
  res.setHeader(
    "Content-Type",
    file.endsWith(".js")
      ? "text/javascript"
      : file.endsWith(".css")
        ? "text/css"
        : file.endsWith(".woff2")
          ? "font/woff2"
          : "text/html",
  );
  res.end(fs.readFileSync(file));
});
if (process.env.MONITOR_FIXTURE_ONLY) {
  server.listen(18412, "127.0.0.1", () => console.log("Monitor fixture: http://127.0.0.1:18412/monitor/"));
} else (async () => {
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  const url = `http://127.0.0.1:${server.address().port}/monitor/`;
  const browser = await chromium.launch({
    headless: true,
    ...(process.env.CHROMIUM ? { executablePath: process.env.CHROMIUM } : {}),
  });
  try {
    // The fallback is authored in HTML, so it survives MIME rejection and disabled JS.
    for (const script of ['i18n.js', 'monitor.js', 'launcher.js', 'logs.js', 'console-shell.js']) {
      const broken = await browser.newPage();
      try {
        await broken.route(`**/monitor/${script}`, route => route.fulfill({
          contentType: 'text/plain', headers: {'X-Content-Type-Options': 'nosniff'},
          body: fs.readFileSync(path.join(directory, script), 'utf8'),
        }));
        await broken.goto(url);
        assert.equal(await broken.locator('[data-ui-pending]').isVisible(), true, script);
        assert.match(await broken.locator('[data-ui-pending]').textContent(), /界面尚未初始化/);
      } finally { await broken.close(); }
    }
    const noScripts = await browser.newPage({javaScriptEnabled: false});
    try {
      await noScripts.goto(url);
      assert.equal(await noScripts.locator('[data-ui-pending]').isVisible(), true);
    } finally { await noScripts.close(); }
    const page = await browser.newPage({
        viewport: { width: 1440, height: 1100 },
      }),
      errors = [];
    page.on("pageerror", (e) => errors.push(e.message));
    await page.goto(url);
    assert.equal(await page.locator('[data-ui-pending]').isVisible(), false);
    await page.waitForFunction(
      () => document.querySelector('[data-gpu="load"]')?.textContent === "42",
    );
    assert.equal(await page.locator("#app").getAttribute("data-theme"), "dark");
    assert.equal(await page.locator('[data-gpu="power"]').textContent(), "—");
    fixtureScenario={backend:'radiance',monitorCache:{host_payload_bytes:1048576,host_payload_budget_bytes:8388608,
      retained_sessions:3,history_hits:5,history_evictions:2,disk_payload_bytes:0,disk_payload_budget_bytes:0},
      active:[{...record(),status:'queued',prefill_tps:null,decode_tps:null,
        image:{decode_ms:1,preprocess_ms:2,queue_ms:3,encode_ms:0,cache_hits:1,cache_misses:0}}]};
    await page.waitForFunction(()=>document.querySelector('[data-value="queue"]').textContent==='活动 1 · 排队 1');
    assert.equal(await page.locator('[data-value="sessions"]').textContent(),'3');
    assert.equal(await page.locator('[data-value="history"]').textContent(),'5 / 2');
    await page.locator('[data-requests] button').first().click();
    assert.match(await page.locator('[data-detail-body]').textContent(),/图片解码.*1.0 \/ 2.0 \/ 3.0 \/ 0.0/);
    assert.match(await page.locator('[data-detail-body]').textContent(),/图片缓存命中.*1 \/ 0/);
    await page.locator('[data-language-select]').selectOption('en');
    assert.match(await page.locator('[data-detail-body]').textContent(),/Image cache hits \/ misses: 1 \/ 0/);
    await page.locator('[data-language-select]').selectOption('zh-CN');
    await page.locator('[data-close]').click();

    assert.equal(await page.locator('[data-bar="host"]').evaluate(e=>e.style.width),'12.5%');
    assert.notEqual(await page.locator('.s-state').getAttribute('data-stale'),'true');
    fixtureScenario={backend:'llamacpp',monitorCache:{host_payload_bytes:1048576,host_payload_budget_bytes:8388608,
      disk_payload_bytes:0,disk_payload_budget_bytes:0,logical_payload_bytes:2097152,physical_payload_bytes:1048576,
      sharing_credit_bytes:1048576,shared_forks:1,physical_state_bytes:1048576,retained_sessions:2}};
    await page.waitForFunction(()=>document.querySelector('[data-cache-metrics]')?.textContent.includes('KV 共享节省下界'));
    assert.match(await page.locator('[data-cache-metrics]').textContent(),/RAM\/SSD 物理计费/);
    assert.equal(await page.locator('[data-bar="host"]').evaluate(e=>e.style.width),'12.5%');
    assert.match(await page.locator('[data-cache-metrics]').textContent(),/2.00 MiB/);
    fixtureScenario={};
    for (const backend of ['llamacpp', 'ninfer', 'polystrata']) {
      const media = {has_media:true, status:'processing', generated_tokens:0};
      fixtureScenario = {backend,nativeRequestMetrics:media};
      await page.locator('[data-media-hint]').waitFor({state:'visible'});
      assert.match(await page.locator('[data-service-state]').textContent(), /处理图像输入/);
      assert.match(await page.locator('[data-requests]').textContent(), /处理图像输入/);
      fixtureScenario.nativeRequestMetrics = {...media,status:'queued'};
      await page.locator('[data-media-hint]').waitFor({state:'hidden'});
      assert.match(await page.locator('[data-service-state]').textContent(), /排队/);
      fixtureScenario.nativeRequestMetrics = {...media,status:'reading'};
      await page.locator('[data-media-hint]').waitFor({state:'visible'});
      fixtureScenario.nativeRequestMetrics = {...media,status:'generating',generated_tokens:1};
      await page.locator('[data-media-hint]').waitFor({state:'hidden'});
      fixtureScenario.nativeRequestMetrics = {...media,has_media:false};
      await page.waitForFunction(() => document.querySelector('[data-service-state]').textContent === '处理中');
      assert.equal(await page.locator('[data-media-hint]').isVisible(), false);
    }
    fixtureScenario = {model:'Qwen3.8 · 图像输入示例',active:[{...record(),id:'image',has_media:true,status:'reading',generated_tokens:0,decode_tps:null,prefill_tps:null},
      {...record(),id:'text'}]};
    await page.locator('[data-media-hint]').waitFor({state:'visible'});
    assert.match(await page.locator('[data-service-state]').textContent(), /生成中/);
    assert.match(await page.locator('[data-media-hint]').textContent(), /1 个请求/);
    await page.selectOption('[data-language-select]', 'en');
    assert.match(await page.locator('[data-media-hint]').textContent(), /Processing input containing images/);
    await page.selectOption('[data-language-select]', 'zh-CN');
    if (process.env.MONITOR_SCREENSHOT_DIR) {
      await page.screenshot({path:path.join(process.env.MONITOR_SCREENSHOT_DIR,'monitor-image-desktop.png')});
      await page.locator('.s-state').screenshot({path:path.join(process.env.MONITOR_SCREENSHOT_DIR,'monitor-image-card.png')});
      await page.locator('.s-cpu').screenshot({path:path.join(process.env.MONITOR_SCREENSHOT_DIR,'monitor-cpu-card.png')});
      await page.setViewportSize({width:390,height:844});
      await page.screenshot({path:path.join(process.env.MONITOR_SCREENSHOT_DIR,'monitor-image-mobile.png')});
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false);
      await page.setViewportSize({width:1440,height:1100});
    }
    fixtureScenario.failure = 503;
    await page.waitForFunction(() => document.querySelector('.s-state').dataset.stale === 'true');
    assert.equal(await page.locator('.s-media-spinner').evaluate(e => getComputedStyle(e).animationPlayState), 'paused');
    fixtureScenario = {idle:true};
    await page.locator('[data-media-hint]').waitFor({state:'hidden'});
    fixtureScenario = {};
    for (const backend of ['llamacpp', 'ninfer', 'polystrata']) {
      fixtureScenario = {backend,sharedDisk:true};
      await page.waitForFunction(() => document.querySelector('[data-value="system-disk"]').textContent === '12.5');
      assert.equal(await page.locator('[data-value="system-disk-write"]').textContent(),'6.3');
      fixtureScenario = {backend,sharedDisk:true,staleDisk:true};
      await page.waitForFunction(() => document.querySelector('[data-value="system-disk"]').textContent === '—');
      assert.equal(await page.locator('[data-value="system-disk-write"]').textContent(),'—');
    }
    fixtureScenario = {sharedDisk:true,noDisk:true};
    await page.waitForTimeout(1200);
    assert.equal(await page.locator('[data-value="system-disk"]').textContent(),'—');
    fixtureScenario = {sharedDisk:true,diskFailure:true};
    await page.waitForTimeout(1200);
    assert.equal(await page.locator('[data-value="decode-mean"]').textContent(),'8.0');
    fixtureScenario = {liveLlama:true};
    await page.waitForFunction(() => document.querySelector('[data-value="decode"]').textContent === '23.5');
    assert.equal(await page.locator('[data-value="decode-mean"]').textContent(),'8.0');
    fixtureScenario = {liveLlama:true,staleSpeed:true};
    await page.waitForFunction(() => document.querySelector('[data-value="decode"]').textContent === '—');
    fixtureScenario = {backend:'ninfer'};
    await page.waitForFunction(() => document.querySelector('[data-value="decode"]').textContent === '58.4');
    assert.equal(await page.locator('[data-value="decode-mean"]').textContent(),'8.0');
    assert.equal(await page.locator('[data-value="decode-label"]').textContent(),'实时 · 5s');
    assert.equal(await page.locator('[data-value="queue"]').textContent(),'活动 1 · 排队 2');
    fixtureScenario={backend:'ninfer',nativeStats:{queue:{depth:2,entries:[{request_id:'<img onerror=bad()>',prompt:'must-not-render'}]},
      kvmem:{history_misses:3,disk_errors:2,disk_written_bytes:2097152,statistics_bytes:1048576},
      throughput_tokens_per_second:{decode:4.5},experts:{hits:10,lookups:20}}};
    await page.waitForFunction(()=>document.querySelector('[data-cache-metrics]')?.textContent.includes('历史未命中'));
    assert.match(await page.locator('[data-cache-metrics]').textContent(),/2.00 MiB/);
    assert.equal(await page.locator('[data-cache-metrics] .s-metric-warning').textContent(),'2');
    await page.locator('[data-backend-metrics] > details > summary').click();
    const queueSection=page.locator('[data-metric-group="ninfer/queue"]');
    await queueSection.locator('summary').click();
    assert.match(await queueSection.textContent(),/<img onerror=bad\(\)>/);
    assert.doesNotMatch(await queueSection.textContent(),/must-not-render/);
    await page.waitForTimeout(1200);
    assert.equal(await queueSection.getAttribute('open'),'');
    assert.match(await page.locator('[data-backend-sections]').textContent(),/启动以来平均吞吐/);
    assert.equal(await page.locator('[data-backend-sections] img').count(),0);
    fixtureScenario = {backend:'ninfer',nativeRequestMetrics:{
      native_request_id:'chatcmpl-exact',decode_tps_live:21,decode_tps_window_ms:5000,
      decode_tps_sampled_at_ms:Date.now(),prefill_ms:250,ttft_ms:250,decode_ms:1000,
      mtp_drafts_offered:30,mtp_drafts_accepted:20,drafts_offered:40,drafts_accepted:25,
      speculative:{ngram_drafted_tokens:10,ngram_accepted_tokens:5}}};
    await page.waitForFunction(()=>document.querySelector('[data-value="decode"]').textContent==='21.0');
    assert.match(await page.locator('[data-value="prefill-times"]').textContent(),/0.25s/);
    assert.match(await page.locator('[data-value="speculative-live"]').textContent(),/MTP.*接受 20 \/ 提议 30.*N-gram.*5.*10/);
    await page.evaluate(()=>window.dispatchEvent(new CustomEvent('kvmem-runtime-status',
      {detail:{state:'ready',active_config:{parameters:{Backend:'ninfer',SpeculativeMode:'mtp',DraftTokens:4}}}})));
    assert.match(await page.locator('[data-runtime-key="SpeculativeMode"]').textContent(),/MTP/);
    assert.doesNotMatch(await page.locator('[data-runtime-config]').textContent(),/提议 40|接受 25/);
    delete fixtureScenario.nativeRequestMetrics.mtp_drafts_offered;
    delete fixtureScenario.nativeRequestMetrics.mtp_drafts_accepted;
    await page.waitForFunction(() => /推测.*接受 25 \/ 提议 40/.test(
      document.querySelector('[data-value="speculative-live"]').textContent));
    assert.match(await page.locator('[data-value="speculative-live"]').textContent(),/推测.*接受 25 \/ 提议 40/);
    await page.evaluate(()=>window.dispatchEvent(new CustomEvent('kvmem-runtime-status',{detail:{state:'stopped'}})));
    assert.equal(await page.locator('[data-live-speed]').getAttribute('title'),'');
    fixtureScenario = {backend:'ninfer',logFailure:true};
    await page.waitForFunction(() => document.querySelector('[data-value="decode"]').textContent === '—');
    assert.equal(await page.locator('[data-value="decode-mean"]').textContent(),'8.0');
    // Prefill uses completed new work, excluding a large reused prefix. Its
    // native clock updates per chunk; final timing replaces it at decode start.
    const prefillRow={...record(),id:'prefill-progress',status:'reading',generated_tokens:0,
      prompt_tokens:16384,cached_tokens:8192,prompt_processed_tokens:9216,
      prefill_elapsed_ms:512,prefill_tps:null,prefill_ms:null,ttft_ms:null};
    const expectPrefill = value => page.waitForFunction(value =>
      document.querySelector('[data-value="prefill"]').textContent === value, value);
    fixtureScenario={backend:'ninfer',active:[prefillRow]};
    await expectPrefill('2,000');
    assert.match(await page.locator('[data-value="prefill-tokens"]').textContent(),/新算 1,024 · 复用 8,192/);
    assert.equal(await page.locator('[data-value="prefill-times"]').textContent(),'预填充中 · 累计平均 · 0.51s');
    fixtureScenario.active=[{...prefillRow,prompt_processed_tokens:11264,prefill_elapsed_ms:2048}];
    await expectPrefill('1,500');
    await page.locator('[data-requests] button').first().click();
    assert.match(await page.locator('[data-detail-body]').textContent(),/预填充 1,500.0 tok\/s/);
    assert.match(await page.locator('[data-detail-body]').textContent(),/累计平均 · 2.05s/);
    await page.locator('[data-close]').click();
    await page.locator('[data-language-select]').selectOption('en');
    assert.match(await page.locator('[data-value="prefill-times"]').textContent(),/Prefilling · Cumulative mean/);
    await page.locator('[data-language-select]').selectOption('zh-CN');
    for(const missing of [{prefill_elapsed_ms:0},{prefill_elapsed_ms:null},{cached_tokens:null},
      {prompt_processed_tokens:8191},{prompt_processed_tokens:16385}]) {
      fixtureScenario.active=[{...prefillRow,...missing,prefill_tps:9999}];
      // A new request id also checks that no previous request's rate is reused.
      fixtureScenario.active[0].id='invalid-'+JSON.stringify(missing);
      await page.waitForFunction(id => document.querySelector('[data-value="request-scope"]').textContent.includes(id),fixtureScenario.active[0].id);
      await expectPrefill('—');
    }
    fixtureScenario.active=[{...prefillRow,prompt_processed_tokens:8192}];
    await expectPrefill('0');
    fixtureScenario.active=[prefillRow,{...prefillRow,id:'other-prefill',prompt_processed_tokens:8704,prefill_elapsed_ms:1024}];
    await page.locator('[data-request-select] option[value="other-prefill"]').waitFor({state:'attached'});
    await page.selectOption('[data-request-select]','other-prefill');
    await expectPrefill('500');
    await page.selectOption('[data-request-select]','prefill-progress');
    await expectPrefill('2,000');
    const finalPrefill={...prefillRow,status:'generating',generated_tokens:1,prefill_tps:1234,prefill_ms:6638,ttft_ms:6638};
    fixtureScenario.active=[finalPrefill];
    await expectPrefill('1,234');
    assert.equal(await page.locator('[data-value="prefill-times"]').textContent(),'预填充 6.64s · 首 token 6.64s');
    recent=[{...finalPrefill,status:'completed'}];fixtureScenario.active=[];
    await page.waitForFunction(() => document.querySelector('[data-value="request-scope"]').textContent.includes('历史请求'));
    await expectPrefill('1,234');
    await page.selectOption('[data-request-select]','');recent=[];
    fixtureScenario = {backend:'polystrata'};
    await page.waitForFunction(() => document.querySelector('[data-value="decode"]').textContent === '8.0');
    await expectPrefill('50');
    assert.equal(await page.locator('[data-backend-metrics]').isVisible(),false);
    // An admission quote is still unverified during restore. Do not render it as an actual hit.
    fixtureScenario.active=[{...record(),id:'restore-pending',planned_cached_tokens:90,
      reuse_accounting_scope:'admission_quote'}];
    await page.waitForFunction(()=>document.querySelector('[data-value="reused-tokens"]').textContent.startsWith('— /'));
    assert.equal(await page.locator('[data-value="reuse"]').textContent(),'未采集');
    fixtureScenario.active=[{...fixtureScenario.active[0],reuse_accounting_scope:'admission_quote_with_independent_native_actual',actual_cached_tokens:null}];
    await page.waitForTimeout(1200);
    assert.match(await page.locator('[data-value="reused-tokens"]').textContent(),/^— \//);
    fixtureScenario.active=[{...fixtureScenario.active[0],actual_cached_tokens:0,cache_source:'miss'}];
    await page.waitForFunction(()=>document.querySelector('[data-value="reused-tokens"]').textContent.startsWith('0 /'));
    assert.equal(await page.locator('[data-value="reuse"]').textContent(),'0%');
    fixtureScenario.active=undefined;
    recent=[{...record(),id:'expert-history',status:'completed',expert_hit_rate:.75,drafts_offered:40,drafts_accepted:30,decode_ms:2500,
      planned_cached_tokens:90,actual_cached_tokens:0,cache_source:'miss'}];
    await page.waitForFunction(()=>document.querySelector('[data-requests]')?.textContent.includes('已完成'));
    await page.locator('[data-requests] button').last().click();
    assert.match(await page.locator('[data-detail-body]').textContent(),/专家命中率 75.0%/);
    assert.match(await page.locator('[data-detail-body]').textContent(),/接受 30/);
    assert.match(await page.locator('[data-detail-body]').textContent(),/计划复用: 90 tokens/);
    assert.match(await page.locator('[data-detail-body]').textContent(),/实际复用: 0 tokens/);
    assert.match(await page.locator('[data-detail-body]').textContent(),/缓存来源: miss/);
    assert.match(await page.locator('[data-detail-body]').textContent(),/复用 0/);
    await page.locator('[data-close]').click();recent=[];
    fixtureScenario = {};
    await page.waitForFunction(() => document.querySelector('[data-value="decode"]').textContent === '—');
    // Definitions are available on demand at readable size, with keyboard dismissal.
    await page.locator('[popovertarget="monitor-note-decode"]').first().click();
    assert.equal(await page.locator('#monitor-note-decode').isVisible(), true);
    assert.equal(await page.locator('#monitor-note-decode p').evaluate(el=>getComputedStyle(el).fontSize), '14px');
    await page.keyboard.press('Escape');
    assert.equal(await page.locator('#monitor-note-decode').isVisible(), false);
    // Run configuration sources are distinct from per-request counters and editable drafts.
    const runtime = detail => page.evaluate(detail=>window.dispatchEvent(new CustomEvent('kvmem-runtime-status',{detail})),detail);
    const setting = key => page.locator(`[data-runtime-key="${key}"]`);
    const value = key => setting(key).locator('.s-runtime-value').textContent();
    await runtime({state:'ready',instance:'config-a',active_config:{parameters:{Backend:'llamacpp',Budget:10001,Reserve:1001,
      KvTypeK:'q8_0',KvTypeV:'q4_0',SpeculativeMode:'dflash2',DraftTokens:5,DraftPolicy:'fixed',Concurrency:2,ThinkingBudget:0,
      MaxTokens:0,FastPrefill:false,DiskEnabled:false,DiskMiB:4096,ApiKey:'secret-never-render',WorkerArgs:['--secret','not-a-setting'] }},
      token_settings:{block_tokens:32,requested:{Budget:10001,Reserve:1001},effective:{Budget:9984,Reserve:992,Prefill:512}},
      effective_scoring:{scorer:'page-softmax',agg:'mean',graph_alpha:0,lrfu:false,lambda:0.5,top_frac:0.9,block_tokens:32}});
    assert.equal(await value('SpeculativeMode'),'DFlash2');
    assert.equal(await value('KvType'),'K q8_0 / V q4_0');
    assert.equal(await value('Budget'),'9,984');
    assert.equal(await setting('Budget').getAttribute('data-source'),'plan');
    assert.match(await page.locator('[data-runtime-adjustments]').textContent(),/10,001 tokens → 启动 9,984 tokens/);
    assert.equal(await page.locator('[data-runtime-config-details]').isVisible(),false);
    assert.equal(await page.locator('.s-runtime-source').count(),0);
    assert.doesNotMatch(await page.locator('[data-runtime-primary]').innerText(),/tokens|启动计划|服务报告|启动配置/);
    assert.equal(await value('DiskMiB'),'关闭');
    assert.equal(await value('ThinkingBudget'),'不单设上限');
    assert.equal(await value('Lrfu'),'关闭');
    assert.equal(await value('GraphAlpha'),'0');
    assert.equal(await setting('BlockTokens').getAttribute('data-source'),'service');
    assert.doesNotMatch(await page.locator('[data-runtime-config]').textContent(),/secret-never-render|not-a-setting|MTP 未开启/);
    await page.locator('[data-runtime-toggle]').click();
    await page.waitForTimeout(1200);
    assert.equal(await page.locator('[data-runtime-toggle]').getAttribute('aria-expanded'),'true');

    // Native algorithm/length win over launch intent, and expose a difference without inventing activity.
    fixtureScenario={props:{speculative:{algorithm:'dflash2',draft_tokens:3,adaptive:false,ngram_draft_tokens:0},
      kvmem:{enabled:true,defaults:{enable_thinking:false,reasoning_budget_tokens:0}},
      total_slots:2,default_generation_settings:{n_ctx:32768,params:{n_predict:-1}}}};
    await page.waitForFunction(()=>document.querySelector('[data-runtime-key="DraftTokens"] .s-runtime-value')?.textContent==='3 tokens');
    assert.equal(await setting('DraftTokens').getAttribute('data-source'),'service');
    assert.equal(await setting('DraftTokens').getAttribute('data-differs'),'true');
    assert.equal(await value('ThinkingMode'),'关闭');
    assert.equal(await value('ThinkingBudget'),'立即结束思考');
    assert.equal(await value('MaxTokens'),'随剩余上下文');
    assert.equal(await setting('MaxTokens').getAttribute('data-differs'),null,'0 at launch and native -1 both mean remaining context');
    // A changed launcher instance invalidates old service values immediately, including same-backend restart.
    await runtime({state:'starting',instance:'config-b',active_config:{parameters:{Backend:'llamacpp',Context:65536,SpeculativeMode:'none'}}});
    assert.equal(await value('Context'),'65,536');
    assert.equal(await setting('Context').getAttribute('data-source'),'launch');
    assert.equal(await value('SpeculativeMode'),'关闭');
    assert.equal(await value('Budget'),'—');
    await runtime({state:'stopped',instance:'config-c'});
    assert.equal(await value('Context'),'—');
    assert.equal(await value('SpeculativeMode'),'—');

    fixtureScenario={backend:'polystrata',props:{kvmem:{budget_tokens:8192,reserve_tokens:512,kv_type:'fp16',block_tokens:32,
      host_mib:2048,disk_mib:0,ssd:false,retained_sessions:3,mtp:true,mtp_drafts:2,vram_reserve_mib:256},
      effective_settings:{effective:{max_tokens:4096,reasoning_effort:'none',reasoning_budget_tokens:0,preserve_thinking:false}},total_slots:1}};
    await runtime({state:'ready',instance:'config-poly',active_config:{parameters:{Backend:'polystrata',Budget:8192,RetainedSessions:3}}});
    await page.waitForFunction(()=>document.querySelector('[data-runtime-key="KvType"] .s-runtime-value')?.textContent==='f16');
    assert.equal(await value('SpeculativeMode'),'MTP');
    assert.equal(await setting('Budget').getAttribute('data-source'),'service');
    assert.equal(await value('RetainedSessions'),'2');
    assert.equal(await value('HostMiB'),'2,048 MiB');
    assert.equal(await value('ThinkingBudget'),'自动预留答案空间');
    assert.equal(await value('ThinkingMode'),'关闭');
    assert.equal(await value('PreserveThinking'),'关闭');
    assert.equal(await value('VramReserveMiB'),'256 MiB');

    fixtureScenario={backend:'ninfer',props:{kvmem:{selected_tokens:16384,reserve_tokens:1024,sink_tokens:64,recent_tokens:128,
      host_payload_budget_bytes:4*1024**3,capabilities:{disk_budget_bytes:8*1024**3,ssd_enabled:true,max_active_requests:2,ngram_min_match:12}},
      speculative:{algorithm:'dflash2',draft_tokens:5,adaptive:false,ngram_draft_tokens:16},
      execution:{monitor_devices:[],stage_layers:[20,20]},total_slots:2}};
    await runtime({state:'ready',instance:'config-ninfer',active_config:{parameters:{Backend:'ninfer',Budget:16384,Reserve:1024,
      Prefill:512,Context:32768,KvType:'int8',SpeculativeMode:'dflash2',DraftTokens:5,NgramDrafts:16,FastPrefill:false,
      Concurrency:2,MaxTokens:4096,ThinkingMode:'default',ThinkingBudget:0,PreserveThinking:false,
      DiskEnabled:true,DiskPath:'C:/cache/<img src=x onerror=bad()>',ListenHost:'127.0.0.1',Port:18200}},
      token_settings:{block_tokens:64,effective:{Prefill:512}},effective_scoring:{scorer:'page-softmax',agg:'mean',graph_alpha:0.1,lrfu:true,query_max_tokens:512,query_replay_max_tokens:2048}});
    await page.waitForFunction(()=>document.querySelector('[data-runtime-key="NgramDrafts"] .s-runtime-value')?.textContent==='16 tokens');
    assert.equal(await value('Budget'),'16,384');
    assert.equal(await value('Reserve'),'1,024');
    assert.equal(await value('FastPrefill'),'关闭');
    assert.equal(await setting('FastPrefill').getAttribute('data-source'),'launch');
    assert.equal(await value('StageLayers'),'20 / 20');
    assert.equal(await page.locator('[data-runtime-config] img').count(),0);
    assert.match(await value('DiskPath'),/<img/);
    fixtureScenario={...fixtureScenario,props:{...fixtureScenario.props,execution:{mode:'tensor',tensor_attention:'heads',
      tensor_split:[3,1],tensor_shares:[{rank:0,kv_heads:[0,6]},{rank:1,kv_heads:[6,8]}],stage_layers:[]}}};
    const tensorRuntime={state:'ready',instance:'config-ninfer-tensor',active_config:{parameters:{Backend:'ninfer',TensorParallel:true,
      TensorSplit:[1,1],FastPrefill:false}}};
    await runtime(tensorRuntime);
    await page.waitForFunction(()=>document.querySelector('[data-runtime-key="TensorShares"] .s-runtime-value')?.textContent==='1: [0, 6) / 2: [6, 8)');
    assert.equal(await value('TensorParallel'),'开启');
    assert.equal(await value('TensorAttention'),'heads');
    assert.equal(await value('TensorSplit'),'3 / 1','service-reported TP weights override requested weights');
    assert.equal(await setting('TensorSplit').getAttribute('data-source'),'service');
    assert.equal(await setting('TensorSplit').getAttribute('data-differs'),'true');
    assert.equal(await setting('TensorShares').getAttribute('data-source'),'service');
    assert.equal(await setting('StageLayers').count(),0,'TP does not imply pipeline stage counts');
    fixtureScenario={...fixtureScenario,props:{...fixtureScenario.props,execution:{mode:'single',stage_layers:[]}}};
    await page.waitForFunction(()=>document.querySelector('[data-runtime-key="TensorParallel"] .s-runtime-value')?.textContent==='关闭');
    assert.equal(await setting('TensorShares').count(),0,'a service mode change clears old actual allocation');
    assert.equal(await setting('TensorAttention').count(),0,'reported single-GPU mode overrides a stale TP draft');
    fixtureScenario={...fixtureScenario,props:{...fixtureScenario.props,execution:{mode:'tensor',tensor_attention:'heads',tensor_split:[],tensor_shares:[]}}};
    await page.waitForFunction(()=>document.querySelector('[data-runtime-key="TensorParallel"] .s-runtime-value')?.textContent==='开启');
    assert.equal(await setting('TensorShares').count(),0,'unknown actual allocation must not be inferred from requested weights');
    assert.equal(await setting('TensorSplit').getAttribute('data-source'),'launch');
    await page.selectOption('[data-language-select]','en');
    assert.equal(await page.locator('[data-runtime-toggle]').getAttribute('aria-expanded'),'true');
    assert.equal(await value('FastPrefill'),'Off');
    assert.equal(await page.locator('.s-runtime-source').count(),0);
    assert.equal(await page.locator('[data-runtime-config]').getAttribute('aria-label'),'Current runtime configuration');
    await page.selectOption('[data-language-select]','zh-CN');
    if(process.env.MONITOR_SCREENSHOT_DIR){
      // The screenshots are explicitly labelled fixture previews, not benchmark evidence.
      fixtureScenario={...fixtureScenario,model:'Qwen3.5-27B · 界面预览（模拟数据）',props:{...fixtureScenario.props,
        default_generation_settings:{n_ctx:32768,params:{n_predict:4096}},
        execution:{mode:'single',monitor_devices:[{uuid:'GPU-test-a',name:'NVIDIA GeForce RTX 5060 Ti'}]}}};
      await page.waitForFunction(()=>document.querySelector('[data-value="model"]').textContent.includes('界面预览'));
      fs.mkdirSync(process.env.MONITOR_SCREENSHOT_DIR,{recursive:true});
      await page.screenshot({path:path.join(process.env.MONITOR_SCREENSHOT_DIR,'runtime-details.png'),fullPage:true});
      await page.locator('[data-runtime-toggle]').click();
      await page.screenshot({path:path.join(process.env.MONITOR_SCREENSHOT_DIR,'runtime-overview.png'),fullPage:true});
      await page.setViewportSize({width:390,height:844});
      await page.screenshot({path:path.join(process.env.MONITOR_SCREENSHOT_DIR,'runtime-mobile.png'),fullPage:true});
      assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),'runtime mobile overflow');
      await page.setViewportSize({width:1440,height:1100});
    }
    if(await page.locator('[data-runtime-config-details]').isVisible())await page.locator('[data-runtime-toggle]').click();
    for(const lang of ['zh-CN','en']){
      await page.selectOption('[data-language-select]',lang);
      await page.setViewportSize({width:1366,height:768});
      const geometry=await page.locator('[data-runtime-config]').evaluate(el=>({height:el.getBoundingClientRect().height,
        tops:[...el.querySelectorAll('[data-runtime-primary] .s-runtime-item')].map(x=>Math.round(x.getBoundingClientRect().top))}));
      assert.equal(new Set(geometry.tops).size,1,'key parameters fit on one laptop-width line: '+lang);
      assert.ok(geometry.height<45,'runtime summary leaves space for live monitoring: '+lang);
    }
    await page.selectOption('[data-language-select]','zh-CN');
    await page.setViewportSize({width:1440,height:1100});
    await runtime(null);fixtureScenario = {};
    await page.waitForFunction(()=>document.querySelector('[data-runtime-key="SpeculativeMode"] .s-runtime-value')?.textContent==='—');
    assert.equal(await page.locator("[data-gpus] article").count(), 1);
    assert.equal(await page.locator("[data-service-state]").textContent(), "生成中");
    fixtureScenario = {dual: true, idle: true, missing: true};
    await page.waitForFunction(() => document.querySelectorAll("[data-gpus] article").length === 2);
    assert.equal(await page.locator("[data-service-state]").textContent(), "空闲");
    assert.match(await page.locator('[data-value="request-scope"]').textContent(), /上次请求/);
    assert.equal(await page.locator('[data-gpu="load"]').first().textContent(), "0");
    assert.equal(await page.locator('[data-gpu="status"]').nth(1).textContent(), "读数未采集");
    fixtureScenario = {unknown: true};
    await page.waitForFunction(() => document.querySelectorAll("[data-gpus] article").length === 0);
    assert.match(await page.locator("[data-device-note]").textContent(), /不根据负载猜测/);
    fixtureScenario = {backend:'radiance',props:{execution:{mode:'single',monitor_devices:[
      {name:'AMD Radeon RX 9070 XT',uuid:'GPU-launch-identity',pci_bus_id:'0000:24:00.0'}]}},gpus:[
      {name:'AMD Radeon RX 9070 XT',pci_bus_id:'00000000:24:00.0',used_bytes:14*1024**3,total_bytes:16*1024**3,
        utilization_percent:81,temperature_c:54,power_w:null,unavailable_metrics:['power_w']},
      {name:'AMD Radeon RX 9070 XT',pci_bus_id:'0000:25:00.0',utilization_percent:99,power_w:200}]};
    await page.waitForFunction(() => document.querySelector('[data-gpu="load"]')?.textContent === '81');
    assert.equal(await page.locator('[data-gpu="vram"]').textContent(),'14.0');
    assert.equal(await page.locator('[data-gpu="status"]').textContent(),'部分指标未采集');
    assert.equal(await page.locator('[data-gpu="power"]').textContent(),'—');
    assert.equal(await page.locator('[data-gpu="power"]').getAttribute('title'),'驱动未提供整卡功耗读数');
    assert.ok(!(await page.locator('[data-about="devices"]').textContent()).includes('undefined'));
    fixtureScenario.gpuStale = true;
    await page.waitForFunction(() => document.querySelector('[data-gpu="status"]')?.textContent === '读数已过期');
    assert.equal(await page.locator('[data-gpu="load"]').textContent(),'—');
    fixtureScenario.gpuStale = false;
    fixtureScenario.gpus[0].power_w = 46;
    fixtureScenario.gpus[0].unavailable_metrics = [];
    await page.waitForFunction(() => document.querySelector('[data-gpu="power"]')?.textContent === '46.0');
    assert.equal(await page.locator('[data-gpu="status"]').textContent(),'本服务');
    assert.equal(await page.locator('[data-gpu="power"]').getAttribute('title'),'');
    fixtureScenario = {};
    await page.waitForFunction(() => document.querySelectorAll("[data-gpus] article").length === 1);
    assert.equal(await page.evaluate(() => window.injected), undefined);
    await page.locator("[data-requests] button").click();
    assert.equal(await page.locator("[data-detail]").isVisible(), true);
    await page.locator("[data-close]").click();
    recent = [{ ...record(), status: "completed" }];
    requestId = "new";
    generated = 13;
    await page.waitForFunction(() =>
      document
        .querySelector('[data-value="output"]')
        .textContent.includes("已生成 13 tokens"),
    );
    await page.selectOption("[data-request-select]", "1");
    assert.ok(
      (await page.locator('[data-value="output"]').textContent()).includes(
        "已生成 12 tokens",
      ),
    );
    await page.selectOption("[data-request-select]", "");
    assert.ok(
      (await page.locator('[data-value="output"]').textContent()).includes(
        "已生成 13 tokens",
      ),
    );
    const shot = process.env.MONITOR_SCREENSHOT_DIR;
    // Language changes preserve the request selection, open details and chart samples.
    await page.selectOption("[data-request-select]", "1");
    await page.locator("[data-requests] button").last().click();
    const samples = await page.locator('[data-gpu-chart] path').count();
    await page.selectOption("[data-language-select]", "en");
    assert.equal(await page.locator("html").getAttribute("lang"), "en");
    assert.equal(await page.locator("[data-request-select]").inputValue(), "1");
    assert.equal(await page.locator("[data-detail]").isVisible(), true);
    assert.match(await page.locator('[data-detail-body]').textContent(), /Endpoint:/);
    assert.match(await page.locator('[data-value="output"]').textContent(), /Generated 12 tokens/);
    assert.equal(await page.locator('[data-gpu="status"]').textContent(), "This service");
    if (samples) assert.equal(await page.locator('[data-gpu-chart] path').count(), samples);
    await page.reload();
    assert.equal(await page.locator("[data-language-select]").inputValue(), "en");
    await page.selectOption("[data-language-select]", "zh-CN");
    await page.waitForFunction(() => document.querySelector('[data-service-state]').textContent === '生成中');
    if (shot) {
      fs.mkdirSync(shot, { recursive: true });
      await page.screenshot({
        path: path.join(shot, "monitor-dark.png"),
        fullPage: true,
      });
    }
    await page.selectOption("[data-theme-select]", "blue");
    await page.reload();
    await page.waitForFunction(
      () => document.querySelector('[data-gpu="load"]')?.textContent === "42",
    );
    assert.equal(await page.locator("#app").getAttribute("data-theme"), "blue");
    if (shot)
      await page.screenshot({
        path: path.join(shot, "monitor-blue.png"),
        fullPage: true,
      });
    // Adding configuration must not restyle the existing monitor or about view.
    // Compare rendered geometry/styles with the configuration stylesheet disabled.
    for (const theme of ['dark', 'blue']) {
      await page.selectOption('[data-theme-select]', theme);
      for (const width of [360, 768, 1366, 1920, 2560]) {
        await page.setViewportSize({width, height: width === 1366 ? 768 : 1080});
        for (const view of ['monitor', 'about']) {
          await page.locator(`[data-page=${view}]`).click();
          const comparison = await page.evaluate(view => {
            const stylesheet = document.querySelector('link[href$="/launcher.css"]');
            const elements = [document.querySelector('.s-header'),
              ...document.querySelectorAll(`[data-view=${view}], [data-view=${view}] *`)];
            function signature() {
              return elements.map(el => {
                const style = getComputedStyle(el), rect = el.getBoundingClientRect();
                return [rect.x, rect.y, rect.width, rect.height,
                  ...['display','gridTemplateColumns','flexDirection','gap','padding',
                      'margin','fontSize','fontFamily','borderRadius','backgroundColor']
                    .map(key => style[key])];
              });
            }
            const enabled = signature();
            stylesheet.disabled = true;
            const disabled = signature();
            stylesheet.disabled = false;
            return {enabled, disabled};
          }, view);
          assert.deepEqual(comparison.enabled, comparison.disabled,
            `configuration CSS leaked into ${view}: ${theme} ${width}px`);
        }
      }
    }
    await page.locator('[data-page=monitor]').click();
    for (const width of [360, 768, 1440]) {
      await page.setViewportSize({ width, height: 1000 });
      assert.ok(
        await page.evaluate(
          () => document.documentElement.scrollWidth <= innerWidth,
        ),
        `horizontal overflow ${width}`,
      );
    }
    await page.setViewportSize({width:1366,height:768});
    const comfortableHeight=(await page.locator('[data-view=monitor]').boundingBox()).height;
    await page.locator('[data-density-select]').selectOption('compact');
    assert.ok((await page.locator('[data-view=monitor]').boundingBox()).height < comfortableHeight * .9, 'compact monitor meaningfully reduces height');
    for(const theme of ['dark','blue']) {
      await page.locator('[data-theme-select]').selectOption(theme);
      for(const width of [360,768,1366,1920]) {
        await page.setViewportSize({width,height:768});
        assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),'compact monitor overflow '+width);
      }
    }
    await page.reload();await page.locator('[data-gpu=name]').first().waitFor();
    assert.equal(await page.locator('#app').getAttribute('data-density'),'compact');
    await page.locator('[data-density-select]').selectOption('comfortable');
    failure = 503;
    await page.waitForFunction(
      () =>
        document.querySelector("[data-connection]").textContent === "连接中断",
    );
    assert.equal(
      await page.locator(".s-metrics").getAttribute("data-stale"),
      "true",
    );
    failure = 0;
    instance = "restarted";
    requestId = "2";
    generated = 1;
    await page.locator("[data-retry]").click();
    await page.waitForFunction(() =>
      document
        .querySelector('[data-value="output"]')
        .textContent.includes("已生成 1 tokens"),
    );
    assert.ok(
      (await page.locator("[data-banner-text]").textContent()).includes("重启"),
    );
    requiredKey = true;
    await page.reload();
    await page.locator("#api-key").waitFor({ state: "visible" });
    await page.fill("#api-key", "test-key");
    await page.locator("[data-auth] button").click();
    await page.waitForFunction(
      () => document.querySelector('[data-gpu="load"]')?.textContent === "42",
    );
    assert.equal(
      await page.evaluate(() => localStorage.getItem("kvmem-monitor-key")),
      null,
    );
    assert.equal(new URL(page.url()).search, "");
    await page.locator('[data-page="about"]').click();
    await page.locator("[data-forget]").click();
    await page.locator("#api-key").waitFor({ state: "visible" });
    assert.equal(
      await page.evaluate(() => sessionStorage.getItem("kvmem-monitor-key")),
      null,
    );
    assert.deepEqual(errors, []);
    console.log(
      "PASS: themes, real-value rendering, missing data, layout, details, reconnect, restart, API-key isolation, XSS escaping.",
    );
  } finally {
    await browser.close();
    server.close();
  }
})().catch((e) => {
  console.error(e);
  server.close();
  process.exitCode = 1;
});
