/* Model-free browser contract for the hipfire launch surface. */
const assert = require('node:assert/strict');
const path = require('node:path');

module.exports = async ({ browser, base, info }) => {
  const context = await browser.newContext({ viewport: { width: 1440, height: 1050 } });
  const page = await context.newPage();
  const errors = [];
  const previews = [];
  const gpu = { uuid: 'GPU-00000000-0000-0000-0000-000000000001', name: 'hipfire UI fixture' };
  const parameters = {
    Backend: 'hipfire', Model: '/models/qwen3.8-27b.mq4-xts', Worker: '/bin/hipfire',
    Gpu: gpu.uuid, Context: 65536, Budget: 4096, Reserve: 1024, HostMiB: 16384,
    Prefill: 512, MaxTokens: 1024, Port: 18200, ListenHost: '127.0.0.1',
    SinkTokens: 128, RecentTokens: 128, Concurrency: 1, RetainedSessions: 1,
    KvType: 'q8', SpeculativeMode: 'mtp', DraftPolicy: 'fixed', Vision: false,
    VisionResidency: 'cpu', ThinkingMode: 'default', ThinkingBudget: 0,
    ReasoningEffort: 'xhigh', PreserveThinking: true, QueryScorer: 'page-softmax',
    ScoreAggregation: 'mean', GraphAlpha: 0, Lrfu: false, QueryMaxTokens: 512,
    QueryReplayMaxTokens: 2048, CacheMinTokens: 0, DiskMiB: 0,
  };
  page.on('pageerror', error => errors.push(error.message));
  await page.route('**/api/launcher/info', route => route.fulfill({ json: {
    ...info, backends: [...new Set([...info.backends, 'hipfire'])], gpus: [gpu],
    workers: { ...info.workers, hipfire: { available: true, path: '/bin/hipfire' } },
    status: { state: 'stopped' }, draft: { schema_version: 1, parameters },
  } }));
  await page.route('**/api/launcher/status', route => route.fulfill({ json: { state: 'stopped' } }));
  await page.route('**/api/launcher/inspect-model', route => route.fulfill({ json: {
    schema_version: 1, status: 'ready', can_start: true,
    model: { format: 'HFQ', architecture: 'qwen35', kv_types: ['q8'], source: 'hfq' },
    features: { vision: 'absent', mtp: 'supported', dflash2: 'supported' }, checks: [],
  } }));
  await page.route('**/api/launcher/worker-capabilities', route => route.fulfill({ json: {
    schema_version: 1, backend: 'hipfire', strategy: 'layerwise', requested_worker: '',
    worker: { path: '/bin/hipfire', executable_sha256: 'a'.repeat(64) },
    profile: { available: true, current_layer_q: true, query_replay: false,
      position_policy: 'original', default_resources: {
        activation_ram_bytes: 64 * 1048576, activation_disk_bytes: 32 * 1048576,
        staging_bytes: 8 * 1048576, metadata_bytes: 1024 * 1048576,
        activation_metadata_bytes: 96 * 1048576,
      } },
  } }));
  await page.route('**/api/launcher/preview', route => {
    previews.push(route.request().postDataJSON().parameters);
    return route.fulfill({ json: { argv: ['/bin/hipfire', 'serve'], environment: {} } });
  });
  const field = key => page.locator(`[data-param=${key}]`);
  try {
    await page.goto(base + '/monitor/#config');
    await field('Backend').selectOption('hipfire');
    await field('Model').waitFor();
    assert.equal(await field('KvType').inputValue(), 'q8');
    assert.equal(await field('IdleTimeout').inputValue(), '0');
    assert.equal((await page.locator('body').innerText()).includes('Safetensors'), false);
    await page.locator('[data-feature=mtp][data-state=supported]').waitFor();
    assert.match(await page.locator('[data-feature=mtp]').innerText(), /支持/);
    assert.match(await page.locator('[data-feature=dflash2]').innerText(), /支持/);
    assert.match(await page.locator('[data-feature=vision]').innerText(), /未包含/);
    assert.equal(await page.locator('[data-preset=hipfire-default]').count(), 1);
    assert.equal(await page.locator('[data-preset=hipfire-layerwise]').count(), 1);
    assert.equal(await page.locator('[data-param=ApiKeyFile]').count(), 1);
    assert.equal(await page.locator('[data-param=VisionSidecar]').count(), 1);
    assert.equal(await field('StageLayers').isDisabled(), true);
    assert.equal(await field('DraftTokens').isDisabled(), true);

    await page.locator('[data-preset=hipfire-layerwise]').click();
    assert.equal(await field('RecallStrategy').inputValue(), 'layerwise');
    assert.equal(await field('GraphAlpha').inputValue(), '0');
    await page.locator('#launch-tab-6').click();
    assert.equal(await field('LayerwiseActivationRamMiB').isVisible(), true);

    await page.locator('[data-preset=hipfire-default]').click();
    await page.locator('#launch-tab-1').click();
    await field('SpeculativeMode').selectOption('dflash2');
    assert.equal(await field('DraftModel').isVisible(), true);
    await field('DraftModel').fill('/models/qwen38-27b-dflash-mq4.hfq');
    await page.locator('#launch-tab-4').click();
    await field('Vision').check();
    assert.equal(await field('VisionSidecar').isVisible(), true);
    await field('VisionSidecar').fill('/models/qwen3.8-27b-vision.hfq');
    await page.locator('#launch-tab-1').click();
    await field('SpeculativeMode').selectOption('none');
    await page.locator('#launch-tab-5').click();
    await field('IdleTimeout').fill('300');
    await page.locator('[data-action=preview]').click();
    await page.waitForFunction(() => document.querySelector('[data-check-summary]')?.textContent.includes('检查通过'));
    assert.equal(previews.at(-1).Backend, 'hipfire');
    assert.equal(previews.at(-1).SpeculativeMode, 'none');
    assert.equal(previews.at(-1).Vision, true);
    assert.equal(previews.at(-1).IdleTimeout, 300);
    await page.locator('[data-dialog=check] [data-dialog-close]').click();
    if (process.env.SCREENSHOT_DIR) {
      require('node:fs').mkdirSync(process.env.SCREENSHOT_DIR, { recursive: true });
      await page.screenshot({ path: path.join(process.env.SCREENSHOT_DIR, 'hipfire-launcher.png'), fullPage: true });
    }
    assert.deepEqual(errors, []);
    console.log('PASS: hipfire launcher, Layerwise, DFlash, vision and key-file controls');
  } finally {
    await context.close();
  }
};
