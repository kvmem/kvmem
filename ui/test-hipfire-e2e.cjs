/* Real hipfire + real Chromium E2E. No API route mocking.
 * Required: HIPFIRE_BIN, HIPFIRE_MODEL. Optional: HIPFIRE_DRAFT,
 * HIPFIRE_VISION_SIDECAR, HIPFIRE_GPU, HIPFIRE_E2E_FLOWS.
 */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const os = require('node:os');
const crypto = require('node:crypto');
const { spawn, spawnSync } = require('node:child_process');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');

const root = path.resolve(__dirname, '..');
const binary = process.env.HIPFIRE_BIN;
const model = process.env.HIPFIRE_MODEL;
if (!binary || !model) throw Error('HIPFIRE_BIN and HIPFIRE_MODEL are required');
const draft = process.env.HIPFIRE_DRAFT;
const visionSidecar = process.env.HIPFIRE_VISION_SIDECAR;
const flows = (process.env.HIPFIRE_E2E_FLOWS || 'legacy,layerwise,dflash,vision,auth').split(',');
const stamp = new Date().toISOString().replace(/[:.]/g, '-');
const evidenceRoot = path.join(root, 'logs/webui/hipfire-e2e-' + stamp);
fs.mkdirSync(evidenceRoot, { recursive: true });
const stateDir = path.join(evidenceRoot, 'state');
fs.mkdirSync(stateDir, { recursive: true });
fs.writeFileSync(path.join(stateDir, 'worker-paths.json'), JSON.stringify({ hipfire: binary }));
const consolePort = Number(process.env.HIPFIRE_CONSOLE_PORT || 18599);
const base = `http://127.0.0.1:${consolePort}`;
const rocm = process.env.ROCM_PATH || process.env.HIP_PATH || process.env.HIPFIRE_ROCM_PATH;
const consoleEnv = { ...process.env, PYTHON: process.env.PYTHON || 'python3' };
if (rocm) Object.assign(consoleEnv, { ROCM_PATH: rocm, HIP_PATH: rocm, HIPFIRE_ROCM_PATH: rocm });
const child = spawn(process.env.PYTHON || 'python3', ['-B', path.join(root, 'scripts/local-console.py'),
  '--port', String(consolePort), '--state-dir', stateDir,
  '--profiles-root', path.join(stateDir, 'profiles'), '--backend-root', root], {
  cwd: root, env: consoleEnv, windowsHide: true,
});
let consoleStderr = '';
child.stderr.on('data', chunk => { consoleStderr += chunk.toString(); });
const delay = ms => new Promise(resolve => setTimeout(resolve, ms));
async function until(fn, timeout = 30000, label = 'condition') {
  const end = Date.now() + timeout;
  let last;
  while (Date.now() < end) {
    try { const value = await fn(); if (value) return value; } catch (error) { last = error; }
    await delay(250);
  }
  throw Error(`Timed out waiting for ${label}: ${last?.message || ''}\n${consoleStderr}`);
}
let token;
async function api(action, data) {
  const response = await fetch(base + '/api/launcher/' + action, {
    method: data === undefined ? 'GET' : 'POST',
    headers: data === undefined ? {} : { 'Content-Type': 'application/json', 'X-KVMem-Token': token },
    body: data === undefined ? undefined : JSON.stringify(data),
  });
  const body = await response.json();
  if (!response.ok) throw Error(body.error || `HTTP ${response.status}`);
  return body;
}
async function directJson(port, route, authorization) {
  const response = await fetch(`http://127.0.0.1:${port}${route}`, {
    headers: authorization ? { Authorization: authorization } : {},
  });
  const body = await response.text();
  if (!response.ok) throw Error(`${route} HTTP ${response.status}: ${body}`);
  return body;
}
function writeEvidence(name, value) {
  fs.writeFileSync(path.join(evidenceRoot, name + '.json'), JSON.stringify(value, null, 2));
}
function hash(value) { return crypto.createHash('sha256').update(value).digest('hex'); }

async function waitReady(page) {
  await page.waitForFunction(() => document.querySelector('[data-runtime-state]')?.dataset.state === 'ready',
    null, { timeout: 300000 });
}
async function waitStopped(page) {
  await page.waitForFunction(() => document.querySelector('[data-runtime-state]')?.dataset.state === 'stopped',
    null, { timeout: 120000 });
}
async function selectGpu(page, gpu) {
  const choices = page.locator('[data-gpu-choice]');
  for (let i = 0; i < await choices.count(); i++) await choices.nth(i).uncheck();
  await page.locator(`[data-gpu-choice="${gpu}"]`).check();
}
async function structuredChat(port, authorization) {
  const response = await fetch(`http://127.0.0.1:${port}/v1/chat/completions`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Accept: 'text/event-stream',
      ...(authorization ? { Authorization: authorization } : {}),
    },
    body: JSON.stringify({
      model: 'qwen3.8-27b.mq4-xts',
      stream: true,
      enable_thinking: false,
      max_tokens: 5000,
      messages: [{ role: 'user', content: 'Output only a valid JSON array of exactly 100 objects. Object i must be {"id":i,"text":"item-i"} for i from 1 through 100. No markdown, no explanation, no trailing text.' }],
    }),
  });
  assert.equal(response.status, 200);
  let content = '';
  for (const line of (await response.text()).split('\n')) {
    if (!line.startsWith('data:')) continue;
    const payload = line.slice(5).trim();
    if (payload === '[DONE]') break;
    const event = JSON.parse(payload);
    for (const choice of event.choices || []) content += choice.delta?.content || '';
  }
  return JSON.parse(content);
}

async function sendMessage(page, text, maxTokens = 128) {
  await page.getByRole('button', { name: '参数', exact: true }).click();
  await page.getByLabel(/总输出上限/).fill(String(maxTokens));
  await page.getByRole('button', { name: '完成', exact: true }).click();
  await page.getByPlaceholder(/输入消息/).fill(text);
  const before = await page.locator('article:not(.user)').count();
  const completed = page.waitForResponse(response => response.url().endsWith('/api/chat/completions') && response.request().method() === 'POST');
  await page.getByRole('button', { name: '发送', exact: true }).click();
  const response = await completed;
  assert.equal(response.status(), 200);
  await page.waitForFunction(count => document.querySelectorAll('article:not(.user)').length > count,
    before, { timeout: 300000 });
  await page.getByText('就绪', { exact: true }).waitFor({ timeout: 180000 });
  return page.locator('article:not(.user)').last().innerText();
}

async function runFlow(browser, info, gpu, config, index) {
  const name = config.name;
  const port = 18210 + index;
  const flowDir = path.join(evidenceRoot, name);
  fs.mkdirSync(flowDir, { recursive: true });
  const context = await browser.newContext({ viewport: { width: 1440, height: 1050 } });
  const page = await context.newPage();
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  const evidence = { name, config, port, errors };
  const authorization = config.auth ? 'Bearer ' + fs.readFileSync(config.auth, 'utf8').trim() : undefined;
  try {
    await page.goto(base + '/monitor/#config');
    await page.locator('[data-param=Backend]').selectOption('hipfire');
    await page.locator('[data-preset=hipfire-default]').click();
    await page.locator('[data-param=Model]').fill(config.model || model);
    await page.locator('[data-param=Context]').fill('65536');
    await page.locator('[data-param=Budget]').fill('4096');
    await page.locator('[data-param=Reserve]').fill('1024');
    await page.locator('#launch-tab-5').click();
    await page.locator('[data-param=Port]').fill(String(port));
    await selectGpu(page, gpu);
    if (config.recall === 'layerwise') {
      await page.locator('[data-preset=hipfire-layerwise]').click();
      await page.locator('#launch-tab-5').click();
      await page.locator('[data-param=Port]').fill(String(port));
    }
    if (config.spec === 'dflash2') {
      await page.locator('#launch-tab-1').click();
      await page.locator('[data-param=SpeculativeMode]').selectOption('dflash2');
      await page.locator('[data-param=DraftModel]').fill(config.draft || draft);
    }
    if (config.useVision) {
      await page.locator('#launch-tab-4').click();
      await page.locator('[data-param=Vision]').check();
      await page.locator('[data-param=VisionSidecar]').fill(config.vision || visionSidecar);
      await page.locator('[data-param=VisionResidency]').selectOption(config.residency || 'cpu');
      await page.locator('#launch-tab-1').click();
      await page.locator('[data-param=SpeculativeMode]').selectOption('none');
    }
    if (config.auth) {
      await page.locator('#launch-tab-5').click();
      await page.locator('[data-param=ListenHost]').selectOption('0.0.0.0');
      await page.locator('[data-param=ApiKeyFile]').fill(config.auth);
    }
    await page.locator('#launch-tab-3').click();
    await page.locator('[data-param=ThinkingMode]').selectOption('off');
    const previewResponses = [];
    const collectPreview = response => {
      if (response.url().endsWith('/api/launcher/preview') && response.request().method() === 'POST') previewResponses.push(response);
    };
    page.on('response', collectPreview);
    await page.screenshot({ path: path.join(flowDir, 'config-before-preview.png'), fullPage: true });
    await page.locator('[data-action=preview]').click();
    await page.waitForFunction(() => {
      const text = document.querySelector('[data-check-summary]')?.textContent || '';
      return text.includes('检查通过') || text.includes('检查未通过') || text.includes('需要补全');
    }, null, { timeout: 60000 });
    const summary = await page.locator('[data-check-summary]').textContent();
    if (!summary.includes('检查通过')) throw Error('Preview blocked: ' + summary + ' / ' + await page.locator('[data-launch-message]').textContent());
    const previewResponse = previewResponses.at(-1);
    assert.ok(previewResponse, 'preview response missing');
    evidence.preview = await previewResponse.json();
    page.off('response', collectPreview);
    await page.locator('[data-dialog=check] [data-dialog-close]').click();
    await page.screenshot({ path: path.join(flowDir, 'config.png'), fullPage: true });

    await page.locator('[data-action=start]').click();
    await waitReady(page);
    evidence.status = await api('status');
    evidence.health = JSON.parse(await directJson(port, '/health', authorization));
    if (config.auth) evidence.publicHealth = JSON.parse(await directJson(port, '/health'));
    evidence.props = JSON.parse(await directJson(port, '/props', authorization));
    evidence.metricsBefore = await directJson(port, '/metrics', authorization);
    assert.match(evidence.health.model, /qwen3\.8-27b\.mq4-xts/);
    if (config.auth) {
      assert.equal(evidence.publicHealth.model, undefined);
      assert.equal((await fetch(`http://127.0.0.1:${port}/props`)).status, 401);
      assert.equal((await fetch(`http://127.0.0.1:${port}/props`, {headers:{Authorization:'Bearer wrong'}})).status, 401);
      assert.equal((await fetch(`http://127.0.0.1:${port}/props`, {headers:{Authorization:authorization}})).status, 200);
      const secret = fs.readFileSync(config.auth, 'utf8').trim();
      assert.ok(!JSON.stringify(evidence.preview).includes(secret), 'preview must not contain the bearer token');
      assert.ok(!(await api('log')).text.includes(secret), 'service log must not contain the bearer token');
    }

    await page.goto(base + '/chat/view/index.html');
    const contextResponse = await fetch(base + '/api/chat/context');
    evidence.context = JSON.parse(await contextResponse.text());
    assert.equal(contextResponse.status, 200, JSON.stringify(evidence.context));
    assert.equal(evidence.context.state, 'ready');
    assert.equal(evidence.context.backend, 'hipfire');
    evidence.chatDom = await page.evaluate(() => ({
      textareas: [...document.querySelectorAll('textarea')].map(el => ({placeholder:el.placeholder, aria:el.getAttribute('aria-label'), display:getComputedStyle(el).display, rect:el.getBoundingClientRect().toJSON()})),
      editables: [...document.querySelectorAll('[contenteditable]')].map(el => ({aria:el.getAttribute('aria-label'), placeholder:el.getAttribute('data-placeholder')})),
      bodyText: document.body.innerText.slice(0, 500),
    }));
    await page.getByPlaceholder(/输入消息/).waitFor({ timeout: 30000 });
    await page.getByText('就绪', { exact: true }).waitFor({ timeout: 30000 });
    if (config.useVision) assert.equal(evidence.context.capabilities.vision, true);
    evidence.shortAnswer = await sendMessage(page, 'Reply with only the two letters OK.', 64);
    assert.match(evidence.shortAnswer, /OK/i);
    evidence.propsAfter = JSON.parse(await directJson(port, '/props', authorization));
    if (config.spec === 'mtp') {
      assert.equal(evidence.propsAfter.speculative.enabled, true);
      assert.match(String(evidence.propsAfter.speculative.algorithm), /mtp/i);
    }
    if (config.spec === 'dflash2') {
      assert.equal(evidence.propsAfter.speculative.enabled, true);
      assert.match(String(evidence.propsAfter.speculative.algorithm), /dflash/i);
    }
    evidence.followUpAnswer = await sendMessage(page, 'What two letters did you just output? Answer with the letters only.', 64);
    assert.match(evidence.followUpAnswer, /OK/i);

    if (config.recall === 'layerwise') {
      const structured = await structuredChat(port, authorization);
      assert.equal(structured.length, 100);
      for (let i = 0; i < structured.length; i++) {
        assert.deepEqual(structured[i], { id: i + 1, text: `item-${i + 1}` });
      }
      evidence.structuredOutput = { objects: structured.length, valid: true };
    }

    if (config.needle !== false) {
      const filler = 'A neutral archival sentence records routine observations without hidden markers. '.repeat(1500);
      const passphrase = 'BLUE-ORCHID-742';
      const prompt = filler.slice(0, Math.floor(filler.length / 2)) + '\n' + passphrase + '\n' + filler.slice(Math.floor(filler.length / 2)) +
        '\nQuestion: What is the exact passphrase? Reply with only the passphrase.';
      evidence.needlePrompt = { chars: prompt.length, sha256: hash(prompt), passphrase };
      evidence.needleAnswer = await sendMessage(page, prompt, 512);
      assert.match(evidence.needleAnswer, /BLUE-ORCHID-742/);
    }
    evidence.metricsAfter = await directJson(port, '/metrics', authorization);
    if (config.needle !== false && config.recall !== 'layerwise') assert.match(evidence.metricsAfter, /hipfire_kvmem_spill_pages_total [1-9]/);
    if (config.recall === 'layerwise') {
      assert.match(evidence.metricsAfter, /hipfire_kvmem_layerwise 1/);
      assert.match(evidence.metricsAfter, /hipfire_kvmem_host_bytes [1-9]/);
    }

    if (config.useVision) {
      const png = Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC', 'base64');
      await page.locator('input[type=file]').setInputFiles({ name: 'pixel.png', mimeType: 'image/png', buffer: png });
      evidence.visionAnswer = await sendMessage(page, 'Describe the image in one short sentence.', 96);
      assert.ok(evidence.visionAnswer.length > 0);
    }

    await page.goto(base + '/monitor/#monitor');
    await page.waitForFunction(() => document.querySelector('[data-backend-metrics]')?.textContent.includes('hipfire · KVMem'), null, { timeout: 30000 });
    evidence.monitor = await (await fetch(base + '/api/monitor/snapshot')).json();
    const measured = evidence.monitor.recent.find(row => row.prompt_new_tokens > 0 && row.generated_tokens > 0);
    assert.ok(measured, 'monitor must include per-request token usage');
    assert.ok(measured.prompt_tokens >= measured.prompt_new_tokens);
    assert.ok(measured.elapsed_ms > 0 && measured.limit > 0);
    assert.ok(measured.decode_tps_mean > 0, 'monitor must include hipfire decode timing');
    await page.screenshot({ path: path.join(flowDir, 'monitor.png'), fullPage: true });

    await page.goto(base + '/chat/view/index.html');
    await page.getByRole('button', { name: '参数', exact: true }).click();
    await page.getByLabel(/总输出上限/).fill('2048');
    await page.getByRole('button', { name: '完成', exact: true }).click();
    await page.getByPlaceholder(/输入消息/).fill('Write a detailed 2000-word story about a lighthouse keeper.');
    await page.getByRole('button', { name: '发送', exact: true }).click();
    await page.getByRole('button', { name: '停止生成', exact: true }).waitFor({ timeout: 30000 });
    // The streaming status updates continuously move the composer; force still dispatches a real pointer click.
    await page.getByRole('button', { name: '停止生成', exact: true }).click({ force: true });
    await page.getByText('已停止或中断', { exact: true }).waitFor({ timeout: 30000 });

    await page.goto(base + '/monitor/#config');
    await page.locator('[data-action=stop]').click();
    await waitStopped(page);
    evidence.finalStatus = await api('status');
    let healthReachable = true;
    try { await fetch(`http://127.0.0.1:${port}/health`, { signal: AbortSignal.timeout(2000) }); }
    catch { healthReachable = false; }
    assert.equal(healthReachable, false, 'hipfire health endpoint must be gone after stop');
    assert.deepEqual(errors, []);
    evidence.result = 'PASS';
    writeEvidence(name, evidence);
    console.log(`PASS ${name}: chat, recall, monitor, cancel, stop`);
  } catch (error) {
    evidence.result = 'FAIL';
    evidence.error = error.stack || String(error);
    try { evidence.launchMessage = await page.locator('[data-launch-message]').textContent(); } catch {}
    try { evidence.log = (await api('log')).text; } catch {}
    try { await api('stop', {}); } catch {}
    await page.screenshot({ path: path.join(flowDir, 'failure.png'), fullPage: true }).catch(() => {});
    writeEvidence(name, evidence);
    throw error;
  } finally {
    await context.close();
  }
}

(async () => {
  const capability = spawnSync(binary, ['--kvmem-capabilities'], { encoding: 'utf8' });
  assert.equal(capability.status, 0, capability.stderr);
  const capabilityJson = JSON.parse(capability.stdout);
  assert.equal(capabilityJson.compiled_with_kvmem, true);
  assert.deepEqual(capabilityJson.model_sources, ['hfq']);
  assert.equal(capabilityJson.max_devices, 1);
  const inspect = spawnSync(binary, ['inspect', '--json', model], { encoding: 'utf8', timeout: 30000 });
  assert.equal(inspect.status, 0, inspect.stderr);
  const inspectJson = JSON.parse(inspect.stdout);
  assert.equal(inspectJson.can_serve, true);
  writeEvidence('preflight', { capability: capabilityJson, inspect: inspectJson });
  const info = await until(async () => {
    const response = await fetch(base + '/api/launcher/info');
    if (!response.ok) return null;
    const value = await response.json();
    return value.token ? value : null;
  }, 30000, 'console');
  token = info.token;
  const gpu = process.env.HIPFIRE_GPU || info.gpus[0]?.uuid;
  assert.ok(gpu, 'No GPU detected; set HIPFIRE_GPU');
  writeEvidence('console-info', { backends: info.backends, gpus: info.gpus, worker: info.workers.hipfire });
  const browser = await chromium.launch({ headless: true,
    ...(process.env.CHROMIUM ? { executablePath: process.env.CHROMIUM } : {}) });
  const configs = {
    legacy: { name: 'legacy', spec: 'mtp', recall: 'legacy', needle: true },
    layerwise: { name: 'layerwise', spec: 'mtp', recall: 'layerwise', needle: true },
    dflash: { name: 'dflash', spec: 'dflash2', recall: 'legacy', needle: false, draft },
    vision: { name: 'vision', spec: 'none', recall: 'legacy', needle: false, useVision: true, vision: visionSidecar },
    auth: { name: 'auth', spec: 'mtp', recall: 'legacy', needle: false, auth: path.join(stateDir, 'hipfire.key') },
  };
  if (configs.auth.auth) {
    fs.writeFileSync(configs.auth.auth, crypto.randomBytes(32).toString('hex') + '\n');
    fs.chmodSync(configs.auth.auth, 0o600);
  }
  try {
    for (const [index, name] of flows.entries()) {
      const config = configs[name];
      if (!config) throw Error(`Unknown flow ${name}`);
      if (name === 'dflash' && !config.draft) throw Error('HIPFIRE_DRAFT is required for dflash flow');
      if (name === 'vision' && !config.vision) throw Error('HIPFIRE_VISION_SIDECAR is required for vision flow');
      await runFlow(browser, info, gpu, config, index);
    }
  } finally {
    await browser.close();
    await new Promise(resolve => {
      const timer = setTimeout(resolve, 5000);
      child.once('exit', () => { clearTimeout(timer); resolve(); });
      child.kill('SIGTERM');
    });
  }
  console.log(`Evidence: ${evidenceRoot}`);
})().catch(error => {
  console.error(error);
  child.kill('SIGTERM');
  setTimeout(() => { try { child.kill('SIGKILL'); } catch {} }, 5000);
  process.exitCode = 1;
});
