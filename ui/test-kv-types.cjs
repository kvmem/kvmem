/* Exercise independent K/V controls, legacy migration and actual preview payloads.
 * Only device discovery and preview results are fixtures; no inference starts. */
const assert=require('node:assert/strict');
const path=require('node:path');
module.exports=async({browser,base,info})=>{
 const context=await browser.newContext({viewport:{width:1366,height:900}});
 const page=await context.newPage(),errors=[],previews=[],mutations=[];
 const parameters={...info.draft.parameters,QueryMaxTokens:512,QueryReplayMaxTokens:2048,Backend:'llamacpp',Model:'C:/models/kv-test.gguf',Worker:'C:/workers/llama.exe',Gpu:'GPU-kv-fixture',KvType:'q5_0'};
 const fixture={...info,gpus:[{uuid:parameters.Gpu,name:'NVIDIA GeForce KV test GPU'}],status:{state:'stopped'},draft:{schema_version:1,parameters}};
 page.on('pageerror',error=>errors.push(error.message));
 page.on('request',request=>{if(request.method()==='POST')mutations.push(new URL(request.url()).pathname);});
 await page.route('**/api/launcher/info',route=>route.fulfill({json:fixture}));
 await page.route('**/api/launcher/status',route=>route.fulfill({json:{state:'stopped'}}));
 await page.route('**/api/launcher/preview',route=>{previews.push(route.request().postDataJSON().parameters);return route.fulfill({json:{argv:['fixture-worker'],environment:{}}});});
 const field=key=>page.locator(`[data-param=${key}]`);
 const pair=async()=>[await field('KvTypeK').inputValue(),await field('KvTypeV').inputValue()];
 const closeCheck=()=>page.locator('[data-dialog=check] [data-dialog-close]').click();
 const saved=()=>page.waitForFunction(()=>document.querySelector('[data-draft-state]').textContent.includes('已自动保存'));
 async function check(){
  const before=previews.length;await page.locator('[data-action=preview]').click();
  await page.waitForFunction(()=>document.querySelector('[data-check-summary]')?.textContent.includes('检查通过'));
  assert.equal(previews.length,before+1);const result=previews.at(-1);await closeCheck();return result;
 }
 try{
  await page.goto(base+'/monitor/#config');await field('KvTypeK').waitFor();
  await page.locator('[data-language-select]').selectOption('zh-CN');
  assert.equal(await field('KvType').count(),0,'llama has two controls, no competing combined override');
  assert.deepEqual(await pair(),['q5_0','q5_0'],'legacy KvType initializes both sides');
  for(const key of ['KvTypeK','KvTypeV'])assert.deepEqual(await field(key).locator('option').evaluateAll(items=>items.map(item=>item.value)),['q8_0','q5_0','q4_0','f16','f32']);
  await field('KvTypeK').selectOption('q8_0');await field('KvTypeV').selectOption('q4_0');
  const mixedBits=await check();assert.equal(mixedBits.KvTypeK,'q8_0');assert.equal(mixedBits.KvTypeV,'q4_0');assert.equal(Object.hasOwn(mixedBits,'KvType'),false);
  if(process.env.SCREENSHOT_DIR)await page.screenshot({path:path.join(process.env.SCREENSHOT_DIR,'llama-kv-types.png'),fullPage:true});
  await field('KvTypeK').selectOption('f16');
  assert.deepEqual(await pair(),['f16','q4_0'],'changing K never silently changes V');
  assert.ok((await page.locator('[data-kv-pair-status]').textContent()).includes('不能混用浮点与量化'));
  const beforeInvalid=previews.length;await page.locator('[data-action=preview]').click();
  assert.ok((await page.locator('[data-check-summary]').textContent()).includes('需要补全'));
  assert.equal(previews.length,beforeInvalid,'unsupported pairs never reach preview');await closeCheck();
  await field('KvTypeV').selectOption('f32');
  const floats=await check();assert.equal(floats.KvTypeK,'f16');assert.equal(floats.KvTypeV,'f32');
  await page.locator('[data-action=template]').click();await page.locator('[data-template-name]').fill('独立 K/V');await page.locator('[data-template-confirm]').click();
  await field('KvTypeK').selectOption('q8_0');await field('KvTypeV').selectOption('q4_0');
  await page.locator('[data-apply-template]').click();assert.deepEqual(await pair(),['f16','f32']);
  await page.locator('[data-preset-undo]').click();assert.deepEqual(await pair(),['q8_0','q4_0']);
  await saved();await page.reload();await field('KvTypeK').waitFor();assert.deepEqual(await pair(),['q8_0','q4_0'],'the independent pair survives reload');
  await field('Backend').selectOption('ninfer');assert.equal(await field('KvTypeK').count(),0);assert.equal(await field('KvTypeV').count(),0);assert.equal(await field('KvType').inputValue(),'int8');
  await field('Model').fill('C:/models/kv-test.ninfer');const ninfer=await check();assert.equal(ninfer.KvType,'int8');assert.equal(Object.hasOwn(ninfer,'KvTypeK'),false);assert.equal(Object.hasOwn(ninfer,'KvTypeV'),false);
  await field('Backend').selectOption('llamacpp');assert.deepEqual(await pair(),['q8_0','q4_0'],'backend switches retain each pair');
  await page.evaluate(p=>{const key='kvmem-launch-templates-v1',items=JSON.parse(localStorage.getItem(key));items.push({id:'legacy-kv',name:'旧 KV 模板',draft:{version:2,parameters:p}});localStorage.setItem(key,JSON.stringify(items));},parameters);
  await saved();await page.reload();await field('KvTypeK').waitFor();
  await page.locator('[data-template-tab=mine]').click();await page.locator('[data-apply-template=legacy-kv]').click();assert.deepEqual(await pair(),['q5_0','q5_0'],'old templates retain their combined KV choice');
  await page.setViewportSize({width:390,height:900});assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),'independent K/V controls fit narrow screens');
  if(process.env.SCREENSHOT_DIR)await page.screenshot({path:path.join(process.env.SCREENSHOT_DIR,'llama-kv-types-mobile.png'),fullPage:true});
  // Updating static files over an older controller must give readable restart guidance.
  const legacy={...fixture};delete legacy.kv_cache_version;
  await page.route('**/api/launcher/info',route=>route.fulfill({json:legacy}));
  await saved();await page.reload();await field('KvTypeK').waitFor();
  const beforeRestart=previews.length;await page.locator('[data-action=preview]').click();
  assert.ok((await page.locator('[data-dialog=check]').textContent()).includes('重新启动控制台'));assert.equal(previews.length,beforeRestart);
  assert.ok(mutations.every(url=>['/api/launcher/preview','/api/launcher/inspect-model','/api/launcher/worker-capabilities'].includes(url)),
   'KV edits may preview/inspect but never start, stop or save a service');assert.deepEqual(errors,[]);
  console.log('PASS: independent K/V selection, mixed-bit payload, native pair limits, drafts/templates/reload, legacy migration, backend isolation and restart guidance (no inference)');
 }finally{await context.close();}
};
