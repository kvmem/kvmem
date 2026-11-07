/* Backend switches preserve common choices and never silently pick a GPU.
 * Device discovery and preview are fixtures; no model or worker is started. */
const assert=require('node:assert/strict');
const path=require('node:path');
module.exports=async({browser,base,info})=>{
 const context=await browser.newContext({viewport:{width:1366,height:900}});
 const page=await context.newPage(),errors=[],posts=[],previews=[];
 page.on('pageerror',error=>errors.push(error.message));
 page.on('request',request=>{if(request.method()==='POST')posts.push(new URL(request.url()).pathname);});
 const gpus=[{uuid:'GPU-switch-first',name:'NVIDIA GeForce Test GPU A'},{uuid:'GPU-switch-second',name:'NVIDIA GeForce Test GPU B'}];
 const initial={Backend:'ninfer',Model:'C:/models/switch.ninfer',Worker:'C:/workers/ninfer.exe',
  Gpu:gpus.map(g=>g.uuid).reverse().join(','),StageLayers:[24,40],Context:8192,Budget:2048,
  Reserve:1024,HostMiB:256,Prefill:128,MaxTokens:128,Port:18499,ListenHost:'127.0.0.1',
  SinkTokens:128,RecentTokens:64,Concurrency:1,RetainedSessions:2,KvType:'rk4v4',
  DiskEnabled:false,DiskPath:'C:/cache/shared',DiskMiB:512,NoLrfu:false,LrfuLambda:0.4,
  LrfuTopFraction:0.8,ScoreAggregation:'mean',ModelAlias:'shared model',LogLevel:'debug',
  MtpDrafts:3,AdaptiveMtp:true,Vision:true,VisionTokens:512,ThinkingBudget:512};
 await page.route('**/api/launcher/info',route=>route.fulfill({json:{...info,gpus,status:{state:'stopped'},draft:{schema_version:1,parameters:initial}}}));
 await page.route('**/api/launcher/status',route=>route.fulfill({json:{state:'stopped'}}));
 await page.route('**/api/launcher/preview',route=>{
  previews.push(route.request().postDataJSON().parameters);
  return route.fulfill({json:{argv:['fixture-worker'],environment:{}}});
 });
 const field=key=>page.locator(`[data-param=${key}]`);
 const group=index=>page.locator(`#launch-tab-${index}`).click();
 const order=()=>field('Gpu').evaluate(el=>Array.from(el.selectedOptions).map(option=>option.value));
 const sharedKeys=['Context','Budget','Reserve','HostMiB','Prefill','MaxTokens','Port','ListenHost',
  'SinkTokens','RecentTokens','Concurrency','RetainedSessions','DiskPath','DiskMiB','LrfuLambda',
  'LrfuTopFraction','ScoreAggregation','ModelAlias','LogLevel'];
 const values=async()=>Object.fromEntries(await Promise.all(sharedKeys.map(async key=>[key,await field(key).inputValue()])));
 const closeCheck=()=>page.locator('[data-dialog=check] [data-dialog-close]').click();
 try{
  await page.goto(base+'/monitor/#config');await field('Backend').waitFor();
  await page.locator('[data-language-select]').selectOption('zh-CN');
  await field('Context').fill('123456');
  await page.locator('.l-native > summary').click();await page.locator('[data-engine-arguments] > summary').click();await page.locator('[data-native]').fill('["--threads",');
  const shared=await values(),originalOrder=await order();
  assert.deepEqual(originalOrder,[gpus[1].uuid,gpus[0].uuid]);
  const cardShape=()=>page.locator('.l-gpu-choice').evaluateAll(cards=>cards.map(card=>({tag:card.tagName,type:card.querySelector('input').type,text:card.textContent})));
  const ninferShape=await cardShape();
  await field('Backend').selectOption('llamacpp');
  assert.deepEqual(await cardShape(),ninferShape,'the same GPU cards appear for both engines');
  assert.deepEqual(await order(),originalOrder,'switching must not replace a multi-GPU selection with the first GPU');
  assert.equal(await field('Context').inputValue(),'262144','first switch uses engine default');
  assert.equal(await field('Budget').inputValue(),'47104');assert.equal(await field('MaxTokens').inputValue(),'0');
  assert.equal(await field('Port').inputValue(),shared.Port);assert.deepEqual(await page.locator('[data-preset]').evaluateAll(nodes=>nodes.map(n=>n.dataset.preset).sort()),['llamacpp-default','llamacpp-layerwise']);
  assert.equal(await page.locator('[data-gpu-limit]').isVisible(),false,'llama.cpp keeps a two-GPU selection launchable');
  assert.equal(await page.locator('[data-gpu-choice]').first().evaluate(el=>el.checkValidity()),true);
  assert.equal(await field('Model').inputValue(),'','model assets are backend-specific');
  assert.equal(await field('Worker').inputValue(),'');
  assert.equal(await field('KvTypeK').inputValue(),'q8_0','native KV names are backend-specific');
  assert.equal(await field('KvTypeV').inputValue(),'q8_0');
  await field('Model').fill('C:/models/switch.gguf');
  await page.locator('.l-native > summary').click();await page.locator('[data-engine-arguments] > summary').click();
  await page.locator('[data-engine-environment] > summary').click();await field('Worker').fill('C:/workers/llama.exe');await page.locator('[data-native]').fill('["--threads-batch","2"]');
  await group(0);await field('KvTypeK').selectOption('q8_0');await field('KvTypeV').selectOption('q4_0');
  await group(4);await field('Port').fill('18521');
  await page.locator('.l-service-fields').locator('..').locator('summary').click();await field('LogLevel').selectOption('trace');
  const changed=await values();
  // A multi-GPU llama.cpp selection is launchable; weights stay optional and are not sent.
  await page.locator('[data-action=preview]').click();
  await page.locator('[data-check-summary]').waitFor();
  assert.ok(!(await page.locator('[data-check-summary]').textContent()).includes('显卡'),'the GPU selection no longer blocks llama.cpp preview');
  await closeCheck();previews.length=0;
  // Reordering keeps counts attached to their original physical GPU when returning.
  await page.locator('[data-ordered-gpu]').nth(1).locator('[data-gpu-move="-1"]').click();
  await field('Backend').selectOption('ninfer');
  assert.deepEqual(await order(),[gpus[0].uuid,gpus[1].uuid]);
  assert.equal(await field('StageLayers').inputValue(),'40,24');
  assert.deepEqual(await values(),{...shared,Port:changed.Port},'resource edits belong to each engine, listener follows the last edit');
  assert.equal(await field('Model').inputValue(),initial.Model);assert.equal(await field('Worker').inputValue(),initial.Worker);
  assert.equal(await field('KvType').inputValue(),'rk4v4');
  assert.equal(await page.locator('[data-native]').inputValue(),'["--threads",','incomplete backend-specific edits survive switching');
  await page.locator('[data-gpu-choice]').nth(1).uncheck();
  await field('Backend').selectOption('llamacpp');
  assert.equal(await field('Model').inputValue(),'C:/models/switch.gguf');
  assert.equal(await field('Worker').inputValue(),'C:/workers/llama.exe');assert.equal(await field('KvTypeK').inputValue(),'q8_0');assert.equal(await field('KvTypeV').inputValue(),'q4_0');
  assert.deepEqual(await order(),[gpus[0].uuid]);assert.equal(await page.locator('[data-gpu-limit]').isVisible(),false);
  assert.equal(await page.locator('[data-gpu-order]').isVisible(),false,'single-GPU mode has no unnecessary order controls');
  assert.equal(await field('StageLayers').isDisabled(),true);
  for(const key of ['DraftTokens','Vision','ThinkingBudget','ReasoningEffort','PreserveThinking'])assert.equal(await field(key).isDisabled(),false,key);
  await page.locator('[data-language-select]').selectOption('en');assert.deepEqual(await order(),[gpus[0].uuid]);
  await page.locator('[data-language-select]').selectOption('zh-CN');
  await group(4);await field('Concurrency').fill('9');
  await field('Backend').selectOption('ninfer');assert.equal(await field('Concurrency').inputValue(),'1');
  await field('Backend').selectOption('llamacpp');assert.equal(await field('Concurrency').inputValue(),'9','switching backends must retain invalid user input');
  await page.locator('[data-action=preview]').click();await page.locator('[data-check-summary]').waitFor();
  assert.ok((await page.locator('[data-check-summary]').textContent()).includes('需要补全'));assert.equal(previews.length,0);
  await closeCheck();await field('Concurrency').fill('1');
  await page.locator('[data-action=preview]').click();
  await page.waitForFunction(()=>document.querySelector('[data-check-summary]')?.textContent.includes('检查通过'));
  assert.equal(previews.length,1);assert.equal(previews[0].Gpu,gpus[0].uuid);
  assert.equal(previews[0].KvTypeK,'q8_0');assert.equal(previews[0].KvTypeV,'q4_0');assert.equal(Object.hasOwn(previews[0],'KvType'),false);
  assert.equal(Object.hasOwn(previews[0],'StageLayers'),false);
  for(const [key,value] of Object.entries({DraftTokens:3,Vision:true,ThinkingBudget:16384,ReasoningEffort:'xhigh',PreserveThinking:true,MaxTokens:0}))assert.equal(previews[0][key],value,key+' reaches llama');
  await closeCheck();
  if(process.env.SCREENSHOT_DIR)await page.screenshot({path:path.join(process.env.SCREENSHOT_DIR,'backend-switch-llama.png'),fullPage:true});
  await field('Backend').selectOption('ninfer');
  assert.deepEqual(await order(),[gpus[0].uuid]);assert.equal(await field('StageLayers').inputValue(),'');
  if(process.env.SCREENSHOT_DIR)await page.screenshot({path:path.join(process.env.SCREENSHOT_DIR,'backend-switch-ninfer.png'),fullPage:true});
  // Empty selection and an explicitly chosen second card must also survive a switch.
  await page.locator('[data-gpu-choice]').first().uncheck();await field('Backend').selectOption('llamacpp');
  assert.deepEqual(await order(),[]);assert.equal(await page.locator('[data-gpu-choice]:checked').count(),0);
  await page.locator('[data-gpu-choice]').nth(1).check();await field('Backend').selectOption('ninfer');
  assert.deepEqual(await order(),[gpus[1].uuid]);await field('Backend').selectOption('llamacpp');assert.deepEqual(await order(),[gpus[1].uuid]);
  await page.waitForFunction(()=>document.querySelector('[data-draft-state]').textContent.includes('已自动保存'));
  await page.reload();await field('Backend').waitFor();assert.deepEqual(await order(),[gpus[1].uuid]);
  await field('Backend').selectOption('ninfer');assert.equal(await field('Context').inputValue(),'123456','other engine draft survives a reload');
  await field('Backend').selectOption('polystrata');assert.equal(await field('Prefill').inputValue(),'512');assert.equal(await field('DraftTokens').inputValue(),'2');assert.equal(await field('ReasoningEffort').inputValue(),'xhigh');assert.equal(await field('Vision').isChecked(),false);assert.equal(await page.locator('[data-preset=polystrata-default]').count(),1);assert.equal(await page.locator('[data-preset=polystrata-gemma4]').count(),1);
  await field('Backend').selectOption('llamacpp');await page.setViewportSize({width:390,height:900});
  assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),'GPU cards fit a narrow screen');
  if(process.env.SCREENSHOT_DIR)await page.screenshot({path:path.join(process.env.SCREENSHOT_DIR,'backend-switch-mobile.png'),fullPage:true});
  // Preview and model inspection are read-only POSTs; start/stop and every other
  // mutation must remain absent while switching backends.
  assert.ok(posts.every(url=>['/api/launcher/preview','/api/launcher/inspect-model'].includes(url)),
   'switches never mutate a running service');
  assert.deepEqual(errors,[]);
  console.log('PASS: identical GPU cards, shared settings, explicit capability validation, per-engine assets/raw edits, GPU/layer identity, language/reload and no inference');
 }finally{await context.close();}
};
