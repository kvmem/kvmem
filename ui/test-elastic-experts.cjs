/* Native flag round-tripping and lifecycle locking, with no model or service launch. */
const assert=require('node:assert/strict');
module.exports=async({browser,base,info,stage})=>{
 const context=await browser.newContext({viewport:{width:1440,height:1100}}),page=await context.newPage(),errors=[],previews=[];
 const p={...info.draft.parameters,QueryMaxTokens:512,QueryReplayMaxTokens:2048,Backend:'polystrata',Model:'C:/fixture/model.gguf',Gpu:'GPU-elastic-fixture',Context:262144,Budget:32768,Reserve:2048,Prefill:512,RetainedSessions:4,ThinkingMode:'on',QueryScorer:'page-softmax',GraphAlpha:.2,Lrfu:false,
  DraftTokens:2,WorkerArgs:['--threads','8','--elastic-experts-headroom','--elastic-headroom-mib','384','--elastic-physical-reserve-mib','768']};
 let status={state:'ready',pid:12,instance:'elastic-first',active_config:{schema_version:1,parameters:p}};
 page.on('pageerror',e=>errors.push(e.message));
 await page.route('**/api/launcher/info',r=>r.fulfill({json:{...info,query_scoring_version:2,gpus:[{uuid:p.Gpu,name:'Fixture GPU'}],status,draft:{schema_version:1,parameters:p}}}));
 await page.route('**/api/launcher/status',r=>r.fulfill({json:status}));
 await page.route('**/api/launcher/inspect-model',r=>r.fulfill({json:{schema_version:1,status:'ready',can_start:true,checks:[],features:{}}}));
 await page.route('**/api/launcher/preview',r=>{previews.push(r.request().postDataJSON().parameters);return r.fulfill({json:{argv:['fixture-worker'],environment:{}}});});
 if(stage)for(const suffix of ['js','css'])await page.route('**/monitor/launcher.'+suffix,r=>r.fulfill({path:require('node:path').join(stage,'ui/monitor/launcher.'+suffix),contentType:suffix==='js'?'text/javascript':'text/css'}));
 const f=k=>page.locator(`[data-param=${k}]`),mode=f('ElasticExpertsMode'),ceiling=f('ElasticHeadroomMiB'),reserve=f('ElasticPhysicalReserveMiB'),raw=page.locator('[data-native]');
 async function preview(){await page.locator('[data-action=preview]').click();await page.waitForFunction(()=>document.querySelector('[data-check-summary]')?.textContent.includes('检查通过'));await page.locator('[data-dialog=check] [data-dialog-close]').click();const sent=previews.at(-1);assert.equal(Object.keys(sent).some(k=>k.startsWith('Elastic')),false);return sent.WorkerArgs;}
 try{
  await page.goto(base+'/monitor/#config');await mode.waitFor({state:'attached'});await page.getByRole('tab',{name:'弹性专家'}).click();
  await page.waitForFunction(()=>document.querySelector('[data-launch-root]').hasAttribute('data-running-config'));
  assert.equal(await page.locator('#launch-panel-1 [data-elastic-controls]').count(),0);assert.equal(await page.locator('#launch-panel-7 [data-elastic-controls]').count(),1);assert.equal(await mode.inputValue(),'headroom');assert.equal(await ceiling.inputValue(),'384');assert.equal(await reserve.inputValue(),'768');
  for(const input of [mode,ceiling,reserve])assert.equal(await input.isDisabled(),true);
  status={state:'stopped'};await page.waitForFunction(()=>!document.querySelector('[data-launch-root]').hasAttribute('data-running-config'));
  await mode.selectOption('workspace');assert.deepEqual(await preview(),['--threads','8','--elastic-experts']);assert.equal(await ceiling.isDisabled(),true);
  await mode.selectOption('off');assert.deepEqual(await preview(),['--threads','8']);
  await mode.selectOption('headroom');assert.equal(await ceiling.inputValue(),'384');await ceiling.fill('512');await reserve.fill('256');
  const expected=['--threads','8','--elastic-experts-headroom','--elastic-headroom-mib','512','--elastic-physical-reserve-mib','256'];
  assert.deepEqual(await preview(),expected);
  await ceiling.fill('1025');const count=previews.length;await page.locator('[data-action=preview]').click();await page.getByText('检查未通过',{exact:true}).waitFor();assert.equal(previews.length,count);await page.locator('[data-dialog=check] [data-dialog-close]').click();await ceiling.fill('512');
  await page.locator('[data-draft-state]').filter({hasText:'已自动保存'}).waitFor();await page.reload();await mode.waitFor({state:'attached'});await page.getByRole('tab',{name:'弹性专家'}).click();assert.equal(await ceiling.inputValue(),'512');assert.equal(await mode.inputValue(),'headroom');
  await page.locator('.l-native').evaluate(e=>e.open=true);await page.locator('[data-engine-arguments]').evaluate(e=>e.open=true);await raw.fill('[');assert.equal(await mode.isDisabled(),true);assert.equal(await raw.inputValue(),'[');
  await raw.fill('["--threads","4","--elastic-experts-headroom"]');assert.equal(await mode.inputValue(),'headroom');assert.equal(await ceiling.inputValue(),'128');assert.equal(await reserve.inputValue(),'512');
  await raw.fill('["--elastic-experts","--elastic-experts"]');assert.equal(await mode.isDisabled(),true);
  await raw.fill('["--threads","4","--elastic-experts"]');assert.equal(await mode.inputValue(),'workspace');assert.equal(await mode.isEnabled(),true);
  for(const backend of ['ninfer','llamacpp']){
   await f('Backend').selectOption(backend);assert.equal(await page.locator('[data-elastic-controls]').count(),0);
   assert.equal(await page.getByRole('tab',{name:'弹性专家',exact:true}).count(),0);
   assert.equal(await page.locator('#launch-tab-1').getAttribute('aria-selected'),'true');
   assert.equal(await page.locator('.l-panel:not([hidden])').count(),1);
  }
  await f('Backend').selectOption('polystrata');await page.getByRole('tab',{name:'弹性专家'}).click();assert.equal(await mode.inputValue(),'workspace');assert.deepEqual(JSON.parse(await raw.inputValue()),['--threads','4','--elastic-experts']);
  await mode.selectOption('headroom');await ceiling.fill('256');await page.locator('[data-language-select]').selectOption('en');assert.equal(await mode.inputValue(),'headroom');assert.equal(await ceiling.inputValue(),'256');
  if(process.env.SCREENSHOT_DIR)await page.screenshot({path:require('node:path').join(process.env.SCREENSHOT_DIR,'elastic-desktop.png'),fullPage:true});
  await page.setViewportSize({width:390,height:900});assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
  if(process.env.SCREENSHOT_DIR)await page.screenshot({path:require('node:path').join(process.env.SCREENSHOT_DIR,'elastic-mobile.png'),fullPage:true});
  status={state:'ready',pid:13,instance:'elastic-second',active_config:{schema_version:1,parameters:{...p,WorkerArgs:['--elastic-experts']}}};
  await page.waitForFunction(()=>document.querySelector('[data-launch-root]').hasAttribute('data-running-config'));assert.equal(await mode.inputValue(),'workspace');assert.equal(await mode.isDisabled(),true);
  assert.deepEqual(errors,[]);console.log('PASS: elastic modes, numeric bounds, native round-trip, unrelated flags, explicit values/defaults, invalid JSON, backend/draft/language, runtime locking and mobile.');
 }finally{await context.close();}
};
