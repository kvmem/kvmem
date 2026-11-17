/* Real UI, fixture capability responses. No model, worker or GPU execution. */
const assert=require('node:assert/strict');
const path=require('node:path');
module.exports=async({browser,base,info,task})=>{
 const context=await browser.newContext({viewport:{width:1440,height:1100}}),page=await context.newPage(),errors=[];
 const gpu='GPU-00000000-0000-0000-0000-000000000001';
 const parameters={Backend:'ninfer',Model:'C:/fixture/model.ninfer',Worker:'C:/fixture/worker.exe',Gpu:gpu,
  Context:65536,Budget:32768,Reserve:1024,HostMiB:16384,Prefill:512,Concurrency:1,
  KvType:'bf16',SpeculativeMode:'none',DraftTokens:0,DraftPolicy:'fixed',Vision:false,
  SinkTokens:0,RecentTokens:0,MaxTokens:16,QueryScorer:'page-softmax',GraphAlpha:.2,
  RecallStrategy:'legacy',LayerwiseMetadataMiB:384,QueryMaxTokens:512,QueryReplayMaxTokens:2048};
 let response='ready',requests=0,startRequests=0,runtime={state:'stopped'};
 const field=key=>page.locator(`[data-param=${key}]`);
 const layerwiseOption=page.locator('[data-param=RecallStrategy] option[value=layerwise]');
 const capabilityState=state=>page.waitForFunction(value=>document.querySelector('[data-param=RecallStrategy]')?.dataset.capabilityState===value,state);
 const ready=()=>capabilityState('ready');
 const reload=async()=>{await page.reload();await field('Backend').waitFor();await page.getByRole('tab',{name:/召回策略|Recall policy/}).click();};
 page.on('pageerror',e=>errors.push(e.message));
 await page.route('**/api/launcher/info',r=>r.fulfill({json:{...info,worker_capabilities_version:1,gpus:[{uuid:gpu,name:'Fixture GPU',vendor:'nvidia'}],draft:{schema_version:1,parameters},status:runtime}}));
 await page.route('**/api/launcher/status',r=>r.fulfill({json:runtime}));
 await page.route('**/api/launcher/start',r=>{startRequests++;return r.fulfill({status:400,json:{error:'Unexpected start in fixture test'}});});
 await page.route('**/api/launcher/inspect-model',r=>r.fulfill({json:{schema_version:1,status:'ready',can_start:true,model:{},checks:[]}}));
 await page.route('**/api/launcher/worker-capabilities',async r=>{
  requests++;const p=r.request().postDataJSON().parameters;
  if(response==='error')return r.fulfill({status:500,json:{error:'Fixture worker is temporarily unavailable'}});
  await r.fulfill({json:{schema_version:1,backend:p.Backend,strategy:p.RecallStrategy,requested_worker:p.Worker,
   worker:{path:`C:/KVMem/backends/${p.Backend}/worker.exe`,executable_sha256:'a'.repeat(64)},
   profile:p.Backend!=='ninfer'||response==='missing'?null:{available:response!=='unsupported',default_resources:{activation_ram_bytes:268435456,activation_disk_bytes:0,metadata_bytes:1073741824,staging_bytes:16777216,activation_metadata_bytes:134217728},
    checkpoint_retention:{default_target:4,max_target:16},speculative:{none:{available:true},mtp:{available:true,max_draft_tokens:3,history_policies:['first-layer'],default_history_policy:'first-layer'}}}}});
 });
 async function snapshot(name){if(task)await page.locator('.l-recall-panel').screenshot({path:path.join(task,name+'.png')});}
 try{
  await page.goto(base+'/monitor/?lang=zh-CN#config');await field('Backend').waitFor();await page.getByRole('tab',{name:/召回策略|Recall policy/}).click();
  await ready();assert.ok(requests>0,'capabilities are probed before the strategy can be chosen');
  assert.equal(await layerwiseOption.evaluate(el=>el.disabled),false);
  assert.equal(await page.locator('[data-worker-capability-status]').count(),0,'no separate capability card');
  assert.equal(await field('LayerwiseStagingMiB').inputValue(),'','Legacy does not inherit layerwise defaults');
  assert.equal(await page.locator('[data-graph-recall]').isVisible(),true);
  assert.equal(await page.locator('[data-layerwise-budgets]').isVisible(),false);
  assert.equal(await field('QueryMaxTokens').isVisible(),false,'low-frequency scoring limits are collapsed');
  await snapshot('recall-legacy');
  await page.locator('[data-recall-advanced=scoring] > summary').click();await field('QueryMaxTokens').fill('-1');
  await page.locator('[data-recall-advanced=scoring] > summary').click();await page.locator('#launch-tab-0').click();
  await page.locator('[data-action=preview]').click();await page.locator('[data-check-summary]').waitFor();
  assert.equal(await page.locator('#launch-tab-0').getAttribute('aria-selected'),'true','passive configuration checks do not switch tabs');
  assert.equal(await field('QueryMaxTokens').isVisible(),false,'passive checks do not expand hidden fields');
  await page.locator('[data-dialog=check] [data-dialog-close]').click();
  await page.locator('[data-action=start]').click();
  assert.equal(await field('QueryMaxTokens').isVisible(),true,'invalid collapsed setting is revealed with its tab');
  await field('QueryMaxTokens').fill('256');
  // A later invalid recall control must not hide the first invalid field.
  await field('QueryMaxTokens').fill('-1');
  await page.locator('[data-recall-advanced=scoring] > summary').click();
  await page.locator('#launch-tab-1').click();await field('Prefill').fill('-1');
  await page.locator('[data-action=start]').click();
  assert.equal(await field('Prefill').isVisible(),true,'first invalid hardware field stays visible');
  assert.equal(await field('Prefill').evaluate(el=>document.activeElement===el),true,'first invalid field receives focus');
  await field('Prefill').fill('512');await page.getByRole('tab',{name:/召回策略|Recall policy/}).click();
  await page.locator('[data-recall-advanced=scoring] > summary').click();await field('QueryMaxTokens').fill('256');
  await field('RecallStrategy').selectOption('layerwise');await ready();
  assert.equal(await page.locator('[data-graph-recall]').isVisible(),false);
  assert.equal(await page.locator('[data-layerwise-budgets]').isVisible(),true);
  assert.equal(await page.locator('[data-layerwise-drafts]').isVisible(),false,'no draft explanation when drafts are off');
  assert.equal(await page.locator('[data-worker-capability-refresh], [data-worker-capability-retry], [data-worker-capability-details]').count(),0);
  assert.equal(await field('LayerwiseMetadataMiB').inputValue(),'384','explicit resource override survives automatic detection');
  await snapshot('recall-layerwise');
  await page.waitForFunction(()=>JSON.parse(localStorage.getItem('kvmem-launch-draft-v2')||'{}').parameters?.RecallStrategy==='layerwise');
  response='error';await reload();await capabilityState('error');
  assert.equal(await layerwiseOption.evaluate(el=>el.disabled),true);
  assert.match(await layerwiseOption.textContent(),/暂时无法确认支持/);
  assert.equal(await page.locator('[data-action=start]').isDisabled(),true);
  await snapshot('recall-error');
  response='ready';await field('RecallStrategy').focus();await ready();
  assert.equal(await field('LayerwiseMetadataMiB').inputValue(),'384');
  response='unsupported';await reload();await capabilityState('unsupported');
  assert.equal(await layerwiseOption.evaluate(el=>el.disabled),true);
  assert.equal(await page.locator('[data-action=start]').isDisabled(),true);
  response='ready';await reload();await ready();
  for(const backend of ['llamacpp','polystrata','radiance']){
   await field('Backend').selectOption(backend);await page.getByRole('tab',{name:/召回策略|Recall policy/}).click();
   await capabilityState('unsupported');
   assert.equal(await layerwiseOption.evaluate(el=>el.disabled),true,backend+' cannot select an unsupported strategy');
   assert.match(await layerwiseOption.textContent(),/（此推理后端暂不支持）/);
   assert.equal(await field('RecallStrategy').inputValue(),'legacy');
   assert.equal(await field('LayerwiseStagingMiB').inputValue(),'','NInfer display defaults do not leak to '+backend);
   assert.equal(await field('QueryScorer').isEnabled(),true,backend+' retains Legacy controls');
   if(task)await snapshot('recall-unsupported-'+backend);
  }
  await field('Backend').selectOption('ninfer');await page.getByRole('tab',{name:/召回策略|Recall policy/}).click();await ready();
  assert.equal(await field('LayerwiseMetadataMiB').inputValue(),'384','NInfer override survives backend round trip');
  await page.waitForFunction(()=>JSON.parse(localStorage.getItem('kvmem-launch-draft-v2')||'{}').parameters?.Backend==='ninfer');
  response='missing';await reload();await capabilityState('unsupported');
  assert.equal(await layerwiseOption.evaluate(el=>el.disabled),true);
  assert.equal(await field('RecallStrategy').inputValue(),'layerwise','an unavailable saved choice is not silently changed');
  assert.equal(await page.locator('[data-action=start]').isDisabled(),true,'old NInfer is not assumed capable by name');
  response='ready';await reload();await ready();
  await page.locator('#launch-tab-1').click();await field('SpeculativeMode').selectOption('mtp');
  await page.getByRole('tab',{name:/召回策略|Recall policy/}).click();assert.equal(await field('LayerwiseMtpHistory').isVisible(),true);
  await field('RecallStrategy').selectOption('legacy');
  assert.equal(await field('QueryMaxTokens').inputValue(),'256');
  await field('RecallStrategy').selectOption('layerwise');await ready();
  assert.equal(await field('LayerwiseMetadataMiB').inputValue(),'384');
  await page.waitForFunction(()=>JSON.parse(localStorage.getItem('kvmem-launch-draft-v2')||'{}').parameters?.RecallStrategy==='layerwise');
  await page.reload();await field('Backend').waitFor();await page.getByRole('tab',{name:/召回策略|Recall policy/}).click();await ready();
  assert.equal(await field('LayerwiseMetadataMiB').inputValue(),'384');
  const saved=await page.evaluate(()=>JSON.parse(localStorage.getItem('kvmem-launch-draft-v2')).parameters);
  assert.equal(saved.LayerwiseMetadataMiB,384);assert.equal(saved.LayerwiseStagingMiB,undefined,'displayed worker defaults do not become overrides');
  for(const [width,language,theme] of [[1440,'en','blue'],[390,'zh-CN','dark'],[390,'en','blue']]){
   await page.setViewportSize({width:1440,height:1100});
   await page.locator('[data-language-select]').selectOption(language);await page.locator('[data-theme-select]').selectOption(theme);
   await page.setViewportSize({width,height:1100});
   await page.getByRole('tab',{name:/召回策略|Recall policy/}).click();await ready();
   assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true,`${width}/${language}: no horizontal overflow`);
   await snapshot(`recall-${width}-${language}-${theme}`);
  }
  runtime={state:'ready',instance:'recall-test',active_config:{schema_version:1,parameters:saved},effective_scoring:{strategy:'layerwise'}};
  await page.waitForFunction(()=>document.querySelector('[data-launch-root]').hasAttribute('data-running-config'));
  assert.equal(await field('RecallStrategy').isDisabled(),true);
  assert.equal(await field('RecallStrategy').getAttribute('data-capability-state'),'running');
  assert.equal(startRequests,0,'invalid fields prevent start requests');
  assert.deepEqual(errors,[]);
  console.log('PASS recall layout: disabled unsupported options, automatic checks/retry, old NInfer, retained unavailable selections, first invalid field, persisted overrides, responsive bilingual themes, running lock');
 }finally{await context.close();}
};
