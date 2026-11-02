/* Gemma launch preset and KV inspection; no inference worker is started. */
const assert=require('node:assert/strict');
module.exports=async({browser,base,info})=>{
 const context=await browser.newContext({viewport:{width:1440,height:1100}});
 try{
  const page=await context.newPage(),errors=[],inspections=[];
  page.on('pageerror',error=>errors.push(error.message));
  const parameters={...info.draft.parameters,Backend:'polystrata',Model:'C:/fixture/gemma.gguf',Gpu:'GPU-gemma-fixture',SpeculativeMode:'none',DraftTokens:0,DraftPolicy:'fixed',NgramDrafts:0,KvType:'f16'};
  await page.route('**/api/launcher/info',r=>r.fulfill({json:{...info,model_inspection_version:1,gpus:[{uuid:parameters.Gpu,name:'Fixture GPU'}],status:{state:'stopped'},draft:{schema_version:1,parameters}}}));
  await page.route('**/api/launcher/status',r=>r.fulfill({json:{state:'stopped'}}));
  await page.route('**/api/launcher/inspect-model',r=>{
   const p=r.request().postDataJSON().parameters;inspections.push(p);
   return r.fulfill({json:{schema_version:1,can_start:p.KvType!=='q8_0',status:p.KvType==='q8_0'?'blocked':'ready'}});
  });
  await page.goto(base+'/monitor/#config');
  await page.locator('[data-param=Backend]').waitFor();
  await page.locator('[data-preset=polystrata-gemma4]').click();
  const f=name=>page.locator(`[data-param=${name}]`);
  assert.equal(await f('KvType').inputValue(),'f16');
  assert.equal(await f('ThinkingMode').inputValue(),'on');
  assert.equal(await f('SpeculativeMode').inputValue(),'none');
  assert.equal(await f('DraftTokens').inputValue(),'0');
  await f('Model').fill('C:/fixture/gemma.gguf');
  const health=state=>page.waitForFunction(expected=>document.querySelector('[data-model-health]')?.dataset.state===expected,state);
  await health('ready');assert.equal(inspections.at(-1).KvType,'f16');
  await f('KvType').selectOption('q8_0');
  await health('blocked');assert.equal(inspections.at(-1).KvType,'q8_0');
  await f('KvType').selectOption('int8k');
  await health('ready');assert.equal(inspections.at(-1).KvType,'int8k');
  await page.locator('#launch-tab-1').click();
  await f('SpeculativeMode').selectOption('mtp');
  assert.equal(await f('DraftModel').isEnabled(),true);
  assert.deepEqual(errors,[]);
  console.log('PASS: Gemma preset, KV inspection invalidation and external assistant input');
 }finally{await context.close();}
};
