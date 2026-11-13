/* Browser contract for independent scoring controls, legacy migration and native reports. */
const assert=require('node:assert/strict');
module.exports=async({browser,base,info})=>{
 const context=await browser.newContext({viewport:{width:1366,height:900}}),page=await context.newPage(),previews=[],errors=[];
 let runtime={state:'stopped'};
 page.on('pageerror',e=>errors.push(e.message));
 const parameters={...info.draft.parameters,Backend:'ninfer',Model:'C:/fixture/model.ninfer',
  Worker:'C:/fixture/worker.exe',Gpu:'GPU-scoring-fixture',Context:8192,Budget:2048,Reserve:1024,
  NoLrfu:false,LrfuLambda:.4,LrfuTopFraction:.8,ScoreAggregation:'max'};
 for(const key of ['QueryScorer','GraphAlpha','Lrfu'])delete parameters[key];
 await page.route('**/api/launcher/info',r=>r.fulfill({json:{...info,gpus:[{uuid:parameters.Gpu,name:'Fixture GPU'}],draft:{schema_version:1,parameters},status:{state:'stopped'}}}));
 await page.route('**/api/launcher/status',r=>r.fulfill({json:runtime}));
 await page.route('**/api/launcher/preview',r=>{previews.push(r.request().postDataJSON().parameters);return r.fulfill({json:{argv:['fixture-worker'],environment:{}}});});
 const field=k=>page.locator(`[data-param=${k}]`);
 const openAdvanced=async()=>{await page.getByRole('tab',{name:/召回策略|Recall policy/}).click();await page.locator('[data-recall-advanced=scoring] > summary').click();};
 const preview=async()=>{const n=previews.length;await page.locator('[data-action=preview]').click();await page.waitForFunction(()=>document.querySelector('[data-check-summary]')?.textContent.includes('检查通过'));assert.equal(previews.length,n+1);await page.locator('[data-dialog=check] [data-dialog-close]').click();return previews.at(-1);};
 try{
  await page.goto(base+'/monitor/#config');await field('Backend').waitFor();
  assert.equal(await page.locator('[data-scoring-migration]').count(),1);
  await page.locator('[data-language-select]').selectOption('zh-CN');await openAdvanced();
  assert.equal(await field('QueryScorer').inputValue(),'meanq-cosine');assert.equal(await field('GraphAlpha').inputValue(),'0');assert.equal(await field('Lrfu').isChecked(),true);
  assert.equal(await field('QueryMaxTokens').getAttribute('min'),'0');
  assert.equal(await field('QueryReplayMaxTokens').getAttribute('min'),'0');
  await field('QueryMaxTokens').fill('0');await field('QueryReplayMaxTokens').fill('0');
  const unlimited=await preview();assert.equal(unlimited.QueryMaxTokens,0);assert.equal(unlimited.QueryReplayMaxTokens,0);
  await field('QueryScorer').selectOption('page-softmax');assert.equal(await field('Lrfu').isChecked(),true,'scorer must not toggle history');
  await field('GraphAlpha').fill('.2');await field('ScoreAggregation').selectOption('mean');await field('LrfuLambda').fill('.3');await field('LrfuTopFraction').fill('.5');
  await field('QueryMaxTokens').fill('128');await field('QueryReplayMaxTokens').fill('512');
  let sent=await preview();assert.equal(sent.QueryScorer,'page-softmax');assert.equal(sent.GraphAlpha,.2);assert.equal(sent.Lrfu,true);assert.equal(sent.LrfuLambda,.3);assert.equal(sent.QueryMaxTokens,128);assert.equal(sent.QueryReplayMaxTokens,512);assert.equal(Object.hasOwn(sent,'NoLrfu'),false);
  await field('QueryScorer').selectOption('meanq-cosine');await field('Lrfu').uncheck();assert.equal(await field('LrfuLambda').isDisabled(),true);
  sent=await preview();assert.equal(sent.Lrfu,false,'false must survive transport for cosine');assert.equal(sent.GraphAlpha,.2);assert.equal(Object.hasOwn(sent,'LrfuLambda'),false);assert.equal(Object.hasOwn(sent,'LrfuTopFraction'),false);
  await page.waitForFunction(()=>JSON.parse(localStorage.getItem('kvmem-launch-draft-v2')||'null')?.parameters?.Lrfu===false);
  await page.reload();await field('Backend').waitFor();await openAdvanced();assert.equal(await field('Lrfu').isChecked(),false);assert.equal(await field('LrfuLambda').inputValue(),'0.3');
  // Deliver the observed report through the controller boundary so polling and
  // asynchronous inspection both see the same runtime state.
  runtime={state:'ready',active_config:{schema_version:1,parameters:sent},effective_scoring:{scorer:'page-softmax',agg:'max',graph_alpha:.7,lrfu:true,query_max_tokens:64,query_replay_max_tokens:256},scoring_confirmed:true};
  await page.waitForFunction(()=>document.querySelector('[data-runtime-key="GraphAlpha"] .s-runtime-value')?.textContent==='0.7');
  for(const [key,value] of [['QueryScorer','page-softmax'],['ScoreAggregation','max'],['GraphAlpha','0.7'],['QueryMaxTokens','64 tokens'],['QueryReplayMaxTokens','256 tokens']]){
    const row=page.locator(`[data-runtime-key="${key}"]`);
    assert.equal(await row.locator('.s-runtime-value').textContent(),value);
    assert.equal(await row.getAttribute('data-source'),'service');
  }
  await openAdvanced();assert.equal(await field('GraphAlpha').isDisabled(),true);
  assert.equal(await field('GraphAlpha').inputValue(),'0.2','locked launch settings retain the submitted value');
  assert.equal(await page.locator('[data-runtime-key="GraphAlpha"] .s-runtime-value').textContent(),'0.7','the observed report remains distinct from launch settings');
  if(process.env.SCREENSHOT_DIR)await page.screenshot({path:process.env.SCREENSHOT_DIR+'/query-scoring.png',fullPage:true});
  assert.deepEqual(errors,[]);
 }finally{await context.close();}
};
