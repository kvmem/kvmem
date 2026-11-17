/* No model or real service mutations: exercise the UI's process-owned settings. */
const assert=require('node:assert/strict');
module.exports=async({browser,base,info})=>{
  const context=await browser.newContext({viewport:{width:1440,height:960}}),page=await context.newPage(),errors=[];
  page.on('pageerror',e=>errors.push(e.message));
  const gpu={uuid:'GPU-running-config-fixture',name:'Fixture GPU'};
  const parameters={...info.draft.parameters,QueryMaxTokens:512,QueryReplayMaxTokens:2048,Backend:'polystrata',Model:'C:/models/current.gguf',Gpu:gpu.uuid,Context:65536,Prefill:512,Reserve:2048,RetainedSessions:4,SpeculativeMode:'mtp',DraftTokens:1,DraftPolicy:'adaptive',ThinkingMode:'on',PreserveThinking:true,WorkerArgs:['--elastic-experts']};
  let status={state:'ready',instance:'first',pid:123,url:'http://127.0.0.1:18200',api_key_configured:true,active_config:{schema_version:1,parameters}};
  let stopMode='fail',startMode='normal',statusFails=false,releaseStop,releaseStart,startPayload;
  await context.addInitScript(()=>localStorage.setItem('kvmem-launch-draft-v2',JSON.stringify({version:2,parameters:{Backend:'ninfer',Model:'C:/stale.ninfer',Context:9999},raw:{WorkerArgs:'["stale"]'}})));
  await page.route('**/api/launcher/info',route=>route.fulfill({json:{...info,gpus:[gpu],status}}));
  await page.route('**/api/launcher/status',route=>statusFails?route.fulfill({status:503,json:{error:'fixture disconnected'}}):route.fulfill({json:status}));
  await page.route('**/api/launcher/inspect-model',route=>route.fulfill({json:{schema_version:1,status:'ready',can_start:true,model:{},features:{},checks:[]}}));
  await page.route('**/api/launcher/stop',async route=>{
    if(stopMode==='fail')return route.fulfill({status:500,json:{error:'fixture stop failed'}});
    await new Promise(resolve=>releaseStop=resolve);status={state:'stopped',url:null};await route.fulfill({json:status});
  });
  await page.route('**/api/launcher/start',async route=>{
    startPayload=route.request().postDataJSON();
    if(startMode==='lost'){statusFails=true;return route.abort();}
    await new Promise(resolve=>releaseStart=resolve);
    status={state:'starting',instance:'second',pid:456,url:'http://127.0.0.1:18200',active_config:startPayload};await route.fulfill({json:status});
  });
  async function locked(){
    await page.waitForFunction(()=>document.querySelector('[data-launch-root]').hasAttribute('data-running-config'));
    assert.equal(await page.locator('[data-launch-root] form').locator('input:not(:disabled),select:not(:disabled),textarea:not(:disabled)').count(),0);
    assert.equal(await page.locator('.l-presets button:not(.l-help):not(:disabled)').count(),0);
    assert.equal(await page.locator('[data-action=preview]').isDisabled(),true);
    assert.equal(await page.locator('[data-action=start]').isDisabled(),true);
  }
  try{
    await page.goto(base+'/monitor/#config');await locked();
    assert.equal(await page.locator('[data-param=Backend]').inputValue(),'polystrata');
    assert.equal(await page.locator('[data-param=Context]').inputValue(),'65536');
    assert.equal(await page.locator('[data-param=Model]').inputValue(),parameters.Model);
    assert.equal(await page.locator('[data-native]').inputValue(),'["--elastic-experts"]');
    assert.equal(await page.locator('[data-param=ApiKey]').inputValue(),'');
    assert.ok((await page.locator('[data-param=ApiKey]').getAttribute('placeholder')).includes('已设置'));
    assert.equal(await page.locator('[data-action=stop]').isEnabled(),true);
    await page.getByRole('tab',{name:'思考设置'}).click();
    assert.equal(await page.locator('[data-param=PreserveThinking]').isChecked(),true);
    assert.equal(await page.locator('[data-param=PreserveThinking] + .l-toggle-state [data-toggle-on]').isVisible(),true);
    await page.locator('[data-language-select]').selectOption('en');await locked();
    assert.equal(await page.locator('[data-param=Context]').inputValue(),'65536');
    assert.equal(await page.evaluate(()=>JSON.parse(localStorage.getItem('kvmem-launch-draft-v2')).parameters.Context),9999,'running view never overwrites browser draft');
    await page.locator('[data-action=stop]').click();await page.locator('[data-dialog=error]').waitFor();await locked();
    await page.locator('[data-dialog=error] [data-dialog-close]').click();
    stopMode='success';await page.locator('[data-action=stop]').click();await locked();
    releaseStop();await page.waitForFunction(()=>!document.querySelector('[data-launch-root]').hasAttribute('data-running-config'));
    assert.equal(await page.locator('[data-param=Context]').inputValue(),'65536');
    assert.equal(await page.locator('[data-param=Context]').isEnabled(),true);
    assert.equal(await page.locator('[data-action=preview]').isEnabled(),true);
    assert.equal(await page.locator('[data-param=LogLevel]').isDisabled(),true,'backend capability restrictions survive unlock');
    await page.locator('[data-param=Context]').fill('70000');
    await page.locator('[data-action=start]').click();await locked();
    assert.equal(startPayload.parameters.Context,70000);
    assert.deepEqual(startPayload.parameters.WorkerArgs,['--elastic-experts']);
    releaseStart();await page.waitForFunction(()=>document.querySelector('[data-runtime-state]').dataset.state==='starting');await locked();
    for(const backend of ['ninfer','llamacpp','polystrata']){
      status={...status,state:'ready',instance:backend,active_config:{schema_version:1,parameters:{...parameters,Backend:backend,Context:80000}}};
      await page.waitForFunction(backend=>document.querySelector('[data-param=Backend]').value===backend,backend);await locked();
      assert.equal(await page.locator('[data-param=Context]').inputValue(),'80000');
    }
    await page.reload();await locked();assert.equal(await page.locator('[data-param=Context]').inputValue(),'80000');
    status={state:'exited',exit_code:1};await page.waitForFunction(()=>!document.querySelector('[data-launch-root]').hasAttribute('data-running-config'));
    assert.equal(await page.locator('[data-param=Context]').isEnabled(),true);
    assert.equal(await page.locator('[data-param=Context]').inputValue(),'80000');
    startMode='lost';await page.locator('[data-action=start]').click();
    await page.locator('[data-dialog=error]').waitFor();await locked();
    await page.locator('[data-dialog=error] [data-dialog-close]').click();
    await page.locator('[data-language-select]').selectOption('zh-CN');await locked();
    assert.ok((await page.locator('[data-draft-state]').textContent()).includes('确认服务状态'));
    statusFails=false;status={state:'ready',instance:'recovered',pid:789,active_config:startPayload};
    await page.waitForFunction(()=>document.querySelector('[data-runtime-state]').dataset.state==='ready');await locked();
    assert.deepEqual(errors,[]);
    console.log('PASS: active parameters override stale draft, all backends lock, language/reload, native args, credential masking, tabs, failed/pending stop, successful unlock, restart payload and process exit');
  }finally{await context.close();}
};
