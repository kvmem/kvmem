/* The advanced surface is organized without changing submitted backend settings. */
const assert=require('node:assert/strict'),path=require('node:path'),fs=require('node:fs');
module.exports=async({browser,base,info,stage})=>{
  const context=await browser.newContext({viewport:{width:1440,height:1050}}),page=await context.newPage(),errors=[],sent=[];
  const gpu={uuid:'GPU-engine-settings-fixture',name:'Fixture GPU'};
  const p={...info.draft.parameters,QueryMaxTokens:512,QueryReplayMaxTokens:2048,Backend:'polystrata',Model:'C:/fixture/model.gguf',Gpu:gpu.uuid,Context:8192,Budget:2048,Reserve:1024,Prefill:128,MaxTokens:128,HostMiB:256,SinkTokens:128,RecentTokens:64,Concurrency:1,RetainedSessions:1,ThinkingMode:'on',DraftTokens:2,Worker:'',Python:'',Tokenizer:'',PolystrataSource:'',WorkerArgs:['--threads','4']};
  let status={state:'stopped'};
  page.on('pageerror',error=>errors.push(error.message));
  await page.route('**/api/launcher/info',r=>r.fulfill({json:{...info,gpus:[gpu],status,draft:{schema_version:1,parameters:p},workers:Object.fromEntries(['polystrata','ninfer','llamacpp'].map(b=>[b,{available:true,path:'C:/runtime/'+b+'.exe'}]))}}));
  await page.route('**/api/launcher/status',r=>r.fulfill({json:status}));
  await page.route('**/api/launcher/inspect-model',r=>r.fulfill({json:{schema_version:1,status:'ready',can_start:true,checks:[],features:{}}}));
  await page.route('**/api/launcher/preview',r=>{sent.push(r.request().postDataJSON().parameters);return r.fulfill({json:{argv:['fixture-worker'],environment:{}}});});
  if(stage)for(const ext of ['js','css'])await page.route('**/monitor/launcher.'+ext,r=>r.fulfill({path:path.join(stage,'ui/monitor/launcher.'+ext),contentType:ext==='js'?'text/javascript':'text/css'}));
  const f=key=>page.locator(`[data-param=${key}]`);
  const preview=async()=>{await page.locator('[data-action=preview]').click();await page.waitForFunction(()=>document.querySelector('[data-check-summary]')?.textContent.includes('检查通过'));await page.locator('[data-dialog=check] [data-dialog-close]').click();return sent.at(-1);};
  const shot=async name=>{if(process.env.SCREENSHOT_DIR){fs.mkdirSync(process.env.SCREENSHOT_DIR,{recursive:true});await page.locator('.l-native').screenshot({path:path.join(process.env.SCREENSHOT_DIR,name+'.png')});}};
  try{
    await page.goto(base+'/monitor/#config');await f('Backend').waitFor();
    await page.locator('.l-native > summary').click();
    assert.equal(await f('PolystrataSource').isVisible(),false);
    assert.equal(await page.locator('[data-native]').isVisible(),false);
    assert.equal(await page.locator('[data-engine-environment] .l-engine-status').textContent(),'自动配置');
    await shot('polystrata-advanced');
    await page.locator('[data-engine-environment] > summary').click();
    await f('PolystrataSource').fill('C:/custom frontend');await f('Python').fill('C:/python/python.exe');
    await f('Tokenizer').fill('C:/tokenizer');await f('Worker').fill('C:/custom/worker.exe');
    assert.equal(await page.locator('[data-engine-environment] .l-engine-status').textContent(),'已自定义');
    await shot('polystrata-runtime');
    await page.locator('#launch-tab-4').click();await page.locator('[data-vision-assets] > summary').click();
    await f('VisionWorker').fill('C:/vision/encoder.exe');await f('Mmproj').fill('C:/vision/mmproj.gguf');
    assert.equal(await page.locator('.l-native [data-param=VisionWorker]').count(),0);
    await page.locator('[data-engine-arguments] > summary').click();await page.locator('[data-native]').fill('["--threads","6"]');
    const payload=await preview();assert.equal(payload.PolystrataSource,'C:/custom frontend');assert.equal(payload.Python,'C:/python/python.exe');assert.equal(payload.VisionWorker,'C:/vision/encoder.exe');assert.deepEqual(payload.WorkerArgs,['--threads','6']);
    for(const backend of ['ninfer','llamacpp']){
      await f('Backend').selectOption(backend);await page.locator('.l-native > summary').click();
      assert.equal(await page.locator('[data-param=PolystrataSource]').count(),0);
      assert.equal(await page.locator('.l-native [data-param]').count(),1,'only the executable override remains here');
      if(backend==='ninfer'){
        assert.equal(await page.locator('#launch-panel-1 [data-param=NgramDrafts]').count(),1);
        assert.equal(await page.locator('#launch-panel-1 [data-param=DeviceProfile]').count(),1);
      }
      await shot(backend+'-advanced');
    }
    await f('Backend').selectOption('llamacpp');
    await f('Model').fill('C:/fixture/model.gguf');
    await page.locator('#launch-tab-4').click();
    assert.equal(await f('LoraScale').isDisabled(),true);
    await f('Lora').fill('C:/adapters/rank two.gguf');
    await f('LoraScale').fill('-0.5');
    const adapted=await preview();
    assert.equal(adapted.Lora,'C:/adapters/rank two.gguf');
    assert.equal(adapted.LoraScale,-0.5);
    for(const value of [-0.5,0]){
      await page.locator('#launch-tab-4').click();
      await f('LoraScale').fill(String(value));
      await page.locator('[data-preset="llamacpp-default"]').click();
      assert.equal(await f('Lora').inputValue(),'C:/adapters/rank two.gguf');
      assert.equal(await f('LoraScale').inputValue(),String(value),'preset must retain the adapter scale with its path');
      const reapplied=await preview();
      assert.equal(reapplied.LoraScale,value);
    }
    await page.locator('#launch-tab-4').click();
    await f('Lora').fill('');
    assert.equal(await f('LoraScale').isDisabled(),true);
    const plain=await preview();
    assert.equal(plain.Lora,undefined);assert.equal(plain.LoraScale,undefined);
    await f('Backend').selectOption('polystrata');assert.equal(await f('Lora').count(),0);
    assert.equal(await f('Python').inputValue(),'C:/python/python.exe');assert.deepEqual(JSON.parse(await page.locator('[data-native]').inputValue()),['--threads','6']);
    await page.locator('.l-native > summary').click();await page.locator('[data-engine-environment] > summary').click();
    await page.setViewportSize({width:390,height:900});assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));await shot('runtime-mobile');
    await page.setViewportSize({width:1440,height:1050});await page.locator('[data-language-select]').selectOption('en');
    assert.equal(await f('Python').inputValue(),'C:/python/python.exe');
    status={state:'ready',pid:11,instance:'engine-settings-running',active_config:{schema_version:1,parameters:{...p,Python:'C:/active/python.exe'}}};
    await page.waitForFunction(()=>document.querySelector('[data-launch-root]').hasAttribute('data-running-config'));
    await page.locator('.l-native > summary').click();await page.locator('[data-engine-environment] > summary').click();
    assert.equal(await f('Python').inputValue(),'C:/active/python.exe');assert.equal(await f('Python').isDisabled(),true);
    assert.deepEqual(errors,[]);console.log('PASS: three-engine advanced layout, relocated controls, native/path payloads, backend/language retention, running lock and mobile.');
  }finally{await context.close();}
};
