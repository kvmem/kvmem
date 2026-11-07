/* Actual launcher DOM with model-free HTTP fixtures; no server/model/GPU. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { chromium } = require('playwright');
const root = path.resolve(__dirname, '..');
const gpu = 'GPU-00000000-0000-0000-0000-000000000001';
const profile = {available:true,current_layer_q:true,query_replay:false,position_policy:'original',
  checkpoint_retention:{default_target:10,max_target:256,metadata_default:'auto'},
  minimum_reserve_blocks:3,max_devices:2,max_concurrency:1,parallel_modes:['single','pipeline','tensor'],
  kv_formats:['bf16'],vision_residency:[],speculative:{none:{available:true},
    mtp:{available:true,max_draft_tokens:3,adaptive:false,ngram:false,
      history_policies:['first-layer','upstream-global'],state_modes:['native']},
    dflash2:{available:true,max_draft_tokens:1,adaptive:false,ngram:false,history:'independent-native-window'}}};
const parameters = {Backend:'ninfer',Worker:'worker-a',Model:'metadata-model.ninfer',Gpu:gpu,
  RecallStrategy:'layerwise',SpeculativeMode:'mtp',DraftTokens:5,DraftPolicy:'adaptive',NgramDrafts:7,
  LayerwiseMtpHistory:'first-layer',KvType:'bf16',Vision:false,Context:4096,Budget:1024,Reserve:1024,
  HostMiB:128,Prefill:256,MaxTokens:16,SinkTokens:0,RecentTokens:0};
const info = {token:'fixture',backends:['ninfer'],backend_root:root,gpus:[{uuid:gpu,name:'Fixture GPU'}],
  recall_strategy_version:1,worker_capabilities_version:1,query_scoring_version:2,kv_cache_version:1,
  model_inspection_version:1,model_library_version:0,script_export_version:0,native_picker:false,
  native_options:{ninfer:[]},backend_capabilities:{ninfer:{}},workers:{},status:{state:'stopped'},
  speculative_capabilities:JSON.parse(fs.readFileSync(path.join(root,'scripts/speculative-capabilities.json'),'utf8')),
  draft:{schema_version:1,parameters}};
const selector = name => `[data-param=${name}]`;
async function main(){
  const browser = await chromium.launch({headless:true,
    ...(process.env.KVMEM_TEST_BROWSER?{executablePath:process.env.KVMEM_TEST_BROWSER}:{}),
    args:['--disable-gpu']});
  const page = await browser.newPage();
  const fill=async(name,value)=>{
    const input=page.locator(selector(name));
    await input.evaluate(e=>{for(let p=e.parentElement;p;p=p.parentElement){p.hidden=false;if(p.tagName==='DETAILS')p.open=true;}});
    await input.fill(value);
  };
  const errors=[];page.on('pageerror',e=>errors.push(e.message));
  // Transport cancellation is best-effort: deliberately deliver the old
  // response so the application's own selection/sequence guard is exercised.
  await page.addInitScript(()=>{const nativeFetch=window.fetch;window.fetch=(url,options={})=>nativeFetch(url,
    String(url).includes('/worker-capabilities')?{...options,signal:undefined}:options);});
  let held, releaseHeld;
  const previews=[];
  const checks=[];
  try {
    await page.route('**/*',async route=>{
      const url=new URL(route.request().url());
      if(url.pathname==='/api/launcher/info')return route.fulfill({json:info});
      if(url.pathname==='/api/launcher/status')return route.fulfill({json:info.status});
      if(url.pathname==='/api/launcher/inspect-model')return route.fulfill({json:{schema_version:1,status:'ready',can_start:true,checks:[]}});
      if(url.pathname==='/api/launcher/save')return route.fulfill({json:route.request().postDataJSON()});
      if(url.pathname==='/api/launcher/preview'){previews.push(route.request().postDataJSON());return route.fulfill({status:400,json:{error:'Fixture boundary: no worker is launched'}});}
      if(url.pathname==='/api/launcher/worker-capabilities'){
        const p=route.request().postDataJSON().parameters;
        const result={schema_version:1,backend:p.Backend,strategy:p.RecallStrategy,requested_worker:p.Worker,
          worker:{path:p.Worker,executable_sha256:'a'.repeat(64)},profile:structuredClone(profile)};
        if(p.Worker==='worker-b'){result.profile.speculative.mtp.max_draft_tokens=2;held=true;await new Promise(resolve=>releaseHeld=resolve);}
        if(p.Worker==='worker-c')result.profile.speculative.mtp.max_draft_tokens=1;
        return route.fulfill({json:result}).catch(()=>{});
      }
      if(url.pathname==='/launcher.js')return route.fulfill({contentType:'text/javascript',body:fs.readFileSync(path.join(root,'ui/monitor/launcher.js'))});
      if(url.pathname==='/')return route.fulfill({contentType:'text/html',body:'<!doctype html><html><head><meta charset="utf-8"></head><body><button data-page="config"></button><a data-chat-link></a><main data-launch-root></main><script>window.monitorI18n={language:"en"};</script><script src="/launcher.js"></script></body></html>'});
      return route.fulfill({status:404,body:''});
    });
    await page.goto('http://kvmem-ui.test/');
    await page.waitForFunction(()=>document.querySelector('[data-param=DraftTokens]')?.max==='3');
    assert.equal(await page.getAttribute(selector('LayerwiseCheckpointTarget'),'placeholder'),'10');
    assert.equal(await page.inputValue(selector('LayerwiseCheckpointTarget')),'');
    assert.equal(await page.getAttribute(selector('LayerwiseMetadataMiB'),'placeholder'),'Auto');
    assert.equal(await page.inputValue(selector('DraftTokens')),'5');
    assert.equal(await page.inputValue(selector('DraftPolicy')),'adaptive');
    assert.equal(await page.inputValue(selector('NgramDrafts')),'7');
    assert.equal(await page.locator(selector('DraftTokens')).evaluate(e=>e.validity.rangeOverflow),true);
    assert.match(await page.locator(selector('DraftPolicy')).evaluate(e=>e.validationMessage),/does not qualify adaptive/);
    assert.match(await page.locator(selector('NgramDrafts')).evaluate(e=>e.validationMessage),/does not qualify N-gram/);
    checks.push('actual profile constrains width/adaptive/ngram while retaining explicit unsupported values');
    await page.selectOption(selector('DraftPolicy'),'fixed',{force:true});await fill('NgramDrafts','0');
    await page.selectOption(selector('SpeculativeMode'),'dflash2',{force:true});
    assert.equal(await page.inputValue(selector('DraftTokens')),'5');
    assert.equal(await page.getAttribute(selector('DraftTokens'),'max'),'1');
    assert.equal(await page.inputValue(selector('LayerwiseMtpHistory')),'first-layer');
    await page.selectOption(selector('SpeculativeMode'),'mtp',{force:true});
    await page.selectOption(selector('LayerwiseMtpHistory'),'',{force:true});
    assert.equal(await page.locator(selector('LayerwiseMtpHistory')).evaluate(e=>e.validity.valueMissing),true);
    checks.push('DFlash independent width and OWN policy no default/no silent reset');
    await fill('Worker','worker-b');
    assert.equal(await page.inputValue(selector('Worker')),'worker-b');
    for(let attempt=0;!held&&attempt<80;attempt++)await page.waitForTimeout(25);
    assert.equal(held,true,await page.locator('[data-worker-capability-note]').textContent());
    await fill('Worker','worker-c');
    await page.waitForFunction(()=>document.querySelector('[data-param=DraftTokens]').max==='1');
    releaseHeld();await page.waitForTimeout(100);
    assert.equal(await page.getAttribute(selector('DraftTokens'),'max'),'1');
    assert.match(await page.locator('[data-worker-capability-note]').textContent(),/worker-c/);
    checks.push('late worker-b response cannot overwrite worker-c capabilities');
    await fill('LayerwiseActivationDiskMiB','64');
    assert.equal(await page.locator(selector('DiskEnabled')).isChecked(),false);
    assert.equal(await page.locator(selector('DiskPath')).isEnabled(),true);
    assert.equal(await page.getAttribute(selector('DiskPath'),'required'),'');
    assert.equal(await page.locator(selector('DiskMiB')).isDisabled(),true);
    await fill('DiskPath','/tmp/activation-only');
    await fill('DraftTokens','1');
    await fill('LayerwiseCheckpointTarget','3');
    await page.selectOption(selector('LayerwiseMtpHistory'),'first-layer',{force:true});
    await page.locator('[data-action=preview]').click({force:true});
    for(let attempt=0;!previews.length&&attempt<40;attempt++)await page.waitForTimeout(25);
    assert.equal(previews.length,1);
    assert.equal(previews[0].parameters.DiskMiB,0);
    assert.equal(previews[0].parameters.DiskPath,'/tmp/activation-only');
    assert.equal(previews[0].parameters.LayerwiseActivationDiskMiB,64);
    assert.equal(previews[0].parameters.LayerwiseCheckpointTarget,3);
    assert.equal(previews[0].parameters.LayerwiseMetadataMiB,undefined);
    checks.push('activation-only SSD enables shared directory without canonical quota');
    await page.selectOption(selector('RecallStrategy'),'legacy',{force:true});
    await page.selectOption(selector('SpeculativeMode'),'dflash2',{force:true});
    assert.equal(await page.inputValue(selector('DraftTokens')),'5');
    checks.push('Legacy draft selection retains existing defaults');
    info.draft.parameters={...parameters};delete info.draft.parameters.DraftTokens;
    await page.addInitScript(()=>localStorage.clear());await page.reload();
    await page.waitForFunction(()=>document.querySelector('[data-param=DraftTokens]')?.max==='3');
    assert.equal(await page.inputValue(selector('DraftTokens')),'');
    assert.equal(await page.locator(selector('DraftTokens')).evaluate(e=>e.validity.valueMissing),true);
    checks.push('missing new-mode draft width remains explicit and required');
    assert.deepEqual(errors,[]);
    console.log(JSON.stringify({status:'PASS',scope:'offline actual UI DOM; no model or GPU',browser:browser.version(),checks},null,2));
  } finally {if(releaseHeld)releaseHeld();await browser.close();}
}
main().catch(error=>{console.error(error);process.exitCode=1;});
