/* Offline launcher UI checks. No worker, model, native server or GPU is started. */
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const http=require('node:http');
const os=require('node:os');
const root=path.resolve(__dirname,'..');
const task=process.env.TEST_ARTIFACTS||path.dirname(root);
const temp=path.join(task,'layerwise-browser-temp');fs.mkdirSync(temp,{recursive:true});
os.tmpdir=()=>temp;
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const uuid='GPU-00000000-0000-0000-0000-000000000001';
const profiles=JSON.parse(fs.readFileSync(path.join(root,'scripts/speculative-capabilities.json'),'utf8'));
const capabilities=JSON.parse(fs.readFileSync(path.join(root,'scripts/launcher-capabilities.json'),'utf8')).backends;
let state={state:'stopped'},parameters;
const baseParameters={Backend:'ninfer',Model:'C:/fixture/model.ninfer',Worker:'C:/fixture/worker.exe',Gpu:uuid,
 Context:65536,Budget:32768,Reserve:1024,HostMiB:16384,Prefill:512,Concurrency:1,
 KvType:'bf16',SpeculativeMode:'none',DraftTokens:0,DraftPolicy:'fixed',Vision:false,
 SinkTokens:0,RecentTokens:0,MaxTokens:16,QueryScorer:'page-softmax',GraphAlpha:.2};
function info(){return {token:'offline',backends:Object.keys(capabilities),backend_capabilities:capabilities,
 speculative_capabilities:profiles,workers:{},native_options:Object.fromEntries(Object.keys(capabilities).map(b=>[b,[]])),backend_root:'C:/fixture',platform:'windows',
 native_picker:false,script_export_version:0,query_scoring_version:2,recall_strategy_version:1,kv_cache_version:1,
 model_inspection_version:1,worker_capabilities_version:1,budget_adjustment:true,gpus:[{uuid,name:'Offline test GPU',vendor:'nvidia'}],
 draft:{schema_version:1,parameters},status:state};}
const server=http.createServer((req,res)=>{
 const pathname=new URL(req.url,'http://localhost').pathname;
 const json=(value,code=200)=>{res.writeHead(code,{'Content-Type':'application/json'});res.end(JSON.stringify(value));};
 // Synthetic contracts exercise the generic UI; they do not qualify a backend.
 if(pathname==='/api/launcher/worker-capabilities'){
  let body='';req.on('data',chunk=>body+=chunk);req.on('end',()=>{
   const p=JSON.parse(body).parameters;
   json({schema_version:1,backend:p.Backend,strategy:p.RecallStrategy,requested_worker:p.Worker,
    worker:{path:p.Worker,executable_sha256:'a'.repeat(64)},profile:{available:true,
     max_devices:1,max_concurrency:1,kv_formats:['bf16','f16'],parallel_modes:['single'],
     speculative:{none:{available:true},mtp:{available:true,max_draft_tokens:3,
      adaptive:false,ngram:false,history_policies:['first-layer','upstream-global']},
      dflash2:{available:true,max_draft_tokens:3,history:'independent-native-window'},
      guess:{available:true,max_draft_tokens:3,adaptive:true,ngram:false}}}});
  });return;
 }
 if(pathname==='/api/launcher/info')return json(info());
 if(pathname==='/api/launcher/status')return json(state);
 if(pathname==='/api/launcher/inspect-model')return json({schema_version:1,status:'ready',can_start:true,model:{},checks:[]});
 if(pathname==='/api/launcher/model-directories')return json({directories:[]});
 if(pathname.startsWith('/api/'))return json({error:'Offline fixture does not start services'},400);
 if(!pathname.startsWith('/monitor/'))return json({error:'No inference service'},503);
 const target=path.resolve(root,'ui',pathname.slice(1));
 if(!target.startsWith(path.join(root,'ui')+path.sep))return json({},404);
 const file=pathname.endsWith('/')?path.join(target,'index.html'):target;
 try{res.writeHead(200,{'Content-Type':{'.html':'text/html','.js':'text/javascript','.css':'text/css','.svg':'image/svg+xml'}[path.extname(file)]||'application/octet-stream'});res.end(fs.readFileSync(file));}catch{json({},404);}
});
let browser;
(async()=>{
 await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
 const base='http://127.0.0.1:'+server.address().port;
 browser=await chromium.launch({headless:true,executablePath:process.env.KVMEM_TEST_BROWSER||process.env.CHROMIUM,args:['--disable-gpu']});
 const errors=[];
 async function open(p){
  parameters={...baseParameters,...p};state={state:'stopped'};
  const context=await browser.newContext({viewport:{width:1440,height:1000}}),page=await context.newPage();
  page.on('pageerror',e=>errors.push(e.message));
  await page.goto(base+'/monitor/?lang=en#config');
  try{await page.locator('[data-param=Backend]').waitFor({timeout:10000});}
  catch(error){fs.writeFileSync(path.join(task,'layerwise-ui-failure.html'),await page.content());console.error('Page errors:',errors);throw error;}
  return {context,page};
 }
 async function set(page,key,value){
  const input=page.locator(`[data-param=${key}]`);
  const panel=await input.evaluate(el=>el.closest('[role=tabpanel]')?.id);
  if(panel)await page.locator(`[aria-controls=${panel}]`).click();
  if(await input.evaluate(el=>el.tagName==='SELECT'))await input.selectOption(value);
  else await input.fill(String(value));
 }
 async function preview(page){
  if(await page.locator('[data-param=RecallStrategy]').inputValue()==='layerwise')
   await page.waitForFunction(()=>document.querySelector('[data-param=RecallStrategy]')?.dataset.capabilityState==='ready');
  let posted;
  await page.route('**/api/launcher/preview',async route=>{
   posted=route.request().postDataJSON();await route.fulfill({json:{argv:['offline-worker'],environment:{},preflight:{port_available:true},effective_settings:{}}});
  });
  await page.locator('[data-action=preview]').click();
  await page.waitForFunction(()=>{
   const text=document.querySelector('[data-check-summary]')?.textContent;
   return text&&text!=='Checking…';
  });
  assert.match(await page.locator('[data-check-summary]').textContent(),/passed/,
    await page.locator('[data-dialog=check], dialog[open]').textContent());
  await page.keyboard.press('Escape');return posted.parameters;
 }
 {
  const {context,page}=await open({});
  assert.equal(await page.locator('[data-param=RecallStrategy]').inputValue(),'legacy');
  await set(page,'RecallStrategy','layerwise');
  assert.equal(await page.locator('[data-param=QueryScorer]').isDisabled(),true);
  for(const key of ['LayerwiseActivationRamMiB','LayerwiseActivationDiskMiB','LayerwiseStagingMiB','LayerwiseMetadataMiB','LayerwiseActivationMetadataMiB'])assert.equal(await page.locator(`[data-param=${key}]`).isDisabled(),false,key);
  await set(page,'SpeculativeMode','mtp');await set(page,'DraftTokens',3);
  const history=page.locator('[data-param=LayerwiseMtpHistory]');
  assert.equal(await history.inputValue(),'');assert.equal(await history.evaluate(el=>el.checkValidity()),false);
  await set(page,'LayerwiseMtpHistory','upstream-global');
  await set(page,'LayerwiseActivationRamMiB',0);await set(page,'LayerwiseActivationDiskMiB',128);
  await set(page,'DiskPath','C:/fixture/activation-cache');
  const p=await preview(page);
  assert.equal(p.DiskPath,'C:/fixture/activation-cache');
  assert.equal(p.DiskMiB,0);
  assert.equal(p.LayerwiseMtpHistory,'upstream-global');assert.equal(p.DraftTokens,3);
  assert.equal(p.LayerwiseActivationRamMiB,0);assert.equal(p.LayerwiseActivationDiskMiB,128);
  assert.equal(p.SinkTokens,0);assert.equal(p.RecentTokens,0);
  for(const key of ['QueryScorer','GraphAlpha','QueryMaxTokens','QueryReplayMaxTokens','Lrfu'])assert.equal(key in p,false,key);
  await page.locator('#launch-tab-5').click();
  await page.screenshot({path:path.join(task,'layerwise-launcher-ui.png'),fullPage:true});
  await set(page,'RecallStrategy','legacy');
  const old=await preview(page);assert.equal(old.QueryScorer,'page-softmax');assert.equal(old.GraphAlpha,.2);
  assert.equal('LayerwiseMtpHistory' in old,false);assert.equal('LayerwiseActivationRamMiB' in old,false);
  await set(page,'RecallStrategy','layerwise');
  assert.equal(await history.inputValue(),'upstream-global');
  await set(page,'SpeculativeMode','dflash2');
  assert.equal(await history.isDisabled(),true);
  assert.equal('LayerwiseMtpHistory' in await preview(page),false);
  await context.close();
 }
 {
  const {context,page}=await open({});
  await page.locator('[data-preset=ninfer-layerwise]').click();
  assert.equal(await page.locator('[data-param=RecallStrategy]').inputValue(),'layerwise');
  const p=await preview(page);
  assert.equal(p.KvType,'bf16');assert.equal(p.SpeculativeMode,'none');
  for(const key of ['SinkTokens','RecentTokens','LayerwiseMtpHistory'])assert.equal(key in p,false,key);
  await context.close();
 }
 {
  const {context,page}=await open({Backend:'polystrata',Model:'C:/fixture/gemma.gguf',DraftModel:'C:/fixture/assistant.gguf',KvType:'f16',RecallStrategy:'layerwise',SpeculativeMode:'guess',DraftTokens:2,DraftPolicy:'adaptive'});
  assert.equal(await page.locator('[data-param=SpeculativeMode]').inputValue(),'guess');
  assert.equal(await page.locator('[data-param=LayerwiseMtpHistory]').isDisabled(),true);
  const p=await preview(page);assert.equal(p.SpeculativeMode,'guess');assert.equal('LayerwiseMtpHistory' in p,false);
  assert.equal(await page.locator('[data-param=DraftModel]').isDisabled(),false);
  await context.close();
 }
 for(const prefill of [undefined,512]){
  const p={Backend:'radiance',Model:'C:/fixture/model.rad',KvType:'bf16'};if(prefill===undefined)p.Prefill=null;else p.Prefill=prefill;
  const {context,page}=await open(p);
  assert.equal(await page.locator('[data-param=Prefill]').inputValue(),String(prefill??256));
  await context.close();
 }
 {
  const {context,page}=await open({RecallStrategy:'layerwise',SpeculativeMode:'mtp',DraftTokens:3,LayerwiseMtpHistory:'first-layer'});
  state={state:'ready',instance:'offline-running',active_config:{schema_version:1,parameters},effective_scoring:{strategy:'layerwise'},scoring_confirmed:false};
  await page.waitForFunction(()=>document.querySelector('[data-launch-root]').hasAttribute('data-running-config'));
  assert.equal(await page.locator('[data-param=RecallStrategy]').isDisabled(),true);
  assert.match(await page.locator('[data-runtime-details]').textContent(),/Awaiting full service report/);
  await context.close();
 }
 parameters={...baseParameters};state={state:'stopped'};
 await require('./test-recall-layout.cjs')({browser,base,info:info(),task});
 assert.deepEqual(errors,[]);
 const result={status:'PASS',scope:'offline UI; no native/model/GPU',browser:browser.version(),checks:['strategy isolation','five finite controls','explicit OWN policy','Legacy retained values','DFlash history separation','Guess SharedTarget','Radiance256 and explicit512','running configuration lock','unconfirmed actual report']};
 fs.writeFileSync(path.join(task,'layerwise-ui-result.json'),JSON.stringify(result,null,2)+'\n');
 console.log(JSON.stringify(result,null,2));
})().catch(error=>{console.error(error);process.exitCode=1;}).finally(async()=>{if(browser)await browser.close();server.close();});
