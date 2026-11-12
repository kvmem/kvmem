/* Static inspection and browser races only; never starts an inference worker. */
const assert=require('node:assert/strict');
const fs=require('node:fs'),os=require('node:os'),path=require('node:path');
const {spawn,spawnSync}=require('node:child_process');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const root=path.resolve(__dirname,'..'),state=fs.mkdtempSync(path.join(os.tmpdir(),'kvmem-health-test-'));
const python=process.env.PYTHON||'python',port=Number(process.env.CONSOLE_TEST_PORT||18437),base=`http://127.0.0.1:${port}`;
const model=path.join(state,'text.ninfer'),broken=path.join(state,'broken.ninfer'),worker=path.join(state,'worker.exe'),rad=path.join(state,'ordinary.rad');
fs.writeFileSync(worker,'not executed');fs.writeFileSync(broken,'broken');
const prepare=spawnSync(python,['-c',"import sys;from pathlib import Path;sys.path.insert(0,'tests');from test_model_health import ninfer;ninfer(Path(sys.argv[1]));sys.path.insert(0,'backends/radiance/tests');from draft_control_test import container;container(Path(sys.argv[2]),['dflash.fc.weight'],[('draft.architectures','DFlash2Model'),('draft.dflash_config.block_size','8'),('mtp_num_hidden_layers',1)])",model,rad],{cwd:root,encoding:'utf8',windowsHide:true});
assert.equal(prepare.status,0,prepare.stderr);
fs.writeFileSync(path.join(state,'worker-paths.json'),JSON.stringify({llamacpp:worker,ninfer:worker,polystrata:worker,radiance:worker}));
const child=spawn(python,['-B',path.join(root,'scripts/local-console.py'),'--port',String(port),'--state-dir',state],{cwd:root,windowsHide:true});
let stderr='',browser;child.stderr.on('data',data=>stderr+=data);
const delay=ms=>new Promise(resolve=>setTimeout(resolve,ms));
async function until(fn){for(let i=0;i<100;i++){try{if(await fn())return;}catch{}await delay(100);}throw Error('Timed out: '+stderr);}
(async()=>{
 let info;await until(async()=>{const r=await fetch(base+'/api/launcher/info');if(r.ok){info=await r.json();return true;}});
 assert.equal(info.model_inspection_version,1);
 const post=(action,parameters,token=info.token)=>fetch(base+'/api/launcher/'+action,{method:'POST',headers:{'Content-Type':'application/json','X-KVMem-Token':token},body:JSON.stringify({schema_version:1,parameters})});
 const params={Backend:'ninfer',Model:model,Worker:worker};
 assert.equal((await post('inspect-model',params,'wrong-token')).status,403);
 const live=await (await post('inspect-model',params)).json();assert.equal(live.status,'ready');
 const rejected=await post('start',{...params,Model:broken});assert.equal(rejected.status,400);assert.match((await rejected.json()).error,/inspection blocked/);
 browser=await chromium.launch({headless:true,args:['--disable-gpu'],...(process.env.CHROMIUM?{executablePath:process.env.CHROMIUM}:{})});
 const page=await browser.newPage({viewport:{width:1440,height:1100}}),errors=[];
 page.on('pageerror',e=>errors.push(e.message));
 await page.goto(base+'/monitor/?lang=en#config');await page.locator('[data-param=Backend]').waitFor();
 await page.locator('[data-param=Backend]').selectOption('ninfer');
 await page.locator('[data-param=SpeculativeMode]').evaluate(el=>{el.value='none';el.dispatchEvent(new Event('change',{bubbles:true}));});
 await page.locator('[data-param=Vision]').evaluate(el=>{el.checked=false;el.dispatchEvent(new Event('change',{bubbles:true}));});
 await page.locator('[data-param=Model]').fill(model);
 const card=page.locator('[data-model-health]'),start=page.locator('[data-action=start]');
 await until(async()=>await card.getAttribute('data-state')==='ready');
 assert.equal(await start.isEnabled(),true);
 await page.getByRole('tab',{name:'Hardware & acceleration'}).click();
 await page.locator('[data-param=SpeculativeMode]').selectOption('mtp');
 await until(async()=>await card.getAttribute('data-state')==='blocked');
 await card.locator('[data-health-fix=disable_speculative]').click();
 await until(async()=>await card.getAttribute('data-state')==='ready');
 assert.equal(await page.locator('[data-param=SpeculativeMode]').inputValue(),'none');
 await page.getByRole('tab',{name:'Templates & vision'}).click();
 await page.locator('[data-param=Vision]').check();
 await until(async()=>await card.getAttribute('data-state')==='blocked');
 assert.equal(await start.isDisabled(),true);
 await card.locator('[data-health-fix=disable_vision]').click();
 await until(async()=>await card.getAttribute('data-state')==='ready');
 assert.equal(await start.isEnabled(),true);
 await page.locator('[data-param=Model]').fill(broken);
 await until(async()=>await card.getAttribute('data-state')==='blocked');
 await page.waitForTimeout(2200); // Status polling must not re-enable a blocked launch.
 assert.equal(await start.isDisabled(),true);
 await page.locator('[data-param=Model]').fill(model);
 await until(async()=>await card.getAttribute('data-state')==='ready');
 await page.locator('[data-language-select]').selectOption('zh-CN');
 await until(async()=>await card.getAttribute('data-state')==='ready');
 assert.match(await card.textContent(),/静态检查通过/);
 await card.locator('summary').click();assert.match(await card.textContent(),/Worker 加载兼容性未验证/);
 if(process.env.HEALTH_SCREENSHOT_DIR){fs.mkdirSync(process.env.HEALTH_SCREENSHOT_DIR,{recursive:true});await page.screenshot({path:path.join(process.env.HEALTH_SCREENSHOT_DIR,'health-desktop.png'),fullPage:true});}
 // Late results for an old model cannot replace the current report.
 let delayed=false;
 await page.route('**/api/launcher/inspect-model',async route=>{
   const data=route.request().postDataJSON();
   if(data.parameters.Model.endsWith('slow.ninfer')){delayed=true;await delay(1400);try{await route.fulfill({json:{...live,status:'blocked',can_start:false}});}catch{}return;}
   await route.continue();
 });
 await page.locator('[data-param=Model]').fill(path.join(state,'slow.ninfer'));
 await until(()=>delayed);
 await page.locator('[data-param=Model]').fill(model);
 await until(async()=>await card.getAttribute('data-state')==='ready');
 await page.waitForTimeout(1600);assert.equal(await card.getAttribute('data-state'),'ready');assert.equal(await start.isEnabled(),true);
 await page.unroute('**/api/launcher/inspect-model');
 await page.route('**/api/launcher/inspect-model',route=>route.fulfill({status:503,json:{error:'unavailable'}}));
 await card.locator('[data-health-retry]').click();
 await until(async()=>await card.getAttribute('data-state')==='unavailable');
 assert.equal(await start.isEnabled(),true);
 await page.unroute('**/api/launcher/inspect-model');
 // The real RAD inspector uses metadata and weights, and changing only depth rechecks it.
 await page.locator('[data-param=Backend]').selectOption('radiance');
 await page.locator('[data-param=Vision]').evaluate(el=>{el.checked=false;el.dispatchEvent(new Event('change',{bubbles:true}));});
 await page.locator('[data-param=Model]').fill(rad);
 await page.getByRole('tab',{name:/硬件与加速|Hardware & acceleration/}).click();
 await page.locator('[data-param=SpeculativeMode]').selectOption('dflash2');
 await until(async()=>await card.getAttribute('data-state')==='ready');
 const draftFeature=card.locator('[data-feature=dflash2]');
 assert.equal(await draftFeature.getAttribute('data-state'),'present');
 assert.match(await draftFeature.textContent(),/已包含/);
 assert.equal(await card.locator('[data-feature=mtp]').getAttribute('data-state'),'absent');
 await page.locator('[data-language-select]').selectOption('en');
 await until(async()=>await card.getAttribute('data-state')==='ready');
 assert.match(await draftFeature.textContent(),/Included/);
 await page.locator('[data-param=DraftTokens]').fill('3');
 await until(async()=>await card.getAttribute('data-state')==='blocked');
 assert.match(await card.textContent(),/requires 7/);
 assert.equal(await draftFeature.getAttribute('data-state'),'present','a depth mismatch does not remove the component');
 assert.equal(await start.isDisabled(),true);
 await page.locator('[data-param=DraftTokens]').fill('7');
 await until(async()=>await card.getAttribute('data-state')==='ready');
 await page.locator('[data-param=SpeculativeMode]').selectOption('mtp');
 await until(async()=>await card.getAttribute('data-state')==='blocked');
 await card.locator('[data-health-fix=disable_speculative]').click();
 await until(async()=>await card.getAttribute('data-state')==='ready');
 assert.equal(await draftFeature.getAttribute('data-state'),'present','turning speculation off does not hide embedded weights');
 if(process.env.HEALTH_SCREENSHOT_DIR)await page.screenshot({path:path.join(process.env.HEALTH_SCREENSHOT_DIR,'health-radiance.png'),fullPage:true});
 // A backend switch clears the previous report and checks the new backend's file format.
 for(const backend of info.backends){
   await page.locator('[data-param=Backend]').selectOption(backend);
   await page.locator('[data-param=Model]').fill(broken);
   await until(async()=>await card.getAttribute('data-state')==='blocked');
 }
 await page.setViewportSize({width:390,height:844});
 assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);
 if(process.env.HEALTH_SCREENSHOT_DIR)await page.screenshot({path:path.join(process.env.HEALTH_SCREENSHOT_DIR,'health-mobile.png'),fullPage:true});
 assert.deepEqual(errors,[]);
 console.log('PASS: authenticated live inspection, fresh launch gate, vision recovery, polling, localization, stale response isolation, unavailable fallback, RAD drafter detection/depth changes, installed backends and mobile layout');
})().catch(error=>{console.error(error);process.exitCode=1;}).finally(async()=>{if(browser)await browser.close();child.kill();});
