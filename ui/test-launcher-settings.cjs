/* Model-free settings checks, run by test-launcher.cjs in an isolated browser context.
 * Only device discovery, picker results and preview responses are fixtures. The real
 * launcher UI must preserve drafts and send the correct active configuration payload.
 * Native flag translation is covered by test-backend-launcher.ps1 and console tests.
 */
const assert = require('node:assert/strict');

module.exports = async function settingsChecks({browser,base,info}) {
 const context=await browser.newContext({viewport:{width:1366,height:768}});
 const page=await context.newPage(),errors=[],mutations=[],previews=[];
 page.on('pageerror',error=>errors.push(error.message));
 page.on('request',request=>{if(request.method()==='POST')mutations.push(new URL(request.url()).pathname);});
 const gpus=[{uuid:'GPU-ui-fixture-first',name:'NVIDIA GeForce Test GPU A'},{uuid:'GPU-ui-fixture-second',name:'NVIDIA GeForce Test GPU B'}];
 const fixture={...info,gpus,status:{state:'stopped'},draft:{schema_version:1,parameters:{
  Backend:'ninfer',Model:'C:/models/ui fixture.ninfer',Gpu:gpus[0].uuid,Context:8192,Budget:2048,Reserve:1024,
  HostMiB:256,Prefill:128,MaxTokens:128,Port:18499,ListenHost:'127.0.0.1',SinkTokens:128,RecentTokens:64,
  RetainedSessions:1,KvType:'int8',DiskEnabled:true,DiskPath:'C:/cache/ui fixture',DiskMiB:512,
  NoLrfu:false,ScoreAggregation:'max',MtpDrafts:4,AdaptiveMtp:true,
  NgramDrafts:8,NgramMinMatch:6,Vision:true,VisionTokens:512,VisionResidency:'cpu',ThinkingMode:'default',ThinkingBudget:512,ReasoningEffort:'medium',PreserveThinking:true,
 }}};
 await page.route('**/api/launcher/info',route=>route.fulfill({json:fixture}));
 await page.route('**/api/launcher/status',route=>route.fulfill({json:{state:'stopped'}}));
 await page.route('**/api/launcher/browse-template',route=>route.fulfill({json:{path:'C:/templates/ui fixture.jinja'}}));
 await page.route('**/api/launcher/preview',async route=>{
  previews.push(route.request().postDataJSON().parameters);
  await route.fulfill({json:{argv:['fixture-worker','--kvmem-gen-reserve','1024'],environment:{},token_settings:{block_tokens:64,
    requested:{Budget:10001,Reserve:1001,Prefill:999,SinkTokens:101,RecentTokens:101},
    effective:{Budget:9984,Reserve:960,Prefill:896,SinkTokens:64,RecentTokens:64}}}});
 });
 const field=key=>page.locator(`[data-param=${key}]`);
 const group=index=>page.locator(`#launch-tab-${index}`).click();
 const disabled=async(keys,value=true)=>{for(const key of keys)assert.equal(await field(key).isDisabled(),value,key+' disabled');};
 const saved=()=>page.waitForFunction(()=>document.querySelector('[data-draft-state]').textContent.includes('已自动保存'));
 const preview=async()=>{
  const before=previews.length;
  await page.locator('[data-action=preview]').click();
  await page.waitForFunction(()=>{const text=document.querySelector('[data-check-summary]')?.textContent;return text&&text!=='正在检查…';});
  assert.ok((await page.locator('[data-check-summary]').textContent()).includes('检查通过'),await page.locator('[data-dialog=check]').textContent());
  assert.ok((await page.locator('[data-dialog=check]').textContent()).includes('10001 → 9984 tokens'),'preview shows requested and effective Budget');
  assert.ok((await page.locator('[data-dialog=check]').textContent()).includes('999 → 896 tokens'),'preview shows capped and aligned Prefill');
  assert.equal(previews.length,before+1,'preview reaches controller boundary');
  const result=previews.at(-1);await page.locator('[data-dialog=check] [data-dialog-close]').click();return result;
 };
 try {
  await page.goto(base+'/monitor/#config');await field('Backend').waitFor();
  await page.locator('[data-language-select]').selectOption('zh-CN');
  // Let the asynchronous inspection card settle before measuring tooltip layout.
  await page.waitForFunction(()=>document.querySelector('[data-model-health]')?.getAttribute('aria-busy')==='false');
  assert.equal(await page.locator('.l-core [data-param=Reserve]').count(),1,'R is a primary resource');
  assert.equal(await page.locator('#launch-panel-0 [data-param=HostMiB]').count(),1,'RAM is in cache settings');
  assert.equal(await page.locator('.l-field-help').count(),0,'help does not occupy form rows');
  const reserveHelp=field('Reserve').locator('..').locator('.l-help');
  const tip=page.locator('#'+await reserveHelp.getAttribute('aria-describedby'));
  const beforeHelp=await field('Reserve').boundingBox();
  await reserveHelp.hover();await tip.waitFor({state:'visible'});
  assert.deepEqual(await field('Reserve').boundingBox(),beforeHelp,'hover help does not move inputs');
  await tip.hover();await page.waitForTimeout(220);
  assert.equal(await tip.isVisible(),true,'pointer can enter the help');
  const bounds=await tip.boundingBox();assert.ok(bounds.x>=0&&bounds.y>=0&&bounds.x+bounds.width<=1366&&bounds.y+bounds.height<=768,'help stays in viewport');
  await page.keyboard.press('Escape');assert.equal(await tip.isVisible(),false);
  await reserveHelp.focus();assert.equal(await tip.isVisible(),true,'keyboard focus opens help');
  await page.keyboard.press('Escape');await reserveHelp.click();
  await page.getByRole('heading',{name:'启动配置',exact:true}).click();assert.equal(await tip.isVisible(),false,'outside click closes pinned help');

  await group(0);await field('DiskEnabled').uncheck();
  await disabled(['DiskPath','DiskMiB']);assert.equal(await field('DiskPath').inputValue(),'C:/cache/ui fixture');
  assert.equal(await field('DiskPath').getAttribute('required'),null,'disabled SSD does not require a path');
  assert.ok((await page.locator('[data-graph-recall]').textContent()).includes('历史检索'));
  assert.equal(await field('NoLrfu').count(),0,'legacy switch is migrated');
  assert.equal(await field('QueryScorer').inputValue(),'meanq-cosine');
  assert.equal(await field('ScoreAggregation').inputValue(),'max');
  await group(1);await field('FastPrefill').check();await disabled(['StageLayers']);
  await page.locator('[data-gpu-choice]').nth(1).check();await disabled(['StageLayers'],false);
  await field('StageLayers').fill('40,24');await page.locator('[data-gpu-choice]').nth(1).uncheck();
  await disabled(['StageLayers']);assert.equal(await field('StageLayers').inputValue(),'','GPU count changes clear stale layer allocation');
  assert.equal(await field('SpeculativeMode').inputValue(),'mtp','legacy positive draft length enables the switch');
  if(process.env.SCREENSHOT_DIR)await page.screenshot({path:process.env.SCREENSHOT_DIR+'/mtp-enabled.png',fullPage:true});
  await field('SpeculativeMode').selectOption('none');await disabled(['DraftTokens','DraftPolicy','NgramDrafts','NgramMinMatch']);
  assert.equal(await field('DraftTokens').inputValue(),'4','turning MTP off retains its length');
  if(process.env.SCREENSHOT_DIR)await page.screenshot({path:process.env.SCREENSHOT_DIR+'/mtp-disabled.png',fullPage:true});
  await field('SpeculativeMode').selectOption('mtp');assert.equal(await field('DraftTokens').inputValue(),'4');await page.locator('.l-native > summary').click();
  await field('NgramDrafts').fill('0');await disabled(['NgramMinMatch']);await disabled(['DraftPolicy','NgramDrafts'],false);
  await field('NgramDrafts').fill('8');await field('DraftPolicy').selectOption('fixed');await disabled(['NgramDrafts','NgramMinMatch'],false);
  await field('DraftPolicy').selectOption('adaptive');await field('SpeculativeMode').selectOption('none');
  await group(0);await field('KvType').selectOption('bf16');await disabled(['FastPrefill']);await disabled(['Prefill'],false);
  await group(2);await field('ThinkingMode').selectOption('off');await disabled(['ThinkingBudget']);
  await group(3);await field('Vision').uncheck();await disabled(['VisionTokens','VisionResidency']);
  await page.locator('.l-template-fields').locator('..').locator('summary').click();
  await page.locator('[data-browse-template]').click();await page.waitForFunction(()=>document.querySelector('[data-param=ChatTemplateFile]').value.endsWith('.jinja'));
  await page.locator('[data-reset-template]').click();assert.equal(await field('ChatTemplateFile').inputValue(),'');
  await page.locator('[data-browse-template]').click();await page.waitForFunction(()=>document.querySelector('[data-param=ChatTemplateFile]').value.endsWith('.jinja'));
  await group(4);await page.locator('[data-generate-key]').click();
  assert.match(await field('ApiKey').inputValue(),/^[a-f0-9]{64}$/);
  const secret='test-only-browser-credential';await field('ApiKey').fill(secret);
  const keyRow=field('ApiKey').locator('..');await keyRow.getByRole('button',{name:'显示',exact:true}).click();assert.equal(await field('ApiKey').getAttribute('type'),'text');
  await keyRow.getByRole('button',{name:'隐藏',exact:true}).click();assert.equal(await field('ApiKey').getAttribute('type'),'password');
  await page.locator('.l-service-fields').locator('..').locator('summary').click();
  await field('ModelAlias').fill('UI friendly model');await field('LogLevel').selectOption('debug');
  await saved();
  const draft=await page.evaluate(()=>JSON.parse(localStorage.getItem('kvmem-launch-draft-v2')));
  assert.equal(Object.hasOwn(draft.parameters,'ApiKey'),false,'secret excluded from browser draft');
  assert.equal(draft.parameters.DiskEnabled,false);assert.equal(draft.parameters.DiskMiB,512);
  assert.equal(Object.hasOwn(draft.parameters),false);assert.equal(draft.parameters.VisionTokens,512);
  assert.equal(draft.parameters.DraftPolicy,'adaptive');assert.equal(draft.parameters.NgramDrafts,8);assert.equal(draft.parameters.FastPrefill,true);
  assert.equal(draft.parameters.SpeculativeMode,'none');assert.equal(draft.parameters.DraftTokens,4);
  const inactive=await preview();
  for(const key of ['DiskEnabled','DiskPath','DiskMiB','VisionTokens','VisionResidency','ThinkingBudget','ReasoningEffort','MtpEnabled','DraftTokens','DraftPolicy','NgramDrafts','NgramMinMatch','FastPrefill','StageLayers'])assert.equal(Object.hasOwn(inactive,key),false,'inactive override omitted: '+key);
  assert.equal(inactive.PreserveThinking,true,'preserving past reasoning is independent of the current thinking switch');
  assert.equal(Object.hasOwn(inactive,'NoLrfu'),false);assert.equal(inactive.ScoreAggregation,'max');assert.equal(inactive.Lrfu,true);assert.equal(inactive.ThinkingMode,'off');
  assert.equal(inactive.ApiKey,secret);assert.equal(inactive.ModelAlias,'UI friendly model');assert.equal(inactive.LogLevel,'debug');
  assert.equal(inactive.ChatTemplateFile,'C:/templates/ui fixture.jinja');assert.equal(inactive.RetainedSessions,1);

  await page.locator('[data-action=template]').click();await page.locator('[data-template-name]').fill('Disabled feature values');
  await page.locator('[data-template-confirm]').click();
  const template=await page.evaluate(()=>JSON.parse(localStorage.getItem('kvmem-launch-templates-v1'))[0]);
  assert.equal(Object.hasOwn(template.draft.parameters,'ApiKey'),false,'secret excluded from personal template');
  assert.equal(template.draft.parameters.DiskEnabled,false);assert.equal(template.draft.parameters.NgramDrafts,8);
  assert.equal(template.draft.parameters.SpeculativeMode,'none');assert.equal(template.draft.parameters.DraftTokens,4);
  await page.reload();await field('Backend').waitFor();
  assert.equal(await field('ApiKey').inputValue(),'','secret does not survive reload');
  await disabled(['DiskPath','DiskMiB','VisionTokens','VisionResidency','ThinkingBudget','DraftTokens','DraftPolicy','NgramDrafts','NgramMinMatch','FastPrefill','StageLayers']);
  assert.equal(await field('SpeculativeMode').inputValue(),'none','off remains off after reload despite a retained positive draft length');
  assert.equal(await field('VisionResidency').inputValue(),'cpu');assert.equal(await field('ThinkingBudget').inputValue(),'512');
  await group(0);await field('DiskEnabled').check();
  await page.locator('[data-template-tab=mine]').click();await page.locator('[data-apply-template]').click();
  assert.equal(await field('DiskEnabled').isChecked(),false);
  await disabled(['DiskPath']);assert.equal(await field('DiskMiB').inputValue(),'512');

  await field('Backend').selectOption('llamacpp');
  await field('Model').fill('C:/models/ui fixture.gguf');await page.locator('[data-gpu-choice]').first().check();
  await disabled(['StageLayers','DraftPolicy']);
  await disabled(['Vision','VisionTokens','VisionResidency','ThinkingBudget','DraftTokens','ReasoningEffort','PreserveThinking'],false);
  await group(0);await field('DiskEnabled').check();await field('DiskPath').fill('C:/cache/llama fixture');await field('DiskMiB').fill('512');
  await group(4);await field('Concurrency').fill('8');await disabled(['DiskEnabled','DiskPath','DiskMiB'],false);
  assert.equal(await field('DiskEnabled').isChecked(),true,'concurrency preserves SSD choice');
  const parallel=await preview();assert.equal(parallel.Concurrency,8);assert.equal(parallel.DiskPath,'C:/cache/llama fixture');assert.equal(parallel.DiskMiB,512);
  assert.equal(parallel.ReasoningEffort,'xhigh');assert.equal(parallel.PreserveThinking,true);assert.equal(parallel.Vision,true);assert.equal(parallel.ThinkingBudget,16384);
  await field('Concurrency').fill('1');await disabled(['DiskEnabled','DiskPath','DiskMiB'],false);
  const serial=await preview();assert.equal(serial.DiskPath,'C:/cache/llama fixture');assert.equal(serial.DiskMiB,512);assert.equal(Object.hasOwn(serial,'DiskEnabled'),false);
  await field('Backend').selectOption('ninfer');await disabled(['DiskPath']);await disabled(['VisionTokens','ThinkingBudget','DraftPolicy']);
  await group(0);await field('DiskEnabled').check();await field('KvType').selectOption('int8');
  await group(1);await field('SpeculativeMode').selectOption('mtp');await field('DraftTokens').fill('4');await disabled(['DraftPolicy','NgramDrafts','NgramMinMatch','FastPrefill'],false);
  assert.equal(await field('DraftPolicy').inputValue(),'adaptive');assert.equal(await field('FastPrefill').isChecked(),true);
  await group(2);await field('ThinkingMode').selectOption('default');
  await group(3);await field('Vision').check();await disabled(['VisionTokens','VisionResidency','ThinkingBudget'],false);
  const active=await preview();
  assert.equal(active.ReasoningEffort,'medium');assert.equal(active.PreserveThinking,true);
  assert.equal(active.DraftTokens,4);assert.equal(active.SpeculativeMode,'mtp');assert.equal(Object.hasOwn(active,'MtpEnabled'),false,'the switch is browser state, not a new backend argument');
  for(const [key,value] of Object.entries({DiskMiB:512,VisionTokens:512,VisionResidency:'cpu',ThinkingBudget:512,DraftPolicy:'adaptive',NgramDrafts:8,NgramMinMatch:6,FastPrefill:true}))assert.equal(active[key],value,'reactivated '+key);
  assert.equal(Object.hasOwn(active,'NoLrfu'),false,'C omits the obsolete ranking switch');
  assert.equal(Object.hasOwn(active,'ApiKey'),false);
  await group(1);await field('SpeculativeMode').selectOption('dflash2');
  assert.equal(await field('DraftTokens').inputValue(),'5');await disabled(['DraftPolicy','DraftModel']);
  const dflash=await preview();assert.equal(dflash.SpeculativeMode,'dflash2');assert.equal(dflash.DraftTokens,5);
  assert.equal(Object.hasOwn(dflash,'DraftPolicy'),false);assert.equal(Object.hasOwn(dflash,'MtpDrafts'),false);
  await group(4);await field('Concurrency').fill('8');
  assert.equal(await field('SpeculativeMode').evaluate(el=>el.validity.valid),true,'NInfer DFlash2 permits eight lanes');
  assert.equal((await preview()).Concurrency,8);
  await field('Concurrency').fill('9');
  assert.equal(await field('SpeculativeMode').evaluate(el=>el.validity.valid),false,'NInfer concurrency above eight is blocked');
  await field('Concurrency').fill('1');await group(1);
  await page.locator('[data-gpu-choice]').nth(1).check();
  assert.equal(await field('SpeculativeMode').evaluate(el=>el.validity.valid),true,'NInfer DFlash2 permits layer-pipeline GPUs');
  const stagedDraft=await preview();
  assert.equal(stagedDraft.SpeculativeMode,'dflash2');
  assert.equal(stagedDraft.Gpu.split(',').length,2);
  await page.locator('[data-gpu-choice]').nth(1).uncheck();
  await field('Backend').selectOption('llamacpp');await group(1);
  await field('SpeculativeMode').selectOption('dflash2');await field('DraftModel').fill('C:/models/draft with spaces.gguf');
  const llamaDraft=await preview();assert.equal(llamaDraft.SpeculativeMode,'dflash2');assert.equal(llamaDraft.DraftTokens,5);
  assert.equal(llamaDraft.DraftModel,'C:/models/draft with spaces.gguf');assert.equal(Object.hasOwn(llamaDraft,'NgramDrafts'),false);
  await page.locator('[data-gpu-choice]').nth(1).check();
  assert.equal(await field('SpeculativeMode').evaluate(el=>el.validity.valid),false,'llama.cpp DFlash2 remains single-GPU');
  await page.locator('[data-gpu-choice]').nth(1).uncheck();
  await group(4);await field('Concurrency').fill('8');
  assert.equal(await field('SpeculativeMode').evaluate(el=>el.validity.valid),true,'llama.cpp DFlash2 permits eight lanes');
  assert.equal((await preview()).Concurrency,8);
  await field('Concurrency').fill('9');
  assert.equal(await field('SpeculativeMode').evaluate(el=>el.validity.valid),false,'llama.cpp concurrency above eight is blocked');
  await field('Backend').selectOption('polystrata');await group(0);
  assert.equal(await field('BlockTokens').inputValue(),'32');
  await field('BlockTokens').selectOption('32');await saved();
  const blockDraft=await page.evaluate(()=>JSON.parse(localStorage.getItem('kvmem-launch-draft-v2')));
  assert.equal(blockDraft.parameters.BlockTokens,32,'numeric select saves a number');
  await page.reload();await field('BlockTokens').waitFor();
  assert.equal(await field('BlockTokens').inputValue(),'32','block size survives reload');
  await field('Model').fill('C:/models/ui fixture.gguf');
  await page.locator('[data-gpu-choice]').first().check();
  await group(4);await field('Concurrency').fill('1');
  assert.equal((await preview()).BlockTokens,32,'block size reaches active launch payload');
  await group(3);await page.locator('[data-vision-assets] summary').click();
  const projector="C:/models/视觉 & ' $custom.GGUF",pickerRequests=[];
  let selection=projector;
  await page.route('**/api/launcher/browse-mmproj',async route=>{pickerRequests.push(route.request().postDataJSON());await route.fulfill({json:{path:selection}});});
  await field('Mmproj').fill('');
  await page.locator('[data-browse-mmproj]').click();
  await page.waitForFunction(value=>document.querySelector('[data-param=Mmproj]').value===value,projector);
  assert.deepEqual(pickerRequests.at(-1),{backend:'polystrata',current_path:'C:/models/ui fixture.gguf'});
  selection=null;await page.locator('[data-browse-mmproj]').click();
  await page.waitForFunction(()=>!document.querySelector('[data-browse-mmproj]').disabled);
  assert.equal(await field('Mmproj').inputValue(),projector,'cancelling preserves the chosen projector');
  await saved();await page.reload();await field('Mmproj').waitFor({state:'attached'});
  assert.equal(await field('Mmproj').inputValue(),projector,'projector survives reload');
  assert.equal((await preview()).Mmproj,projector,'chosen projector reaches active launch payload');
  await group(3);await page.locator('[data-vision-assets] summary').click();
  await page.unroute('**/api/launcher/browse-mmproj');
  // Hold the dialog response while the user makes a newer edit on the page.
  async function delayedProjector(edit){
    let release,received;
    const arrived=new Promise(resolve=>received=resolve),pending=new Promise(resolve=>release=resolve);
    await page.route('**/api/launcher/browse-mmproj',async route=>{
      received();await pending;await route.fulfill({json:{path:'C:/models/stale-dialog-choice.gguf'}});
    });
    await page.locator('[data-browse-mmproj]').click();await arrived;
    try{await edit();}finally{release();}
    await page.waitForFunction(()=>!document.querySelector('[data-browse-mmproj]').disabled);
    await page.unroute('**/api/launcher/browse-mmproj');
  }
  const manual='C:/models/new-manual-choice.gguf';
  await delayedProjector(()=>field('Mmproj').fill(manual));
  assert.equal(await field('Mmproj').inputValue(),manual,'late picker cannot overwrite a newer manual path');
  assert.equal((await preview()).Mmproj,manual,'newer path reaches launch preview');
  await delayedProjector(()=>field('Model').fill('C:/models/new-model.gguf'));
  assert.equal(await field('Mmproj').inputValue(),manual,'a projector chosen for an old model cannot replace the current path');
  await saved();await page.reload();await field('Mmproj').waitFor({state:'attached'});
  assert.equal(await field('Mmproj').inputValue(),manual,'newer manual path survives reload after the late response');
  await field('Backend').selectOption('ninfer');
  assert.equal(await field('BlockTokens').count(),0,'block switch is PolyStrata only');
  await field('Model').fill('C:/models/ui fixture.ninfer');
  assert.equal(Object.hasOwn(await preview(),'BlockTokens'),false);
  assert.ok(mutations.every(url=>['/api/launcher/preview','/api/launcher/browse-template','/api/launcher/browse-mmproj','/api/launcher/inspect-model'].includes(url)),'settings checks never start/save/stop inference');
  assert.deepEqual(errors,[]);
  console.log('PASS: help, resource placement, feature dependencies, draft/template retention, credentials, native settings and active preview payloads (no inference)');
 } finally { await context.close(); }
};
