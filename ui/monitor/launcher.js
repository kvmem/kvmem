/* Configuration stays separate from the running service. No shell text is accepted. */
"use strict";
(() => {
  const root = document.querySelector('[data-launch-root]');
  const tr = (zh,en) => window.monitorI18n.language === 'en' ? en : zh;
  let info, busy=false, activeTab=0, runtime=null, startPending=false, launchFailure="";
  let budgetPending=false, budgetAdjustment=null;
  let health=null, healthKey='', healthTimer=null, healthAbort=null, healthSequence=0, healthState='idle';
  // A running process owns the displayed configuration; browser drafts cannot replace it.
  let configLocked=false, configUncertain=false, runningParameters=null, pendingParameters=null, runningView='';
  const lockedControls=new WeakMap();
  const configurationLocked=()=>configLocked||startPending||configUncertain;
  function applyConfigurationLock(){
    const locked=configurationLocked();root.toggleAttribute('data-running-config',locked);
    root.querySelectorAll('form input,form select,form textarea,form button:not(.l-help):not([role=tab]),.l-presets button:not(.l-help),.l-draft-actions button').forEach(el=>{
      if(locked){if(!lockedControls.has(el))lockedControls.set(el,el.disabled);el.dataset.launchLocked=String(!lockedControls.get(el));if(!el.disabled)el.disabled=true;}
      else if(lockedControls.has(el)){el.disabled=lockedControls.get(el);lockedControls.delete(el);delete el.dataset.launchLocked;}
    });
    root.querySelectorAll('.l-draft-actions button').forEach(el=>el.disabled=locked||busy);
    const start=root.querySelector('[data-action=start]'),stop=root.querySelector('[data-action=stop]'); if(start)start.disabled=locked||busy||runtime?.budget_planning||healthBlocks(); if(stop)stop.disabled=busy||!['starting','ready'].includes(runtime?.state);
    const title=root.querySelector('.l-intro h1');if(title)title.textContent=locked?tr('当前启动参数','Current launch parameters'):tr('启动配置','Launch configuration');
    const healthBox=root.querySelector('[data-model-health]');if(healthBox)healthBox.hidden=locked;
    const key=root.querySelector('[data-param=ApiKey]');
    if(key){
      if(locked){
        key.value='';key.dataset.launchKeyHidden='true';
        const configured=startPending?!!pendingParameters?.ApiKey:runtime?.api_key_configured;
        key.placeholder=configured?tr('已设置 · 密钥不回显','Configured · Key not displayed'):tr('未设置','Not configured');
      }else if(key.dataset.launchKeyHidden==='true'){
        key.value=launchKeys[root.querySelector('[data-param=Backend]')?.value]||'';
        key.placeholder='';delete key.dataset.launchKeyHidden;
      }
    }
    draftStatus();renderBudgetAdjustment();
  }
  function syncRunningConfiguration(status){
    const running=['starting','ready'].includes(status.state);
    if(running){
      const parameters=status.active_config?.parameters||runningParameters||pendingParameters||captureDraft().parameters;
      const key=(status.instance||status.pid||'running')+':'+JSON.stringify(parameters);
      if(!configLocked){saveEditing();clearTimeout(healthTimer);healthAbort?.abort();healthSequence++;}
      configLocked=true;runningParameters=structuredClone(parameters);delete runningParameters.ApiKey;
      if(runningView!==key){runningView=key;selectedPreset='';presetUndo=null;render(runningParameters);}
    }else if(configLocked&&!startPending&&!busy&&!configUncertain){
      const parameters=runningParameters;configLocked=false;runningView='';runningParameters=null;
      render(parameters);scheduleSave();
    }
    applyConfigurationLock();
  }
  const perBackend = {};
  // Credentials are kept in memory only, separate from drafts and templates.
  const launchKeys = {};
  let pickingModel = false;
  function recentModels(backend) {
    try {
      const items = JSON.parse(localStorage.getItem('kvmem-recent-models-'+backend) || '[]');
      return Array.isArray(items) ? items.filter(x => typeof x === 'string' && x.length < 8192).slice(0,8) : [];
    } catch { return []; }
  }
  function rememberModel(backend, path) {
    try { localStorage.setItem('kvmem-recent-models-'+backend, JSON.stringify([path,...recentModels(backend).filter(x=>x!==path)].slice(0,8))); } catch {}
  }
  function recentModelLabel(path, items) {
    const parts = path.split(/[\\/]/).filter(Boolean), name = parts.at(-1) || path;
    const peers = items.filter(other => other !== path && other.split(/[\\/]/).at(-1)?.toLowerCase() === name.toLowerCase());
    if (!peers.length) return name;
    const parents = parts.slice(0,-1);
    for (let depth=1;depth<=parents.length;depth++) {
      const suffix = parents.slice(-depth).join('/');
      if (peers.every(other => other.split(/[\\/]/).filter(Boolean).slice(0,-1).slice(-depth).join('/').toLowerCase() !== suffix.toLowerCase())) {
        return name+' · '+suffix;
      }
    }
    return name+' · '+parents.join('/');
  }
  let selectedPreset = '', presetUndo = null;
  const DRAFT_KEY='kvmem-launch-draft-v2', TEMPLATE_KEY='kvmem-launch-templates-v1';
  let saveTimer=null, dirty=false, saveError=false, templateTab='builtin';
  // Paths and physical GPU placement are local choices, never baked into recipes.
  const commonPreset={Context:262144,Budget:47104,Reserve:2048,HostMiB:12288,MaxTokens:0,
    SinkTokens:12288,RecentTokens:1024,Concurrency:1,RetainedSessions:1,ThinkingBudget:16384,
    ReasoningEffort:'xhigh',PreserveThinking:true,SpeculativeMode:'mtp',VisionResidency:'cpu',VisionTokens:1024,
    QueryScorer:'page-softmax',GraphAlpha:0.2,Lrfu:false,ScoreAggregation:'mean',QueryMaxTokens:512,QueryReplayMaxTokens:2048};
  const ninferDefaultPreset={...commonPreset,Backend:'ninfer',ThinkingMode:'default',ThinkingBudget:0,KvType:'int8',Prefill:256,DraftTokens:4,DraftPolicy:'adaptive',NgramDrafts:31,NgramMinMatch:12,FastPrefill:true,DeviceProfile:'auto',Vision:true};
  const presets = [
    {id:'radiance',title:['Radiance AMD','Radiance AMD'],note:['单卡或双卡张量并行、图像、跨轮历史和并发解码。','One or two GPUs with tensor parallelism, images, conversation history and batched decode.'],parameters:{"Backend":"radiance","Context":65536,"Budget":4096,"Reserve":1024,"HostMiB":8192,"SinkTokens":128,"RecentTokens":512,"Concurrency":1,"RetainedSessions":8,"QueryScorer":"page-softmax","ScoreAggregation":"mean","GraphAlpha":0.2,"Lrfu":false,"QueryMaxTokens":512,"QueryReplayMaxTokens":2048,"KvType":"fp8","Prefill":512,"SpeculativeMode":"none","DraftTokens":0,"DraftPolicy":"fixed","Vision":false,"MaxTokens":0,"ThinkingMode":"default","ThinkingBudget":0,"VramReserveMiB":1024}},
    {id:'ninfer-speed',title:['NInfer · 极速档','NInfer · Speed'],
      parameters:{...ninferDefaultPreset,SpeculativeMode:'dflash2',DraftTokens:5,DraftPolicy:'fixed',Budget:36*1024,Reserve:4*1024,NgramDrafts:0},
      note:['启用 DFlash2，历史 KV 预算较少，侧重响应速度。','Uses DFlash2 with a smaller history KV budget, prioritizing response speed.']},
    {id:'ninfer-quality',title:['NInfer · 质量档','NInfer · Quality'],
      parameters:{...ninferDefaultPreset,SpeculativeMode:'mtp',DraftTokens:3,Budget:48*1024,Reserve:4*1024,NgramDrafts:0},
      note:['启用 MTP，历史 KV 预算较大，侧重保留更多历史信息。','Uses MTP with a larger history KV budget, retaining more historical context.']},
    {id:'llamacpp-default',title:['llama.cpp · 默认模板','llama.cpp · Default template'],parameters:{...commonPreset,Backend:'llamacpp',ThinkingMode:'on',KvTypeK:'q8_0',KvTypeV:'q8_0',Prefill:256,DraftTokens:3,DraftPolicy:'fixed',Vision:true}},
    {id:'polystrata-default',title:['PolyStrata · 默认模板','PolyStrata · Default template'],parameters:{...commonPreset,Backend:'polystrata',Budget:36*1024,Reserve:4*1024,ThinkingBudget:0,ThinkingMode:'on',KvType:'q8_0',Prefill:512,DraftTokens:2,DraftPolicy:'adaptive',Vision:false,RetainedSessions:2,VramReserveMiB:256,WorkerArgs:['--elastic-experts-headroom','--elastic-headroom-mib','128','--elastic-physical-reserve-mib','512']},
      note:['256K 上下文 · 36K 驻留 · 4K 生成预留 · 思考不限额','256K context · 36K resident · 4K reserve · no fixed thinking cap']},
  ].map(p=>({...p,note:p.note||['256K 上下文 · 46K 驻留 · 2K 生成预留','256K context · 46K resident · 2K reserve']}));
  const enginePreset=backend=>structuredClone(presets.find(p=>p.parameters.Backend===backend).parameters);
  const BACKENDS_KEY='kvmem-launch-backends-v1';
  const defaults={Backend:'ninfer',Model:'',Worker:'',Gpu:'',Context:262144,Budget:32768,HostMiB:16384,Reserve:1024,Prefill:128,MaxTokens:1024,Port:18200,ListenHost:'127.0.0.1',SinkTokens:12288,RecentTokens:1024};
  // Resource and feature edits belong to each engine; placement and listener stay explicit.
  const sharedSettings=['Gpu','Port','ListenHost'];
  const names={llamacpp:'llama.cpp · Dense',ninfer:'NInfer · Dense',polystrata:'PolyStrata · MoE',radiance:'Radiance · AMD'};
  const modelFormat=b=>info?.backend_capabilities?.[b]?.model_format||(b==='radiance'?'.rad':b==='ninfer'?'.ninfer':'.gguf');
  const groups=[['缓存与历史','Cache & history'],['硬件与加速','Hardware & acceleration'],['思考设置','Thinking settings'],['模板与图像','Templates & vision'],['服务与并发','Service & concurrency'],['召回策略','Recall policy']];
  const make=(tag,attrs={},text)=>{const el=document.createElement(tag);Object.entries(attrs).forEach(([k,v])=>el.setAttribute(k,v));if(text!==undefined)el.textContent=text;return el;};
  function healthParameters(){
    const parameters={};
    for(const key of ['Backend','Model','Worker','Vision','Mmproj','VisionWorker','Tokenizer','ChatTemplateFile','SpeculativeMode','MtpDrafts','DraftModel']){
      const el=root.querySelector(`[data-param=${key}]`);
      if(!el||el.disabled)continue;
      if(el.type==='checkbox')parameters[key]=el.checked;
      else if(el.value!=='')parameters[key]=el.dataset.booleanSelect==='true'?el.value==='true':(el.type==='number'||el.dataset.valueType==='number')?Number(el.value):el.value;
    }
    return parameters;
  }
  function healthBlocks(){return health?.can_start===false;}
  function scheduleHealth(force=false){
    if(!info||configurationLocked())return;
    const parameters=healthParameters(),key=JSON.stringify(parameters);
    if(!force&&key===healthKey)return;
    healthKey=key;health=null;clearTimeout(healthTimer);healthAbort?.abort();
    const sequence=++healthSequence;
    healthState=parameters.Model?.trim()?'checking':'idle';renderHealth();updateStatus(runtime);
    if(healthState==='idle')return;
    if(info.model_inspection_version!==1){healthState='unavailable';renderHealth();return;}
    healthTimer=setTimeout(async()=>{
      const controller=new AbortController();healthAbort=controller;
      const timeout=setTimeout(()=>controller.abort(),12000);
      try{
        const response=await fetch('/api/launcher/inspect-model',{method:'POST',headers:{'Content-Type':'application/json','X-KVMem-Token':info.token},body:JSON.stringify({schema_version:1,parameters}),signal:controller.signal});
        const result=await response.json();
        if(!response.ok||result.schema_version!==1||typeof result.can_start!=='boolean')throw Error('Inspection unavailable');
        if(sequence!==healthSequence)return;
        health=result;healthState=result.status;
      }catch(error){if(sequence!==healthSequence)return;healthState='unavailable';}
      finally{clearTimeout(timeout);if(sequence===healthSequence){renderHealth();updateStatus(runtime);}}
    },force?0:400);
  }
  function renderHealth(){
    const box=root.querySelector('[data-model-health]');if(!box)return;
    const expanded=box.querySelector('details')?.open||false;
    box.replaceChildren();box.dataset.state=healthState;
    box.setAttribute('aria-busy',String(healthState==='checking'));
    const titles={idle:['选择模型后自动体检','Select a model for automatic inspection'],checking:['正在检查模型文件…','Inspecting model files…'],ready:['静态检查通过','Static checks passed'],warning:['检查完成，有待确认项','Checked with items to review'],blocked:['当前配置无法启动','Current configuration cannot start'],unavailable:['体检未完成，可尝试启动','Inspection incomplete; launch may be attempted']};
    const head=make('div',{class:'l-health-head'}),title=make('strong',{'data-health-summary':'',role:'status','aria-live':'polite'},tr(...(titles[healthState]||titles.unavailable)));
    head.append(title);
    if(healthState!=='idle'){
      const retry=make('button',{type:'button',class:'s-button','data-health-retry':''},tr('重新检查','Recheck'));
      retry.disabled=healthState==='checking';retry.onclick=()=>scheduleHealth(true);head.append(retry);
    }
    box.append(head);
    if(healthState==='idle')return;
    const model=health?.model||{},facts=[model.format,model.architecture,model.files?tr(`${model.files} 个文件`,`${model.files} file(s)`):'',model.bytes?`${(model.bytes/1024**3).toFixed(2)} GiB`:''].filter(Boolean);
    if(facts.length)box.append(make('p',{class:'l-health-meta'},facts.join(' · ')));
    box.append(make('p',{class:'l-health-note'},tr('仅检查文件与资源；worker 加载、显存用量与推理结果尚未验证。','Files and resources only; worker loading, VRAM use and inference remain unverified.')));
    const features=make('div',{class:'l-health-features'}),states={present:['已包含','Included'],absent:['未包含','Absent'],unknown:['未确认','Unverified'],external:['需外部文件','External file']};
    for(const [key,zh,en] of [['text','文本','Text'],['vision','图像','Vision'],['mtp','MTP','MTP'],['dflash2','DFlash2','DFlash2'],['tokenizer','分词器','Tokenizer'],['template','聊天模板','Chat template']]){
      const state=health?.features?.[key];if(!state)continue;
      features.append(make('span',{'data-feature':key,'data-state':state},tr(zh,en)+' · '+tr(...(states[state]||states.unknown))));
    }
    if(features.children.length)box.append(features);
    const checks=health?.checks||[],issues=checks.filter(c=>c.status==='error'||c.status==='warning');
    for(const item of issues.slice(0,4)){
      const line=make('div',{class:'l-health-issue','data-severity':item.status});
      line.append(make('strong',{},tr(item.zh,item.en)),make('span',{},item.detail||''));box.append(line);
    }
    const fixes=new Set(checks.filter(c=>c.status==='error'||c.status==='warning').map(c=>c.fix));
    for(const fix of ['disable_vision','disable_speculative']){
      if(!fixes.has(fix))continue;
      const action=make('button',{type:'button',class:'s-button','data-health-fix':fix},fix==='disable_vision'?tr('关闭图像，重新检查','Disable vision and recheck'):tr('关闭推测解码，重新检查','Disable speculation and recheck'));
      action.onclick=()=>{const key=fix==='disable_vision'?'Vision':'SpeculativeMode',el=root.querySelector(`[data-param=${key}]`);if(!el)return;if(el.type==='checkbox')el.checked=false;else el.value='none';el.dispatchEvent(new Event('change',{bubbles:true}));};box.append(action);
    }
    if(checks.length){
      const details=make('details',{'data-health-details':''});details.open=expanded;details.append(make('summary',{},tr('查看检查详情','View check details')));
      const labels={file:['文件结构','File structure'],config:['所选配置','Selected configuration'],worker:['引擎兼容性','Worker compatibility']};
      for(const layer of ['file','config','worker']){
        const items=checks.filter(c=>c.layer===layer);if(!items.length)continue;
        details.append(make('h4',{},tr(...labels[layer])));
        for(const item of items){const line=make('p',{'data-severity':item.status});line.append(make('strong',{},({pass:'✓ ',error:'✕ ',warning:'! ',unknown:'? '}[item.status]||'')+tr(item.zh,item.en)));if(item.detail)line.append(make('small',{},item.detail));details.append(line);}
      }
      const copy=make('button',{type:'button',class:'s-button','data-health-copy':''},tr('复制检查报告','Copy inspection report'));
      copy.onclick=async()=>{try{await navigator.clipboard.writeText(JSON.stringify(health,null,2));copy.textContent=tr('已复制','Copied');}catch{copy.textContent=tr('复制失败，请重试','Copy failed; retry');}};details.append(copy);box.append(details);
    }
  }
  let openHelp=null,helpOpenTimer=null,helpCloseTimer=null,helpId=0;
  function closeHelp(){
    clearTimeout(helpOpenTimer);clearTimeout(helpCloseTimer);
    if(!openHelp)return;
    const current=openHelp;openHelp=null;current.pinned=false;
    current.button.setAttribute('aria-expanded','false');
    if(current.tip.hidePopover&&current.tip.matches(':popover-open'))current.tip.hidePopover();
    current.tip.hidden=true;
  }
  function showHelp(item){
    clearTimeout(helpOpenTimer);clearTimeout(helpCloseTimer);
    if(!item.button.isConnected)return;
    item.reason.textContent=item.input?.dataset.disabledReason||'';item.reason.hidden=!item.reason.textContent;
    if(openHelp!==item){closeHelp();openHelp=item;item.tip.hidden=false;if(item.tip.showPopover)item.tip.showPopover();}
    item.button.setAttribute('aria-expanded','true');
    const anchor=item.button.getBoundingClientRect(),box=item.tip.getBoundingClientRect(),edge=12;
    item.tip.style.left=Math.max(edge,Math.min(anchor.left,innerWidth-box.width-edge))+'px';
    const below=anchor.bottom+8,top=below+box.height<=innerHeight-edge?below:anchor.top-box.height-8;
    item.tip.style.top=Math.max(edge,Math.min(top,innerHeight-box.height-edge))+'px';
  }
  function help(parent,title,text,input=null){
    const id='launch-tooltip-'+(++helpId),button=make('button',{type:'button',class:'l-help','aria-label':tr('关于：','About: ')+title,'aria-describedby':id,'aria-expanded':'false'},'i');
    const tip=make('div',{id,class:'l-help-popover',role:'tooltip',popover:'manual',hidden:''}),reason=make('p',{class:'l-help-reason',hidden:''});tip.append(make('strong',{},title),make('p',{},text),reason);root.append(tip);parent.append(button);
    if(input)input.setAttribute('aria-describedby',id);
    const item={button,tip,input,reason,pinned:false,hover:false,focused:false};
    const leave=()=>{clearTimeout(helpOpenTimer);clearTimeout(helpCloseTimer);helpCloseTimer=setTimeout(()=>{if(openHelp===item&&!item.hover&&!item.focused&&!item.pinned)closeHelp();},140);};
    button.addEventListener('pointerenter',()=>{item.hover=true;clearTimeout(helpCloseTimer);helpOpenTimer=setTimeout(()=>showHelp(item),180);});
    button.addEventListener('pointerleave',()=>{item.hover=false;leave();});
    tip.addEventListener('pointerenter',()=>{item.hover=true;clearTimeout(helpCloseTimer);});
    tip.addEventListener('pointerleave',()=>{item.hover=false;leave();});
    button.addEventListener('focus',()=>{item.focused=true;showHelp(item);});
    button.addEventListener('blur',()=>{item.focused=false;leave();});
    button.addEventListener('click',event=>{event.preventDefault();event.stopPropagation();if(openHelp===item&&item.pinned)closeHelp();else{showHelp(item);item.pinned=true;}});
    return button;
  }
  document.addEventListener('pointerdown',event=>{if(openHelp&&!openHelp.button.contains(event.target)&&!openHelp.tip.contains(event.target))closeHelp();},true);
  document.addEventListener('keydown',event=>{if(event.key==='Escape'&&openHelp){closeHelp();event.preventDefault();event.stopPropagation();}},true);
  window.addEventListener('scroll',event=>{if(openHelp&&event.target!==openHelp.tip)closeHelp();},true);
  window.addEventListener('resize',closeHelp);
  document.addEventListener('change',event=>{if(event.target.matches('[data-density-select]'))closeHelp();});
  function captureDraft(){
    const parameters={};
    root.querySelectorAll('[data-param]').forEach(el=>{
      if((el.disabled&&el.dataset.keepDraft!=='true')||el.dataset.secret==='true')return;
      const value=el.dataset.booleanSelect==='true'?el.value==='true':el.type==='checkbox'?(el.dataset.inverted==='true'?!el.checked:el.checked):el.multiple?Array.from(el.selectedOptions).map(x=>x.value).join(','):(el.type==='number'||el.dataset.valueType==='number')&&el.value!==''?Number(el.value):el.value;
      parameters[el.dataset.param]=el.dataset.idleSessions==='true'&&value!==''?Number(value)+1:value;
    });
    const raw={StageLayers:String(parameters.StageLayers||''),TensorSplit:String(parameters.TensorSplit||''),WorkerArgs:root.querySelector('[data-native]')?.value||'[]'};
    parameters.StageLayers=raw.StageLayers.trim()?raw.StageLayers.split(',').map(x=>Number(x.trim())).filter(Number.isFinite):[];
    parameters.TensorSplit=raw.TensorSplit.trim()?raw.TensorSplit.split(',').map(x=>Number(x.trim())).filter(Number.isFinite):[];
    try{parameters.WorkerArgs=JSON.parse(raw.WorkerArgs);}catch{parameters.WorkerArgs=[];}
    return {version:2,parameters,raw};
  }
  function usableDraft(value){return value&&value.version===2&&value.parameters&&typeof value.parameters==='object'&&info.backends.includes(value.parameters.Backend)&&Array.isArray(value.parameters.StageLayers||[])&&Array.isArray(value.parameters.TensorSplit||[])&&(!value.raw||typeof value.raw==='object');}
  function preferBundledWorker(draft){
    const worker=info?.prefer_bundled_worker&&info.workers?.[draft.parameters.Backend];
    if(!worker?.available||draft.parameters.Worker===worker.path)return draft;
    const migrated=structuredClone(draft);migrated.parameters.Worker=worker.path;return migrated;
  }
  function draftStatus(){
    const el=root.querySelector('[data-draft-state]');if(!el)return;
    if(configurationLocked()){el.textContent=configUncertain?tr('正在确认服务状态，暂不可修改参数','Checking service state; settings remain locked'):tr('本次启动参数 · 停止服务后可修改','Parameters for this run · Stop the service to edit');el.classList.remove('l-error');el.disabled=true;return;}
    el.textContent=saveError?tr('保存失败，点击重试','Save failed. Click to retry'):dirty?tr('正在保存编辑…','Saving edits…'):tr('编辑已自动保存 · 下次启动生效','Edits saved automatically · Apply on next launch');
    el.classList.toggle('l-error',saveError);el.disabled=!saveError;
  }
  function saveEditing(){
    clearTimeout(saveTimer);if(configurationLocked()||(!dirty&&!saveError))return;
    try{const draft=captureDraft();perBackend[draft.parameters.Backend]=draft;localStorage.setItem(DRAFT_KEY,JSON.stringify(draft));localStorage.setItem(BACKENDS_KEY,JSON.stringify(perBackend));dirty=false;saveError=false;}
    catch{saveError=true;}
    draftStatus();
  }
  function scheduleSave(){if(configurationLocked())return;dirty=true;saveError=false;draftStatus();clearTimeout(saveTimer);saveTimer=setTimeout(saveEditing,400);}
  function restoreDraft(draft){
    draft=preferBundledWorker(draft);
    if(configurationLocked()){render(runningParameters||pendingParameters||draft.parameters);return;}
    render(draft.parameters);
    if(draft.raw){const weights=root.querySelector('[data-param=TensorSplit]');if(weights&&typeof draft.raw.TensorSplit==='string'){weights.value=draft.raw.TensorSplit;weights.dispatchEvent(new Event('input'));}const layers=root.querySelector('[data-param=StageLayers]');if(layers&&typeof draft.raw.StageLayers==='string'){layers.value=draft.raw.StageLayers;layers.dispatchEvent(new Event('input'));}const native=root.querySelector('[data-native]');if(native&&typeof draft.raw.WorkerArgs==='string'){native.value=draft.raw.WorkerArgs;native.dispatchEvent(new Event('input'));}}
    draftStatus();
  }
  root.addEventListener('input',event=>{if(!configurationLocked()&&event.target.matches('[data-param], [data-native]')){if(!budgetPending)budgetAdjustment=null;syncDependencies();scheduleSave();scheduleHealth();renderBudgetAdjustment();}});
  root.addEventListener('change',event=>{if(!configurationLocked()&&event.target.matches('[data-param], [data-native]')){if(!budgetPending)budgetAdjustment=null;syncDependencies();scheduleSave();scheduleHealth();renderBudgetAdjustment();}});
  window.addEventListener('pagehide',saveEditing);
  function readTemplates(){
    const values=JSON.parse(localStorage.getItem(TEMPLATE_KEY)||'[]');
    if(!Array.isArray(values))throw Error(tr('个人模板数据无法读取','Cannot read personal templates'));
    return values.filter(x=>x&&typeof x.id==='string'&&typeof x.name==='string'&&usableDraft(x.draft)).slice(0,100).map(x=>{delete x.draft.parameters.ApiKey;return x;});
  }
  function writeTemplates(values){if(values.length>100)throw Error(tr('最多保存 100 个模板','Up to 100 templates can be saved'));localStorage.setItem(TEMPLATE_KEY,JSON.stringify(values));}
  function dialog(title,kind){
    closeHelp();
    const el=make('dialog',{class:'l-dialog','data-dialog':kind,'aria-label':title});
    const head=make('div',{class:'l-heading'}),close=make('button',{type:'button',class:'s-button','data-dialog-close':''},tr('关闭','Close'));
    head.append(make('h2',{class:'s-title'},title),close);el.append(head);root.append(el);
    close.onclick=()=>el.close();el.addEventListener('close',()=>el.remove());el.showModal();return el;
  }
  function showError(error){const el=dialog(tr('操作未完成','Action could not complete'),'error');el.append(make('p',{class:'l-error',role:'alert'},error.message||String(error)));}
  function refreshTemplates(){const current=root.querySelector('.l-presets');if(current){const side=current.parentElement;current.querySelectorAll('.l-help').forEach(button=>{if(openHelp?.button===button)closeHelp();document.getElementById(button.getAttribute('aria-describedby'))?.remove();});current.remove();
    presetPanel(side);}}
  function templateEditor(mode='create',item=null){
    const snapshot=captureDraft(),title=mode==='delete'?tr('删除模板','Delete template'):mode==='rename'?tr('重命名模板','Rename template'):mode==='update'?tr('更新模板','Update template'):tr('另存为模板','Save as template');
    const el=dialog(title,'template'),form=make('form'),error=make('p',{class:'l-error',role:'alert'});
    const label=make('label',{class:'l-field'},tr('模板名称','Template name')),name=make('input',{required:'',maxlength:'80','data-template-name':''});
    name.value=item?.name||'';label.append(name);
    const descLabel=make('label',{class:'l-field'},tr('说明（可选）','Description (optional)')),description=make('textarea',{maxlength:'240',rows:'2','data-template-description':''});description.value=item?.description||'';descLabel.append(description);
    if(mode==='delete')form.append(make('p',{},tr(`删除“${item.name}”？已填入编辑区的配置不受影响。`,`Delete “${item.name}”? The current draft will remain unchanged.`)));
    else form.append(label,descLabel,make('p',{class:'l-muted'},mode==='rename'?tr('仅修改名称和说明。','Only the name and description change.'):tr('保存当前模型路径、显卡顺序和参数；不保存 API 密钥。模板保存在当前浏览器，保存不会启动服务。','Save the current model path, GPU order and settings, excluding the API key. Templates stay in this browser; saving does not start a service.')));
    const submit=make('button',{type:'submit',class:'s-button l-primary','data-template-confirm':''},mode==='delete'?tr('删除模板','Delete template'):mode==='update'?tr('确认更新','Update template'):tr('保存模板','Save template'));
    form.append(error,submit);el.append(form);if(mode!=='delete')name.focus();
    form.onsubmit=event=>{event.preventDefault();try{
      const list=readTemplates(),index=item?list.findIndex(x=>x.id===item.id):-1;
      if(item&&index<0)throw Error(tr('模板已不存在，请重新打开列表','Template no longer exists. Reopen the list.'));
      const trimmed=name.value.trim();if(mode!=='delete'&&!trimmed)throw Error(tr('请填写模板名称','Enter a template name'));
      if(mode!=='delete'&&list.some(x=>x.id!==item?.id&&x.name.toLocaleLowerCase()===trimmed.toLocaleLowerCase()))throw Error(tr('已有同名模板，请使用其他名称','A template with this name already exists'));
      if(mode==='delete')list.splice(index,1);
      else{const entry={id:item?.id||(crypto.randomUUID?.()||Array.from(crypto.getRandomValues(new Uint8Array(16)),x=>x.toString(16).padStart(2,'0')).join('')),name:trimmed,description:description.value.trim(),draft:mode==='rename'?item.draft:snapshot};if(index<0)list.push(entry);else list[index]=entry;}
      writeTemplates(list);templateTab='mine';el.close();refreshTemplates();message(mode==='delete'?tr('模板已删除。','Template deleted.'):tr('模板已保存到“我的模板”。','Saved to My templates.'));
    }catch(e){error.textContent=e.message;}};
  }
  function focusSetting(key){
    let el=key==='WorkerArgs'?root.querySelector('[data-native]'):root.querySelector(`[data-param=${key}]`);
    if(key==='Gpu')el=root.querySelector('[data-gpu-choice]')||el;
    const panel=el?.closest('[role=tabpanel]');if(panel)root.querySelector(`[aria-controls=${panel.id}]`)?.click();
    for(let details=el?.closest('details');details;details=details.parentElement?.closest('details'))details.open=true;
    el?.scrollIntoView({block:'center',behavior:'smooth'});el?.focus();
  }
  function launchData(){
    const data=collect(),p=data.parameters;
    if(info.query_scoring_version!==2)throw Error(tr('启动配置已更新，请完整更新并重启控制台，再刷新页面。','Launch settings changed. Update and restart the complete console, then refresh.'));
    if(p.Backend==='llamacpp'&&info.kv_cache_version!==1)throw Error(tr('K/V 独立格式设置已更新，请先关闭原启动窗口并重新启动控制台，然后刷新页面。','Independent K/V settings have been updated. Close the original startup window, restart the console, then refresh this page.'));
    const disk=root.querySelector('[data-param=DiskEnabled]');
    if(disk?.checked&&!disk.disabled){
      if(!p.DiskPath?.trim())throw Error(tr('启用 SSD 缓存后，请填写 SSD 缓存目录。','Enter an SSD cache directory when SSD cache is enabled.'));
      if(p.DiskMiB!==undefined&&p.DiskMiB<=0)throw Error(tr('启用 SSD 缓存后，SSD 缓存预算必须大于 0。','The SSD cache budget must be positive when SSD cache is enabled.'));
    }else{delete p.DiskPath;delete p.DiskMiB;}
    if(!p.Vision){delete p.VisionResidency;delete p.VisionTokens;}
    return data;
  }
  async function checkConfiguration(){
    const el=dialog(tr('检查配置','Check configuration'),'check'),summary=make('p',{'data-check-summary':'',role:'status'},tr('正在检查…','Checking…')),list=make('div',{class:'l-check-list'});el.append(summary,list);
    const errors=[];
    function row(label,value,key,ok=true){const card=make('div',{class:'l-check-row'});card.append(make('strong',{},label),make('span',{class:ok?'':'l-error'},String(value)));if(key){const fix=make('button',{type:'button',class:'s-button'},tr('前往修改','Edit setting'));fix.onclick=()=>{el.close();focusSetting(key);};card.append(fix);}list.append(card);if(!ok)errors.push(label);}
    let data;
    try{
      data=launchData();const p=data.parameters;
      row(tr('推理引擎','Backend'),names[p.Backend]||p.Backend,'Backend');
      row(tr('模型文件','Model file'),p.Model||tr('请选择模型','Choose a model'),'Model',!!p.Model);
      const ids=(p.Gpu||'').split(',').filter(Boolean),missing=ids.filter(id=>!info.gpus.some(g=>g.uuid===id));
      const gpuIssue=root.querySelector('[data-gpu-choice]')?.validationMessage||'';
      const gpuSummary=ids.length?ids.map((id,i)=>`${i+1}. ${info.gpus.find(g=>g.uuid===id)?.name||id}${p.StageLayers?.[i]?tr(` · ${p.StageLayers[i]} 层`,` · ${p.StageLayers[i]} layers`):''}`).join('\n'):tr('请选择显卡','Choose a GPU');
      row(tr('显卡顺序','GPU order'),gpuSummary+(gpuIssue?'\n'+gpuIssue:''),'Gpu',ids.length>0&&!missing.length&&!gpuIssue);
      row(tr('上下文 / GPU 历史 / RAM 缓存','Context / GPU history / RAM cache'),`${p.Context??'—'} tokens / ${p.Budget??'—'} tokens / ${p.HostMiB??'—'} MiB`,'Context');
      row(tr('KV 缓存格式','KV cache format'),p.Backend==='llamacpp'?`K: ${p.KvTypeK||'—'} / V: ${p.KvTypeV||'—'}`:p.KvType||'—',p.Backend==='llamacpp'?'KvTypeK':'KvType');
      for(const input of root.querySelectorAll('form [data-param]'))if(input.dataset.param!=='Gpu'&&input.willValidate&&!input.checkValidity())row(input.getAttribute('aria-label')||input.dataset.param,input.validationMessage,input.dataset.param,false);
      if(errors.length){summary.textContent=tr('需要补全或修改配置','Some settings need attention');return;}
      const result=await api('preview',data);if(!el.isConnected)return;
      row(tr('后端程序','Backend executable'),result.argv[0],'Worker');
      if(result.token_settings){
        const plan=result.token_settings,labels={Budget:tr('历史 KV 预算','History KV budget'),Reserve:tr('生成预留空间','Generation reserve'),Prefill:tr('预填充块大小','Prefill chunk'),SinkTokens:tr('保留前缀','Retained prefix'),RecentTokens:tr('保留后缀','Retained suffix')};
        row(tr('实际 KV 块大小','Effective KV block size'),`${plan.block_tokens} tokens`,p.Backend==='polystrata'?'BlockTokens':null);
        for(const [key,label] of Object.entries(labels)){const before=plan.requested[key],after=plan.effective[key];row(label,`${before===after?after:`${before} → ${after}`} tokens`+(before===after?'':tr(' · 已自动调整',' · Adjusted automatically')),key);}
      }
      row(tr('对外模型名','Served model name'),result.effective_settings?.model_alias||p.ModelAlias||tr('沿用模型名称','Use model name'),'ModelAlias');
      row(tr('API 访问密钥','API access key'),p.ApiKey?tr('已设置 · 预览隐藏密钥','Configured · Hidden in preview'):tr('未设置','Not configured'),'ApiKey');
      const reserveAt=result.argv.indexOf(p.Backend==='polystrata'?'--reserve':'--kvmem-gen-reserve'),outputAt=result.argv.indexOf(p.Backend==='polystrata'?'--max-tokens':p.Backend==='ninfer'?'--default-max-tokens':'-n');
      if(reserveAt>=0&&outputAt>=0){const reserve=Number(result.argv[reserveAt+1]),requested=Number(result.argv[outputAt+1]);row(tr('默认输出预算 / 追加空间','Default output budget / append workspace'),`${requested} / ${reserve} tokens · `+tr('追加空间可循环复用，输出预算可超过 R','Append workspace is reusable; output budget may exceed R'),'Reserve');}
      const port=result.preflight?.port??p.Port??18200,available=result.preflight?.port_available;
      row(tr('服务端口','Service port'),`${result.preflight?.listen_host||p.ListenHost||'127.0.0.1'}:${port} · `+(available===true?tr('可用','Available'):available===false?tr('已占用或无法绑定；启动前需释放或更换','In use or unavailable; release or change before starting'):tr('启动时检查','Checked on start')),'Port',available!==false);
      summary.textContent=errors.length?tr('配置需要调整','Configuration needs attention'):tr('启动前检查通过','Preflight checks passed');
      if(result.warnings)el.append(make('p',{class:'l-muted'},result.warnings));
      const advanced=make('details',{class:'l-check-command'});advanced.append(make('summary',{},tr('高级：查看启动命令','Advanced: launch command')));
      const quoted=result.argv.map(x=>"'"+x.replaceAll("'","''")+"'").join(' ');
      const command=Object.entries(result.environment||{}).map(([k,v])=>`$env:${k} = '${String(v).replaceAll("'","''")}'`).join('\n')+'\n& '+quoted;
      const code=make('pre',{class:'l-result'},command),copy=make('button',{type:'button',class:'s-button','data-copy-command':''},tr('复制启动命令','Copy launch command'));copy.onclick=async()=>{try{await navigator.clipboard.writeText(command);copy.textContent=tr('已复制','Copied');}catch{copy.textContent=tr('复制失败，请手动选择文本','Copy failed; select the text manually');}};advanced.append(code,copy);el.append(advanced);
      if(p.ApiKey)advanced.append(make('p',{class:'l-muted'},tr('复制的命令含 <redacted> 占位符，手动运行前需替换为密钥。页面启动服务会使用当前填写的密钥。','The copied command contains a <redacted> placeholder. Replace it before running manually. Starting here uses the key entered on this page.')));
    }catch(error){if(!el.isConnected)return;summary.textContent=tr('检查未通过','Check failed');const text=error.message||String(error);const key=['BlockTokens','Prefill','Budget','Reserve','Context','MaxTokens','SinkTokens','RecentTokens','HostMiB','RetainedSessions','VramReserveMiB'].find(k=>text.includes(k))||(/JSON|原生参数|WorkerArgs/i.test(text)?'WorkerArgs':/Worker|后端程序/i.test(text)?'Worker':/Model|模型/i.test(text)?'Model':/StageLayers|层数/i.test(text)?'StageLayers':/Gpu|GPU|显卡/i.test(text)?'Gpu':/Port|端口/i.test(text)?'Port':null);row(tr('需要处理','Needs attention'),text,key,false);}
    finally{if(el.isConnected)el.append(make('p',{class:'l-muted'},tr('检查不会加载模型。兼容性和实际资源需求仍需启动验证；检查通过也不会自动启动。','This check does not load a model. Compatibility and resource needs are verified at startup; passing does not start a service.')));}
  }
  function message(text,error=false){const el=root.querySelector('[data-launch-message]');if(el){el.textContent=text;el.classList.toggle('l-error',error);el.setAttribute('role',error?'alert':'status');}}
  async function api(path,data,timeout=60000){const response=await fetch('/api/launcher/'+path,{method:data===undefined?'GET':'POST',headers:data===undefined?{}:{'Content-Type':'application/json','X-KVMem-Token':info.token},body:data===undefined?undefined:JSON.stringify(data),signal:AbortSignal.timeout(timeout)});const value=await response.json();if(!response.ok)throw Error(value.error||'HTTP '+response.status);return value;}
  function collect(){const params={};root.querySelectorAll('[data-param]').forEach(el=>{if(el.disabled||el.dataset.uiOnly==='true')return;const k=el.dataset.param;if(el.type==='checkbox'){const checked=el.dataset.inverted==='true'?!el.checked:el.checked;if(checked||el.dataset.explicitBoolean==='true'||k==='PreserveThinking')params[k]=checked;}else if(el.multiple){params[k]=Array.from(el.selectedOptions).map(x=>x.value).join(',');}else if(el.value!==''){params[k]=el.dataset.booleanSelect==='true'?el.value==='true':(el.type==='number'||el.dataset.valueType==='number')?Number(el.value)+(el.dataset.idleSessions==='true'?1:0):el.value;}});for(const key of ['StageLayers','TensorSplit'])if(params[key])params[key]=params[key].split(',').map(x=>Number(x.trim()));const raw=root.querySelector('[data-native]');if(raw&&!raw.checkValidity())throw Error(raw.validationMessage);if(raw?.value.trim()){const args=JSON.parse(raw.value);if(!Array.isArray(args))throw Error(tr('原生参数必须为 JSON 数组','Native arguments must be a JSON array'));params.WorkerArgs=args;}return {schema_version:1,parameters:params};}
  function elasticControls(parent,p,raw){
    const section=make('section',{class:'l-elastic l-wide','data-elastic-controls':''});
    section.append(make('h3',{class:'s-title'},tr('专家缓存','Expert cache')));
    const grid=make('div',{class:'l-fields'});section.append(grid);parent.prepend(section);
    const mode=field(grid,{},'ElasticExpertsMode','弹性专家','Elastic experts','text',[
      ['off',tr('关闭','Off')],['workspace',tr('弹性分配','Elastic allocation')],['headroom',tr('弹性分配 + 显存借用','Elastic allocation + VRAM borrowing')]
    ],['生成阶段将空闲工作空间用于专家缓存；显存借用在 Windows 上进一步按显存余量扩大缓存，实际分配由引擎动态调整。','Reuse idle workspace for the expert cache during generation. On Windows, VRAM borrowing can expand the cache further; the engine adjusts actual allocations dynamically.']);
    mode.closest('label').classList.add('l-wide');
    const ceiling=field(grid,{ElasticHeadroomMiB:p.ElasticHeadroomMiB??128},'ElasticHeadroomMiB','额外借用上限 · MiB','Borrowing ceiling · MiB','number',null,
      ['可超出系统显存预算的上限，范围 0–1024 MiB；不是保证分配的数量。','Maximum excess above the system VRAM budget, 0–1024 MiB; not a guaranteed allocation.']);ceiling.max='1024';
    const reserve=field(grid,{ElasticPhysicalReserveMiB:p.ElasticPhysicalReserveMiB??512},'ElasticPhysicalReserveMiB','物理显存保留 · MiB','Physical VRAM reserve · MiB','number',null,
      ['借用时至少保留的物理空闲显存，范围 256–4096 MiB。','Minimum physical free VRAM while borrowing, 256–4096 MiB.']);reserve.min='256';reserve.max='4096';
    for(const input of [mode,ceiling,reserve]){input.dataset.uiOnly='true';input.dataset.keepDraft='true';}
    const switches=new Set(['--elastic-experts','--elastic-experts-headroom']);
    const numbers=new Map([['--elastic-headroom-mib',ceiling],['--elastic-physical-reserve-mib',reserve]]);
    function parse(){
      const args=JSON.parse(raw.value||'[]');if(!Array.isArray(args)||args.some(x=>typeof x!=='string'))throw Error('array');
      const other=[],seen=new Set(),values={};let workspace=false,headroom=false;
      for(let i=0;i<args.length;i++){
        const flag=args[i];
        if(switches.has(flag)||numbers.has(flag)){
          if(seen.has(flag))throw Error('duplicate');seen.add(flag);
          if(flag==='--elastic-experts')workspace=true;
          else if(flag==='--elastic-experts-headroom')headroom=true;
          else{
            const input=numbers.get(flag),value=args[++i];
            if(!/^\d{1,6}$/.test(value??'')||Number(value)<Number(input.min)||Number(value)>Number(input.max))throw Error('range');
            values[flag]=value;
          }
        }else other.push(flag);
      }
      return {other,values,mode:headroom?'headroom':workspace?'workspace':'off'};
    }
    function dependencies(valid){
      const reason=valid?tr('选择“弹性分配 + 显存借用”后可设置。','Select elastic allocation + VRAM borrowing first.'):tr('请先修正引擎专属参数中的原生 JSON。','Correct the native JSON in engine-specific settings first.');
      dependency('ElasticExpertsMode',valid,reason);
      for(const key of ['ElasticHeadroomMiB','ElasticPhysicalReserveMiB'])dependency(key,valid&&mode.value==='headroom',reason);
      ceiling.required=reserve.required=valid&&mode.value==='headroom';
      applyConfigurationLock();
    }
    function read(){
      try{
        const parsed=parse();mode.value=parsed.mode;
        for(const [flag,input] of numbers){
          if(parsed.values[flag]!==undefined)input.value=parsed.values[flag];
          else if(parsed.mode==='headroom')input.value=input===ceiling?'128':'512';
        }
        raw.setCustomValidity('');dependencies(true);
      }catch{
        raw.setCustomValidity(tr('原生参数须为 JSON 字符串数组，弹性参数不可重复且需在范围内。','Native options must be a JSON string array with unique, valid elastic settings.'));
        dependencies(false);
      }
    }
    function write(){
      if(configurationLocked())return;
      try{
        const parsed=parse(),args=parsed.other;
        dependencies(true);
        if(mode.value==='workspace')args.push('--elastic-experts');
        if(mode.value==='headroom'){
          if(!ceiling.checkValidity()||!reserve.checkValidity()){
            raw.setCustomValidity(tr('请检查弹性专家的借用上限与物理显存保留范围。','Check the elastic borrowing ceiling and physical VRAM reserve ranges.'));return;
          }
          args.push('--elastic-experts-headroom','--elastic-headroom-mib',ceiling.value,'--elastic-physical-reserve-mib',reserve.value);
        }
        raw.value=JSON.stringify(args);raw.setCustomValidity('');
      }catch{read();}
    }
    raw.addEventListener('input',read);raw.addEventListener('change',read);
    for(const input of [mode,ceiling,reserve]){input.addEventListener('input',write);input.addEventListener('change',write);}
    read();
  }
  function field(parent,p,key,zh,en,type='text',options=null,note=null,disabled=false){
    const label=make('label',{class:'l-field l-row'}),title=tr(zh,en);
    if(type==='checkbox'){
      label.classList.add('l-toggle');
      const copy=make('span',{class:'l-toggle-copy'});copy.append(make('span',{},title));
      const input=make('input',{type:'checkbox',role:'switch','aria-label':title,'data-param':key});
      input.checked=!!p[key];input.disabled=disabled;input.dataset.backendSupported=String(!disabled);
      if(note||disabled)help(copy.firstChild,title,(disabled?tr('当前启动器对此后端暂未开放此设置。','This launcher does not expose this setting for this backend yet. '):'')+(note?tr(...note):''),input);
      const control=make('span',{class:'l-toggle-control'}),state=make('span',{class:'l-toggle-state','aria-hidden':'true'});
      state.append(make('span',{'data-toggle-on':''},tr('已开启','On')),make('span',{'data-toggle-off':''},tr('已关闭','Off')),make('span',{'data-toggle-unavailable':''},tr('不可用','Unavailable')));
      control.append(input,state);label.append(copy,control);parent.append(label);return input;
    }
    label.append(make('span',{},title));let input;
    if(options){input=make('select');options.forEach(([value,text])=>input.append(make('option',{value},text)));}else input=make('input',{type});
    input.dataset.param=key;input.disabled=disabled;input.dataset.backendSupported=String(!disabled);input.setAttribute('aria-label',title);
    if(type==='number'){input.dataset.valueType='number';input.min='0';input.step='1';}input.value=p[key]??'';label.append(input);
    if(note||disabled)help(label.firstChild,title,(disabled?tr('当前启动器对此后端暂未开放此设置。','This launcher does not expose this setting for this backend yet. '):'')+(note?tr(...note):''),input);
    if(disabled)label.append(make('small',{class:'l-unavailable'},tr('此后端暂不可用','Unavailable for this backend')));
    parent.append(label);return input;
  }

  function dependency(key,enabled,reason){
    const input=root.querySelector(`[data-param=${key}]`);
    if(!input||input.dataset.backendSupported==='false')return;
    // Disabled by a feature dependency, not unsupported by this backend.
    // Preserve values for drafts/templates while collect() omits the override.
    input.dataset.keepDraft='true';input.disabled=!enabled;input.dataset.disabledReason=enabled?'':reason;
    const state=input.closest('label')?.querySelector('[data-toggle-unavailable]');
    if(state)state.textContent=tr('未生效','Inactive');
  }
  function syncDependencies(){
    const get=key=>root.querySelector(`[data-param=${key}]`);
    if(!get('Backend'))return;
    dependency('DiskEnabled',true,'');
    const disk=!!get('DiskEnabled')?.checked&&!get('DiskEnabled').disabled,vision=!!get('Vision')?.checked;
    for(const key of ['DiskPath','DiskMiB'])dependency(key,disk,tr('先开启 SSD 缓存。','Enable SSD cache first.'));
    if(get('DiskPath'))get('DiskPath').required=disk;
    for(const key of ['VisionTokens','VisionResidency'])dependency(key,vision,tr('先开启图像输入。','Enable vision input first.'));
    dependency('ThinkingBudget',get('ThinkingMode')?.value!=='off',tr('默认思考模式已关闭；重新开启或选择默认后可设置预算。','Default thinking is off. Select on or default to set a budget.'));
    for(const option of get('ReasoningEffort')?.options||[])if(option.value==='none'&&get('Backend').value==='polystrata')option.disabled=get('ThinkingMode')?.value!=='off';
    dependency('ReasoningEffort',get('ThinkingMode')?.value!=='off',tr('默认思考模式已关闭；选择默认后可设置思考强度。','Default thinking is off. Select default to set reasoning effort.'));
    const history=!!get('Lrfu')?.checked;
    for(const key of ['LrfuLambda','LrfuTopFraction'])dependency(key,history,tr('先开启历史累计。','Enable LRFU history first.'));
    const fraction=get('LrfuTopFraction');if(fraction)fraction.setCustomValidity(fraction.value!==''&&Number(fraction.value)<=0?tr('候选比例必须大于 0。','The reward fraction must be greater than 0.'): '');
    const backend=get('Backend').value,mode=get('SpeculativeMode')?.value||'none';
    const profiles=info.speculative_capabilities?.[backend]||{none:{},...(backend==='ninfer'?{mtp:{max_draft_tokens:15,adaptive:true,ngram:true,max_concurrency:8,max_devices:8}}:{})};
    const cap=profiles[mode]||{},enabled=mode!=='none';
    dependency('DraftTokens',enabled,tr('先选择草稿方式。','Select a draft method first.'));
    if(get('DraftTokens')){get('DraftTokens').required=enabled;get('DraftTokens').max=String(cap.max_draft_tokens||15);}
    dependency('DraftPolicy',enabled&&!!cap.adaptive,tr('所选后端或草稿方式仅支持固定长度。','The selected backend or draft method supports fixed length only.'));
    dependency('NgramDrafts',enabled&&!!cap.ngram,tr('所选后端或草稿方式不支持 N-gram。','The selected backend or draft method does not support N-gram.'));
    dependency('NgramMinMatch',enabled&&!!cap.ngram&&Number(get('NgramDrafts')?.value)>0,tr('先启用 N-gram 草稿。','Enable N-gram drafts first.'));
    dependency('DraftModel',enabled&&backend==='llamacpp',tr('NInfer 使用模型内置草稿组件；llama.cpp 需先选择草稿方式。','NInfer uses an embedded draft component. Select a draft method for llama.cpp first.'));
    if(get('DraftModel'))get('DraftModel').required=enabled&&cap.draft_asset==='required_gguf';
    const selectedGpu=get('Gpu'),selectedCount=selectedGpu?.multiple?selectedGpu.selectedOptions.length:(selectedGpu?.value||'').split(',').filter(Boolean).length;
    const specReason=enabled&&(!profiles[mode]||Number(get('Concurrency')?.value||1)>(cap.max_concurrency||1)||selectedCount>(cap.max_devices||1))?
      tr('当前草稿方式不支持所选并发或显卡数量。','The selected draft method does not support this concurrency or GPU count.'):'';
    get('SpeculativeMode')?.setCustomValidity(specReason);
    for(const option of get('SpeculativeMode')?.options||[]){const c=profiles[option.value];option.disabled=!c||(option.value!=='none'&&(Number(get('Concurrency')?.value||1)>c.max_concurrency||selectedCount>c.max_devices));}
    dependency('FastPrefill',['int8','rk8v4','rk4v4','rk4v4-e8','rk2v4-e8'].includes(get('KvType')?.value),tr('当前 KV 格式不支持快速预填充。','The current KV format does not support fast prefill.'));
    const gpu=get('Gpu'),gpuCount=gpu?.multiple?gpu.selectedOptions.length:(gpu?.value||'').split(',').filter(Boolean).length;
    const tensor=get('TensorParallel');
    dependency('TensorParallel',backend==='ninfer'&&gpuCount>1,tr('至少选择两张显卡后才能启用张量并行。','Select at least two GPUs to enable tensor parallelism.'));
    const heads=!!tensor?.checked&&!tensor.disabled;
    dependency('StageLayers',gpuCount>1&&!heads,tr('层数分配用于非 TP 多卡模式。','Layer allocation applies to multiple GPUs without TP.'));
    dependency('TensorSplit',heads,tr('先启用张量并行。','Enable tensor parallelism first.'));
    const weights=get('TensorSplit');
    if(weights){const parts=weights.value.split(',');weights.setCustomValidity(heads&&weights.value.trim()&&(parts.length!==gpuCount||parts.some(x=>!x.trim()||!Number.isFinite(Number(x))||Number(x)<=0)||!Number.isFinite(parts.reduce((total,x)=>total+Number(x),0)))?tr('请按显卡顺序填写等量的正数权重，或留空自动分配。','Enter one positive weight per GPU in order, or leave empty for automatic allocation.'):'');}
    const keyType=get('KvTypeK'),valueType=get('KvTypeV'),pairStatus=root.querySelector('[data-kv-pair-status]');
    if(keyType&&valueType){
      const quantized=['q8_0','q5_0','q4_0'],mixed=keyType.value&&valueType.value&&quantized.includes(keyType.value)!==quantized.includes(valueType.value);
      const reason=mixed?tr('当前后端不能混用浮点与量化 K/V。请两边都选 q8_0、q5_0 或 q4_0，或两边都选 f16、f32。','This backend cannot mix floating-point and quantized K/V. Use q8_0, q5_0 or q4_0 for both sides, or f16/f32 for both.'):'';
      keyType.setCustomValidity(reason);valueType.setCustomValidity(reason);
      if(pairStatus){pairStatus.textContent=reason;pairStatus.hidden=!mixed;}
    }
    if(openHelp)showHelp(openHelp);
  }

  function inputActions(input){
    const row=make('div',{class:'l-input-actions'});input.replaceWith(row);row.append(input);return row;
  }

  function credentialField(parent,p){
    const input=field(parent,{ApiKey:launchKeys[p.Backend]||''},'ApiKey','API 访问密钥（可选）','API access key (optional)','password',null,['仅在当前页面保留，不写入草稿或模板；刷新后如需重启服务，请重新填写。监控自动连接当前服务；外部聊天客户端需填写此密钥。','Kept only on this page, excluded from drafts and templates. Re-enter after refreshing to restart. Monitoring connects automatically; external chat clients need this key.']);
    input.closest('label').classList.add('l-wide');input.dataset.secret='true';input.autocomplete='off';input.spellcheck=false;input.maxLength=512;input.pattern='[A-Za-z0-9._~+\\x2f\\x2d]+=*';
    input.addEventListener('input',()=>{launchKeys[p.Backend]=input.value;});
    const row=inputActions(input),show=make('button',{type:'button',class:'s-button','aria-pressed':'false'},tr('显示','Show')),generate=make('button',{type:'button',class:'s-button','data-generate-key':''},tr('生成','Generate'));
    show.onclick=()=>{const visible=input.type==='password';input.type=visible?'text':'password';show.setAttribute('aria-pressed',String(visible));show.textContent=visible?tr('隐藏','Hide'):tr('显示','Show');};
    generate.onclick=()=>{const bytes=crypto.getRandomValues(new Uint8Array(32));input.value=Array.from(bytes,x=>x.toString(16).padStart(2,'0')).join('');input.dispatchEvent(new Event('input',{bubbles:true}));};
    row.append(show,generate);
  }

  function templateFileField(parent,p){
    const advanced=make('details',{class:'l-advanced l-wide'});advanced.append(make('summary',{},tr('高级 · 自定义聊天模板','Advanced · Custom chat template')));const grid=make('div',{class:'l-fields l-template-fields'});advanced.append(grid);parent.append(advanced);
    const input=field(grid,p,'ChatTemplateFile','模板文件（可选）','Template file (optional)','text',null,['留空使用模型与引擎默认模板。只选择模板文件，不在页面中编辑模板。','Leave empty to use the model and engine default. Choose a template file; template editing is not provided here.']);input.closest('label').classList.add('l-wide');
    const row=inputActions(input),browse=make('button',{type:'button',class:'s-button','data-browse-template':''},tr('选择…','Choose…')),reset=make('button',{type:'button',class:'s-button','data-reset-template':''},tr('恢复默认','Use default'));
    browse.onclick=async()=>{if(pickingModel)return;pickingModel=true;browse.disabled=true;try{const result=await api('browse-template',{backend:p.Backend,current_path:input.value||p.Model||''},315000);if(result.path&&input.isConnected){input.value=result.path;input.dispatchEvent(new Event('input',{bubbles:true}));}}catch(error){showError(error);}finally{pickingModel=false;browse.disabled=false;}};
    reset.onclick=()=>{input.value='';input.dispatchEvent(new Event('input',{bubbles:true}));};row.append(browse,reset);
    if(info.native_picker===false)browse.hidden=true;
  }

  function modelPicker(parent, p) {
    const box = make('div', {class:'l-field l-model-picker'});
    const label = make('label', {for:'launch-model'}, tr('模型文件','Model file'));
    const row = make('div', {class:'l-model-row'});
    const input = make('input', {id:'launch-model', type:'text', 'data-param':'Model', required:'',
      placeholder:tr('点击选择模型，也可粘贴文件路径','Choose a model, or paste a file path')});
    input.value = p.Model || '';
    input.title = input.value;
    input.addEventListener('input',()=>{input.title=input.value;});
    const browse = make('button', {type:'button', class:'s-button', 'data-browse-model':''}, tr('选择模型…','Choose model…'));
    browse.disabled = pickingModel;
    if(info.native_picker===false){browse.hidden=true;input.placeholder=tr('输入此 Linux 主机上的模型完整路径','Enter the full model path on this Linux host');}
    row.append(input, browse);box.append(label, row);
    const recent = make('select', {'aria-label':tr('最近选择的模型','Recently selected models'),'data-recent-model':''});
    function refreshRecent() {
      const items = recentModels(p.Backend);
      recent.replaceChildren(make('option',{value:''},tr(`最近选择（${items.length}）`,`Recent models (${items.length})`)));
      for (const path of items) recent.append(make('option',{value:path,title:path},recentModelLabel(path,items)));
      recent.hidden = !items.length;
    }
    refreshRecent();
    recent.onchange = () => { if(recent.value){input.value=recent.value;input.dispatchEvent(new Event('input',{bubbles:true}));recent.value='';} };
    box.append(recent);
    const scan = make('button',{type:'button',class:'s-button l-scan-trigger','data-scan-models':''},tr('查找模型…','Find models…'));
    const modelTools=make('div',{class:'l-model-tools'});
    modelTools.append(make('small',{},(modelFormat(p.Backend))+tr(' · 本机文件',' · Local file')),scan);
    const dialog = make('dialog',{class:'l-model-dialog','aria-labelledby':'model-scan-title'});
    const head = make('div',{class:'l-heading'});
    const close = make('button',{type:'button',class:'s-button'},tr('关闭','Close'));
    head.append(make('h2',{id:'model-scan-title',class:'s-title'},tr('查找模型','Find models')+' · '+(modelFormat(p.Backend))),close);
    const folderLabel=make('label',{class:'l-field'},tr('模型所在文件夹','Model folder'));
    const directory=make('input',{type:'text','data-scan-directory':'',placeholder:tr('粘贴文件夹路径，或点击浏览文件夹','Paste a folder path, or browse for a folder')});
    folderLabel.append(directory);
    const folderActions=make('div',{class:'l-actions l-scan-actions'});
    const other=make('button',{type:'button',class:'s-button','data-scan-folder':''},tr('浏览文件夹…','Browse folder…'));
    const submit=make('button',{type:'button',class:'s-button l-primary','data-scan-submit':''},tr('扫描此目录并记住','Scan and remember folder'));
    const rescan=make('button',{type:'button',class:'s-button','data-scan-all':'','aria-describedby':'model-scan-locations-help'},tr('扫描常用位置','Scan known locations'));
    folderActions.append(other,submit,rescan);
    if(info.native_picker===false){other.hidden=true;directory.placeholder=tr('输入此 Linux 主机上的文件夹路径','Enter a folder path on this Linux host');}
    const locationHelp=make('p',{id:'model-scan-locations-help',class:'l-muted l-scan-help'},tr('常用位置：已记住的模型目录、当前模型所在文件夹、程序所在文件夹，以及附近名为 models 或 model 的文件夹。不会扫描整个硬盘；其他位置请在上方选择或粘贴目录。本次具体位置见“扫描范围”。','Known locations include remembered model folders, the current model’s folder, the program folders, and nearby folders named models or model. This does not scan entire drives. For other locations, choose or paste a folder above. Expand Search locations to see the actual folders searched.'));
    const savedRow=make('div',{class:'l-scan-saved-row'});
    const saved=make('select',{'data-scan-saved':'','aria-label':tr('已记住的模型目录','Remembered model folders')});
    const forget=make('button',{type:'button',class:'s-button','data-scan-forget':'',disabled:'',title:tr('只移除目录记录，不删除模型文件','Remove the folder record; keep all model files')},tr('移除目录记录','Forget folder'));
    savedRow.append(saved,forget);
    const folderError=make('p',{class:'l-error',role:'alert','data-scan-folder-error':'',hidden:''});
    const progress = make('p',{class:'l-muted','aria-live':'polite','data-scan-status':''});
    const scope = make('details',{class:'l-scan-scope'});scope.append(make('summary',{},tr('扫描范围','Search locations')));
    const paths = make('div',{class:'l-muted'});scope.append(paths);
    const filter=make('input',{type:'search','data-scan-filter':'','aria-label':tr('筛选模型','Filter models'),placeholder:tr('按模型名称或路径筛选结果','Filter results by name or path')});
    const formatHint=make('p',{class:'l-muted','data-scan-format-hint':'',hidden:''});
    const results = make('div',{class:'l-scan-results','data-scan-results':''});
    dialog.append(head,folderLabel,folderActions,locationHelp,savedRow,folderError,
      make('p',{class:'l-muted'},tr('扫描指定目录后会记住位置，下次可直接选择。模型文件保留在本机。','Scanned folders are remembered for next time. Model files stay on this computer.')),
      progress,scope,filter,formatHint,results,make('p',{class:'l-muted'},tr('选择结果只填入启动配置。文件名和大小不能证明模型兼容，启动时还会检查。','Selecting a result only fills the launch settings. File names and sizes do not prove compatibility; startup checks still apply.')));
    box.append(modelTools,dialog);
    close.onclick=()=>dialog.close();
    let scanning=false, scanTask=null, scanGeneration=0, latest=null;
    function folderMessage(text='',error=true){folderError.textContent=text;folderError.hidden=!text;folderError.classList.toggle('l-error',error);folderError.classList.toggle('l-muted',!error);folderError.setAttribute('role',error?'alert':'status');}
    async function refreshLocations(selected='',generation=scanGeneration){
      const data=await api('model-directories');
      if(!dialog.isConnected||!dialog.open||generation!==scanGeneration)return;
      saved.replaceChildren(make('option',{value:''},tr('已记住的目录（选择后扫描）','Remembered folders (select to scan)')));
      for(const item of data.directories||[])saved.append(make('option',{value:item.path},item.path+(item.available?'':tr(' · 当前不可访问',' · Unavailable'))));
      saved.value=selected;forget.disabled=!saved.value;
    }
    function showLocations(result){
      paths.replaceChildren();
      const labels={scanned:['已扫描','Scanned'],covered:['已包含','Included'],partial:['部分扫描','Partly scanned'],unavailable:['不存在或无法访问','Missing or inaccessible'],pending:['尚未扫描','Not scanned']};
      for(const item of result.locations||result.roots.map(path=>({path,status:'scanned'}))){
        const row=make('div',{class:'l-scan-location'});
        row.append(make('span',{},item.path),make('small',{},tr(...(labels[item.status]||labels.pending))));paths.append(row);
      }
      if(!paths.childElementCount)paths.textContent=tr('尚未找到常用模型目录，请在上方选择模型文件夹。','No known model folder was found. Choose a model folder above.');
      scope.open=!result.models.length||result.limited||!!result.skipped;
    }
    function renderResults(){
      if(!latest)return;
      const query=filter.value.trim().toLocaleLowerCase(),models=latest.models.filter(model=>(model.name+' '+model.path).toLocaleLowerCase().includes(query));
      results.replaceChildren();
      progress.textContent=query?tr(`显示 ${models.length} / ${latest.models.length} 个候选模型`,`Showing ${models.length} of ${latest.models.length} candidate models`):tr(`找到 ${latest.models.length} 个候选模型`,`Found ${latest.models.length} candidate models`);
      const reasons={time:['扫描时间已达上限','Time limit reached'],entries:['检查的文件数量已达上限','Entry limit reached'],results:['候选数量已达上限','Result limit reached'],depth:['部分目录层级较深','Some folders are too deeply nested']};
      if(latest.limited)progress.textContent+=tr(' · 仅显示部分结果：',' · Partial results: ')+(latest.limit_reasons?.length?latest.limit_reasons.map(key=>tr(...(reasons[key]||['已达扫描限制','Scan limit reached']))).join('；'):tr('已达扫描限制','Scan limit reached'))+tr('，可选择更具体的文件夹。','; choose a more specific folder.');
      if(latest.skipped)progress.textContent+=tr(' · 部分文件或目录无法读取。',' · Some files or folders could not be read.');
      if(!models.length)results.append(make('p',{class:'l-scan-empty'},query?tr('没有符合筛选条件的模型。','No models match this filter.'):tr(`扫描范围内未找到 ${modelFormat(p.Backend)} 模型。请在上方添加实际存放模型的文件夹，或直接使用“选择模型”。`,`No ${modelFormat(p.Backend)} models were found in these locations. Add the folder containing your models above, or use Choose model.`)));
      for(const model of models){
        const pick=make('button',{type:'button',class:'l-model-candidate','data-model-candidate':model.path});
        const empty=model.bytes===0||model.warnings?.includes('empty');
        pick.disabled=!!empty;
        const unit=model.bytes>=1024**3?['GiB',1024**3]:model.bytes>=1024**2?['MiB',1024**2]:model.bytes>=1024?['KiB',1024]:['B',1];
        pick.append(make('span',{},model.name),make('small',{},`${(model.bytes/unit[1]).toLocaleString(window.monitorI18n.language,{maximumFractionDigits:2})} ${unit[0]} · ${model.format}`),make('small',{class:'l-candidate-path'},model.path));
        if(empty)pick.append(make('small',{class:'l-error'},tr('文件为空，可能尚未下载完成。','This file is empty; its download may be incomplete.')));
        pick.onclick=()=>{if(!input.isConnected||!dialog.open)return;input.value=model.path;rememberModel(p.Backend,model.path);refreshRecent();input.dispatchEvent(new Event('input',{bubbles:true}));dialog.close();input.focus();message(tr('已选用候选模型；下次启动生效。','Candidate selected; applies on next launch.'));};
        results.append(pick);
      }
      const different=p.Backend==='ninfer'?'gguf':'ninfer',count=latest.other_formats?.[different]||0;
      formatHint.hidden=!count;formatHint.textContent=count?tr(`扫描范围还发现 ${count} 个 .${different} 文件；当前后端需要 ${modelFormat(p.Backend)} 模型。`,`The scanned locations also contain ${count} .${different} files; this backend needs ${modelFormat(p.Backend)} models.`):'';
    }
    dialog.addEventListener('close',()=>{scanGeneration++;scanTask?.abort();scanning=false;});
    filter.oninput=renderResults;
    async function runScan(searchDirectory='') {
      if(scanning||!dialog.open)return;
      const generation=++scanGeneration,controller=new AbortController();scanTask=controller;
      const timer=setTimeout(()=>controller.abort(),12000);
      scanning=true;other.disabled=true;submit.disabled=true;rescan.disabled=true;saved.disabled=true;forget.disabled=true;filter.disabled=true;results.replaceChildren();formatHint.hidden=true;folderMessage();
      progress.textContent=tr('正在扫描模型目录…','Scanning model folders…');paths.textContent=tr('正在确定扫描范围…','Determining search locations…');
      try {
        const response=await fetch('/api/launcher/scan-models',{method:'POST',headers:{'Content-Type':'application/json','X-KVMem-Token':info.token},body:JSON.stringify({backend:p.Backend,current_path:input.value,worker_path:root.querySelector('[data-param=Worker]')?.value||'',...(searchDirectory?{directory:searchDirectory}:{})}),signal:controller.signal});
        const result=await response.json();if(!response.ok)throw Error(result.error||'HTTP '+response.status);
        if(generation!==scanGeneration||!dialog.open||!input.isConnected)return;
        latest=result;showLocations(result);
        if(searchDirectory&&result.roots[0])directory.value=result.roots[0];
        try{await refreshLocations(searchDirectory?directory.value:'',generation);}catch(error){if(generation===scanGeneration&&dialog.open)folderMessage(error.message);}
        if(generation===scanGeneration&&dialog.open&&input.isConnected)renderResults();
      }catch(error){if(generation===scanGeneration&&dialog.open){latest=null;progress.textContent=tr('本次扫描未完成，可以修改目录后重试。','This scan did not complete. Adjust the folder and try again.');folderMessage(error.name==='AbortError'?tr('扫描超时，请选择更具体的模型文件夹，或直接选择模型文件。','The scan timed out. Choose a more specific folder or select the model file directly.'):error.message);}}
      finally{clearTimeout(timer);if(generation===scanGeneration){scanning=false;other.disabled=false;submit.disabled=false;rescan.disabled=false;saved.disabled=false;forget.disabled=!saved.value;filter.disabled=false;scanTask=null;}}
    }
    scan.onclick=()=>{if(busy)return;if(info.model_library_version!==1){showError(Error(tr('模型查找功能已更新，请先关闭原启动窗口并重新启动控制台，然后刷新页面。','Model discovery has been updated. Close the original startup window, restart the console, then refresh this page.')));return;}dialog.showModal();other.disabled=false;submit.disabled=false;rescan.disabled=false;filter.disabled=false;filter.value='';runScan();};
    submit.onclick=()=>{const value=directory.value.trim();if(!value){folderMessage(tr('请先填写模型文件夹路径，或点击“浏览文件夹”。','Enter a model folder path or click Browse folder first.'));directory.focus();return;}filter.value='';runScan(value);};
    directory.onkeydown=event=>{if(event.key==='Enter'){event.preventDefault();submit.click();}};
    rescan.onclick=()=>{directory.value='';filter.value='';runScan();};
    saved.onchange=()=>{forget.disabled=!saved.value;if(saved.value){directory.value=saved.value;filter.value='';runScan(saved.value);}};
    forget.onclick=async()=>{if(!saved.value||scanning)return;try{await api('model-directories',{action:'remove',path:saved.value});directory.value='';await refreshLocations();runScan();}catch(error){folderMessage(error.message);}};
    other.onclick=async()=>{
      if(pickingModel||scanning)return;
      pickingModel=true;other.disabled=true;
      try {
        folderMessage(tr('请在系统窗口中选择模型文件夹；也可取消后直接粘贴路径。','Choose the model folder in the system dialog, or cancel and paste its path.'),false);
        const result=await api('browse-model-folder',{backend:p.Backend,current_path:directory.value||input.value},315000);
        if(!dialog.open||!input.isConnected)return;
        if(result.path){directory.value=result.path;filter.value='';await runScan(result.path);}else folderMessage(tr('已取消选择，当前扫描结果保持不变。','Folder selection cancelled; current results are unchanged.'),false);
      }catch(error){if(dialog.open)folderMessage(error.message);}
      finally{pickingModel=false;other.disabled=false;}
    };
    browse.onclick = async () => {
      if(pickingModel || busy) return;
      pickingModel = true; browse.disabled = true;
      const original = input.value;
      browse.textContent = tr('等待选择…','Waiting…');
      message(tr('请在系统文件选择窗口中选择模型。','Choose a model in the system file picker.'));
      try {
        const response = await fetch('/api/launcher/browse-model',{method:'POST',headers:{'Content-Type':'application/json','X-KVMem-Token':info.token},
          body:JSON.stringify({backend:p.Backend,current_path:original}),signal:AbortSignal.timeout(315000)});
        const result = await response.json();
        if(!response.ok) throw Error(result.error || 'HTTP '+response.status);
        if(result.path) {
          rememberModel(p.Backend,result.path);
          if(input.isConnected && input.value === original){input.value=result.path;input.dispatchEvent(new Event('input',{bubbles:true}));refreshRecent();message(tr('已选择模型；下次启动生效。','Model selected; applies on next launch.'));}
          else message(tr('已加入该引擎的最近选择列表。','Added to the recent models for that backend.'));
        } else message(tr('已取消，原模型保持不变。','Cancelled; the original model is unchanged.'));
      } catch(error) {message(error.message,true);}
      finally {pickingModel=false;root.querySelectorAll('[data-browse-model]').forEach(button=>{button.disabled=false;button.textContent=tr('选择模型…','Choose model…');});}
    };
    parent.append(box);
  }
  function applyPreset(preset) {
    if (busy || !info.backends.includes(preset.parameters.Backend)) return;
    try {
      const before = captureDraft().parameters;
      presetUndo = {draft: captureDraft(), selected: selectedPreset, tab: activeTab};
      perBackend[before.Backend] = captureDraft();
      const assets = perBackend[preset.parameters.Backend]?.parameters || {};
      const next = {...defaults, ...structuredClone(preset.parameters)};
      for (const key of ['Model','Worker','ChatTemplateFile','Mmproj','VisionWorker','Tokenizer','PolystrataSource','Python','DraftModel','DiskPath']) {
        if (assets[key]) next[key] = assets[key];
      }
      next.Port = before.Port ?? defaults.Port;
      next.ListenHost = before.ListenHost ?? defaults.ListenHost;
      next.Gpu = before.Gpu || '';
      next.StageLayers=before.Backend===next.Backend?before.StageLayers||[]:[];next.TensorParallel=before.Backend===next.Backend?!!before.TensorParallel:false;next.TensorSplit=before.Backend===next.Backend?before.TensorSplit||[]:[];
      selectedPreset = preset.id;
      activeTab = 0;
      render(next);
      scheduleSave();
      message(tr('模板已填入编辑区，尚未启动。可以撤销本次套用。','Template applied to the draft; no service started. You can undo this change.'));
      root.querySelector(`[data-preset="${preset.id}"]`).focus();
    } catch (error) { message(error.message, true); }
  }
  function presetPanel(side) {
    const panel = make('section', {class:'s-card l-presets'});
    const title=make('h2',{class:'s-title'},tr('启动模板','Launch templates'));panel.append(title);
    help(title,tr('启动模板','Launch templates'),tr('套用参数后再点击启动，模型路径按引擎保留。草稿与个人模板保存在当前浏览器，不包含 API 密钥；清除网站数据会删除这些设置。','Apply a recipe, then start. Model paths are retained per backend. Drafts and templates stay in this browser without API keys; clearing site data removes them.'));
    const list = make('div', {class:'l-preset-list'});
    const switcher=make('div',{class:'l-template-tabs'});
    for(const [id,zh,en] of [['builtin','内置模板','Built-in'],['mine','我的模板','My templates']]){const tab=make('button',{type:'button',class:'s-button','data-template-tab':id,'aria-pressed':String(templateTab===id)},tr(zh,en));tab.onclick=()=>{templateTab=id;refreshTemplates();};switcher.append(tab);}panel.append(switcher);
    for (const preset of templateTab==='builtin'?presets.filter(p=>p.parameters.Backend===root.querySelector('[data-param=Backend]').value):[]) {
      const available = info.backends.includes(preset.parameters.Backend);
      const button = make('button', {type:'button', class:'l-preset', 'data-preset':preset.id,
        'aria-pressed':String(selectedPreset === preset.id)});
      button.disabled = busy || !available;
      button.append(make('span', {}, tr(...preset.title)), make('small', {},
        available ? tr(...preset.note) : tr('当前后端目录未安装此引擎','Not installed in this backend checkout')));
      button.onclick = () => applyPreset(preset);
      list.append(button);
    }
    if(templateTab==='mine'){
      try{
        const saved=readTemplates().filter(item=>item.draft.parameters.Backend===root.querySelector('[data-param=Backend]').value);
        if(!saved.length)list.append(make('p',{class:'l-muted'},tr('还没有个人模板。点击页面顶部“另存为模板”保存当前设置。','No templates yet. Use Save as template above to keep these settings.')));
        for(const item of saved){
          const wrapper=make('div',{class:'l-user-template','data-user-template':item.id}),button=make('button',{type:'button',class:'l-preset','data-apply-template':item.id,'aria-pressed':String(selectedPreset===item.id)});
          button.append(make('span',{},item.name),make('small',{},item.description||names[item.draft.parameters.Backend]));
          button.onclick=()=>{if(busy)return;presetUndo={draft:captureDraft(),selected:selectedPreset,tab:activeTab};selectedPreset=item.id;restoreDraft(structuredClone(item.draft));scheduleSave();message(tr('模板已填入编辑区，尚未启动。可以撤销本次套用。','Template applied to the draft; no service started. You can undo this change.'));};
          const menu=make('details',{class:'l-template-menu'});menu.append(make('summary',{},tr('管理','Manage')));
          for(const [mode,zh,en] of [['update','用当前设置更新','Update from current settings'],['rename','重命名','Rename'],['delete','删除','Delete']]){const action=make('button',{type:'button',class:'s-button','data-template-edit':mode},tr(zh,en));action.onclick=()=>templateEditor(mode,item);menu.append(action);}
          wrapper.append(button,menu);list.append(wrapper);
        }
      }catch(error){list.append(make('p',{class:'l-error'},error.message));}
    }
    panel.append(list);
    if (presetUndo) {
      const undo = make('button', {type:'button', class:'s-button l-secondary', 'data-preset-undo':''}, tr('撤销套用','Undo template'));
      undo.disabled = busy;
      undo.onclick = () => {
        if (busy) return;
        const previous = presetUndo;
        presetUndo = null; selectedPreset = previous.selected; activeTab = previous.tab;
        restoreDraft(previous.draft);scheduleSave();
        message(tr('已恢复套用模板前的配置。','Restored the configuration before applying the template.'));
        root.querySelector('[data-param=Backend]').focus();
      };
      panel.append(undo);
    }
    const log=side.querySelector('.l-log');if(log)side.insertBefore(panel,log);else side.append(panel);
  }
  function render(p){
    if(configLocked&&runningParameters)p=runningParameters;
    // Migrate old templates in memory; only canonical fields are submitted.
    p={...p,SpeculativeMode:p.SpeculativeMode??(p.MtpEnabled===false?'none':Number(p.MtpDrafts)>0?'mtp':'none'),
      DraftTokens:p.DraftTokens??p.MtpDrafts??(p.Backend==='polystrata'?2:4),DraftPolicy:p.DraftPolicy??(p.AdaptiveMtp?'adaptive':'fixed')};
    const legacyScoring=!p.QueryScorer&&('NoLrfu' in p||'LrfuLambda' in p||'LrfuTopFraction' in p);
    const scorer=p.QueryScorer||(legacyScoring&&(!p.NoLrfu||p.ScoreAggregation==='max'||'LrfuLambda' in p||'LrfuTopFraction' in p)?'meanq-cosine':'page-softmax');
    p={...p,QueryScorer:scorer,ScoreAggregation:p.ScoreAggregation??(scorer==='meanq-cosine'?'max':'mean'),
      GraphAlpha:p.GraphAlpha??(scorer==='meanq-cosine'?0:0.2),Lrfu:p.Lrfu??('NoLrfu' in p?!p.NoLrfu:scorer==='meanq-cosine'),
      QueryMaxTokens:p.QueryMaxTokens??512,QueryReplayMaxTokens:p.QueryReplayMaxTokens??2048};
    let refreshGpuOrder=()=>{};
    const b=p.Backend,n=b==='ninfer',l=b==='llamacpp',poly=b==='polystrata',rad=b==='radiance';
    if(n)p.CheckpointExtraRecomputeTokens??=0;
    // Resolve the launcher's omitted values before rendering; keep explicit 0/false.
    const concreteDefaults={Prefill:poly?512:defaults.Prefill,ThinkingBudget:0,Concurrency:1,VisionTokens:1024,NgramDrafts:0,NgramMinMatch:12,
      DeviceProfile:'off',LogLevel:'info',QueryMaxTokens:512,ReplayMaxTokens:2048};
    for(const [key,value] of Object.entries(concreteDefaults))if(p[key]===undefined||p[key]===null||p[key]==='')p[key]=value;
    if(p.RetainedSessions===undefined||p.RetainedSessions===null||p.RetainedSessions==='')p.RetainedSessions=poly?Math.max(4,Number(p.Concurrency)+1):4;
    if(!p.ThinkingMode||p.ThinkingMode==='default')p.ThinkingMode=(n||rad)?'default':poly?(p.ReasoningEffort==='none'?'off':'on'):'off';
    if(poly&&!p.ReasoningEffort)p.ReasoningEffort='medium';
    if(poly&&p.ReasoningEffort==='high')p.ReasoningEffort='xhigh';
    if(poly&&p.SpeculativeMode==='mtp')p.DraftPolicy='adaptive';
    if(poly&&p.PreserveThinking===undefined)p.PreserveThinking=true;
    if(poly&&p.BlockTokens===undefined)p.BlockTokens=32;
    if(poly&&p.VramReserveMiB===undefined)p.VramReserveMiB=256;
    if(rad&&p.VramReserveMiB===undefined)p.VramReserveMiB=1024;
    closeHelp();root.replaceChildren();
    const heading=make('div',{class:'l-heading l-page-heading'}),intro=make('div',{class:'l-intro'}),savedState=make('button',{type:'button',class:'l-save-state l-muted','data-draft-state':'','aria-live':'polite'});savedState.onclick=saveEditing;intro.append(make('h1',{},tr('启动配置','Launch configuration')),savedState);heading.append(intro);root.append(heading);
    const layout=make('div',{class:'l-layout'}),main=make('div',{class:'l-main'}),side=make('aside',{class:'l-side'});layout.append(main,side);root.append(layout);
    const form=make('form',{'aria-label':tr('启动配置','Launch configuration')});form.style.display='contents';main.append(form);form.addEventListener('submit',e=>e.preventDefault());
    const core=make('section',{class:'s-card l-essential'});core.append(make('h2',{class:'s-title'},tr('模型与资源','Model & resources')));const fields=make('div',{class:'l-fields l-core'});core.append(fields);form.append(core);
    const backend=field(fields,p,'Backend','推理引擎','Backend','text',Object.entries(names).filter(([v])=>info.backends.length!==1||info.backends.includes(v)).map(([v,t])=>[v,t]),['按 KVMem 当前支持范围区分：Dense 为稠密模型，MoE 为混合专家模型。PolyStrata 目前支持 Qwen3.5 MoE（包括 Tiel）。','Labels describe current KVMem support: Dense for dense models, MoE for mixture-of-experts models. PolyStrata currently supports Qwen3.5 MoE, including Tiel.']);Array.from(backend.options).forEach(o=>{o.disabled=!info.backends.includes(o.value);if(o.disabled)o.textContent+=' · '+tr('未安装','not installed');});
    modelPicker(fields,p);
    fields.append(make('section',{class:'l-model-health l-wide','data-model-health':'','aria-label':tr('模型体检','Model inspection')}));
    const gpu=field(fields,p,'Gpu','推理显卡','Inference GPUs','text',info.gpus.map(g=>[g.uuid,g.name+' · '+g.uuid.slice(-8)]),['选择要使用的显卡。多卡按勾选顺序排列，可前移／后移；切换引擎保留当前选择。','Choose the GPUs to use. Multiple GPUs follow selection order and can be reordered. Switching engines retains your selection.']);
    gpu.closest('label').classList.add('l-gpu-field');
    gpu.multiple=true;gpu.size=2;Array.from(gpu.options).forEach(o=>o.selected=(p.Gpu||'').split(',').includes(o.value));
    gpu.required=true;
    if(!info.gpus.length){gpu.replaceChildren(make('option',{value:''},tr('未检测到显卡','No GPU detected')));}
    if(info.gpus.length){
      const old=gpu.closest('label'),box=make('div',{class:old.className});
      box.append(...old.childNodes);old.replaceWith(box);
      gpu.hidden=true;gpu.required=false;gpu.setAttribute('aria-hidden','true');
      const choices=make('div',{class:'l-gpu-choices',role:'group','aria-label':tr('推理显卡','Inference GPUs')});
      const checks=[],options=new Map(Array.from(gpu.options).map(option=>[option.value,option]));
      let order=[...new Set((p.Gpu||'').split(',').filter(id=>options.has(id)))];
      const ordered=make('div',{class:'l-gpu-order','data-gpu-order':'','aria-label':tr('显卡执行顺序','GPU execution order')});
      const gpuLimit=make('p',{class:'l-muted l-gpu-limit','data-gpu-limit':'',hidden:''});
      function clearLayers(){
        let cleared=false;
        for(const key of ['StageLayers','TensorSplit']){const input=root.querySelector(`[data-param=${key}]`);if(input?.value.trim()){input.value='';cleared=true;}}
        if(cleared)message(tr('显卡选择已改变，原分配已清空，请重新设置。','GPU selection changed. Previous allocation cleared; configure it again.'));
      }
      for(const device of info.gpus){
        const option=Array.from(gpu.options).find(x=>x.value===device.uuid);
        const card=make('label',{class:'l-gpu-choice'}),check=make('input',{type:'checkbox','data-gpu-choice':device.uuid});
        check.checked=option.selected;
        const caption=make('span');caption.append(make('strong',{},device.name.replace(/^NVIDIA GeForce /,'')),make('small',{},'UUID · '+device.uuid.slice(-8)));
        card.append(check,caption);choices.append(card);checks.push(check);
        check.onchange=()=>{clearLayers();order=check.checked?[...order.filter(id=>id!==device.uuid),device.uuid]:order.filter(id=>id!==device.uuid);sync();gpu.dispatchEvent(new Event('input',{bubbles:true}));};
      }
      function drawOrder(){
        ordered.replaceChildren();ordered.hidden=order.length<2;
        if(order.length<2)return;
        ordered.append(make('span',{class:'l-muted'},tr('执行顺序 · 从前到后','Execution order · first to last')));
        const input=root.querySelector('[data-param=StageLayers]'),layers=(input?.value||'').split(',').map(x=>x.trim());
        const valid=layers.length===order.length&&layers.every(x=>/^[1-9]\d*$/.test(x));
        const tensor=root.querySelector('[data-param=TensorParallel]'),split=root.querySelector('[data-param=TensorSplit]');
        const heads=!!tensor?.checked&&!tensor.disabled,weights=(split?.value||'').split(',').map(x=>x.trim());
        const validWeights=weights.length===order.length&&weights.every(x=>x&&Number.isFinite(Number(x))&&Number(x)>0)&&Number.isFinite(weights.reduce((total,x)=>total+Number(x),0));
        order.forEach((id,i)=>{
          const device=info.gpus.find(g=>g.uuid===id),row=make('div',{class:'l-gpu-order-row','data-ordered-gpu':id,tabindex:'-1'});
          const allocation=rad?tr('张量并行','Tensor parallel'):heads?(validWeights?tr('权重 '+weights[i],'Weight '+weights[i]):split?.value.trim()?tr('待校验','Needs validation'):tr('自动分配','Automatic')):(valid?tr(layers[i]+' 层',layers[i]+' layers'):input?.value.trim()?tr('待校验','Needs validation'):tr('自动分配','Automatic'));
          row.append(make('span',{class:'l-gpu-position'},String(i+1)),make('span',{class:'l-gpu-order-name'},device.name.replace(/^NVIDIA GeForce /,'')+' · '+id.slice(-8)),make('small',{'data-gpu-layers':''},allocation));
          for(const [delta,zh,en] of [[-1,'前移','Move earlier'],[1,'后移','Move later']]){
            const move=make('button',{type:'button',class:'s-button','data-gpu-move':String(delta),'aria-label':tr(zh,en)+' · '+device.name+' · '+id.slice(-8)},tr(zh,en));
            move.disabled=i+delta<0||i+delta>=order.length;
            move.onclick=()=>{
              const target=i+delta;
              if(input?.value.trim()){
                if(valid){[layers[i],layers[target]]=[layers[target],layers[i]];input.value=layers.join(',');}
                else{input.value='';message(tr('原层数分配不完整，已清空；请按新的显卡顺序重新设置。','Incomplete layer allocation cleared; configure it for the new GPU order.'));}
              }
              const split=root.querySelector('[data-param=TensorSplit]');
              if(split?.value.trim()){const weights=split.value.split(',').map(x=>x.trim());if(weights.length===order.length&&weights.every(x=>x&&Number.isFinite(Number(x))&&Number(x)>0)){[weights[i],weights[target]]=[weights[target],weights[i]];split.value=weights.join(',');}else{split.value='';message(tr('原 TP 权重不完整，已清空，请重新设置。','Incomplete TP weights cleared; configure them again.'));}}
              [order[i],order[target]]=[order[target],order[i]];sync();gpu.dispatchEvent(new Event('input',{bubbles:true}));
              Array.from(ordered.querySelectorAll('[data-ordered-gpu]')).find(el=>el.dataset.orderedGpu===id)?.focus();
            };
            row.append(move);
          }
          ordered.append(row);
        });
      }
      function sync(){
        for(const id of [...order,...info.gpus.map(g=>g.uuid).filter(id=>!order.includes(id))]){const option=options.get(id);gpu.append(option);option.selected=order.includes(id);}
        checks.forEach(check=>{check.checked=order.includes(check.dataset.gpuChoice);check.closest('label').classList.toggle('is-selected',check.checked);});
        const maxDevices=rad?(info.backend_capabilities?.[b]?.supported_combinations?.text?.max_devices||1):(l||poly)?1:8;
        const tooMany=order.length>maxDevices;
        const foreignGpu=rad&&order.some(id=>{const vendor=info.gpus.find(g=>g.uuid===id)?.vendor;return vendor&&vendor.toLowerCase()!=='amd';});
        const reason=tooMany?(rad?tr(`当前后端最多支持 ${maxDevices} 张显卡。请调整选择后启动。`,`This backend supports up to ${maxDevices} GPUs. Adjust the selection before starting.`):tr('当前后端仅接入单卡。已保留原选择，请只勾选一张显卡后再启动。','This backend currently exposes one GPU. Your selection is retained; select just one GPU before starting.')):foreignGpu?tr('Radiance 需要 AMD 显卡，请更改当前选择。','Radiance requires an AMD GPU; change the selected device.'):!order.length?tr('请至少选择一张显卡','Select at least one GPU'):'';
        gpuLimit.textContent=reason;gpuLimit.hidden=!reason;gpuLimit.classList.toggle('l-error',tooMany||foreignGpu);
        checks[0].setCustomValidity(reason);choices.setAttribute('aria-invalid',String(!!reason));drawOrder();
      }
      gpu.addEventListener('change',()=>{order=Array.from(gpu.selectedOptions).map(option=>option.value);sync();});sync();gpu.after(choices);box.append(ordered,gpuLimit);
      refreshGpuOrder=drawOrder;
    }
    field(fields,p,'Context','最大上下文 · tokens','Context limit · tokens','number');
    const outputLimit=field(fields,p,'MaxTokens','默认最大输出 · tokens','Default output limit · tokens','number',null,['聊天请求可以覆盖默认输出预算。温度、top-p 等采样参数由聊天请求设置。','Chat requests can override the default output budget. Temperature, top-p and other sampling controls are set by chat requests.']);
    const outputMode=make('select',{'data-output-mode':'','aria-label':tr('输出上限模式','Output limit mode')});
    for(const [value,label] of [['unlimited',tr('不限制','Unlimited')],['limited',tr('指定上限','Token limit')]])outputMode.append(make('option',{value},label));
    outputMode.value=p.MaxTokens!==''&&Number(p.MaxTokens)===0?'unlimited':'limited';
    outputLimit.before(outputMode);outputLimit.hidden=outputMode.value==='unlimited';
    outputLimit.addEventListener('input',()=>{outputMode.value=outputLimit.value!==''&&Number(outputLimit.value)===0?'unlimited':'limited';outputLimit.hidden=outputMode.value==='unlimited';});
    outputMode.onchange=()=>{outputLimit.value=outputMode.value==='unlimited'?'0':String(Number(outputLimit.value)||16384);outputLimit.hidden=outputMode.value==='unlimited';outputLimit.dispatchEvent(new Event('input',{bubbles:true}));};
    if(poly)field(fields,p,'BlockTokens','KV 管理块大小 · tokens','KV block size · tokens','number',[[32,'32'],[64,'64']],['启动时固定；32 提供更细的召回粒度，并增加统计与评分开销。','Fixed at startup. 32 enables finer recall and increases statistics and scoring overhead.']);
    const budgetInput=field(fields,p,'Budget','GPU 历史 KV 预算 · tokens','GPU history KV budget · tokens','number',null,['每条活动请求选中的历史 KV，不含生成预留空间；不是整卡显存上限','Selected history KV per active request, excluding append workspace; not a device VRAM limit']);
    if(p.Backend==='ninfer'&&info.budget_adjustment!==false){
      const row=make('div',{class:'l-budget-value'}),button=make('button',{type:'button',class:'s-button l-budget-auto','data-budget-auto':''},tr('自动调整','Auto adjust'));
      budgetInput.replaceWith(row);row.append(budgetInput,button);
      help(row,tr('自动调整 budget','Automatically adjust budget'),tr('根据其他设置自动调整 budget：保持当前 gen reserve、KV 类型等设置，加载一次模型并测算带显存余量的最大 budget。仅填回 budget，不启动服务；仍需实际启动验证。','Adjust budget from the other settings: keep gen reserve, KV type and other options, load the model once and estimate the largest budget with memory headroom. Fill in budget without starting the service; startup validation is still needed.'),button);
      row.after(make('small',{'data-budget-status':'',role:'status','aria-live':'polite',hidden:''}));
      button.onclick=event=>{event.preventDefault();adjustBudget();};
    }
    field(fields,p,'Reserve','生成预留空间 · tokens','Generation reserve · tokens','number',null,['每条活动请求的追加 KV 工作空间；可循环复用，输出预算可以超过此值','Append KV workspace per active request; reusable, so output budget may exceed this value']);
    const resourceFields=make('div',{class:'l-resource-grid l-wide'});
    for(const key of ['Context','MaxTokens','Budget','Reserve']){
      const label=fields.querySelector(`[data-param=${key}]`).closest('label');
      label.classList.add('l-resource-field');resourceFields.append(label);
    }
    fields.append(resourceFields);
    const section=make('section',{class:'s-card'}),tabs=make('div',{class:'l-tabs',role:'tablist','aria-label':tr('共性参数分组','Common settings groups')});section.append(tabs);form.append(section);const panels=[];
    const visibleGroups=poly?[...groups,['弹性专家','Elastic experts']]:groups;
    if(activeTab>=visibleGroups.length)activeTab=1;
    visibleGroups.forEach((g,i)=>{const tab=make('button',{type:'button',class:'l-tab',role:'tab',id:'launch-tab-'+i,'aria-selected':String(activeTab===i),'aria-controls':'launch-panel-'+i},tr(...g));tabs.append(tab);const panel=make('div',{class:'l-panel',id:'launch-panel-'+i,role:'tabpanel','aria-labelledby':tab.id});panel.hidden=activeTab!==i;const grid=make('div',{class:'l-fields'});panel.append(grid);panels.push(grid);section.append(panel);tab.onclick=()=>{activeTab=i;tabs.querySelectorAll('button').forEach((x,j)=>x.setAttribute('aria-selected',String(i===j)));section.querySelectorAll('.l-panel').forEach((x,j)=>x.hidden=i!==j);};});
    field(panels[0],p,'HostMiB','RAM 历史缓存 · MiB','RAM history cache · MiB','number',null,['活动与空闲历史共享的原生 KV 数据硬预算；不含模型、统计索引等其他内存；1024 MiB = 1 GiB','Shared hard budget for active and idle native KV data; excludes model weights, statistics and other memory; 1024 MiB = 1 GiB']);
    if(l){
      const kv=['q8_0','q5_0','q4_0','f16','f32'],aliases={fp16:'f16',fp32:'f32',q8:'q8_0',q5:'q5_0',q4:'q4_0'};
      const canonical=value=>aliases[value]||value;
      const types={KvTypeK:canonical(p.KvTypeK||p.KvType||'q8_0'),KvTypeV:canonical(p.KvTypeV||p.KvType||'q8_0')};
      const kvField=make('div',{class:'l-field l-row l-kv-field',role:'group','aria-label':tr('KV 格式','KV format')});
      const kvTitle=make('span',{},tr('KV 格式','KV format')),pair=make('div',{class:'l-kv-pair'});
      const keyType=field(pair,types,'KvTypeK','K 缓存格式','K cache format','text',kv.map(v=>[v,v]));
      const valueType=field(pair,types,'KvTypeV','V 缓存格式','V cache format','text',kv.map(v=>[v,v]));
      for(const [input,side] of [[keyType,'K'],[valueType,'V']]){const label=input.closest('label');label.className='l-kv-side';label.firstChild.textContent=side;}
      keyType.required=true;valueType.required=true;
      help(kvTitle,tr('KV 格式','KV format'),tr('K 和 V 可分别选择：q8_0、q5_0、q4_0 可互配；f16、f32 可互配。浮点与量化不能混搭。','Choose K and V independently: q8_0, q5_0 and q4_0 can be combined; f16 and f32 can be combined. Floating-point and quantized types cannot be mixed.'));
      kvField.append(kvTitle,pair,make('p',{class:'l-muted l-error l-kv-status','data-kv-pair-status':'','aria-live':'polite',hidden:''}));panels[0].append(kvField);
    }else{
      const kv=rad?['fp8','bf16']:poly?['f16','q8_0']:['int8','bf16','fp8','nvfp4','k8v4','rk8v4','rk4v4','rk4v4-e8','rk2v4-e8'];
      field(panels[0],{...p,KvType:p.KvType||kv[0]},'KvType','KV 格式','KV format','text',kv.map(v=>[v,v]));
    }
    // UI state only: the worker still receives the existing path/budget pair.
    const diskEnabled=field(panels[0],{DiskEnabled:typeof p.DiskEnabled==='boolean'?p.DiskEnabled:!!p.DiskPath?.trim()&&p.DiskMiB!==0},'DiskEnabled','启用 SSD 缓存','Enable SSD cache','checkbox',null,['关闭后不传 SSD 参数；目录与预算仍保存在草稿和模板中，重新开启时可继续使用。','When off, no SSD arguments are passed. The directory and budget stay in drafts and templates for later reuse.']);
    diskEnabled.dataset.uiOnly='true';diskEnabled.closest('label').classList.add('l-wide');
    field(panels[0],p,'DiskPath','SSD 缓存目录','SSD cache directory','text',null,['开启 SSD 缓存时必填；配合正的 SSD 缓存预算使用。','Required when SSD cache is enabled, together with a positive SSD cache budget.']);
    const diskBudget=field(panels[0],{DiskMiB:4096,...p},'DiskMiB','SSD 缓存预算 · MiB','SSD cache budget · MiB','number');
    diskBudget.min='1';
    field(panels[0],p,'SinkTokens','保留开头 · tokens','Keep prefix · tokens','number');field(panels[0],p,'RecentTokens','保留最近 · tokens','Keep recent · tokens','number');
    const retained=field(panels[0],{...p,RetainedSessions:poly?Math.max(0,p.RetainedSessions-1):p.RetainedSessions},'RetainedSessions','额外保留的空闲会话','Idle sessions to retain','number',null,['不含活动会话；PolyStrata 当前运行槽之外的历史保留目标，实际数量还受共享内存预算约束。','Excludes active sessions. Retention is also bounded by the shared memory budget.']);
    if(poly){retained.dataset.idleSessions='true';retained.max='15';}
    const recallSection=(zh,en,badgeZh,badgeEn,noteZh,noteEn)=>{
      const box=make('section',{class:'l-recall-section l-wide'}),head=make('div',{class:'l-recall-heading'}),title=make('h3',{class:'s-title'},tr(zh,en));
      help(title,tr(zh,en),tr(noteZh,noteEn));
      head.append(title,make('span',{class:'l-recall-badge'},tr(badgeZh,badgeEn)));
      const fields=make('div',{class:'l-fields'});box.append(head,fields);panels[5].append(box);return fields;
    };
    panels[5].classList.add('l-recall-panel');
    const scoring=recallSection('历史检索','History retrieval','按问题召回','Query recall',
      '按当前问题召回相关历史。评分、图传播与历史累计决定历史内容的相关性。',
      'Recall history relevant to the current question. Scoring, graph propagation and accumulated history determine relevance.');
    scoring.parentElement.setAttribute('data-graph-recall','');
    const retention=recallSection('驻留与会话保留','Resident & session retention','注意力保留','Attention retention',
      '空间不足时，按累计注意力保留驻留内容；开头与最近内容按下方预算保护。空闲会话保留还受共享内存预算限制。',
      'Under memory pressure, retain resident content by accumulated attention. Protect the prefix and recent tokens using the budgets below. Idle sessions also share the memory budget.');
    ['SinkTokens','RecentTokens','RetainedSessions'].forEach(k=>{const el=panels[0].querySelector(`[data-param="${k}"]`);if(el)retention.append(el.closest('label'));});
    if(n){const extra=field(retention,p,'CheckpointExtraRecomputeTokens','快照额外重算上限 · tokens','Checkpoint extra recompute limit · tokens','number',null,['0 关闭。允许用附近较早点替代低收益快照；执行中的 Query 回放点仍受保护。重试可能需要重新保存快照或读取 SSD，因此该值不是耗时上限。','0 disables this optimization. Nearby earlier states can replace low-value checkpoints; active Query replay points remain protected. Retries may rebuild checkpoints or read SSD, so this is not a latency limit.']);extra.min='0';extra.max='2147483647';extra.required=true;}
    if(legacyScoring)scoring.before(make('p',{class:'l-muted','data-scoring-migration':''},tr('已将旧检索设置转换为明确参数；请检查下方评分器和历史累计选项。','Legacy retrieval settings were converted to explicit controls. Review the scorer and LRFU settings below.')));
    field(scoring,p,'QueryScorer','召回评分算法','Recall scoring algorithm','text',[['page-softmax','Page softmax'],['meanq-cosine','Mean-Q cosine']],['评分器、图传播与历史累计可以独立组合。','Scorer, graph propagation and LRFU can be combined independently.']);
    field(scoring,p,'ScoreAggregation','分组查询聚合','GQA score aggregation','text',[['mean',tr('平均','Mean')],['max',tr('最大','Max')]]);
    const alpha=field(scoring,p,'GraphAlpha','图传播强度','Graph strength','number',null,['范围 0–1；0 关闭传播。cosine 仅传播正向证据，负分保留在原页。','Range 0–1; 0 disables propagation. Cosine transports positive evidence and retains negative residuals locally.']);alpha.max='1';alpha.step='any';alpha.required=true;
    const history=field(scoring,p,'Lrfu','历史累计 · LRFU','LRFU history','checkbox',null,['对历次问题的重要性做衰减累计。top-fraction 只筛正向奖励，负分仍保留。','Accumulate decayed relevance across queries. Top-fraction gates positive rewards while retaining negative evidence.']);history.dataset.explicitBoolean='true';
    const decay=field(scoring,{LrfuLambda:0.5,...p},'LrfuLambda','历史衰减系数','History decay','number');decay.max='1';decay.step='any';decay.required=true;
    const fraction=field(scoring,{LrfuTopFraction:0.9,...p},'LrfuTopFraction','正向奖励候选比例','Positive reward fraction','number');fraction.min='0';fraction.max='1';fraction.step='any';fraction.required=true;
    for(const [key,zh,en,note] of [['QueryMaxTokens','问题评分上限 · tokens','Query score limit · tokens',['仅使用当前问题末尾的这些 tokens 评分。','Score with at most this many trailing tokens of the current question.']],['QueryReplayMaxTokens','完整回放上限 · tokens','Causal replay limit · tokens',['超过上限时保持驻留视图并使用 Pressure；不会截断因果回放。','Above this limit, retain the resident view with Pressure; causal replay is not truncated.']]]){const limit=field(scoring,p,key,zh,en,'number',null,note);limit.min='1';limit.required=true;}
    field(panels[1],p,'Prefill','预填充块大小 · tokens','Prefill chunk · tokens','number');
    if(poly||rad){const vram=field(panels[1],p,'VramReserveMiB','显存预留 · MiB','VRAM reserve · MiB','number',null,['原生引擎最低为 256 MiB；与生成追加空间的 token 预算分别设置。','Worker minimum: 256 MiB. Separate from the generation workspace measured in tokens.']);vram.min='256';vram.max='65536';}
    const tensor=field(panels[1],p,'TensorParallel','张量并行 · heads','Tensor parallelism · heads','checkbox',null,
      ['将每层计算分配到多张显卡。默认关闭；启用后只使用 heads 注意力。','Distribute each layer across GPUs. Off by default; uses heads attention.'],!n);
    const tensorSplit=field(panels[1],{...p,TensorSplit:(p.TensorSplit||[]).join(',')},'TensorSplit','TP 权重（可选）','TP weights (optional)','text',null,
      ['依次对应上方显卡，例如 3,1；留空由后端根据可用显存和连接情况分配。实际比例受 head 和量化分块约束。','In GPU order, e.g. 3,1. Empty lets the backend allocate using available memory and links. Heads and quantization constrain the actual split.'],!n);
    tensor.addEventListener('change',refreshGpuOrder);tensorSplit.addEventListener('input',refreshGpuOrder);
    const layerSplit=field(panels[1],{...p,StageLayers:(p.StageLayers||[]).join(',')},'StageLayers','多卡层数分配（可选）','Layer split (optional)','text',null,[n?'依次对应上方第 1、2…张显卡，用逗号分隔；留空由后端分配':'当前接入暂不支持设置多卡层数',n?'Comma-separated counts for GPU 1, 2, … above; empty uses backend allocation':'Layer assignment is unavailable in this integration'],!n);
    layerSplit.addEventListener('input',refreshGpuOrder);refreshGpuOrder();
    const specModes=info.speculative_capabilities?.[b]||{none:{},...(n?{mtp:{max_draft_tokens:15,adaptive:true,ngram:true,max_concurrency:8,max_devices:8,draft_asset:'embedded'}}:{})};
    const specMode=field(panels[1],p,'SpeculativeMode','草稿方式','Draft method','text',[
      ['none',tr('关闭','Off')],['mtp','MTP'],['dflash2','DFlash2']
    ],poly?['单请求执行可使用 MTP；进入批处理时使用普通生成。','MTP is available in solo execution; batches use ordinary decode.']:['需要模型支持。可用选项取决于后端、并发和显卡数量。','Requires model support. Availability depends on backend, concurrency and GPU count.']);
    for(const option of specMode.options)option.disabled=!specModes[option.value];
    const draftTokens=field(panels[1],p,'DraftTokens','草稿长度 · tokens','Draft length · tokens','number',null,
      ['主模型会验证草稿；实际长度还受草稿模型限制。DFlash2 首次选择使用 5。','The target verifies the drafts. The draft model can impose a smaller limit. DFlash2 starts at 5.']);
    draftTokens.min='1';
    specMode.addEventListener('change',()=>{if(specMode.value==='dflash2')draftTokens.value='5';else if(specMode.value!=='none'&&Number(draftTokens.value)<=0)draftTokens.value=String(rad?3:poly?2:4);});
    field(panels[1],p,'DraftPolicy','草稿长度策略','Draft length policy','text',[
      ['fixed',tr('固定','Fixed')],['adaptive',tr('自动调整','Adaptive')]
    ],['自动调整仅在后端和所选草稿方式支持时生效。','Adaptive mode applies only when supported by the backend and draft method.']);
    const draftModel=field(panels[1],p,'DraftModel','草稿模型 GGUF','Draft model GGUF','text',null,
      ['llama.cpp 的 DFlash2 需要与主模型匹配的独立草稿文件；NInfer 使用模型内置草稿组件。','DFlash2 on llama.cpp requires a separate draft matching the target. NInfer uses an embedded draft component.'],n||poly||rad);
    if(!n&&!poly&&!rad&&info.native_picker!==false){const row=inputActions(draftModel),choose=make('button',{type:'button',class:'s-button l-secondary'},tr('选择文件','Choose file'));row.append(choose);choose.onclick=async()=>{try{const result=await api('browse-model',{backend:'llamacpp',current_path:draftModel.value});if(result.path){draftModel.value=result.path;draftModel.dispatchEvent(new Event('change',{bubbles:true}));}}catch(e){message(e.message,true);}};}
    if(n)field(panels[1],p,'FastPrefill','启用快速预填充','Enable fast prefill','checkbox',null,['支持 int8、rk8v4、rk4v4、rk4v4-e8、rk2v4-e8；还需模型支持。与 MTP、普通预填充块大小分别设置。','Supports int8, rk8v4, rk4v4, rk4v4-e8 and rk2v4-e8; model support is also required. Independent of MTP and the standard prefill chunk size.'],!n);
    const thinking=field(panels[2],p,'ThinkingMode','默认思考模式','Default thinking mode','text',[[(n||rad)?'default':'on',(n||rad)?tr('跟随模型','Model default'):tr('开启','On')],['off',tr('关闭','Off')]],['需要模型与模板支持。聊天请求中的显式设置优先。','Requires model/template support. Explicit chat settings take priority.']);
    const thinkingBudget=field(panels[2],p,'ThinkingBudget','默认思考预算 · tokens','Default thinking budget · tokens','number',null,poly?['0：根据输出上限自动给答案留空间；正数：思考 token 上限，也受总输出预算约束。','0: automatically reserve answer space within the output limit. Positive values cap reasoning tokens, subject to the total output budget.']:['0：不设置单独的思考限额，仍受总输出预算约束。正数：思考 token 上限。','0: no separate reasoning cap; the total output budget still applies. Positive values cap reasoning tokens.']);
    thinkingBudget.required=true;
    field(panels[2],p,'ReasoningEffort','默认思考强度（可选）','Default reasoning effort (optional)','text',[
      ...(!poly?[['',tr('不附加强度指令','No additional effort instruction')]]:[]),...(poly?['none','low','medium','xhigh']:['none','minimal','low','medium','high','xhigh','max']).map(x=>[x,x])
    ],['默认思考强度，具体支持取决于模型；与思考 token 预算分别设置。','Default reasoning effort; supported levels depend on the model. Separate from the thinking-token budget.']);
    if(n){
      const preserve=field(panels[2],{PreserveThinking:p.PreserveThinking===true?'true':'false'},'PreserveThinking','历史思考','Previous reasoning','text',[['true',tr('开启','On')],['false',tr('跟随模型','Model default')]],['当前 NInfer 启动接口只提供强制保留开关；跟随模型不等于关闭。聊天请求仍可单独覆盖。','NInfer only provides a force-preserve startup switch. Model default does not mean off. Chat requests can override it.']);preserve.dataset.booleanSelect='true';
    }else{
    field(panels[2],p,'PreserveThinking','保留历史思考','Preserve previous reasoning','checkbox',null,poly?['保留客户端回传的已结束轮次思考；内置聊天页自动回传。不会按回答文本猜测或补回缺失思考。','Retain reasoning supplied by the client. Built-in chat sends it automatically; missing reasoning is never guessed from answer text.']:['在后续提示中保留已结束轮次的 assistant 思考内容。','Retain closed-turn assistant reasoning in later prompts.']);
    }
    field(panels[3],p,'Vision','启用图像输入','Enable vision','checkbox',null,['需要模型及配套资源支持','Requires compatible model and resources']);
    field(panels[3],p,'VisionTokens','图像 token 上限（可选）','Image token limit (optional)','number',null,['开启图像输入时生效。','Applies when vision input is enabled.']);field(panels[3],{VisionResidency:'resident',...p},'VisionResidency','图像编码器存放位置','Vision encoder placement','text',[['resident','GPU'],['cpu','CPU']],['开启图像输入时生效；关闭后保留原选择。','Applies when vision input is enabled; the selection is retained when off.']);
    templateFileField(panels[3],p);
    field(panels[4],{ListenHost:'127.0.0.1',...p},'ListenHost','监听地址','Listen address','text',[['127.0.0.1',tr('127.0.0.1 · 仅本机','127.0.0.1 · This computer')],['0.0.0.0',tr('0.0.0.0 · 所有网卡','0.0.0.0 · All interfaces')]],['选择所有网卡后，其他设备可通过本机 IP 和端口访问推理服务','All interfaces allows other devices to access inference through this computer’s IP and port']);field(panels[4],p,'Port','服务端口','Service port','number');
    const concurrency=field(panels[4],p,'Concurrency','同时处理的请求数（可选）','Simultaneous requests (optional)','number',null,(n||rad)?['范围 1–8；留空为 1。并发增加会提高 KV 和显存需求。','Range 1–8; empty uses 1. More concurrent requests require more KV and VRAM.']:['范围 1–8；留空为 1。当前控制台使用单卡；并发增加会提高 KV 和显存需求。','Range 1–8; empty uses 1. This console uses one GPU; more concurrent requests require more KV and VRAM.']);concurrency.min='1';concurrency.max='8';
    credentialField(panels[4],p);
    const serviceAdvanced=make('details',{class:'l-advanced l-wide'});serviceAdvanced.append(make('summary',{},tr('高级 · 服务标识与日志','Advanced · Service identity & logs')));const serviceFields=make('div',{class:'l-fields l-service-fields'});serviceAdvanced.append(serviceFields);panels[4].append(serviceAdvanced);
    field(serviceFields,p,'ModelAlias','对外模型名（可选）','Served model name (optional)','text',null,poly?['供 API 客户端使用；留空使用 kvmem-polystrata-qwen35。','API model name; empty uses kvmem-polystrata-qwen35.']:['供 API 客户端使用的稳定名称；留空沿用模型名称。','A stable name for API clients; leave empty to inherit the model name.']);
    field(serviceFields,p,'LogLevel','日志级别','Log level','text',[['info','Info'],['debug','Debug'],['trace','Trace'],['warning','Warning'],['error','Error'],['off',tr('关闭日志','Off')]],['通常使用 Info；排查问题时可临时选择 Debug。','Use Info normally; select Debug temporarily for troubleshooting.']);
    if(n){
      const ngram=field(panels[1],p,'NgramDrafts','N-gram 推测 token 数','N-gram draft tokens','number',null,['依赖支持 N-gram 的草稿方式；0 或留空关闭，1–63 启用。并发时后端可能缩小实际推测宽度。','Requires a draft method supporting N-gram. 0 or empty disables it; 1–63 enables it. The backend may reduce its width under concurrency.']);ngram.max='63';
      const minMatch=field(panels[1],p,'NgramMinMatch','N-gram 最短匹配','N-gram minimum match','number',null,['启用 N-gram 时生效，范围 4–64。','Applies when N-gram is enabled; range 4–64.']);minMatch.min='4';minMatch.max='64';
      field(panels[1],p,'DeviceProfile','设备性能配置','Device profile','text',[['off',tr('关闭','Off')],['auto',tr('自动','Auto')]]);

    }
    if(poly||l){
      const assets=make('details',{class:'l-advanced l-wide','data-vision-assets':''});assets.append(make('summary',{},tr('图像配套文件','Vision assets')));
      const grid=make('div',{class:'l-fields l-vision-assets'});assets.append(grid);panels[3].append(assets);
      for(const [key,zh,en] of [...(poly?[['VisionWorker','图像编码程序','Vision encoder executable']]:[]),['Mmproj','图像投影 GGUF','Vision projection GGUF']]){
        const input=field(grid,p,key,zh,en,'text',null,key==='Mmproj'?['选择与主模型配套的 GGUF，文件名不限。留空时优先查找同名配套文件，否则使用同目录唯一的 mmproj GGUF；多个候选时请手动选择。','Choose a GGUF matching the model; any filename is accepted. If empty, prefer a model-named projector, then the only adjacent mmproj GGUF. Select explicitly when ambiguous.']:null);input.placeholder=tr('自动检测','Auto-detect');input.closest('label').classList.remove('l-row');input.title=input.value;
        if(key==='Mmproj'&&info.native_picker!==false&&!info.script_export_version){
          const row=inputActions(input),choose=make('button',{type:'button',class:'s-button l-secondary','data-browse-mmproj':''},tr('选择文件','Choose file'));row.append(choose);
          choose.onclick=async()=>{choose.disabled=true;try{const result=await api('browse-mmproj',{backend:b,current_path:input.value||root.querySelector('[data-param=Model]').value},315000);if(result.path){input.value=result.path;input.title=result.path;input.dispatchEvent(new Event('change',{bubbles:true}));}}catch(e){message(e.message,true);}finally{choose.disabled=false;}};
        }
      }
    }
    const native=make('details',{class:'s-card l-native'});
    const nativeSummary=make('summary',{class:'l-engine-summary'});
    nativeSummary.append(make('span',{},tr('高级设置','Advanced settings')),make('span',{class:'l-engine-badge'},names[b].split(' · ')[0]));native.append(nativeSummary);
    function engineDisclosure(key,zh,en,badge){
      const box=make('details',{class:'l-engine-disclosure',['data-engine-'+key]:''});
      const summary=make('summary',{});summary.append(make('span',{},tr(zh,en)));
      const status=make('span',{class:'l-engine-status'},badge);summary.append(status);box.append(summary);native.append(box);
      const grid=make('div',{class:'l-fields l-engine-fields'});box.append(grid);return {box,grid,status};
    }
    const environment=engineDisclosure('environment','自定义运行环境','Custom runtime',tr('自动配置','Automatic'));
    const ng=environment.grid;
    const detected=info.workers?.[b];
    const worker=field(ng,p,'Worker','后端程序','Backend executable','text',null,
      detected?.available?['留空使用已检测到的程序：'+detected.path,'Leave empty to use the detected executable: '+detected.path]:['未检测到后端程序，请填写已安装程序的完整路径。','No backend executable detected. Enter the full path to an installed executable.']);
    worker.placeholder=tr('自动检测','Auto-detect');worker.closest('label').classList.add('l-wide');
    if(rad){for(const [key,zh,en] of [['RadianceHome','Radiance 资源目录','Radiance resource directory'],['RocmPath','ROCm 运行库目录','ROCm runtime directory']])field(ng,p,key,zh,en);}
    if(poly){
      for(const [key,zh,en,placeholderZh,placeholderEn] of [
        ['Python','Python 程序','Python executable','使用内置环境','Use bundled runtime'],
        ['Tokenizer','Tokenizer 目录','Tokenizer directory','从模型自动准备','Prepare from model'],
        ['PolystrataSource','Frontend 目录','Frontend directory','使用随附的 PolyStrata','Use bundled PolyStrata']
      ]){
        const input=field(ng,p,key,zh,en);input.placeholder=tr(placeholderZh,placeholderEn);
        if(key==='PolystrataSource')input.closest('label').classList.add('l-wide');
      }
      help(environment.box.querySelector('summary > span'),'PolyStrata',tr('这些路径由控制台自动解析，仅在使用自建环境或排查安装问题时覆盖。留空恢复自动配置。','The console resolves these paths automatically. Override only for a custom installation or troubleshooting; leave empty to restore automatic configuration.'));
    }
    for(const input of ng.querySelectorAll('input')){
      input.closest('label').classList.remove('l-row');
      const updateTitle=()=>input.title=input.value;updateTitle();input.addEventListener('input',updateTitle);
    }
    const environmentStatus=()=>{
      const custom=[...ng.querySelectorAll('input')].some(input=>input.value.trim());
      environment.status.textContent=custom?tr('已自定义','Customized'):detected?.available?tr('自动配置','Automatic'):tr('需要配置','Setup needed');
      environment.box.dataset.custom=String(custom);
    };
    ng.addEventListener('input',environmentStatus);environmentStatus();
    const argumentsPanel=engineDisclosure('arguments','原生启动参数','Native launch arguments','JSON');
    const nativeLabel=make('label',{class:'l-field l-wide'}),nativeTitle=make('span',{},tr('参数数组','Argument array'));nativeLabel.append(nativeTitle);
    const raw=make('textarea',{'data-native':'',rows:'4',spellcheck:'false','aria-label':tr('原生选项 · JSON 参数数组','Native options · JSON argument array')});raw.value=JSON.stringify(p.WorkerArgs||[]);nativeLabel.append(raw);
    help(nativeTitle,tr('原生启动参数','Native launch arguments'),tr('仅用于界面未提供的选项。弹性专家等常用设置请使用对应页签；此处保留完整参数供高级检查。允许的选项：','Use for options without a dedicated control. Configure common settings such as elastic experts in their tabs; the full arguments remain available here for advanced inspection. Allowed options: ')+(info.native_options[b].join(', ')||tr('无','None')),raw);
    argumentsPanel.grid.append(nativeLabel);form.append(native);
    if(poly)elasticControls(panels[6],p,raw);
    const actions=make('div',{class:'l-actions l-draft-actions'});const save=make('button',{type:'button',class:'s-button','data-action':'template'},tr('另存为模板','Save as template'));const preview=make('button',{type:'button',class:'s-button','data-action':'preview'},tr('检查配置','Check configuration'));actions.append(preview,save);heading.append(actions);const output=make('pre',{class:'l-result l-notice','data-launch-message':'','aria-live':'polite'});heading.after(output);
    if(info.script_export_version){const generate=make('button',{type:'button',class:'s-button','data-action':'export'},tr('生成启动脚本','Generate launch script'));generate.onclick=exportDialog;actions.append(generate);}
    if(info.script_export_version){const services=make('button',{type:'button',class:'s-button','data-services-open':''},tr('后台服务管理','Background services'));services.onclick=servicesDialog;actions.append(services);}
    const status=make('section',{class:'s-card l-runtime'}),statusTitle=make('h2',{class:'s-title'},tr('当前运行服务','Running service'));status.append(statusTitle,make('p',{class:'l-status s-badge','data-runtime-state':''},tr('读取中','Loading')));help(statusTitle,tr('当前运行服务','Running service'),tr('只管理此控制台启动的服务。停止会卸载模型并清空其内存缓存。','Only services started here are managed. Stopping unloads the model and its in-memory cache.'));const progress=make('div',{class:'l-launch-progress','data-launch-progress':'',hidden:''});progress.append(make('p',{'data-launch-phase':'',role:'status'}),make('div',{class:'l-launch-track',role:'progressbar','data-launch-track':''}),make('small',{class:'l-muted','data-launch-help':''}));status.append(progress,make('dl',{'data-runtime-details':''}));const lifecycle=make('div',{class:'l-actions'});const start=make('button',{type:'button',class:'s-button l-primary','data-action':'start'},tr('启动服务','Start service'));const stop=make('button',{type:'button',class:'s-button','data-action':'stop',disabled:''},tr('停止服务','Stop service'));lifecycle.append(start,stop);status.append(lifecycle);side.append(status);
    progress.after(make('section',{class:'l-memory-recovery','data-memory-recovery':'','aria-label':tr('显存恢复建议','Memory recovery recommendations'),hidden:''}));
    presetPanel(side);
    const log=make('details',{class:'s-card l-log'});log.append(make('summary',{},tr('启动日志','Startup log')));const refresh=make('button',{type:'button',class:'s-button l-secondary'},tr('刷新日志','Refresh log'));log.append(refresh,make('pre',{class:'l-result','data-log':''}));side.append(log);refresh.onclick=async()=>{try{log.querySelector('[data-log]').textContent=(await api('log')).text||tr('暂无日志','No log yet');}catch(e){message(e.message,true);}};
    const liveLog=make('button',{type:'button',class:'s-button l-secondary'},tr('打开实时日志 →','Open live logs →'));log.insertBefore(liveLog,refresh);liveLog.onclick=()=>document.querySelector('[data-page=logs]').click();
    const origin=make('div',{class:'l-environment'});origin.append(make('span',{class:'l-muted'},tr('后端安装目录','Backend checkout')),make('p',{class:'l-muted',style:'overflow-wrap:anywhere'},info.backend_root));log.append(origin);
    backend.onchange=()=>{
      try{
        const target=backend.value,previous=captureDraft();previous.parameters.Backend=b;perBackend[b]=previous;
        const draft=structuredClone(perBackend[target]||{version:2,parameters:{...defaults,...enginePreset(target)}});
        const oldOrder=(draft.parameters.Gpu||'').split(',').filter(Boolean);
        for(const key of sharedSettings){
          // PolyStrata's measured default and explicit edits belong to its own draft.
          if(key==='Prefill'&&(target==='polystrata'||b==='polystrata'))continue;
          if(key==='RetainedSessions'&&Object.hasOwn(previous.parameters,key)){
            draft.parameters[key]=previous.parameters[key]+(target==='polystrata'?1:0)-(b==='polystrata'?1:0);continue;
          }
          if(Object.hasOwn(previous.parameters,key))draft.parameters[key]=previous.parameters[key];
        }
        const newOrder=(draft.parameters.Gpu||'').split(',').filter(Boolean),layers=draft.parameters.StageLayers||[];
        let layersCleared=false;
        if(oldOrder.join(',')!==newOrder.join(',')){
          if(layers.length===oldOrder.length&&layers.every(x=>Number.isInteger(x)&&x>0)&&oldOrder.length===newOrder.length&&newOrder.every(id=>oldOrder.includes(id))){
            draft.parameters.StageLayers=newOrder.map(id=>layers[oldOrder.indexOf(id)]);
            if(draft.raw)draft.raw.StageLayers=draft.parameters.StageLayers.join(',');
          }else{
            layersCleared=!!(draft.raw?.StageLayers?.trim()||layers.length);
            draft.parameters.StageLayers=[];if(draft.raw)draft.raw.StageLayers='';
          }
        }
        if(oldOrder.join(',')!==newOrder.join(',')){
          const weights=typeof draft.raw?.TensorSplit==='string'&&draft.raw.TensorSplit.trim()?draft.raw.TensorSplit.split(',').map(x=>x.trim()?Number(x):NaN):draft.parameters.TensorSplit||[];
          if(weights.length===oldOrder.length&&weights.every(x=>Number.isFinite(x)&&x>0)&&Number.isFinite(weights.reduce((total,x)=>total+x,0))&&oldOrder.length===newOrder.length&&newOrder.every(id=>oldOrder.includes(id))){draft.parameters.TensorSplit=newOrder.map(id=>weights[oldOrder.indexOf(id)]);if(draft.raw)draft.raw.TensorSplit=draft.parameters.TensorSplit.join(',');}
          else{draft.parameters.TensorSplit=[];if(draft.raw)draft.raw.TensorSplit='';}
        }
        selectedPreset=perBackend[target]?'':presets.find(p=>p.parameters.Backend===target).id;restoreDraft(draft);scheduleSave();
        if(layersCleared)message(tr('显卡选择已改变，原层数分配已清空，请重新设置。','GPU selection changed. Previous layer allocation cleared; configure it again.'));
      }catch(e){backend.value=b;message(e.message,true);}
    };
    [save,preview,start,stop].forEach(button=>button.onclick=()=>act(button.dataset.action));
    if(info.script_export_version){start.textContent=tr('生成并启动','Generate and start');root.querySelectorAll('[data-browse-template],[data-browse-model],[data-scan-folder]').forEach(button=>button.hidden=true);}
    if(rad){
      const residency=root.querySelector('[data-param=VisionResidency]');
      if(residency){residency.value='resident';residency.disabled=true;residency.dataset.backendSupported='false';}
      for(const option of root.querySelector('[data-param=LogLevel]').options)option.disabled=!['info','debug','trace'].includes(option.value);
    }
    if(poly){
      for(const key of ['LogLevel']){const input=root.querySelector(`[data-param=${key}]`);if(input){input.disabled=true;input.dataset.backendSupported='false';}}
    }
    syncDependencies();
    scheduleHealth(true);
    updateStatus(runtime);
    draftStatus();
  }
  function updateStatus(status,confirmed=false){runtime=status;if(!status)return;if(confirmed)configUncertain=false;syncRunningConfiguration(status);window.dispatchEvent(new CustomEvent('kvmem-runtime-status',{detail:status}));const running=['starting','ready'].includes(status.state);const label=root.querySelector('[data-runtime-state]');if(!label)return;const translations={stopped:['未启动','Stopped'],starting:['正在加载模型','Loading model'],ready:['服务已就绪','Ready'],exited:['进程已退出','Process exited']};label.textContent=tr(...(translations[status.state]||translations.stopped));label.dataset.state=status.state;const dl=root.querySelector('[data-runtime-details]');dl.replaceChildren();const p=status.active_config?.parameters,o=status.effective_scoring;for(const [zh,en,value] of [['引擎','Backend',p?names[p.Backend]:null],['模型','Model',p?.Model],['监听地址','Listen address',p?p.ListenHost||'127.0.0.1':null],['本机地址','Local URL',status.url],['进程','PID',status.pid],['退出码','Exit code',status.exit_code],['KV 块大小','KV block size',o?.block_tokens],['评分器','Query scorer',o?.scorer],['聚合','Aggregation',o?.agg],['图传播强度','Graph strength',o?.graph_alpha],['历史累计','LRFU',o?.lrfu===undefined?null:o.lrfu?tr('开启','On'):tr('关闭','Off')],['问题／回放上限','Query / replay limits',o?.query_max_tokens===undefined?null:`${o.query_max_tokens} / ${o.query_replay_max_tokens??'—'}`],['配置确认','Configuration',p?(status.scoring_confirmed?tr('服务已确认','Confirmed by service'):status.scoring_mismatch?tr('与启动计划不一致','Differs from launch plan'):tr('等待完整服务报告','Awaiting full service report')):null],['程序校验值','Worker SHA-256',status.worker?.executable_sha256?.slice(0,12)]]){if(value!==null&&value!==undefined){dl.append(make('dt',{},tr(zh,en)),make('dd',{title:String(value)},zh==='模型'?String(value).split(/[\\/]/).pop():String(value)));}}root.querySelector('[data-action=start]').disabled=busy||running||status.budget_planning||configurationLocked()||healthBlocks();root.querySelector('[data-action=stop]').disabled=busy||!running;const chat=document.querySelector('[data-chat-link]');if(info){chat.hidden=false;chat.href='/chat/';chat.removeAttribute('aria-disabled');}renderLaunchProgress();renderMemoryRecovery();renderBudgetAdjustment();}
  function renderBudgetAdjustment(){
    const button=root.querySelector('[data-budget-auto]'),status=root.querySelector('[data-budget-status]');if(!button||!status)return;
    const gpu=root.querySelector('[data-param=Gpu]'),selection=gpu?.multiple?Array.from(gpu.selectedOptions).map(x=>x.value):(gpu?.value||'').split(',').filter(Boolean);
    const tensor=root.querySelector('[data-param=TensorParallel]');
    const unavailable=selection.length!==1||(tensor?.checked&&!tensor.disabled);
    button.disabled=busy||configurationLocked()||runtime?.budget_planning||unavailable;
    button.dataset.disabledReason=unavailable?tr('请选择一张显卡；自动调整暂支持 NInfer 单卡。','Select one GPU; automatic adjustment currently supports single-GPU NInfer.'):'';
    button.textContent=budgetPending?tr('测算中…','Planning…'):tr('自动调整','Auto adjust');
    status.hidden=!budgetAdjustment;status.classList.toggle('l-error',budgetAdjustment?.kind==='error');
    if(!budgetAdjustment)return;
    if(budgetPending)status.textContent=tr(`正在加载模型并测算… ${Math.floor((Date.now()-budgetAdjustment.started)/1000)} 秒`,`Loading model and planning… ${Math.floor((Date.now()-budgetAdjustment.started)/1000)} s`);
    else if(budgetAdjustment.kind==='success')status.textContent=tr(`已调整为 ${budgetAdjustment.budget.toLocaleString()} · 耗时 ${(budgetAdjustment.elapsed/1000).toFixed(1)} 秒；待启动验证。`,`Set to ${budgetAdjustment.budget.toLocaleString()} · ${(budgetAdjustment.elapsed/1000).toFixed(1)} s; startup validation pending.`);
    else status.textContent=budgetAdjustment.message;
  }
  async function adjustBudget(){
    if(busy||configurationLocked()||runtime?.budget_planning)return;
    let timer;
    try{
      const configuration=launchData(),key=JSON.stringify(configuration);
      busy=true;budgetPending=true;budgetAdjustment={started:Date.now()};applyConfigurationLock();renderMemoryRecovery();renderBudgetAdjustment();
      timer=setInterval(renderBudgetAdjustment,1000);
      const result=await api('adjust-budget',configuration,240000);
      if(JSON.stringify(launchData())!==key)throw Error(tr('设置已改变，测算结果未应用；请重新测算。','Settings changed; the result was not applied. Run adjustment again.'));
      const input=root.querySelector('[data-param=Budget]');
      input.value=String(result.changes.Budget);input.dispatchEvent(new Event('input',{bubbles:true}));
      budgetAdjustment={kind:'success',budget:result.changes.Budget,elapsed:result.elapsed_ms};
    }catch(error){budgetAdjustment={kind:'error',message:error.message||String(error)};}
    finally{clearInterval(timer);budgetPending=false;busy=false;applyConfigurationLock();updateStatus(runtime);renderBudgetAdjustment();}
  }
  function rankMemoryOptions(options){
    const qualityOrder={int8:0,rk8v4:1,nvfp4:2};
    return [...options].sort((a,b)=>{
      if(a.available!==b.available)return a.available?-1:1;
      if(a.available){
        const aBudget=a.changes.Budget,bBudget=b.changes.Budget;
        const aMeetsTarget=aBudget>=32768,bMeetsTarget=bBudget>=32768;
        if(aMeetsTarget!==bMeetsTarget)return aMeetsTarget?-1:1;
        if(!aMeetsTarget&&aBudget!==bBudget)return bBudget-aBudget;
      }
      return qualityOrder[a.id]-qualityOrder[b.id];
    });
  }
  function renderMemoryRecovery(){
    const box=root.querySelector('[data-memory-recovery]');if(!box)return;
    const diagnosis=runtime?.memory_diagnosis,trial=runtime?.memory_trial;
    const instance=runtime?.instance;
    const renderKey=JSON.stringify([instance,runtime?.state,diagnosis,trial,busy,configurationLocked(),window.monitorI18n.language]);
    if(box.dataset.renderKey===renderKey)return;box.dataset.renderKey=renderKey;
    box.hidden=!diagnosis&&!trial;box.dataset.state=diagnosis?'error':'trial';box.replaceChildren();if(box.hidden)return;
    if(!diagnosis){
      box.append(make('h2',{class:'s-title'},tr('显存推荐方案','Memory recommendation')),
        make('p',{},trial.startup_verified?tr('推荐方案已通过启动验证；推理峰值显存尚未验证。','Recommendation passed startup validation; peak inference memory remains unverified.'):runtime.state==='exited'?tr('推荐方案的进程已退出，请检查日志。','The recommended launch exited; check the log.'):tr('正在验证推荐方案，等待服务就绪。','Validating recommendation; waiting for readiness.')));return;
    }
    box.append(make('h2',{class:'s-title'},tr('启动时显存不足','Insufficient GPU memory at startup')));
    const reasons={single_ninfer_only:['暂仅支持 NInfer 单卡推荐。','Recommendations currently support single-GPU NInfer.'],missing_native_plan:['此次失败未生成后端内存规划，暂无法给出可靠的推荐参数。','No native memory plan was produced for this failure; reliable settings are unavailable.'],invalid_native_plan:['后端规划报告与本次启动参数不一致，无法使用该建议。','The native plan does not match this launch; its recommendations cannot be used.']};
    if(diagnosis.reason){box.append(make('p',{},tr(...(reasons[diagnosis.reason]||reasons.missing_native_plan))));return;}
    const mib=n=>`${Math.ceil(n/1048576).toLocaleString()} MiB`;
    box.append(make('p',{},tr(`后端需要 ${mib(diagnosis.required_bytes)}；模型加载后可用 ${mib(diagnosis.available_bytes)}。`,`Runtime requires ${mib(diagnosis.required_bytes)}; ${mib(diagnosis.available_bytes)} available after model loading.`)),
      make('p',{class:'l-muted'},tr(`后端已分别规划，耗时 ${(diagnosis.planning_us/1000).toFixed(1)} ms；需重新启动验证。格式变化可能影响质量和速度。`,`Native plans computed in ${(diagnosis.planning_us/1000).toFixed(1)} ms; restart to validate. Format changes may affect quality and speed.`)));
    const options=rankMemoryOptions(diagnosis.options);
    for(const [index,option] of options.entries()){
      const card=make('div',{class:'l-memory-option','data-memory-option':option.id});
      const heading=make('div',{class:'l-memory-heading'});
      heading.append(make('h3',{},`${index+1}. ${option.id.toUpperCase()} KV`));
      const recommended=index===0&&option.available;
      if(recommended){
        card.dataset.memoryRecommended='true';
        heading.append(make('span',{class:'l-memory-badge'},tr('推荐','Recommended')));
      }
      card.append(heading);
      if(!option.available){card.append(make('p',{},option.reason==='planning_unavailable'?tr('后端无法完成此格式的规划，此方案暂不可用。','The backend could not plan this format; this option is unavailable.'):tr('最低合法 budget 仍无法容纳，保留 reserve，此方案不可用。','Even the minimum legal budget does not fit with the retained reserve; this option is unavailable.')));box.append(card);continue;}
      const c=option.changes,old=diagnosis.original;
      if(recommended)card.append(make('p',{class:'l-memory-reason'},c.Budget>=32768?
        tr(`budget ≥ 32K，达标方案中 ${option.id.toUpperCase()} 的 KV 精度最高。`,`Budget ≥ 32K; ${option.id.toUpperCase()} is the highest-precision qualifying KV type.`):
        tr('可用方案均低于 32K，此方案 budget 最大；并列时优先 KV 精度。','All available budgets are below 32K; this has the largest budget, with KV precision breaking ties.')));
      card.append(make('p',{},`budget ${old.Budget.toLocaleString()} → ${c.Budget.toLocaleString()} · gen reserve ${old.Reserve.toLocaleString()} → ${c.Reserve.toLocaleString()}`));
      const details=make('details');details.append(make('summary',{},tr('估算详情','Estimate details')),
        make('p',{class:'l-muted'},tr(`预计运行空间 ${mib(option.estimated_runtime_bytes)}，另留 ${mib(option.safety_bytes)} 余量；不含已加载模型。`,`Estimated runtime ${mib(option.estimated_runtime_bytes)}, plus ${mib(option.safety_bytes)} headroom; excludes loaded weights.`)),
        make('p',{class:'l-muted'},tr(`prefill ${c.Prefill}；含静态 CUDA 图预留，未额外运行 GPU 探测。`,`prefill ${c.Prefill}; includes static CUDA graph allowance, without additional GPU probes.`)),
        make('p',{class:'l-muted'},tr('先将超过 4K 的 gen reserve 降到 4096，否则保留原值，再减少 budget。','Cap gen reserve at 4096, retaining smaller values, then reduce budget.')));card.append(details);
      if(option.id==='nvfp4'&&c.FastPrefill===false)card.append(make('p',{class:'l-muted'},tr('同时关闭快速预填充以匹配 NVFP4。','Also disables fast prefill for NVFP4.')));
      const button=make('button',{type:'button',class:'s-button l-primary','data-memory-retry':option.id},tr('应用此方案并重新启动','Apply and restart'));
      button.disabled=busy||configurationLocked();button.onclick=()=>retryMemory(option.id,instance);card.append(button);box.append(card);
    }
    box.append(make('p',{class:'l-muted'},tr('推荐基于本次失败配置；修改配置后请重新启动以获取新建议。','Recommendations apply to this failed launch. After editing settings, start again to get new recommendations.')));
  }
  async function retryMemory(option,instance){
    if(busy||configurationLocked())return;
    try{
      const configuration=launchData();busy=true;renderMemoryRecovery();applyConfigurationLock();
      const result=await api('retry-memory',{option,instance,configuration});
      launchFailure='';updateStatus(result,true);
      message(tr('已应用推荐并提交启动，等待服务验证。','Recommendation applied and launch submitted; awaiting validation.'));
    }catch(error){showError(error);try{updateStatus(await api('status'),true);}catch{configUncertain=true;}}
    finally{busy=false;updateStatus(runtime);}
  }
  function renderLaunchProgress(){
    const box=root.querySelector('[data-launch-progress]');if(!box)return;
    const state=startPending?'preparing':launchFailure?'failed':runtime?.state;
    box.hidden=!['preparing','starting','ready','exited','failed'].includes(state);
    if(box.hidden)return;
    const labels={preparing:['检查配置并启动进程…','Checking settings and starting process…'],starting:['正在加载模型，等待服务就绪…','Loading model; waiting for readiness…'],ready:['启动完成，可以开始聊天','Ready to chat'],exited:['服务进程已退出','Service process exited'],failed:['启动请求未完成','Launch request did not complete']};
    box.dataset.state=state;
    const phase=box.querySelector('[data-launch-phase]'),track=box.querySelector('[data-launch-track]');
    phase.textContent=tr(...labels[state]);track.setAttribute('aria-label',phase.textContent);
    track.hidden=state==='failed'||state==='exited';
    if(state==='ready')track.setAttribute('aria-valuenow','100');else track.removeAttribute('aria-valuenow');
    box.querySelector('[data-launch-help]').textContent=state==='failed'?launchFailure:state==='exited'?(runtime?.memory_diagnosis?tr('检测到显存不足，请查看紧接下方的调整建议。','GPU memory shortage detected. Review the recommendations immediately below.'):tr('请查看下方运行日志中的退出原因。','Check the service log below for the cause.')):state==='ready'?tr('右上角聊天入口已可用。','Chat is available in the top right.'):tr('请稍候。大模型首次加载可能需要较长时间。','Please wait. Loading a large model for the first time may take a while.');
  }
  function exportDialog(){
    if(busy||configurationLocked())return;
    if(!root.querySelector('form').reportValidity())return;
    const dialog=make('dialog',{class:'l-model-dialog','aria-label':tr('生成启动脚本','Generate launch script')}),form=make('form',{method:'dialog'});
    const heading=make('div',{class:'l-heading'}),close=make('button',{type:'button',class:'s-button'},tr('关闭','Close'));
    heading.append(make('h2',{class:'s-title'},tr('生成启动脚本','Generate launch script')),close);
    const nameLabel=make('label',{class:'l-field'},tr('配置名称','Profile name'));
    const name=make('input',{type:'text',required:'',pattern:'[a-z0-9][a-z0-9_-]{0,63}','data-export-profile':''});name.value='default-'+root.querySelector('[data-param=Backend]').value;nameLabel.append(name);
    const scopeLabel=make('label',{class:'l-field'},tr('后台服务类型','Background service type'));
    const scope=make('select',{'data-export-scope':''});scope.append(make('option',{value:'user'},tr('用户服务','User service')),make('option',{value:'system'},tr('系统服务','System service')));scopeLabel.append(scope);
    const userLabel=make('label',{class:'l-field'},tr('运行账户','Runtime account'));
    const user=make('input',{type:'text','data-export-user':''});user.value=info.runtime_user||'';userLabel.append(user);userLabel.hidden=true;scope.onchange=()=>{userLabel.hidden=scope.value!=='system';user.required=scope.value==='system';};
    const secretLabel=make('label',{class:'l-field'}),store=make('input',{type:'checkbox','data-export-store-key':''});secretLabel.append(store,make('span',{},tr('将 API 密钥保存到本机私有文件，供后台启动使用','Save the API key to a private local file for background startup')));secretLabel.hidden=!launchData().parameters.ApiKey;
    const note=make('p',{class:'l-muted'},tr('每次生成保留独立版本，可直接在终端运行。服务文件同时生成；安装、启动和开机自启分别操作。','Each export preserves a separate version that can run from a terminal. A service file is included; installation, startup and boot activation are separate actions.'));
    const row=make('div',{class:'l-actions'}),generate=make('button',{type:'button',class:'s-button l-primary','data-export-generate':''},tr('生成文件','Generate files')),start=make('button',{type:'button',class:'s-button',disabled:'','data-export-start':''},tr('启动此版本','Start this version'));
    const register=make('button',{type:'button',class:'s-button',disabled:'','data-export-register':''},tr('登记后台服务','Register background service'));
    const result=make('pre',{class:'l-result','data-export-result':'',role:'status'});row.append(generate,start,register);form.append(heading,nameLabel,scopeLabel,userLabel,secretLabel,note,row,result);dialog.append(form);root.append(dialog);
    let exported=null;
    close.onclick=()=>dialog.close();dialog.addEventListener('close',()=>dialog.remove(),{once:true});form.onsubmit=event=>event.preventDefault();
    generate.onclick=async()=>{if(!form.reportValidity())return;generate.disabled=true;start.disabled=true;result.textContent=tr('正在检查并生成…','Checking and generating…');try{const config=launchData();exported=await api('export',{config,profile_id:name.value,scope:scope.value,...(scope.value==='system'?{user:user.value}:{}),store_api_key:store.checked});result.textContent=tr('启动脚本：','Launch script: ')+exported.script+'\n\n'+tr('服务文件：','Service file: ')+exported.unit+'\n\n'+tr('版本：','Revision: ')+exported.revision+(exported.requires_api_key_environment?'\n\n'+tr('独立运行前需设置 KVMEM_API_KEY；本控制台当前仍保留此密钥。','Set KVMEM_API_KEY before running independently; this console still has the current key.'):'');start.disabled=false;}catch(error){exported=null;result.textContent=error.message;}finally{generate.disabled=false;}};
    start.onclick=async()=>{if(!exported||busy)return;start.disabled=true;generate.disabled=true;busy=true;try{const state=await api('start-revision',{profile_id:exported.profile_id,revision:exported.revision});updateStatus(state,true);dialog.close();message(tr('已启动生成的脚本版本，请查看服务状态。','Started the generated version; check service status.'));}catch(error){result.textContent=error.message;}finally{busy=false;start.disabled=false;generate.disabled=false;updateStatus(runtime);}};
    const originalGenerate=generate.onclick;
    generate.onclick=async()=>{register.disabled=true;await originalGenerate();if(exported){register.disabled=exported.requires_api_key_environment;if(exported.install_commands?.length)result.textContent+='\n\n'+tr('先在终端安装，再点击登记：\n','Install in a terminal, then register:\n')+exported.install_commands.join('\n');}};
    register.onclick=async()=>{if(!exported)return;register.disabled=true;try{await api('service',{action:'register',profile_id:exported.profile_id,revision:exported.revision,scope:exported.scope});dialog.close();await servicesDialog();}catch(error){result.textContent=error.message;}finally{register.disabled=false;}};
    dialog.showModal();name.focus();
  }
  async function servicesDialog(){
    const dialog=make('dialog',{class:'l-model-dialog','aria-label':tr('后台服务管理','Background services')}),body=make('div');
    const close=make('button',{type:'button',class:'s-button'},tr('关闭','Close'));
    dialog.append(make('h2',{class:'s-title'},tr('后台服务管理','Background services')),close,body);root.append(dialog);
    close.onclick=()=>dialog.close();dialog.addEventListener('close',()=>dialog.remove(),{once:true});dialog.showModal();
    async function refresh(){
      body.replaceChildren(make('p',{},tr('读取服务状态…','Reading service status…')));
      try{
        const data=await api('services');body.replaceChildren();
        body.append(make('p',{class:'l-muted'},tr('后台服务关闭控制台后仍会运行。生成脚本后，先登记服务，再分别选择启动与自启。','Background services survive closing the console. Generate a script, register it, then choose startup and boot activation separately.')));
        if(data.selected){
          const disconnect=make('button',{type:'button',class:'s-button','data-service-action':'disconnect'},tr('解除当前连接（不停止服务）','Disconnect current service (keep it running)'));
          const error=make('p',{role:'status'});body.append(disconnect,error);
          disconnect.onclick=async()=>{disconnect.disabled=true;try{await api('service',{action:'disconnect',profile_id:data.selected.profile,scope:data.selected.scope});updateStatus(await api('status'),true);await refresh();}catch(e){error.textContent=e.message;disconnect.disabled=false;}};
        }
        if(!data.capabilities.available||!data.capabilities.user)body.append(make('p',{class:'l-muted'},data.capabilities.reason||tr('用户会话服务不可用，可使用独立脚本或系统服务。','User services are unavailable; use independent scripts or system services.')));
        if(!data.capabilities.linger)body.append(make('p',{class:'l-muted'},tr('用户服务“启用自启”在登录时生效。如需无人登录运行，请在终端按需执行：','Enabling a user service starts it at login. For unattended startup, explicitly run in a terminal:')+' loginctl enable-linger '+(info.runtime_user||'<user>')));
        if(!data.services.length)body.append(make('p',{},tr('还没有登记的服务，请先生成启动脚本。','No services are registered. Generate a launch script first.')));
        for(const service of data.services){
          const card=make('section',{class:'s-card'}),actions=make('div',{class:'l-actions'}),error=make('p',{role:'status'});
          card.append(make('h3',{},service.profile_id+' · '+service.scope),make('p',{},service.error||`${service.active} / ${service.substate} · ${service.enabled?tr('已启用自启','Enabled'):tr('未启用自启','Disabled')} · ${service.revision||''}`));
          if(!service.error)for(const [action,zh,en] of [['select','查看／连接','View / connect'],['start','启动','Start'],['stop','停止','Stop'],['restart','重启','Restart'],[service.enabled?'disable':'enable',service.enabled?'取消自启':'启用自启',service.enabled?'Disable startup':'Enable startup'],['remove','移除登记','Remove registration']]){
            if(action==='remove'&&service.scope==='system')continue;
            const button=make('button',{type:'button',class:'s-button','data-service-action':action},tr(zh,en));
            button.onclick=async()=>{actions.querySelectorAll('button').forEach(b=>b.disabled=true);try{await api('service',{action,profile_id:service.profile_id,scope:service.scope});updateStatus(await api('status'),true);await refresh();}catch(e){error.textContent=e.message;actions.querySelectorAll('button').forEach(b=>b.disabled=false);}};
            actions.append(button);
          }
          card.append(actions,error);body.append(card);
        }
      }catch(error){body.replaceChildren(make('p',{role:'status'},error.message));}
    }
    await refresh();
  }
  async function act(action){
    if(busy||(configurationLocked()&&action!=='stop'))return;
    if(action==='template'){templateEditor();return;}
    if(action==='preview'){await checkConfiguration();return;}
    try{
      const data=action==='stop'?{}:launchData();
      if(action==='start'&&!root.querySelector('form').reportValidity())return;
      saveEditing();busy=true;root.querySelectorAll('[data-action]').forEach(button=>button.disabled=true);
      if(action==='start'){pendingParameters=structuredClone(data.parameters);startPending=true;launchFailure='';applyConfigurationLock();renderLaunchProgress();root.querySelector('[data-launch-progress]').scrollIntoView({block:'nearest',behavior:'smooth'});}
      const result=await api(action,data);startPending=false;launchFailure='';updateStatus(result,true);
      message(action==='start'?tr('启动请求已提交，可查看右侧状态和日志。','Launch submitted; see status and logs.'):tr('服务已停止，配置页面仍可使用。','Service stopped; configuration remains available.'));
    }catch(error){if(action==='start'){launchFailure=error.message||String(error);try{updateStatus(await api('status'),true);}catch{configUncertain=true;}}showError(error);}
    finally{startPending=false;busy=false;root.querySelectorAll('[data-action]').forEach(button=>button.disabled=false);if(configUncertain)applyConfigurationLock();else updateStatus(runtime);}
  }
  async function poll(){if(info&&!busy){try{updateStatus(await api('status'),true);}catch(e){message(e.message,true);}}setTimeout(poll,document.hidden?5000:2000);}
  // The shared i18n bindings translate navigation, including its mobile label.
  window.addEventListener('kvmem-language-change',()=>{if(info){try{restoreDraft(captureDraft());}catch(e){message(e.message,true);}}});
  async function init(){try{info=await api('info');let saved;try{saved=JSON.parse(localStorage.getItem(DRAFT_KEY)||'null');}catch{saveError=true;}try{const drafts=JSON.parse(localStorage.getItem(BACKENDS_KEY)||'{}');for(const [key,value] of Object.entries(drafts))if(usableDraft(value)&&value.parameters.Backend===key)perBackend[key]=value;}catch{saveError=true;}if(usableDraft(saved))restoreDraft(saved);else render(info.draft.parameters);updateStatus(info.status,true);if(location.pathname!=='/chat/'&&!location.hash&&!['starting','ready'].includes(info.status.state))document.querySelector('[data-page=config]').click();poll();}catch(e){root.replaceChildren(make('section',{class:'s-card'},tr('启动配置需要独立的本地控制台。请运行 scripts/local-console.py，然后打开它显示的地址；推理服务停止后，这个控制台仍然可用。','Launch settings require the independent local console. Run scripts/local-console.py and open its printed URL; the console remains available when inference stops.')));}}
  init();
  document.documentElement.dataset.launcherReady = "true";
})();
