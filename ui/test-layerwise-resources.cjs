// Execute production UI helpers in a model-free DOM harness. No browser/GPU,
// package installation, HTTP service or synthetic copy of their implementation.
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const vm=require('node:vm');
const source=fs.readFileSync(path.join(__dirname,'monitor/launcher.js'),'utf8');
function production(name){
  const start=source.indexOf(`  function ${name}(`);
  assert.ok(start>=0,`missing production helper ${name}`);
  const lineEnd=source.indexOf('\n',start);
  if(source.slice(start,lineEnd).trimEnd().endsWith('}'))return source.slice(start,lineEnd);
  const end=source.indexOf('\n  }',start);
  assert.ok(end>start,`missing helper end ${name}`);return source.slice(start,end+4);
}
const resourceLine=source.split('\n').find(line=>line.includes('const layerwiseResources='));
const element=(param,value='')=>({value,type:'number',disabled:false,dataset:{param,valueType:'number'}});
const inputs={Backend:element('Backend','ninfer'),Worker:element('Worker','worker-A'),RecallStrategy:element('RecallStrategy','layerwise'),
  LayerwiseMetadataMiB:element('LayerwiseMetadataMiB'),LayerwiseActivationDiskMiB:element('LayerwiseActivationDiskMiB')};
for(const key of ['Backend','Worker','RecallStrategy']){inputs[key].type='text';delete inputs[key].dataset.valueType;}
const tasks=[],fetches=[];
const context={console,structuredClone,AbortController,info:{worker_capabilities_version:1,token:'fixture'},workerKey:'',workerContract:null,
  workerError:'',workerTimer:null,workerAbort:null,workerSequence:0,runtime:null,
  configLocked:true,configUncertain:false,startPending:false,busy:false,runningParameters:null,pendingParameters:null,runningView:'',
  render:()=>{},applyConfigurationLock:()=>{},saveEditing:()=>{},healthTimer:null,healthAbort:null,healthSequence:0,
  selectedPreset:'',presetUndo:null,
  configurationLocked:()=>false,tr:(zh,en)=>en,syncDependencies:()=>{},updateStatus:()=>{},
  setTimeout:(fn,ms)=>{tasks.push({fn,ms});return tasks.length;},clearTimeout:()=>{},
  fetch:()=>new Promise(resolve=>fetches.push(resolve)),
  root:{querySelector(selector){const match=selector.match(/^\[data-param=([^\]]+)\]$/);return match?inputs[match[1]]||null:null;},
    querySelectorAll(){return Object.values(inputs);}}};
vm.createContext(context);
vm.runInContext(resourceLine+'\n'+['workerSelection','workerProbeSelection','currentWorkerProfile','workerResourceDefaults','syncWorkerResourceDefaults',
  'resourceOverride','markResourceEdit','captureDraft','collect','scheduleWorkerCapabilities','syncRunningConfiguration'].map(production).join('\n'),context);
const normalized=x=>JSON.parse(JSON.stringify(x));
const one={available:true,default_resources:{metadata_bytes:1073741824,activation_disk_bytes:0}};
assert.deepEqual(normalized(context.workerResourceDefaults(one)),{LayerwiseMetadataMiB:1024,LayerwiseActivationDiskMiB:0});
assert.deepEqual(normalized(context.workerResourceDefaults({available:true})),{});
for(const invalid of [null,[],{metadata_bytes:true},{metadata_bytes:0},{metadata_bytes:-1}])
  assert.throws(()=>context.workerResourceDefaults({default_resources:invalid}),/resource default/);
context.syncWorkerResourceDefaults(one);
assert.equal(inputs.LayerwiseMetadataMiB.value,'1024');
assert.equal(inputs.LayerwiseActivationDiskMiB.value,'0');
assert.equal(context.collect().parameters.LayerwiseMetadataMiB,undefined);
assert.equal(context.captureDraft().parameters.LayerwiseMetadataMiB,undefined);
assert.equal(context.collect().parameters.LayerwiseActivationDiskMiB,undefined);
console.log('PASS backend byte defaults display without materializing config/argv overrides');

inputs.LayerwiseMetadataMiB.value='256';context.markResourceEdit({target:inputs.LayerwiseMetadataMiB});
context.syncWorkerResourceDefaults({default_resources:{metadata_bytes:2147483648}});
assert.equal(inputs.LayerwiseMetadataMiB.value,'256');
assert.equal(context.collect().parameters.LayerwiseMetadataMiB,256);
assert.equal(context.captureDraft().parameters.LayerwiseMetadataMiB,256);
console.log('PASS explicit256 remains explicit256 across new backend defaults');

inputs.LayerwiseMetadataMiB.value='';context.markResourceEdit({target:inputs.LayerwiseMetadataMiB});
context.syncWorkerResourceDefaults(one);
inputs.Worker.value='worker-B';context.syncWorkerResourceDefaults(null);
assert.equal(inputs.LayerwiseMetadataMiB.value,'');
context.syncWorkerResourceDefaults({default_resources:{metadata_bytes:2147483648}});
assert.equal(inputs.LayerwiseMetadataMiB.value,'2048');
assert.equal(context.collect().parameters.LayerwiseMetadataMiB,undefined);
context.syncWorkerResourceDefaults({default_resources:{metadata_bytes:3221225472}});
assert.equal(inputs.LayerwiseMetadataMiB.value,'3072');
context.syncWorkerResourceDefaults({default_resources:{metadata_bytes:1073741825}});
assert.equal(inputs.LayerwiseMetadataMiB.value,'');
assert.equal(context.collect().parameters.LayerwiseMetadataMiB,undefined);
console.log('PASS worker change/refresh clears stale defaults; non-MiB bytes are never rounded');

(async()=>{
  inputs.Worker.value='worker-A';context.scheduleWorkerCapabilities(true);
  const first=tasks.find(x=>x.ms===0);const firstPending=first.fn();
  inputs.Worker.value='worker-B';context.scheduleWorkerCapabilities(true);
  const second=tasks.filter(x=>x.ms===0)[1];const secondPending=second.fn();
  fetches[1]({ok:true,json:async()=>({schema_version:1,backend:'ninfer',strategy:'layerwise',requested_worker:'worker-B',
    worker:{executable_sha256:'b'.repeat(64)},profile:{default_resources:{metadata_bytes:2147483648},available:true}})});
  await secondPending;
  assert.equal(inputs.LayerwiseMetadataMiB.value,'2048');
  fetches[0]({ok:true,json:async()=>({schema_version:1,backend:'ninfer',strategy:'layerwise',requested_worker:'worker-A',
    worker:{executable_sha256:'a'.repeat(64)},profile:one})});
  await firstPending;
  assert.equal(context.workerContract.requested_worker,'worker-B');
  assert.equal(inputs.LayerwiseMetadataMiB.value,'2048');
  assert.equal(context.captureDraft().parameters.LayerwiseMetadataMiB,undefined);
  console.log('PASS actual asynchronous capability handler rejects an out-of-order worker response');
  const running={Backend:'ninfer',RecallStrategy:'layerwise'};
  context.syncRunningConfiguration({state:'ready',instance:'actual-report-fixture',active_config:{parameters:running},
    effective_scoring:{strategy:'layerwise',metadata_bytes:1073741824,activation_disk_bytes:0}});
  assert.equal(inputs.LayerwiseMetadataMiB.value,'1024');
  assert.equal(inputs.LayerwiseActivationDiskMiB.value,'0');
  assert.equal(context.runningParameters.LayerwiseMetadataMiB,undefined);
  assert.equal(running.LayerwiseMetadataMiB,undefined);
  console.log('PASS normalized running limits display without writing a missing user configuration key');
})().catch(error=>{console.error(error);process.exitCode=1;});
