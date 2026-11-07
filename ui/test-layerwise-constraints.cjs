// Model-free DOM checks execute the actual production helpers. No worker/model/GPU.
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const vm=require('node:vm');
const source=fs.readFileSync(path.join(__dirname,'monitor/launcher.js'),'utf8');
function production(name){
 const start=source.indexOf(`  function ${name}(`),end=source.indexOf('\n  }',start);
 assert.ok(start>=0&&end>start,name);return source.slice(start,end+4);
}
const element=value=>({value:String(value),max:'63',message:'',setCustomValidity(message){this.message=message;}});
const inputs={Budget:element(64),Reserve:element(64),Concurrency:element(1),NgramDrafts:element(63)};
const context={tr:(zh,en)=>en,root:{querySelector(selector){return inputs[selector.match(/data-param=([^\]]+)/)[1]];}}};
vm.createContext(context);
vm.runInContext(['workerTokenConstraints','syncWorkerTokenConstraints'].map(production).join('\n'),context);
const sync=(profile={},spec={ngram:true},layerwise=true)=>context.syncWorkerTokenConstraints(profile,spec,layerwise,true);
const profile={minimum_window_tokens:256};
const spec={ngram:true,max_ngram_draft_tokens:63,max_ngram_draft_tokens_concurrent:15};
sync(profile,spec);
assert.match(inputs.Budget.message,/256/);assert.equal(inputs.Budget.value,'64');
assert.equal(inputs.NgramDrafts.max,'63');assert.equal(inputs.NgramDrafts.message,'');
inputs.Budget.value='192';sync(profile,spec);assert.equal(inputs.Budget.message,'');
console.log('PASS worker minimum window checks Budget+Reserve, preserves entered values');
inputs.Concurrency.value='2';sync(profile,spec);
assert.equal(inputs.NgramDrafts.max,'15');assert.match(inputs.NgramDrafts.message,/limits.*15.*retained/);
assert.equal(inputs.NgramDrafts.value,'63');
inputs.NgramDrafts.value='15';sync(profile,spec);assert.equal(inputs.NgramDrafts.message,'');
inputs.NgramDrafts.value='0';sync(profile,spec);assert.equal(inputs.NgramDrafts.message,'');
inputs.NgramDrafts.value='63';inputs.Concurrency.value='1';sync(profile,spec);
assert.equal(inputs.NgramDrafts.max,'63');assert.equal(inputs.NgramDrafts.message,'');
console.log('PASS concurrent N-gram bound rejects excess; switch back clears stale error without clamping');
sync({}, {ngram:true});assert.equal(inputs.NgramDrafts.max,'63');assert.equal(inputs.NgramDrafts.message,'');
sync({}, {ngram:false});assert.match(inputs.NgramDrafts.message,/does not qualify/);assert.equal(inputs.NgramDrafts.value,'63');
sync(profile,spec,false);assert.equal(inputs.Budget.message,'');assert.equal(inputs.NgramDrafts.message,'');
assert.equal(inputs.NgramDrafts.max,'63');
console.log('PASS missing optional fields retain boolean contract and Legacy clears new constraints');
for(const key of ['minimum_window_tokens','max_ngram_draft_tokens','max_ngram_draft_tokens_concurrent']){
 for(const value of [null,true,0,-1,1.5,'15']){
  const p=key==='minimum_window_tokens'?{[key]:value}:{};
  const s=key==='minimum_window_tokens'?{ngram:true}:{ngram:true,[key]:value};
  sync(p,s);assert.match(inputs.Budget.message,/Invalid worker token constraint/);
  assert.equal(inputs.NgramDrafts.value,'63');
 }
}
console.log('PASS malformed declared numeric constraints refuse instead of guessing a limit');
