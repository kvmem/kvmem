// Pure functions from the shipped chat: no network or model required.
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const app=fs.readFileSync(path.join(__dirname,'../backends/polystrata/terminal/serve/web/kvmem-app.js'),'utf8');
const start=app.indexOf('function assistantMessages(m) {'),end=app.indexOf('\nfunction setBusy(',start);
const messages=new Function(app.slice(start,end)+';return assistantMessages;')();
assert.deepEqual(messages({text:'Done.',reasoning:'correct task-specific reasoning'}),[{role:'assistant',content:'Done.',reasoning_content:'correct task-specific reasoning'}]);
const history=messages({text:'beforeafter',reasoning:'reasonAreasonB',tools:[{id:'1',name:'tool',at:6,rat:7,round:0,result:'result',state:'done',arguments:{x:1}}]});
assert.equal(history[0].content,'before');assert.equal(history[0].reasoning_content,'reasonA');
assert.equal(history[1].role,'tool');assert.equal(history[2].content,'after');assert.equal(history[2].reasoning_content,'reasonB');
const defaultCode=app.match(/^const DEFAULTS = .*;$/m)[0];
const settingsCode=app.slice(app.indexOf('let settings ='),app.indexOf('let messages ='));
const nodes={},$=id=>nodes[id]??={};let payload={model:'model',server_instance:'a',shared:false,effective:{reasoning_effort:'none',max_tokens:32000,reasoning_budget_tokens:0,preserve_thinking:true,temperature:.7,top_p:.8,top_k:20}};
let stored={scope:'old',overrides:{thinking:'high'}};
const store={get:()=>stored,set:(_,v)=>{stored=v;}};
(async()=>{
 const api=new Function('$','store','fetch','headers','toast',defaultCode+'\nlet sharedOn=false;\n'+settingsCode+';return {refreshChatSettings,get:()=>({...settings}),override:x=>{chatOverrides=x;resolveChatSettings();},wire:()=>({...chatOverrides})};')($,store,async()=>({ok:true,json:async()=>payload}),()=>({}),()=>{});
 await api.refreshChatSettings();assert.equal(api.get().thinking,'none');assert.equal(api.get().max,32000);assert.deepEqual(api.wire(),{});
 api.override({thinking:'low',budget:128});assert.equal(api.get().thinking,'low');assert.equal(api.get().budget,128);
 payload={...payload,server_instance:'b'};await api.refreshChatSettings();assert.equal(api.get().thinking,'none');assert.deepEqual(api.wire(),{});
 console.log('PASS: client-owned reasoning/tool spans, actual service defaults, explicit chat overrides, restart reset');
})().catch(e=>{console.error(e);process.exitCode=1;});
