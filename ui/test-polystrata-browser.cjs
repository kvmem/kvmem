// Review regressions against the shipped HTML/JS; transport/device data are fixtures.
const fs=require('node:fs'),path=require('node:path'),http=require('node:http'),assert=require('node:assert/strict');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const source=path.resolve(__dirname,'..'),evidence=path.resolve(source,'../validation');
const sent=[],previews=[],errors=[];
let instance='first';
const settings=()=>({model:'tiel',server_instance:instance,shared:false,defaults:{},effective:{reasoning_effort:'none',reasoning_budget_tokens:0,max_tokens:32000,preserve_thinking:true,temperature:.7,top_p:.8,top_k:20}});
const info={token:'fixture-token',backends:['polystrata','ninfer','llamacpp'],backend_root:source,kv_cache_version:1,query_scoring_version:2,
 gpus:[{uuid:'GPU-12345678',name:'RTX 5050 test fixture'}],workers:{polystrata:{path:'C:/fixture/worker.exe',available:true}},
 native_options:{polystrata:['--threads'],ninfer:['--threads'],llamacpp:['--threads']},
 speculative_capabilities:JSON.parse(fs.readFileSync(path.join(source,'scripts/speculative-capabilities.json'),'utf8')),
 status:{state:'stopped'},draft:{schema_version:1,parameters:{Backend:'polystrata',Model:'C:/fixture/tiel.gguf',Gpu:'GPU-12345678',
 Context:262144,Budget:32768,Reserve:2048,HostMiB:4096,Prefill:512,MaxTokens:32000,KvType:'q8_0',RetainedSessions:4,MtpDrafts:0,Port:18200,SinkTokens:12288,RecentTokens:1024}}};
const record={id:'1',started_at_ms:Date.now(),status:'completed',finish_reason:'length',finish_phase:'thinking',prompt_tokens:6518,cached_tokens:6456,
 generated_tokens:128,decode_tps:42,decode_tps_mean:42,prefill_tps:53,prefill_ms:1170,ttft_ms:1200,prompt_new_tokens:62,elapsed_ms:5000,limit:128,drafts_offered:20,drafts_accepted:15};
const server=http.createServer(async(req,res)=>{
 const url=new URL(req.url,'http://localhost'),p=url.pathname;
 let body='';for await(const chunk of req)body+=chunk;
 const json=data=>{res.setHeader('Content-Type','application/json');res.end(JSON.stringify(data));};
 if(p==='/settings')return json(settings());
 if(p==='/health')return json({status:'ok',model:'tiel',max_context:262144});
 if(p==='/metrics')return json({live:{state:'idle'},engine:{},requests:[]});
 if(p==='/mcp')return json({tools:0,servers:[]});
 if(p==='/api/models')return json({models:[]});
 if(p==='/api/launcher/info')return json(info);
 if(p==='/api/launcher/status')return json(info.status);
 if(p==='/api/launcher/preview'){previews.push(JSON.parse(body));return json({argv:['python','--max-tokens','32000','--reserve','2048'],environment:{},effective_settings:{model_alias:'kvmem-polystrata-qwen35'}});}
 if(p==='/props')return json({model_alias:'tiel',default_generation_settings:{n_ctx:262144},effective_settings:settings(),execution:{mode:'single',monitor_devices:[]},kvmem:{mtp:true,vram_reserve_mib:256}});
 if(p==='/api/monitor/snapshot')return json({schema_version:1,backend:'polystrata',server_instance:'test',sampled_at_ms:Date.now(),active:[],recent:[record],queued_count:2,hardware:{sampled_at_ms:Date.now(),gpus:[],ram_total_bytes:32*2**30,ram_used_bytes:30*2**30,system_disk_read_bytes_per_second:2**20},kvmem:{retained_sessions:3,history_hits:0,history_evictions:0,host_payload_bytes:200,host_payload_budget_bytes:1000,disk_read_bytes:0,sampled_at_ms:Date.now()}});
 if(p==='/slots')return json([]);
 if(p==='/v1/chat/completions'){
  sent.push(JSON.parse(body));res.setHeader('Content-Type','text/event-stream');
  res.end('data: '+JSON.stringify({choices:[{delta:{reasoning_content:'exact reasoning'},finish_reason:null}]})+'\n\n'+
   'data: '+JSON.stringify({choices:[{delta:{},finish_reason:'length'}],usage:{completion_tokens:12},timings:{predicted_per_second:40}})+'\n\ndata: [DONE]\n\n');return;
 }
 const file=p.startsWith('/monitor/')?path.join(source,'ui',p.endsWith('/')?p+'index.html':p):path.join(source,'backends/polystrata/serve/web',p==='/'?'index.html':p.replace(/^\/web\//,''));
 if(!fs.existsSync(file)||!fs.statSync(file).isFile()){res.statusCode=404;res.end('{}');return;}
 const ext=path.extname(file);res.setHeader('Content-Type',({'.js':'text/javascript','.html':'text/html','.css':'text/css','.svg':'image/svg+xml'})[ext]||'application/octet-stream');res.end(fs.readFileSync(file));
});
let browser;
(async()=>{
 await new Promise(r=>server.listen(0,'127.0.0.1',r));const base='http://127.0.0.1:'+server.address().port;
 browser=await chromium.launch({headless:true,executablePath:process.env.CHROMIUM});
 const page=await browser.newPage({viewport:{width:1366,height:900}});page.on('pageerror',e=>errors.push(e.message));
 await page.goto(base+'/');await page.locator('#chat-thinking-summary').filter({hasText:'关闭'}).waitFor();
 await page.locator('#thinking-btn').click();assert.equal(await page.locator('#s-max').inputValue(),'32000');assert.equal(await page.locator('#s-budget').inputValue(),'0');
 await page.locator('#drawer-close').click();await page.locator('#input').fill('first');await page.locator('#send-btn').click();
 await page.getByText('思考阶段达到输出上限',{exact:false}).waitFor();assert.equal(sent[0].reasoning_effort,undefined);assert.equal(sent[0].max_tokens,undefined);
 await page.locator('#thinking-btn').click();await page.locator('#s-thinking [data-v=low]').click();await page.locator('#s-budget').fill('128');await page.locator('#s-apply').click();
 await page.locator('#input').fill('second');await page.locator('#send-btn').click();await page.waitForFunction(()=>document.querySelectorAll('.st-msg').length>=4&&!document.querySelector('#send-btn').disabled);
 assert.equal(sent[1].reasoning_effort,'low');assert.equal(sent[1].reasoning_budget_tokens,128);assert.equal(sent[1].messages[1].reasoning_content,'exact reasoning');
 // Regenerating an earlier answer uses the same overrides and excludes later turns.
 await page.locator('[data-msg-regen]').first().click();
 await page.locator('[data-swipe-count]').first().filter({hasText:'2/2'}).waitFor();
 assert.equal(sent[2].reasoning_effort,'low');assert.equal(sent[2].reasoning_budget_tokens,128);
 assert.equal(sent[2].messages.length,1);assert.equal(sent[2].messages[0].content,'first');
 await page.locator('[data-msg-prev]').first().click();
 await page.locator('[data-swipe-count]').first().filter({hasText:'1/2'}).waitFor();
 await page.locator('[data-msg-next]').first().click();
 await page.locator('[data-swipe-count]').first().filter({hasText:'2/2'}).waitFor();
 await page.locator('#thinking-btn').click();await page.screenshot({path:path.join(evidence,'ui-consistency-chat.png'),fullPage:true,animations:'disabled'});
 await page.locator('#s-reset').click();await page.waitForFunction(()=>document.querySelector('#s-thinking [data-v=none]').getAttribute('aria-checked')==='true');
 await page.setViewportSize({width:390,height:844});assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);
 await page.screenshot({path:path.join(evidence,'ui-consistency-chat-mobile.png'),fullPage:true,animations:'disabled'});
 await page.setViewportSize({width:1366,height:900});await page.goto(base+'/monitor/#config');await page.locator('[data-param=Backend]').waitFor();
 await page.locator('#launch-tab-6').click();const idle=page.locator('[data-param=RetainedSessions]');assert.equal(await idle.inputValue(),'3');
 await page.locator('#launch-tab-1').click();assert.equal(await page.locator('[data-param=VramReserveMiB]').inputValue(),'256');await page.locator('[data-param=SpeculativeMode]').selectOption('mtp');assert.equal(await page.locator('[data-param=DraftTokens]').inputValue(),'2');
 await page.locator('#launch-tab-3').click();assert.equal(await page.locator('[data-param=PreserveThinking]').isChecked(),true);await page.locator('[data-param=PreserveThinking]').uncheck();
 await page.locator('[data-action=preview]').click();await page.locator('[data-check-summary]').filter({hasText:'检查通过'}).waitFor();
 assert.equal(previews[0].parameters.RetainedSessions,4);assert.equal(previews[0].parameters.PreserveThinking,false);assert.equal(previews[0].parameters.VramReserveMiB,256);
 await page.locator('[data-dialog=check] [data-dialog-close]').click();await page.reload();await page.locator('[data-param=Backend]').waitFor();await page.locator('#launch-tab-6').click();assert.equal(await idle.inputValue(),'3');
 await page.locator('[data-page=monitor]').click();await page.locator('[data-value=decode-mean]').filter({hasText:'42'}).waitFor();
 assert.equal(await page.locator('[data-live-speed]').isVisible(),false);assert.ok((await page.locator('[data-value=prefill-tokens]').textContent()).includes('62'));
 assert.ok((await page.locator('[data-requests]').textContent()).includes('思考阶段达到输出上限'));
 await page.screenshot({path:path.join(evidence,'ui-consistency-monitor.png'),fullPage:true});
 info.status={state:'ready',url:base,pid:12345,active_config:info.draft};
 await page.reload();await page.locator('[data-chat-link][href="/chat/"]').waitFor({state:'attached'});
 assert.equal(await page.locator('[data-page=chat]').isVisible(),true);
 assert.equal(await page.locator('[data-chat-link]').getAttribute('href'),'/chat/');
 assert.deepEqual(errors,[]);console.log('PASS: chat thinking controls/defaults/overrides/reasoning, mobile layout, idle-session migration, MTP default, explicit false, mean/finish/TTFT monitor');
})().catch(e=>{console.error(e);process.exitCode=1;}).finally(async()=>{if(browser)await browser.close();server.close();});
