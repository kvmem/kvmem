// Built unified chat with the real adapter for three native props contracts; no GPU.
const assert=require('node:assert/strict'),fs=require('node:fs'),http=require('node:http'),path=require('node:path');
const {spawnSync}=require('node:child_process');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const directory=path.resolve(process.env.UI_DIST||path.join(__dirname,'../build/share/kvmem/ui'));
assert.ok(fs.existsSync(path.join(directory,'index.html')),'Built chat index is missing: '+directory);
const types={'.html':'text/html','.js':'text/javascript','.css':'text/css','.json':'application/json','.svg':'image/svg+xml','.png':'image/png','.ico':'image/x-icon','.woff2':'font/woff2'};
const server=http.createServer((request,response)=>{
 const filename=path.resolve(directory,'.'+decodeURIComponent(new URL(request.url,'http://localhost').pathname));
 const target=filename===directory?path.join(directory,'index.html'):filename;
 if(!target.startsWith(directory+path.sep)||!fs.existsSync(target)||!fs.statSync(target).isFile()){response.writeHead(404);return response.end();}
 response.setHeader('Content-Type',types[path.extname(target)]||'application/octet-stream');fs.createReadStream(target).pipe(response);
});
function settings(backend){
 const props={model_alias:'native-'+backend,default_generation_settings:{n_ctx:8192,params:{n_predict:-1,temperature:.2}},modalities:{vision:false}};
 if(backend==='ninfer')props.kvmem={enabled:true,selected_tokens:768};
 if(backend==='llamacpp')props.kvmem={generation_limit:8192,defaults:{enable_thinking:false,reasoning_budget_tokens:0,chat_template_kwargs:{}},sampling:{non_thinking:{temperature:.7},thinking:{temperature:.6}}};
 const input={state:{state:'ready',instance:'fixture-'+backend,url:'http://localhost:99',active_config:{parameters:{Backend:backend}}},props,
  settings:backend==='polystrata'?{effective:{reasoning_effort:'none',max_tokens:0,reasoning_budget_tokens:32,preserve_thinking:true}}:null};
 const script='import json,sys;sys.path.insert(0,"scripts");from console_chat import context_payload;print(json.dumps(context_payload(**json.load(sys.stdin))))';
 const result=spawnSync(process.env.PYTHON||'python',['-X','utf8','-c',script],{cwd:path.join(__dirname,'..'),input:JSON.stringify(input),encoding:'utf8'});
 assert.equal(result.status,0,result.stderr);return JSON.parse(result.stdout);
}
(async()=>{
 await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
 const browser=await chromium.launch({headless:true,...(process.env.CHROMIUM?{executablePath:process.env.CHROMIUM}:{})});
 try{
  for(const backend of ['llamacpp','ninfer','polystrata']){
   const context=await browser.newContext(),page=await context.newPage(),errors=[],bad=[],sent=[];
   const ctx=settings(backend);let exposedContext=ctx,timings={predicted_n:1,predicted_ms:.001,predicted_per_second:1000000};
   const running={state:'ready',instance:'previous-service',active_config:{parameters:{Model:'C:\\models\\display-'+backend+'.gguf'}}};
   page.on('pageerror',e=>errors.push(e.message));page.on('response',r=>{if(r.status()>=400)bad.push(r.url());});
   await page.route('**/api/launcher/info',route=>route.fulfill({json:{token:'fixture-token',status:running}}));
   await page.route('**/api/chat/context',route=>route.fulfill({json:exposedContext}));
   await page.route('**/api/chat/requests/*',route=>route.fulfill({json:{state:'completed'}}));
   await page.route('**/api/chat/cancel',route=>route.fulfill({json:{cancelled:true}}));
   await page.route('**/api/chat/completions',route=>{
    sent.push(route.request().postDataJSON());
    assert.equal(route.request().headers()['x-kvmem-token'],'fixture-token');
    assert.equal(route.request().headers()['x-kvmem-instance'],ctx.instance);
    const speculative=timings.draft_n ? backend==='llamacpp' ? {backend:'draft-mtp',drafted_tokens:8,accepted_tokens:4} : backend==='ninfer' ? {backend:'mtp',drafted_tokens:8,accepted_tokens:4,mtp_drafted_tokens:6,mtp_accepted_tokens:3,ngram_drafted_tokens:2,ngram_accepted_tokens:1} : null : null;
    return route.fulfill({contentType:'text/event-stream',body:'data: '+JSON.stringify({kvmem:speculative ? {speculative}:undefined,choices:[{delta:{content:'Fixture reply'},finish_reason:'stop'}],timings})+'\n\ndata: [DONE]\n\n'});
   });
   try{
    await page.goto(`http://127.0.0.1:${server.address().port}/`);
    await page.getByText(ctx.model,{exact:true}).waitFor();
    // A display name may only come from the same running instance, never the draft.
    running.instance=ctx.instance;
    await page.getByText('display-'+backend,{exact:true}).waitFor();
    for(const state of ['stopped','starting','exited']){
     exposedContext={schema_version:1,state,instance:null,effective:{},capabilities:{},backend:null,model:null};
     await page.waitForResponse(async response=>response.url().endsWith('/api/chat/context')&&(await response.json()).state===state);
     await page.getByText('尚未启动模型',{exact:true}).waitFor();
     assert.equal(await page.locator('.model-heading').textContent(),'尚未启动模型KVMem');
     assert.equal(await page.getByRole('button',{name:'参数',exact:true}).isDisabled(),true);
    }
    exposedContext=ctx;await page.getByText('display-'+backend,{exact:true}).waitFor();
    assert.deepEqual(errors,[],'non-ready service states must not throw or leave the previous model displayed');
    const message=page.getByLabel('消息输入',{exact:true});
    const initialHeight=await message.evaluate(el=>el.clientHeight);
    await message.fill('Line one\nLine two\nLine three\nLine four\nLine five');
    await page.waitForFunction(height=>document.querySelector('textarea').clientHeight>height,initialHeight);
    const viewport=page.viewportSize();
    await message.fill('Long draft line\n'.repeat(15));
    await page.setViewportSize({width:390,height:280});
    await page.waitForFunction(()=>{
     const input=document.querySelector('textarea'),button=document.querySelector('button[type="submit"]'),rect=button.getBoundingClientRect();
     return rect.bottom<=innerHeight&&rect.top>=0&&input.clientHeight<200&&input.scrollHeight>input.clientHeight;
    });
    await page.setViewportSize(viewport);
    await page.waitForFunction(()=>document.querySelector('textarea').clientHeight===200);
    await message.fill('');
    await page.waitForFunction(height=>document.querySelector('textarea').clientHeight===height,initialHeight);
    await page.getByRole('button',{name:'参数',exact:true}).click();
    const dialog=page.getByRole('dialog',{name:'生成参数'}),thinking=dialog.getByLabel('思考',{exact:true});
    assert.equal(await dialog.getByLabel('temperature',{exact:true}).inputValue(),backend==='llamacpp'?'0.7':'0.2');
    assert.equal(await thinking.inputValue(),backend==='ninfer'?'unknown':'false');
    const output=dialog.getByLabel('总输出上限 · tokens',{exact:true});
    assert.equal(await output.inputValue(),'');assert.equal(await output.getAttribute('placeholder'),'未设独立输出上限');
    assert.equal(await output.evaluate(el=>el.checkValidity()),true);
    await output.fill('64');await output.press('Tab');await output.fill('');await output.press('Tab');
    assert.equal(await output.inputValue(),'');assert.equal(await output.getAttribute('placeholder'),'未设独立输出上限');
    await output.fill('16');await output.press('Tab');
    await output.fill('8193');await output.press('Tab');
    await dialog.getByRole('alert').filter({hasText:'max_tokens'}).waitFor();
    assert.equal(await output.inputValue(),'16','invalid value must not look accepted');
    await dialog.getByLabel('关闭面板').click();
    const send=async()=>{
     const count=await page.locator('.message-footer').count();
     await page.getByLabel('消息输入',{exact:true}).fill('Hello');
     await page.getByRole('button',{name:/^发送/}).click();
     await page.locator('.message-footer').nth(count).waitFor();
     await page.getByRole('button',{name:'参数',exact:true}).waitFor();
     return page.locator('.message-footer').last();
    };
    let footer=await send();
    assert.equal(sent[0].model,ctx.model);assert.equal(sent[0].max_tokens,16);
    assert.equal(Object.hasOwn(sent[0],'chat_template_kwargs'),false,'untouched defaults stay server-owned');
    assert.equal(await footer.getByText('1 token',{exact:true}).count(),1);
    assert.equal(await footer.getByText(/tok\/s/).count(),0,'one token has no decode-step rate');
    for(const [native,expected] of [
     [{predicted_n:5,predicted_ms:2000,predicted_per_second:12.5,draft_n:8,draft_n_accepted:4},'平均 12.5 tok/s'],
     [{predicted_n:5,predicted_ms:.001,predicted_per_second:0},null],
     [{predicted_n:5,predicted_ms:.001},null],
     [{predicted_n:5,predicted_ms:.001,predicted_per_second:-1},null],
    ]){
     timings=native;footer=await send();
     assert.equal(await footer.getByText('5 token',{exact:true}).count(),1);
     if(expected)assert.equal(await footer.getByText(expected,{exact:true}).count(),1);
     else assert.equal(await footer.getByText(/tok\/s/).count(),0,'do not fabricate rate from elapsed time');
     if(native.draft_n) {
       if(backend!=='polystrata') assert.match(await footer.locator('.message-meta').textContent(),backend==='llamacpp' ? /MTP 接受 \/ 提议 4 \/ 8/ : /MTP 接受 \/ 提议 3 \/ 6/);
       await footer.locator('summary').click();
       assert.match(await footer.locator('.message-details').textContent(),/推测解码提议 \/ 接受：8 \/ 4/);
       assert.match(await footer.locator('.message-details').textContent(),/解码耗时：2.00s/);
     }
    }
    await page.getByRole('button',{name:'参数',exact:true}).click();
    await thinking.selectOption('true');
    assert.equal(await dialog.getByLabel('temperature',{exact:true}).inputValue(),backend==='llamacpp'?'0.6':'0.2');
    await dialog.getByLabel('关闭面板').click();await send();
    assert.equal(sent.at(-1).chat_template_kwargs.enable_thinking,true);
    if(backend==='polystrata')assert.equal(sent.at(-1).reasoning_effort,'xhigh');
    await page.reload();await page.locator('.message-footer').nth(5).waitFor();
    assert.equal(await page.locator('form').getByRole('button',{name:'删除对话',exact:true}).count(),0);
    await page.locator('.history-menu summary').click();
    page.once('dialog',dialog=>dialog.dismiss());
    await page.getByRole('button',{name:'删除对话',exact:true}).click();
    assert.equal(await page.locator('.message-footer').count(),6,'cancelled deletion keeps the current history');
    assert.deepEqual(errors,[]);assert.deepEqual(bad,[]);
    console.log('PASS unified chat settings, means, thinking and history:',backend);
   }finally{await context.close();}
  }
 }finally{await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;}).finally(()=>server.close());
