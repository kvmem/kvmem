/* Standalone, model-free browser coverage of the real log page and polling contract. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const http = require('node:http');
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
let version=1, hits=0, status=200, session='first', backend='ninfer', running=true, delay=0, legacyMode=false;
let text='[2026-10-24 15:42:18.021] [info] ninfer-serve: Model loaded\n'+
  '2026-10-24 15:42:18.022  WARN memory | Headroom is low\n'+
  '2026-10-24 15:42:18.023  DEBUG serve | prefix=128\n'+
  '[strata] ERROR: <img src=x onerror=alert(1)>\n';
const original=text;
const server=http.createServer((req,res)=>{
  const url=new URL(req.url,'http://localhost');
  if(url.pathname==='/api/launcher/logs'){
    hits++;
    if(legacyMode){res.writeHead(404);return res.end('{}');}
    const body={protocol:1,session,backend,model:'Browser fixture · no model loaded',running,exit_code:null,started_at:1792827738000,updated_at:1792827738000+version*1000,available:!!text,bytes:text.length,limit_bytes:262144,truncated:false,pending:false,revision:String(version),text:url.searchParams.get('revision')===String(version)?null:text};
    return setTimeout(()=>{res.writeHead(status,{'Content-Type':'application/json'});res.end(JSON.stringify(body));},delay);
  }
  if(legacyMode && url.pathname==='/api/launcher/log'){res.writeHead(status,{'Content-Type':'application/json'});return res.end(JSON.stringify({text}));}
  if(url.pathname.startsWith('/api/')||['/props','/slots','/stats'].includes(url.pathname)){res.writeHead(404,{'Content-Type':'application/json'});return res.end('{}');}
  const file=path.join(__dirname,'monitor',url.pathname==='/monitor/'?'index.html':path.basename(url.pathname));
  if(!fs.existsSync(file)){res.writeHead(404);return res.end();}
  const types={'.js':'text/javascript','.css':'text/css','.html':'text/html','.woff2':'font/woff2'};
  res.writeHead(200,{'Content-Type':types[path.extname(file)]||'text/plain'});res.end(fs.readFileSync(file));
});
const sleep=ms=>new Promise(resolve=>setTimeout(resolve,ms));
async function until(fn){for(let i=0;i<100;i++){if(await fn())return;await sleep(100);}throw new Error('Timed out');}
(async()=>{
  await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
  const browser=await chromium.launch({headless:true,...(process.env.CHROMIUM?{executablePath:process.env.CHROMIUM}:{})});
  try{
    const page=await browser.newPage({viewport:{width:1440,height:1040}}),errors=[];
    page.on('pageerror',error=>errors.push(error.message));
    await page.goto('http://127.0.0.1:'+server.address().port+'/monitor/#logs',{waitUntil:'domcontentloaded',timeout:60000});
    await page.locator('.s-log-row').first().waitFor();
    assert.equal(await page.locator('[data-view=logs]').isVisible(),true);
    assert.equal(await page.locator('.s-log-row').count(),4);
    assert.equal(await page.locator('[data-log-rows] img').count(),0);
    assert.equal(await page.locator('[data-log-rows]').textContent(),original);
    assert.equal(await page.locator('.s-log-time,.s-log-level,.s-log-source,.s-log-number').count(),0);
    assert.ok(await page.locator('.s-log-token-warn').count()>0);
    assert.ok(await page.locator('.s-log-token-error').count()>0);
    await page.locator('[data-log-search]').fill('[info] ninfer');
    assert.equal((await page.locator('mark').allTextContents()).join(''),'[info] ninfer');
    await page.locator('[data-log-search]').fill('prefix');assert.equal(await page.locator('.s-log-row').count(),1);
    await page.locator('[data-log-format=raw]').click();assert.ok((await page.locator('[data-log-rows]').textContent()).includes('DEBUG serve | prefix=128'));
    assert.equal(await page.locator('[class^=s-log-token-]').count(),0);
    await page.locator('[data-log-search]').fill('');
    const downloadEvent=page.waitForEvent('download');await page.locator('[data-log-export]').click();
    const download=await downloadEvent;assert.equal(fs.readFileSync(await download.path(),'utf8'),original);
    await page.locator('[data-log-format=styled]').click();
    await page.locator('[data-log-pause]').click();
    text+='INFO while paused\n';version++;
    const pausedHits=hits;await sleep(1250);assert.equal(hits,pausedHits);assert.equal(await page.locator('.s-log-row').count(),4);
    await page.locator('[data-log-pause]').click();await until(async()=>await page.locator('.s-log-row').count()===5);
    // No polling on another view, then an immediate catch-up on return.
    await page.locator('[data-page=about]').click();await sleep(150);const hiddenHits=hits;
    await sleep(1200);assert.equal(hits,hiddenHits);
    await page.goBack();await until(async()=>await page.locator('[data-view=logs]').isVisible());
    for(const id of ['llamacpp','polystrata','ninfer']){
      backend=id;version++;
      text=id==='llamacpp'?'slot print_timing: eval time = 12 ms\n':id==='polystrata'?'[strata] thinking: 128 tokens, 29.7 tok/s\n':original;
      await until(async()=>await page.locator('[data-log-backend]').textContent()===({llamacpp:'llama.cpp',polystrata:'PolyStrata',ninfer:'NInfer'}[id]));
      assert.ok(await page.locator('.s-log-row').count()>0);
    }
    status=503;await until(async()=>!(await page.locator('[data-log-notice]').getAttribute('hidden'))&& (await page.locator('[data-log-state]').textContent()).includes('连接中断'));
    assert.equal(await page.locator('.s-log-row').count(),4);
    status=200;running=false;version++;
    await until(async()=>(await page.locator('[data-log-state]').textContent()).includes('已停止'));
    assert.equal(await page.locator('.s-log-row').count(),4);
    session='new-service';backend='polystrata';running=true;version++;
    text=Array.from({length:2100},(_,i)=>'[strata] token '+i).join('\n')+'\n';
    await until(async()=>await page.locator('.s-log-row').count()===2000);
    await page.locator('[data-log-scroll]').evaluate(el=>el.scrollTop=300);
    await until(async()=>!(await page.locator('[data-log-follow]').isChecked()));
    assert.ok(await page.locator('[data-log-latest]').isVisible());
    const top=await page.locator('[data-log-scroll]').evaluate(el=>el.scrollTop);
    text+='[strata] token 2100\n';version++;
    await until(async()=>(await page.locator('.s-log-message').last().textContent())==='[strata] token 2100');
    assert.ok(Math.abs(await page.locator('[data-log-scroll]').evaluate(el=>el.scrollTop)-top)<60);
    await page.locator('[data-log-latest]').click();assert.ok(await page.locator('[data-log-follow]').isChecked());
    await page.setViewportSize({width:390,height:960});
    await until(async()=>await page.locator('[data-log-scroll]').evaluate(el=>el.scrollHeight-el.clientHeight-el.scrollTop<5));
    assert.equal(await page.locator('.s-log-time,.s-log-level').count(),0);
    text='[strata] 29.7 tok/s\n\n  tensor    type\n\tq_proj    Q4_K\n'+'long '.repeat(200)+'\n';version++;
    await until(async()=>await page.locator('.s-log-row').count()===5);
    assert.equal(await page.locator('[data-log-rows]').textContent(),text);
    await page.locator('[data-log-wrap]').uncheck();
    await until(async()=>await page.locator('[data-log-scroll]').evaluate(el=>el.scrollWidth>el.clientWidth));
    assert.ok(await page.locator('.s-log-token-metric').count()>0);
    await page.locator('[data-log-wrap]').check();
    await until(async()=>await page.locator('[data-log-scroll]').evaluate(el=>el.scrollWidth<=el.clientWidth+1));
    await page.setViewportSize({width:1440,height:1040});
    text=original+'[strata] thinking: 128 tokens, 29.7 tok/s\n'+'slot print_timing: eval time = 12 ms\n';version++;
    await until(async()=>await page.locator('.s-log-row').count()===6);
    await page.locator('[data-theme-select]').selectOption('dark');
    if(process.env.SCREENSHOT_DIR){fs.mkdirSync(process.env.SCREENSHOT_DIR,{recursive:true});await page.screenshot({path:path.join(process.env.SCREENSHOT_DIR,'logs-desktop.png'),fullPage:true});}
    await page.locator('[data-language-select]').selectOption('en');
    assert.equal(await page.locator('[data-page=logs]').textContent(),'Backend logs');
    assert.equal(await page.locator('[data-log-pause]').textContent(),'Pause refresh');
    await page.locator('[data-density-select]').selectOption('compact');
    await page.locator('[data-theme-select]').selectOption('blue');
    for(const width of [736,390,320]){
      await page.setViewportSize({width,height:1000});
      const size=await page.locator('[data-view=logs]').evaluate(el=>({width:el.clientWidth,scroll:el.scrollWidth}));
      assert.ok(size.scroll<=size.width+1,JSON.stringify({width,...size}));
    }
    if(process.env.SCREENSHOT_DIR)await page.screenshot({path:path.join(process.env.SCREENSHOT_DIR,'logs-mobile-light.png'),fullPage:true});
    status=404;await until(async()=>(await page.locator('[data-log-notice]').textContent()).includes('no log API'));
    legacyMode=true;status=200;await page.reload();
    await until(async()=>(await page.locator('[data-log-updated]').textContent()).includes('Legacy console'));
    assert.equal(await page.locator('.s-log-row').count(),6);
    assert.ok((await page.locator('[data-log-label=limit]').textContent()).includes('32 KiB'));
    await page.evaluate(()=>window.dispatchEvent(new CustomEvent('kvmem-runtime-status',{detail:{state:'ready',instance:'legacy-run',active_config:{parameters:{Backend:'polystrata',Model:'C:/models/tiel.gguf'}}}})));
    await until(async()=>(await page.locator('[data-log-model]').textContent())==='tiel.gguf');
    assert.equal(await page.locator('[data-log-backend]').textContent(),'PolyStrata');
    assert.deepEqual(errors,[]);
    console.log('PASS: original text/blank lines/indentation, inline colors, cross-token search, wrap/plain mode, three backends, export, pause, hidden polling, navigation, disconnection/stop/restart, 2000-line cap, scroll anchoring, EN/CN, themes, 320px layout and legacy fallback');
  }finally{await browser.close();await new Promise(resolve=>server.close(resolve));}
})().catch(error=>{console.error(error);process.exitCode=1;server.close();});
