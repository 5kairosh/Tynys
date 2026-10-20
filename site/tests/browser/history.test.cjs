// Real Chromium + IndexedDB. Stub only inference: these tests exercise history,
// not medical model accuracy. Run: npm ci && npx playwright install chromium && npm test.
const {test,before,after}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs/promises');
const path=require('node:path');
const http=require('node:http');
const {createRequire}=require('node:module');
const loader=process.env.TYNYS_PLAYWRIGHT_PATH?createRequire(path.join(process.env.TYNYS_PLAYWRIGHT_PATH,'package.json')):require;
const {chromium}=loader('playwright');
const STATIC=path.resolve(__dirname,'../../static');
const PNG=Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+a+p8AAAAASUVORK5CYII=','base64');
const preview='data:image/png;base64,'+PNG.toString('base64');
let server,browser,base;
let predictCount=0,reportCount=0,reportBody;
function prediction(filename='sample.png'){
  return {filename,model_class:'PNEUMONIA',pneumonia_score:.9,raw_score:.9,threshold:.745,
    model_id:'test-model-v1',width:256,height:256,elapsed_ms:25,preview:filename.endsWith('.dcm')?preview:null,
    assessment:{policy:'ood_guard_v1',status:'review_required',title:'Сенімді бағалау мүмкін емес',message:'OOD сынағы.',notice:'Оқу сынағы.'},
    xray_check:{available:true,is_xray:false},calibration:{T:1},explanation:{available:false},
    comparison:{available:false},quality:{available:true,status:'warning',summary:'Сурет сапасына назар аударыңыз',warnings:[{code:'low_detail',title:'Сурет бұлыңғыр болуы мүмкін',action:'Түпнұсқаны тексеріңіз.'}],notice:'Техникалық сынақ.'}};
}
before(async()=>{
  server=http.createServer(async(req,res)=>{
    try{
      if(req.url==='/api/status'){res.setHeader('Content-Type','application/json');res.end(JSON.stringify({ready:true,model_id:'test-model-v1',config:{threshold:.745},metrics:[]}));return;}
      if(req.url==='/api/predict'||req.url==='/api/report'){
        const chunks=[];for await(const chunk of req)chunks.push(chunk);const body=Buffer.concat(chunks);
        if(req.url==='/api/report'){reportCount++;reportBody=body;res.setHeader('Content-Type','application/pdf');res.end('%PDF-1.4\n%%EOF');return;}
        predictCount++;res.setHeader('Content-Type','application/json');res.end(JSON.stringify(prediction(/filename="([^"]+)"/.exec(body.toString())?.[1]||'sample.png')));return;
      }
      const name=req.url.split('?')[0]==='/'?'index.html':req.url.split('?')[0].replace(/^\/static\//,'');
      const target=path.resolve(STATIC,name);
      if(!target.startsWith(STATIC+path.sep)){res.writeHead(404);res.end();return;}
      res.setHeader('Content-Type',name.endsWith('.js')?'text/javascript':name.endsWith('.css')?'text/css':name.endsWith('.html')?'text/html':'application/octet-stream');
      res.end(await fs.readFile(target));
    }catch{res.writeHead(404);res.end();}
  });
  await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
  base='http://127.0.0.1:'+server.address().port;
  browser=await chromium.launch({headless:true,executablePath:process.env.TYNYS_CHROMIUM_PATH||undefined});
});
after(async()=>{await browser?.close();await new Promise(resolve=>server?.close(resolve));});
async function ready(page){await page.goto(base,{waitUntil:'domcontentloaded'});await page.locator('#save-history').waitFor();await page.waitForFunction(()=>!document.getElementById('save-history').disabled);}
async function contextTest(fn){const context=await browser.newContext();context.setDefaultTimeout(30000);try{await fn(context,await context.newPage());}finally{await context.close();}}
async function upload(page,name='sample.png'){
  await page.waitForFunction(()=>!document.getElementById('image-input').disabled);
  await page.locator('#image-input').setInputFiles({name,mimeType:'image/png',buffer:PNG});
  await page.locator('#analyze').click();
  await page.waitForFunction(()=>!document.body.classList.contains('busy')&&document.getElementById('result-title').textContent.length>0);
}
async function storage(page){return page.evaluate(async()=>{const s=await new TynysHistory.Store().read();return {enabled:s.enabled,records:s.records.length};});}
async function count(page,n){await page.waitForFunction(n=>document.querySelectorAll('#history-list>li').length===n,n);}

test('default opt-out never persists image history',()=>contextTest(async(context,page)=>{
  await ready(page);assert.equal(await page.locator('#save-history').isChecked(),false);
  await upload(page);assert.deepEqual(await storage(page),{enabled:false,records:0});
  assert.equal(await page.locator('.history-saved').textContent(),'Тек осы бетте');
  await page.reload({waitUntil:'domcontentloaded'});await count(page,0);assert.equal((await context.cookies()).length,0);
  assert.equal(await page.evaluate(()=>localStorage.length),0);
}));

test('opt-in restores exact file, score, quality and OOD without inference',()=>contextTest(async(context,page)=>{
  await ready(page);await upload(page,'saved.dcm');await page.locator('#save-history').check();
  await page.waitForFunction(()=>document.getElementById('history-storage-status').textContent.includes('Сақтау қосылды'));
  const before=predictCount;await page.reload({waitUntil:'domcontentloaded'});await count(page,1);
  await page.locator('.history-open').click();
  assert.equal(predictCount,before);assert.equal(await page.locator('#score-value').innerText(),'0,900');
  assert.match(await page.locator('#quality-panel').innerText(),/бұлыңғыр/);
  assert.match(await page.locator('#result-title').innerText(),/Сенімді бағалау/);
  assert.equal(await page.locator('#preview-image').getAttribute('src'),preview);
  const report=reportCount;
  await page.locator('#download-pdf').click();await page.waitForFunction(()=>!document.getElementById('download-pdf').disabled);
  assert.equal(reportCount,report+1);assert.ok(reportBody.includes(PNG));
  assert.ok(reportBody.includes(Buffer.from('saved.dcm')));
}));

test('turning storage off removes disk copies and keeps only page memory',()=>contextTest(async(context,page)=>{
  await ready(page);await page.locator('#save-history').check();await upload(page);
  await page.locator('#save-history').uncheck();
  await page.waitForFunction(()=>document.getElementById('history-storage-status').textContent.includes('Сақтау өшірілді'));
  assert.deepEqual(await storage(page),{enabled:false,records:0});await count(page,1);
  await page.reload({waitUntil:'domcontentloaded'});await count(page,0);assert.equal(await page.locator('#save-history').isChecked(),false);
}));

test('one entry can be deleted permanently',()=>contextTest(async(context,page)=>{
  await ready(page);await page.locator('#save-history').check();await upload(page);
  await page.locator('.history-delete').click();await count(page,0);
  assert.deepEqual(await storage(page),{enabled:true,records:0});
  await page.reload({waitUntil:'domcontentloaded'});await count(page,0);
}));

test('clear removes originals/results, resets consent and clears the open result',()=>contextTest(async(context,page)=>{
  await ready(page);await page.locator('#save-history').check();await upload(page);
  await page.locator('#clear-history').click();await count(page,0);
  assert.deepEqual(await storage(page),{enabled:false,records:0});
  assert.equal(await page.locator('#result-content').isVisible(),false);
  await page.reload({waitUntil:'domcontentloaded'});await count(page,0);assert.equal(await page.locator('#save-history').isChecked(),false);
}));

test('another tab sees deletions and stale consent cannot resurrect deleted images',()=>contextTest(async(context,page)=>{
  await ready(page);await page.locator('#save-history').check();await upload(page);
  const second=await context.newPage();await ready(second);await count(second,1);
  await second.evaluate(async()=>{const state=await new TynysHistory.Store().read();window.staleHistory={record:state.records[0],token:state.generation};});
  await page.locator('#clear-history').click();await count(page,0);await count(second,0);
  const stale=await second.evaluate(async()=>{try{await new TynysHistory.Store().append(staleHistory.record,staleHistory.token);return false;}catch(e){return e instanceof TynysHistory.HistoryChanged;}});
  assert.equal(stale,true);assert.deepEqual(await storage(second),{enabled:false,records:0});
}));

test('expiry and 20-entry limit are enforced by real IndexedDB',()=>contextTest(async(context,page)=>{
  await ready(page);const result=prediction();
  const limits=await page.evaluate(async result=>{
    let now=Date.now();const store=new TynysHistory.Store({now:()=>now});
    const records=Array.from({length:25},(_,i)=>TynysHistory.recordFor({id:'entry-'+i,file:new File(['original'],'entry.png'),time:new Date(now-i*1000),result}));
    await store.enable(records);const initial=await store.read();
    now+=8*24*60*60*1000;const expired=await store.read();
    return {count:initial.records.length,newest:initial.records[0].id,expired:expired.records.length};
  },result);
  assert.deepEqual(limits,{count:20,newest:'entry-0',expired:0});
}));

test('80 MiB storage budget prunes oldest images',()=>contextTest(async(context,page)=>{
  await ready(page);
  const answer=await page.evaluate(async result=>{
    const now=Date.now(),file=new File([new Uint8Array(12*1024*1024)],'large.png');
    const rows=Array.from({length:8},(_,i)=>TynysHistory.recordFor({id:'large-'+i,file,time:new Date(now-i),result}));
    const store=new TynysHistory.Store();await store.enable(rows);const state=await store.read();
    return state.records.map(r=>r.id);
  },prediction());
  assert.deepEqual(answer,['large-0','large-1','large-2','large-3','large-4','large-5']);
}));

test('corrupt saved entries are removed instead of breaking restoration',()=>contextTest(async(context,page)=>{
  await ready(page);
  const answer=await page.evaluate(async()=>{
    const s=new TynysHistory.Store();await s.enable([]);
    await s.transaction(({entries})=>entries.put({id:'bad',version:1,file:'corrupt'}));
    return (await s.read()).records.length;
  });assert.equal(answer,0);
}));

test('quota failure rolls back consent and does not claim saved',()=>contextTest(async(context,page)=>{
  await ready(page);await upload(page);
  await page.evaluate(()=>{const original=IDBObjectStore.prototype.put;IDBObjectStore.prototype.put=function(...args){if(this.name==='entries')throw new DOMException('Quota exceeded','QuotaExceededError');return original.apply(this,args);};});
  await page.locator('#save-history').check();
  await page.waitForFunction(()=>document.getElementById('history-storage-status').textContent.includes('орындалмады'));
  assert.equal(await page.locator('#save-history').isChecked(),false);
  assert.deepEqual(await storage(page),{enabled:false,records:0});await count(page,1);
}));

test('blocked IndexedDB still permits session analysis and honest clearing',()=>contextTest(async(context,page)=>{
  await context.addInitScript(()=>Object.defineProperty(window,'indexedDB',{get(){throw new DOMException('Denied','SecurityError');}}));
  await ready(page);await upload(page);await count(page,1);
  await page.locator('#clear-history').click();await count(page,0);
  assert.match(await page.locator('#history-storage-status').textContent(),/жою расталмады/);
}));

test('saved history opens when inference server is unavailable',()=>contextTest(async(context,page)=>{
  await ready(page);await page.locator('#save-history').check();await upload(page);
  await page.route('**/api/**',route=>route.abort());await page.reload({waitUntil:'domcontentloaded'});await count(page,1);
  await page.locator('.history-open').click();assert.equal(await page.locator('#score-value').innerText(),'0,900');
  assert.equal(await page.locator('#analyze').isEnabled(),false);
  const download=page.waitForEvent('download');await page.locator('#download-result').click();
  assert.equal((await download).suggestedFilename(),'tynys-result.txt');
}));

test('saved filenames are rendered as text, not markup',()=>contextTest(async(context,page)=>{
  await ready(page);await page.locator('#save-history').check();await upload(page,'<img src=x onerror=alert(1)>.png');
  await page.reload({waitUntil:'domcontentloaded'});await count(page,1);
  assert.equal(await page.locator('.history-item-content>strong img').count(),0);
  assert.match(await page.locator('.history-item-content>strong').textContent(),/<img/);
}));

test('failed append leaves previous records intact and the new result session-only',()=>contextTest(async(context,page)=>{
  await ready(page);await page.locator('#save-history').check();await upload(page,'old.png');
  await page.evaluate(()=>{const original=IDBObjectStore.prototype.put;IDBObjectStore.prototype.put=function(...args){if(this.name==='entries')throw new DOMException('Quota exceeded','QuotaExceededError');return original.apply(this,args);};});
  await upload(page,'new.png');await count(page,2);
  assert.deepEqual(await storage(page),{enabled:true,records:1});
  assert.match(await page.locator('#history-storage-status').textContent(),/сақталмады/);
  assert.equal(await page.locator('.history-saved').first().textContent(),'Тек осы бетте');
  await page.reload({waitUntil:'domcontentloaded'});await count(page,1);assert.equal(await page.locator('.history-item-content>strong').textContent(),'old.png');
}));

test('failed disk deletion is reported honestly and can be retried',()=>contextTest(async(context,page)=>{
  await ready(page);await page.locator('#save-history').check();await upload(page);
  await page.evaluate(()=>{window.originalClear=IDBObjectStore.prototype.clear;IDBObjectStore.prototype.clear=function(){throw new DOMException('Denied','UnknownError');};});
  await page.locator('#clear-history').click();await count(page,0);
  assert.match(await page.locator('#history-storage-status').textContent(),/жою расталмады/);
  assert.deepEqual(await storage(page),{enabled:true,records:1});
  await page.evaluate(()=>{IDBObjectStore.prototype.clear=window.originalClear;});
  await page.locator('#clear-history').click();
  await page.waitForFunction(()=>document.getElementById('history-storage-status').textContent.includes('тарих тазаланды'));
  assert.deepEqual(await storage(page),{enabled:false,records:0});
}));

test('different browser profiles do not share saved originals',()=>contextTest(async(context,page)=>{
  await ready(page);await page.locator('#save-history').check();await upload(page);
  const separate=await browser.newContext();try{
    const other=await separate.newPage();await ready(other);await count(other,0);
    assert.deepEqual(await storage(other),{enabled:false,records:0});
  }finally{await separate.close();}
}));

// The aggregate report has an independent denominator from full-cohort metrics.
async function auditStatus(page,report){
  await page.route('**/api/status',route=>route.fulfill({json:{ready:true,model_id:'test-model-v1',config:{threshold:.745},metrics:[],selective_evaluation:report}}));
}
async function actualAudit(){return {available:true,...JSON.parse(await fs.readFile(path.resolve(__dirname,'../../model/selective_evaluation.json'),'utf8'))};}
test('audit cards show actual accepted counts and download exactly that evidence',()=>contextTest(async(context,page)=>{
  const report=await actualAudit(),before=predictCount;await auditStatus(page,report);await ready(page);
  for(const key of ['internal_test','external_openi']){
    const cohort=report.cohorts[key],card=page.locator(`[data-cohort="${key}"]`);
    assert.match(await card.innerText(),new RegExp(`${cohort.accepted.count} / ${cohort.baseline.count}`));
    await card.locator('summary').click();
    assert.match(await card.innerText(),new RegExp(`${cohort.accepted.TN} / ${cohort.accepted.FP} / ${cohort.accepted.FN} / ${cohort.accepted.TP}`));
  }
  const pending=page.waitForEvent('download');await page.locator('#selective-download').click();
  const download=await pending;assert.deepEqual(JSON.parse(await fs.readFile(await download.path(),'utf8')),report);
  assert.equal(predictCount,before);assert.equal(await page.locator('#save-history').isChecked(),false);
  assert.match(await page.locator('.audit-limit').innerText(),/клиникалық сенімділік дәлелденбеген/);
}));
test('zero coverage never renders perfect or zero selective accuracy',()=>contextTest(async(context,page)=>{
  const report=await actualAudit();
  for(const cohort of Object.values(report.cohorts)){
    cohort.accepted={count:0,normal:0,pneumonia:0,TN:0,FP:0,FN:0,TP:0,errors:0,accuracy:null,error_rate:null,sensitivity:null,specificity:null};
    cohort.withheld={...cohort.baseline};cohort.coverage=0;cohort.withheld_rate=1;
  }
  await auditStatus(page,report);await ready(page);
  assert.equal(await page.getByText('Бағалау мүмкін емес',{exact:true}).count(),Object.keys(report.cohorts).length);
  assert.equal(await page.getByText('Қорытынды берілген сурет жоқ',{exact:true}).count(),Object.keys(report.cohorts).length);
  assert.match(await page.locator('#selective-finding').innerText(),/Шектеу анықталды/);
}));
test('unavailable audit cannot block image upload or analysis',()=>contextTest(async(context,page)=>{
  await auditStatus(page,{available:false,message:'Параметрлер сәйкес емес.'});await ready(page);
  assert.match(await page.locator('#selective-status').innerText(),/сәйкес емес/);
  assert.equal(await page.locator('#selective-download').isDisabled(),true);
  await upload(page);assert.match(await page.locator('#result-title').innerText(),/Сенімді бағалау/);
}));
test('audit remains usable on narrow screens without horizontal overflow',()=>contextTest(async(context,page)=>{
  const report=await actualAudit();await page.setViewportSize({width:390,height:844});await auditStatus(page,report);await ready(page);
  await page.locator('#selective-title').scrollIntoViewIfNeeded();
  assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth));
  assert.equal(await page.locator('#selective-cards .audit-card').count(),Object.keys(report.cohorts).length);
}));

test('status refresh preserves an opened audit explanation',()=>contextTest(async(context,page)=>{
  await auditStatus(page,await actualAudit());await ready(page);
  const details=page.locator('[data-cohort="internal_test"] details');await details.locator('summary').click();
  const response=page.waitForResponse('**/api/status');
  await page.evaluate(()=>window.dispatchEvent(new Event('online')));await response;
  assert.equal(await details.getAttribute('open'),'');
}));

test('v2 accepted prediction renders a class and the actual OOD distance',()=>contextTest(async(context,page)=>{
  const r=prediction('current.png');r.assessment={...r.assessment,policy:'ood_guard_v2',status:'classified',message:'Train/validation тексеруі.'};
  r.xray_check={available:true,is_xray:true,within_reference:true,distance:28.2,threshold:29.3,parameter_id:'current-ood'};
  await page.route('**/api/predict',route=>route.fulfill({json:r}));await ready(page);await upload(page,'current.png');
  assert.equal(await page.locator('#result-title').innerText(),'Пневмония санатына жіктелді');
  assert.match(await page.locator('#xray-warning').innerText(),/28,20; OOD шегі: 29,30/);
  assert.equal(await page.locator('#score-value').innerText(),'0,900');
}));
test('legacy saved result keeps its original guard outcome after v2 upgrade',()=>contextTest(async(context,page)=>{
  await ready(page);await page.locator('#save-history').check();await upload(page,'legacy.png');
  const before=predictCount;
  await page.route('**/api/status',route=>route.fulfill({json:{ready:true,model_id:'test-model-v1',config:{threshold:.745},metrics:[],assessment_policy:{name:'ood_guard_v2'},xray_check:{parameter_id:'current-ood'}}}));
  await page.reload({waitUntil:'domcontentloaded'});await count(page,1);await page.locator('.history-open').click();
  assert.equal(predictCount,before);assert.match(await page.locator('#result-title').innerText(),/Сенімді бағалау/);
  assert.match(await page.locator('#history-feedback').innerText(),/OOD ережесі жаңарған/);
  assert.equal(await page.locator('#score-value').innerText(),'0,900');
}));
test('v2 report discloses train calibration roles and previous-policy coverage',()=>contextTest(async(context,page)=>{
  await auditStatus(page,await actualAudit());await ready(page);
  assert.match(await page.locator('#ood-calibration-detail').innerText(),/5401 train/);
  assert.match(await page.locator('#ood-calibration-detail').innerText(),/1352 validation/);
  assert.match(await page.locator('[data-cohort="internal_test"] .audit-change').innerText(),/бұрын 96,76%/);
  assert.match(await page.locator('[data-cohort="external_openi"] .audit-change').innerText(),/бұрын 16,00%/);
}));

async function actualGeneralization(){return JSON.parse(await fs.readFile(path.join(STATIC,'research/generalization-stage6.json'),'utf8'));}
async function generalizationStatus(page,report){
  await page.route('**/api/status',route=>route.fulfill({json:{ready:true,model_id:report.active_model.id,config:{threshold:report.active_model.threshold},calibration:{T:report.active_model.temperature},metrics:[]}}));
}
test('stage6 shows both gains and internal tradeoff and downloads the frozen evidence',()=>contextTest(async(context,page)=>{
  const report=await actualGeneralization(),before=predictCount;await generalizationStatus(page,report);await ready(page);
  await page.waitForFunction(()=>!document.getElementById('generalization-download').disabled);
  assert.equal(await page.locator('#generalization-cards .audit-card').count(),3);
  for(const key of ['rsna_confirmation','external_openi','internal_test']){
    const v=report.cohorts[key];assert.match(await page.locator(`#generalization-cards [data-cohort="${key}"]`).innerText(),new RegExp(`${v.baseline.errors} → ${v.candidate.errors} / ${v.candidate.count}`));
  }
  assert.match(await page.locator('.study-tradeoff').innerText(),/бір қате артық/);
  const pending=page.waitForEvent('download');await page.locator('#generalization-download').click();const d=await pending;
  assert.deepEqual(JSON.parse(await fs.readFile(await d.path(),'utf8')),report);assert.equal(predictCount,before);
  await page.setViewportSize({width:390,height:844});assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth));
}));
test('missing external study cannot block analysis',()=>contextTest(async(context,page)=>{
  const report=await actualGeneralization();await generalizationStatus(page,report);
  await page.route('**/static/research/**',route=>route.fulfill({status:404,body:'missing'}));await ready(page);
  await page.waitForFunction(()=>document.getElementById('generalization-status').textContent.includes('қолжетімсіз'));
  await upload(page);assert.match(await page.locator('#result-title').innerText(),/Сенімді бағалау/);
}));
test('source branch survives saved history and TXT export without re-inference',()=>contextTest(async(context,page)=>{
  const r=prediction('routed.png');r.domain_routing={available:true,selected_expert:'rsna_adapted',rsna_domain_score:.99,notice:'Дереккөзге ұқсастық; диагноз емес.'};
  await page.route('**/api/predict',route=>route.fulfill({json:r}));await ready(page);await page.locator('#save-history').check();await upload(page,'routed.png');
  assert.match(await page.locator('#routing-note').innerText(),/RSNA/);const before=predictCount;
  await page.reload({waitUntil:'domcontentloaded'});await count(page,1);await page.locator('.history-open').click();
  assert.equal(predictCount,before);assert.match(await page.locator('#routing-note').innerText(),/RSNA/);
  const pending=page.waitForEvent('download');await page.locator('#download-result').click();const d=await pending;
  assert.match(await fs.readFile(await d.path(),'utf8'),/Модель тармағы: rsna_adapted/);
}));
test('a mismatched study cannot advertise old performance for another model',()=>contextTest(async(context,page)=>{
  await ready(page);await page.waitForFunction(()=>document.getElementById('generalization-status').textContent.includes('сәйкес емес'));
  assert.equal(await page.locator('#generalization-download').isDisabled(),true);assert.equal(await page.locator('#generalization-cards .audit-card').count(),0);
  await upload(page);assert.equal(await page.locator('#score-value').innerText(),'0,900');
}));

test('stage6 evidence recovers after server disconnect without reload or inference',()=>contextTest(async(context,page)=>{
  const report=await actualGeneralization();let online=true;
  await page.route('**/api/status',route=>route.fulfill({json:online?{ready:true,model_id:report.active_model.id,config:{threshold:report.active_model.threshold},calibration:{T:report.active_model.temperature},metrics:[]}:{ready:false,metrics:[]}}));
  await ready(page);await page.waitForFunction(()=>!document.getElementById('generalization-download').disabled);const before=predictCount;
  online=false;let response=page.waitForResponse('**/api/status');await page.evaluate(()=>window.dispatchEvent(new Event('online')));await response;
  await page.waitForFunction(()=>document.getElementById('generalization-download').disabled);assert.equal(await page.locator('#generalization-cards .audit-card').count(),0);
  online=true;response=page.waitForResponse('**/api/status');await page.evaluate(()=>window.dispatchEvent(new Event('online')));await response;
  await page.waitForFunction(()=>!document.getElementById('generalization-download').disabled);assert.equal(await page.locator('#generalization-cards .audit-card').count(),3);assert.equal(predictCount,before);
}));

// Service recovery uses transport faults, not alternate medical predictions.
test('429 overload keeps the selected file and requires a manual retry',()=>contextTest(async(context,page)=>{
  await ready(page);let requests=0;
  await page.route('**/api/predict',route=>{requests++;return route.fulfill({status:429,headers:{'Retry-After':'5'},contentType:'application/json',body:JSON.stringify({detail:'Сервер бос емес.'})});});
  await page.locator('#image-input').setInputFiles({name:'retry.png',mimeType:'image/png',buffer:PNG});
  await page.locator('#analyze').click();
  await page.waitForFunction(()=>!document.body.classList.contains('busy'));
  assert.match(await page.locator('#error-message').innerText(),/5 секунд/);
  assert.equal(await page.locator('#file-name').innerText(),'retry.png');
  assert.equal(await page.locator('#analyze').isEnabled(),true);
  assert.equal((await storage(page)).records,0);
  await page.waitForTimeout(200);assert.equal(requests,1);
  await page.unroute('**/api/predict');await page.locator('#analyze').click();
  await page.waitForFunction(()=>!document.body.classList.contains('busy'));
  assert.match(await page.locator('#result-title').innerText(),/Сенімді бағалау/);
}));

test('client deadline releases controls without creating history or retrying',()=>contextTest(async(context,page)=>{
  await page.addInitScript(()=>{const original=AbortSignal.timeout.bind(AbortSignal);AbortSignal.timeout=ms=>original(ms===190000?30:ms);});
  await ready(page);let requests=0;
  await page.route('**/api/predict',async route=>{requests++;await new Promise(resolve=>setTimeout(resolve,150));await route.abort().catch(()=>{});});
  await page.locator('#image-input').setInputFiles({name:'timeout.png',mimeType:'image/png',buffer:PNG});
  await page.locator('#analyze').click();
  await page.waitForFunction(()=>!document.body.classList.contains('busy'));
  assert.match(await page.locator('#error-message').innerText(),/күту уақыты аяқталды/);
  assert.equal(await page.locator('#analyze').isEnabled(),true);
  assert.equal(await page.locator('#file-name').innerText(),'timeout.png');
  assert.equal(await page.locator('#result-content').isVisible(),false);
  assert.deepEqual(await storage(page),{enabled:false,records:0});assert.equal(requests,1);
}));

test('PDF overload preserves the prior result and re-enables download',()=>contextTest(async(context,page)=>{
  await ready(page);await upload(page);const title=await page.locator('#result-title').innerText();let requests=0;
  await page.route('**/api/report',route=>{requests++;return route.fulfill({status:503,headers:{'Retry-After':'5'},contentType:'application/json',body:JSON.stringify({detail:'Талдау кезегін күту уақыты аяқталды.'})});});
  await page.locator('#download-pdf').click();
  await page.locator('#error-message').waitFor({state:'visible'});
  await page.waitForFunction(()=>!document.getElementById('download-pdf').disabled);
  assert.match(await page.locator('#error-message').innerText(),/кезегін күту/);
  assert.equal(await page.locator('#result-title').innerText(),title);
  assert.equal(await page.locator('#download-pdf').isEnabled(),true);assert.equal(requests,1);
}));

test('full service becomes available again without losing the selected file',()=>contextTest(async(context,page)=>{
  let accepting=false;
  await page.route('**/api/status',route=>route.fulfill({contentType:'application/json',body:JSON.stringify({ready:true,model_id:'test-model-v1',config:{threshold:.745},metrics:[],service:{accepting,active:1,waiting:2}})}));
  await ready(page);
  await page.locator('#image-input').setInputFiles({name:'waiting.png',mimeType:'image/png',buffer:PNG});
  assert.equal(await page.locator('#analyze').isEnabled(),false);
  assert.match(await page.locator('#header-status').innerText(),/бос емес/);
  assert.match(await page.locator('#service-status').innerText(),/қайта жіберілмейді/);
  accepting=true;
  await page.waitForFunction(()=>!document.getElementById('analyze').disabled);
  assert.equal(await page.locator('#file-name').innerText(),'waiting.png');
  assert.match(await page.locator('#header-status').innerText(),/Модель дайын/);
}));

// Disposable passwords and files, in fresh browser contexts only. Never user history.
const VAULT_PASSWORD='Test-only long vault passphrase 2026';
async function protect(page){
  await page.locator('#protect-history').click();
  await page.locator('#vault-password').fill(VAULT_PASSWORD);
  await page.locator('#vault-confirm').fill(VAULT_PASSWORD);
  await page.locator('#vault-submit').click();
  await page.waitForFunction(()=>document.getElementById('vault-badge').textContent==='ШИФРЛАНҒАН');
}
async function unlock(page,password=VAULT_PASSWORD){
  await page.locator('#unlock-history').click();await page.locator('#vault-password').fill(password);await page.locator('#vault-submit').click();
  await page.waitForFunction(()=>document.getElementById('vault-badge').textContent==='ШИФРЛАНҒАН');
}
async function rawVault(page){return page.evaluate(async()=>new TynysHistory.Store().transaction(({rows,prefs})=>({
  rows:rows.map(r=>({keys:Object.keys(r),version:r.version,file:r.file instanceof Blob,
    filename:r.filename,result:r.result,iv:r.iv?Array.from(r.iv):null,
    cipher:r.ciphertext?Array.from(new Uint8Array(r.ciphertext)):null})),
  prefs:JSON.stringify(prefs)
})));}

test('vault migration encrypts files, names and results, then restores exact data after unlock',()=>contextTest(async(context,page)=>{
  await ready(page);await page.locator('#save-history').check();await upload(page,'private-test.dcm');
  await protect(page);const raw=await rawVault(page);
  assert.equal(raw.rows.length,1);assert.equal(raw.rows[0].version,2);
  assert.equal(raw.rows[0].file,false);assert.equal(raw.rows[0].filename,undefined);assert.equal(raw.rows[0].result,undefined);
  assert.ok(!JSON.stringify(raw).includes(VAULT_PASSWORD));assert.ok(!JSON.stringify(raw).includes('private-test.dcm'));
  const before=predictCount;
  await page.reload({waitUntil:'domcontentloaded'});await count(page,0);
  await page.waitForFunction(()=>document.getElementById('vault-badge').textContent==='ҚҰЛЫПТАУЛЫ');
  assert.equal(await page.locator('#result-content').isVisible(),false);
  await unlock(page);await count(page,1);await page.locator('.history-open').click();
  assert.equal(predictCount,before);assert.equal(await page.locator('#score-value').innerText(),'0,900');
  assert.equal(await page.locator('#preview-image').getAttribute('src'),preview);
  const exact=await page.evaluate(async password=>{const store=new TynysHistory.Store();const state=await store.unlock(password);return {file:Array.from(new Uint8Array(await state.records[0].file.arrayBuffer())),result:state.records[0].result,keyExtractable:store.key.extractable};},VAULT_PASSWORD);
  assert.deepEqual(Buffer.from(exact.file),PNG);assert.deepEqual(exact.result,prediction('private-test.dcm'));assert.equal(exact.keyExtractable,false);
}));

test('wrong password preserves ciphertext and clears both password inputs',()=>contextTest(async(context,page)=>{
  await ready(page);await upload(page);await protect(page);const before=await rawVault(page);
  await page.locator('#lock-history').click();await page.locator('#unlock-history').click();
  await page.locator('#vault-password').fill('Incorrect disposable password');await page.locator('#vault-submit').click();
  await page.locator('#vault-dialog-error').waitFor({state:'visible'});
  assert.match(await page.locator('#vault-dialog-error').innerText(),/Пароль дұрыс емес/);
  assert.equal(await page.locator('#vault-password').inputValue(),'');assert.equal(await page.locator('#vault-confirm').inputValue(),'');
  assert.deepEqual(await rawVault(page),before);
  await page.locator('#vault-password').fill(VAULT_PASSWORD);await page.locator('#vault-submit').click();
  await page.waitForFunction(()=>document.getElementById('vault-badge').textContent==='ШИФРЛАНҒАН');await count(page,1);
}));

test('failed encryption transaction leaves the prior plaintext history intact',()=>contextTest(async(context,page)=>{
  await ready(page);await page.locator('#save-history').check();await upload(page,'migration-test.png');
  await page.evaluate(()=>{window.vaultOriginalPut=IDBObjectStore.prototype.put;IDBObjectStore.prototype.put=function(value,...args){if(this.name==='entries'&&value.version===2)throw new DOMException('Quota test','QuotaExceededError');return window.vaultOriginalPut.call(this,value,...args);};});
  await page.locator('#protect-history').click();await page.locator('#vault-password').fill(VAULT_PASSWORD);await page.locator('#vault-confirm').fill(VAULT_PASSWORD);await page.locator('#vault-submit').click();
  await page.locator('#vault-dialog-error').waitFor({state:'visible'});
  const raw=await rawVault(page);assert.equal(raw.rows.length,1);assert.equal(raw.rows[0].file,true);assert.equal(raw.rows[0].filename,'migration-test.png');assert.ok(!raw.prefs.includes('vault'));
  assert.equal(await page.locator('#vault-badge').innerText(),'ПАРОЛЬСІЗ');
  await page.locator('#close-vault').click();await page.evaluate(()=>{IDBObjectStore.prototype.put=window.vaultOriginalPut;});
}));

test('ciphertext tampering fails authentication without deleting the record',()=>contextTest(async(context,page)=>{
  await ready(page);await upload(page);await protect(page);await page.locator('#lock-history').click();
  await page.evaluate(async()=>new TynysHistory.Store().transaction(({entries,rows})=>{new Uint8Array(rows[0].ciphertext)[0]^=1;entries.put(rows[0]);}));
  const before=await rawVault(page);
  await page.locator('#unlock-history').click();await page.locator('#vault-password').fill(VAULT_PASSWORD);await page.locator('#vault-submit').click();
  await page.locator('#vault-dialog-error').waitFor({state:'visible'});
  assert.match(await page.locator('#vault-dialog-error').innerText(),/өшіp|өшірілген жоқ|оқу мүмкін/);
  assert.deepEqual(await rawVault(page),before);await count(page,0);
}));

test('ciphertext is bound to record identity and expiry metadata',()=>contextTest(async(context,page)=>{
  await ready(page);const result=await page.evaluate(async({password,prediction})=>{
    const cases=[];
    for(const field of ['id','createdAt','expiresAt']){
      const store=new TynysHistory.Store({name:'aad-test-'+field});
      const record=TynysHistory.recordFor({id:'original',file:new File(['engineering bytes'],'aad.png'),time:new Date(),result:prediction});
      await store.enable([record]);await store.protect(password);
      await store.transaction(({entries,rows})=>{const row=rows[0];entries.delete(row.id);if(field==='id')row.id='swapped';else row[field]-=1000;entries.put(row);});
      store.lock();let rejected=false;try{await store.unlock(password);}catch{rejected=true;}
      cases.push({field,rejected,remaining:await store.transaction(({rows})=>rows.length)});
    }
    return cases;
  },{password:VAULT_PASSWORD,prediction:prediction()});
  assert.ok(result.every(r=>r.rejected&&r.remaining===1),JSON.stringify(result));
}));

test('new tabs stay locked, and a cross-tab lock suppresses an in-flight result',()=>contextTest(async(context,page)=>{
  await ready(page);await upload(page);await protect(page);
  const other=await context.newPage();await ready(other);
  assert.equal(await other.locator('#vault-badge').innerText(),'ҚҰЛЫПТАУЛЫ');await count(other,0);
  await unlock(other);await count(other,1);
  let release,started;const requestStarted=new Promise(resolve=>started=resolve);
  await page.route('**/api/predict',async route=>{started();await new Promise(resolve=>release=resolve);await route.fulfill({contentType:'application/json',body:JSON.stringify(prediction('late.png'))});});
  await page.locator('#image-input').setInputFiles({name:'late.png',mimeType:'image/png',buffer:PNG});await page.locator('#analyze').click();await requestStarted;
  await other.locator('#lock-history').click();await page.waitForFunction(()=>document.getElementById('vault-badge').textContent==='ҚҰЛЫПТАУЛЫ');
  release();await page.waitForFunction(()=>!document.body.classList.contains('busy'));
  await count(page,0);assert.equal(await page.locator('#result-content').isVisible(),false);assert.equal(await page.locator('#preview').isVisible(),false);
  assert.equal(await page.locator('#result-title').textContent(),'');assert.equal(await page.locator('#file-name').textContent(),'');
  assert.equal((await rawVault(page)).rows.length,1);
  await page.unroute('**/api/predict');
  await upload(page,'locked-session.png');assert.equal((await rawVault(page)).rows.length,1);
  assert.equal(await page.locator('.history-saved').innerText(),'Тек осы бетте');
}));

test('locking another tab suppresses a pending PDF download',()=>contextTest(async(context,page)=>{
  await ready(page);await upload(page);await protect(page);
  const other=await context.newPage();await ready(other);await unlock(other);
  let release,started,downloads=0;const requestStarted=new Promise(resolve=>started=resolve);
  page.on('download',()=>downloads++);
  await page.route('**/api/report',async route=>{started();await new Promise(resolve=>release=resolve);await route.fulfill({contentType:'application/pdf',body:'%PDF-1.4\n%%EOF'});});
  await page.locator('#download-pdf').click();await requestStarted;await other.locator('#lock-history').click();
  await page.waitForFunction(()=>document.getElementById('vault-badge').textContent==='ҚҰЛЫПТАУЛЫ');
  release();await page.waitForFunction(()=>!document.getElementById('download-pdf').disabled);
  assert.equal(downloads,0);assert.equal(await page.locator('#result-content').isVisible(),false);
}));

test('five-minute idle lock and cached-page exit remove visible decrypted data',()=>contextTest(async(context,page)=>{
  await ready(page);await upload(page);await protect(page);
  await page.evaluate(()=>{const original=Date.now;Date.now=()=>original()+301000;});
  await page.waitForFunction(()=>document.getElementById('vault-badge').textContent==='ҚҰЛЫПТАУЛЫ');
  assert.equal(await page.locator('#result-content').isVisible(),false);await count(page,0);
  await unlock(page);await page.locator('.history-open').click();
  await page.evaluate(()=>window.dispatchEvent(new PageTransitionEvent('pagehide',{persisted:true})));
  await count(page,0);assert.equal(await page.locator('#result-content').isVisible(),false);
  assert.equal((await rawVault(page)).rows.length,1);
}));

test('locked expiry is cleaned, encrypted limits apply and v1 clients fail closed',()=>contextTest(async(context,page)=>{
  await ready(page);
  const state=await page.evaluate(async({password,result})=>{
    let now=Date.now();const store=new TynysHistory.Store({name:'encrypted-limit-test',now:()=>now});
    const rows=Array.from({length:25},(_,i)=>TynysHistory.recordFor({id:'encrypted-'+i,file:new File(['engineering'],'test.png'),time:new Date(now-i),result}));
    await store.protect(password,rows);const initial=await store.read();
    store.lock();const oldDenied=await new Promise(resolve=>{const req=indexedDB.open('encrypted-limit-test',1);req.onerror=()=>resolve(req.error.name==='VersionError');req.onsuccess=()=>{req.result.close();resolve(false);};});
    now+=8*24*60*60*1000;const expired=await store.read();
    return {initial:initial.records.length,oldDenied,expired:expired.storedCount,locked:expired.locked};
  },{password:VAULT_PASSWORD,result:prediction()});
  assert.deepEqual(state,{initial:20,oldDenied:true,expired:0,locked:true});
}));

test('a migration race cannot overwrite another tab append',()=>contextTest(async(context,page)=>{
  await ready(page);const state=await page.evaluate(async({password,result})=>{
    const first=new TynysHistory.Store({name:'vault-race-test'}),second=new TynysHistory.Store({name:'vault-race-test'});
    const record=id=>TynysHistory.recordFor({id,file:new File(['test'],'race.png'),time:new Date(),result});
    const enabled=await first.enable([record('first')]);
    const original=crypto.subtle.deriveKey.bind(crypto.subtle);let resume,signal;
    const entered=new Promise(resolve=>signal=resolve);
    crypto.subtle.deriveKey=async(...args)=>{signal();await new Promise(resolve=>resume=resolve);return original(...args);};
    const pending=first.protect(password).then(()=>false,error=>error instanceof TynysHistory.HistoryChanged);
    await entered;await second.append(record('second'),enabled.generation);resume();const rejected=await pending;delete crypto.subtle.deriveKey;
    const saved=await first.read();return {rejected,count:saved.records.length,protected:saved.protected};
  },{password:VAULT_PASSWORD,result:prediction()});
  assert.deepEqual(state,{rejected:true,count:2,protected:false});
}));

test('passwords and keys never leave the browser, and cancel has no storage effect',()=>contextTest(async(context,page)=>{
  const requests=[];page.on('request',request=>requests.push(request.url()+' '+(request.postData()||'')));
  await ready(page);await page.locator('#protect-history').click();await page.locator('#vault-password').fill(VAULT_PASSWORD);await page.locator('#close-vault').click();
  assert.equal(await page.locator('#vault-password').inputValue(),'');assert.deepEqual(await storage(page),{enabled:false,records:0});
  await upload(page);await protect(page);await page.locator('#lock-history').click();await unlock(page);
  assert.ok(requests.every(value=>!value.includes(VAULT_PASSWORD)));assert.ok(!(await rawVault(page)).prefs.includes('CryptoKey'));
  assert.equal(await page.evaluate(()=>localStorage.length+sessionStorage.length),0);assert.equal((await context.cookies()).length,0);
}));

test('missing secure crypto disables protection but keeps analysis usable',()=>contextTest(async(context,page)=>{
  await page.addInitScript(()=>Object.defineProperty(window,'isSecureContext',{value:false,configurable:true}));
  await ready(page);assert.equal(await page.locator('#protect-history').isEnabled(),false);await upload(page);
  assert.equal(await page.locator('#result-content').isVisible(),true);assert.deepEqual(await storage(page),{enabled:false,records:0});
}));


test('v1 database upgrade keeps legacy records until the user elects protection',()=>contextTest(async(context,page)=>{
  await ready(page);const result=await page.evaluate(async prediction=>{
    const record=TynysHistory.recordFor({id:'v1-existing',file:new File(['legacy test bytes'],'legacy.png'),time:new Date(),result:prediction});
    await new Promise((resolve,reject)=>{const request=indexedDB.open('legacy-v1-fixture',1);request.onupgradeneeded=()=>{request.result.createObjectStore('entries',{keyPath:'id'});request.result.createObjectStore('settings',{keyPath:'key'});};request.onsuccess=()=>{const db=request.result,tx=db.transaction(['entries','settings'],'readwrite');tx.objectStore('entries').put(record);tx.objectStore('settings').put({key:'consent',enabled:true,generation:'legacy-consent'});tx.oncomplete=()=>{db.close();resolve();};tx.onabort=()=>reject(tx.error);};});
    const store=new TynysHistory.Store({name:'legacy-v1-fixture'});const state=await store.read();
    return {enabled:state.enabled,protected:state.protected,name:state.records[0].filename,bytes:await state.records[0].file.text(),result:state.records[0].result};
  },prediction('legacy.png'));
  assert.deepEqual(result,{enabled:true,protected:false,name:'legacy.png',bytes:'legacy test bytes',result:prediction('legacy.png')});
}));

test('encrypted byte budget accounts for ciphertext and prunes older large entries',()=>contextTest(async(context,page)=>{
  await ready(page);const result=await page.evaluate(async({password,prediction})=>{
    const store=new TynysHistory.Store({name:'encrypted-bytes-test'}),file=new File([new Uint8Array(5*1024*1024)],'engineering-large.png'),now=Date.now();
    const rows=Array.from({length:17},(_,i)=>TynysHistory.recordFor({id:'size-'+i,file,time:new Date(now-i),result:prediction}));
    await store.protect(password,rows);
    return store.transaction(({rows})=>({count:rows.length,used:rows.reduce((n,row)=>n+row.ciphertext.byteLength+2048,0),versions:rows.map(r=>r.version)}));
  },{password:VAULT_PASSWORD,prediction:prediction()});
  assert.equal(result.count,15);assert.ok(result.used<=80*1024*1024);assert.ok(result.versions.every(v=>v===2));
}));


test('blank passwords and inconsistent consent cannot create or expose a vault',()=>contextTest(async(context,page)=>{
  await ready(page);const result=await page.evaluate(async({password,result})=>{
    const store=new TynysHistory.Store({name:'vault-consent-test'});
    const row=TynysHistory.recordFor({id:'consent-row',file:new File(['test'],'consent.png'),time:new Date(),result});
    await store.enable([row]);let blankRejected=false,disabledRejected=false;
    try{await store.protect('            ');}catch{blankRejected=true;}
    const stillPlain=(await store.read()).records.length===1;
    await store.protect(password);await store.transaction(({settings,prefs})=>settings.put({...prefs,enabled:false}));
    try{await store.read();}catch{disabledRejected=true;}
    return {blankRejected,stillPlain,disabledRejected,cipherRecords:await store.transaction(({rows})=>rows.filter(r=>r.version===2).length)};
  },{password:VAULT_PASSWORD,result:prediction()});
  assert.deepEqual(result,{blankRejected:true,stillPlain:true,disabledRejected:true,cipherRecords:1});
}));
