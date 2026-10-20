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
