'use strict';
const $=id=>document.getElementById(id);
let selected=[], results=[], status=null, busy=false, previewUrl=null;
let selectionRevision=0,analysisRequest=null,retryRevision=null;
let analysisHistory=[], activeHistoryId=null;
let historyReady=false,historyWorking=false,historyEnabled=false,historyGeneration=null,pendingHistorySync=false,historyDeletionPending=false;
const historyStore=new TynysHistory.Store();
let historyProtected=false,historyLocked=false,storedHistoryCount=0,privacyEpoch=0;
let vaultMode='protect',lastHistoryActivity=Date.now();
const HISTORY_IDLE_MS=5*60*1000;
let historyChannel=null;
try{if(typeof BroadcastChannel==='function')historyChannel=new BroadcastChannel('tynys-history-events-v1');}catch{}
let statusLoading=false, statusTimer=null, connectionError=false;
function scheduleStatusCheck(){
  clearTimeout(statusTimer);
  statusTimer=setTimeout(()=>loadStatus(),status?.ready&&status?.service?.accepting!==false?15000:3000);
}
const decimal=(n,d=3)=>Number(n).toFixed(d).replace('.',',');
const percent=n=>decimal(n*100,2)+'%';
function error(message){$('error-message').textContent=message;$('error-message').hidden=!message;}
function updateButton(){
  const locked=busy||historyWorking||!historyReady;
  $('analyze').disabled=locked||!selected.length||!status?.ready||status?.service?.accepting===false;
  const selectionLocked=historyWorking||!historyReady||(busy&&!analysisRequest);
  $('image-input').disabled=selectionLocked;
  $('replace-image').disabled=selectionLocked;
  $('remove-image').disabled=selectionLocked;
  $('retry-analysis').disabled=locked||!selected.length;
  $('clear-history').disabled=locked||(!analysisHistory.length&&!historyEnabled&&!historyDeletionPending);
  $('save-history').disabled=locked;
  for(const id of ['protect-history','unlock-history','lock-history'])$(id).disabled=locked;
  $('protect-history').disabled=locked||!TynysHistory.secureAvailable();
  $('download-pdf').disabled=locked||!status?.ready||status?.service?.accepting===false;
  for(const button of $('history-list').querySelectorAll('button'))button.disabled=locked;
  if(pendingHistorySync&&!locked){pendingHistorySync=false;queueMicrotask(()=>void syncHistory());}
}
function requestError(message,retryable=true){return Object.assign(new Error(message),{retryable});}
function connectionFailure(e){
  return requestError(e.name==='TimeoutError'||e.name==='AbortError'
    ?'Жауапты күту уақыты аяқталды. Күту тоқтатылды. «Қайта көру» батырмасын басыңыз.'
    :'Серверге қосылмады. Интернетті тексеріп, «Қайта көру» батырмасын басыңыз. Сурет автоматты түрде қайта жіберілмейді.');
}
async function apiResponse(url,options={}){
  let response;
  try{
    const seconds=url==='/api/report'?280:url==='/api/predict'?190:8;
    response=await fetch(url,{...options,signal:options.signal||AbortSignal.timeout(seconds*1000)});
  }catch(e){
    throw connectionFailure(e);
  }
  if(!response.ok){
    const data=await response.json().catch(()=>({}));
    const retry=Number(response.headers.get('Retry-After'));
    const message=typeof data.detail==='string'?data.detail:'Сұрауды орындау мүмкін болмады.';
    throw requestError(message+(retry>0&&retry<=300?` Қайта жіберу автоматты емес; ${retry} секунд күтіңіз.`:''),[408,425,429,500,502,503,504].includes(response.status));
  }
  return response;
}
async function api(url,options){
  const response=await apiResponse(url,options);
  try{return await response.json();}catch(e){
    if(e.name==='TimeoutError'||e.name==='AbortError'||e instanceof TypeError)throw connectionFailure(e);
    throw requestError('Серверден дұрыс жауап алынбады. «Қайта көру» батырмасын басыңыз.');
  }
}
function renderService(){
  const box=$('service-status');
  box.classList.toggle('service-offline',!status?.ready);
  if(!status?.ready){box.textContent='Сервермен байланыс тексерілуде. Сервер оянып жатқанда алғашқы қосылу біраз уақыт алуы мүмкін. Қосылым автоматты тексеріледі.';return;}
  const service=status.service;
  box.textContent=service?.accepting===false?'Сервер бос емес. Кезек босағанда талдау батырмасы автоматты қосылады. Таңдалған сурет қайта жіберілмейді.':service?.active||service?.waiting?`Сервер жұмыс істеп тұр: ${service.active} талдау орындалуда, ${service.waiting} сұрау кезекте. Суреттер кезекпен өңделеді.`:'Сервер дайын. Суреттер кезекпен өңделеді; орын толса, қайта жіберу туралы хабарлама шығады.';
}
async function loadStatus(){
  if(statusLoading)return;
  if(busy){scheduleStatusCheck();return;}
  statusLoading=true;
  try{status=await api('/api/status',{signal:AbortSignal.timeout(8000)});
  renderService();
  renderSelectiveEvaluation(status.selective_evaluation);
  void renderGeneralizationEvaluation(status);
  if(connectionError&&status.ready){if(retryRevision===null)error('');connectionError=false;}
$('header-status').textContent=status.ready?(status.service?.accepting===false?'Сервер бос емес':'Модель дайын'):'Модель қолжетімсіз';$('status-dot').classList.toggle('ready',status.ready);if(!status.ready)error(status.message||'Модель қосылмаған.');const m=status.metrics?.find(x=>x.model==='MobileNetV2');const cnn=status.metrics?.find(x=>x.model==='CNN');const ensemble=status.metrics?.find(x=>x.model==='Ensemble');if(m){$('metric-accuracy').textContent=percent(m.accuracy);$('metric-recall').textContent=percent(m.recall);$('metric-count').textContent=Math.round(m.TN+m.FP+m.FN+m.TP);const body=$('metrics-table');body.replaceChildren();for(const [key,label,type] of [['accuracy','Accuracy','percent'],['precision','Precision','percent'],['recall','Recall / sensitivity','percent'],['specificity','Specificity','percent'],['f1','F1-score','decimal'],['roc_auc','ROC-AUC','decimal']]){const tr=document.createElement('tr');for(const value of [label,cnn?cnn[key]:null,m[key],status.ensemble?.available?ensemble?.[key]:null]){const td=document.createElement('td');td.textContent=typeof value==='string'?value:value==null?'—':type==='percent'?percent(value):decimal(value,4);tr.append(td);}body.append(tr);}}const external=status.external_validation;$('external-note').textContent=external?.available?`Сыртқы тексеру (OpenI, ${external.metrics.count} сурет): accuracy ${percent(external.metrics.accuracy)}, recall ${percent(external.metrics.recall)}. Ішкі тесттен бөлек. Топтау есеп ID-іне сүйенеді.`:(external?.message||'Сыртқы бағалау қолжетімсіз.');const paired=status.mcnemar;$('mcnemar-note').textContent=paired?.available?`Ішкі test: McNemar p=${Number(paired.p_value).toExponential(3)}, α=0,05. ${paired.significant?'Айырма статистикалық маңызды.':'Статистикалық маңыздылық анықталмады.'} Бұл осы тестке қатысты.`:(paired?.message||'McNemar нәтижесі қолжетімсіз.');if(status.config)$('research-threshold').textContent=decimal(status.config.threshold);updateButton();}catch(e){
    status=null;connectionError=true;renderService();
    void renderGeneralizationEvaluation(null);
    renderSelectiveEvaluation({available:false,message:'Серверге қосылмады. Бағалау есебі сервер қосылғанда қайта тексеріледі.'});
    $('header-status').textContent='Серверге қосылмады';
    $('status-dot').classList.remove('ready');
    if(retryRevision===null)error('Серверге қосылмады. Интернетті және сервердің қосылғанын тексеріңіз. Қосылым автоматты түрде қайта тексеріледі.');
    updateButton();
  }finally{statusLoading=false;scheduleStatusCheck();}
}
window.addEventListener('online',()=>loadStatus());
window.addEventListener('focus',()=>loadStatus());
function analysisProgress(loading,label='Сурет талдануда…'){
  $('analysis-progress').hidden=!loading;$('analysis-progress-text').textContent=loading?label:'';
  $('result-panel').setAttribute('aria-busy',String(loading));
  $('analyze-label').textContent=loading?label:'Суретті талдау';
}
function invalidateAnalysis(){
  selectionRevision++;retryRevision=null;$('retry-analysis').hidden=true;
  const previous=analysisRequest;analysisRequest=null;
  if(previous){previous.controller.abort();busy=false;document.body.classList.remove('busy');}
  analysisProgress(false);
}
function resetResult(){
  $('empty-result').hidden=false;$('result-content').hidden=true;$('result-tag').textContent='КҮТУДЕ';
  for(const id of ['attention-tools','attention-unavailable','comparison-panel','quality-panel'])$(id).hidden=true;
  for(const id of ['result-title','result-category','result-explanation','score-value','result-time','result-model','routing-note','comparison-note','quality-summary','attention-caption'])$(id).textContent='';
  for(const id of ['batch-results','comparison-table','quality-warnings'])$(id).replaceChildren();
  $('score-fill').style.width='0%';results=[];
}
function selectFiles(files){
  if((busy&&!analysisRequest)||historyWorking||!historyReady)return;
  const list=Array.from(files);if(!list.length)return;
  clearSelection();renderHistory();updateButton();
  if(list.length>10){error('Бір ретте 10 суретке дейін таңдаңыз.');return;}
  for(const f of list){
    if(!/\.(jpe?g|png|dcm)$/i.test(f.name)){error('Тек JPG, JPEG, PNG немесе DICOM таңдаңыз.');return;}
    if(f.size>12*1024*1024){error('Әр суреттің көлемі 12 МБ-тан аспауы керек.');return;}
    if(!f.size){error('Бос файлды жүктеу мүмкін емес.');return;}
  }
  selected=list;previewUrl=URL.createObjectURL(selected[0]);$('preview-image').src=previewUrl;
  $('preview-image').alt='Жүктелген рентген суреті';
  const isDicom=/\.dcm$/i.test(selected[0].name);$('preview-image').hidden=isDicom;$('dicom-placeholder').hidden=!isDicom;
  $('preview').hidden=false;$('dropzone').hidden=true;$('file-info').hidden=false;$('replace-image').hidden=false;
  $('file-name').textContent=selected[0].name;
  $('file-size').textContent=(selected[0].size/1024).toFixed(0)+' КБ'+(selected.length>1?` · ${selected.length} сурет таңдалды`:'');
  $('file-list').textContent=selected.length>1?selected.slice(1).map(f=>f.name).join(' · '):'';updateButton();
}
$('image-input').addEventListener('change',e=>selectFiles(e.target.files));
$('dropzone').addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();$('image-input').click();}});
for(const type of ['dragenter','dragover'])$('dropzone').addEventListener(type,e=>{e.preventDefault();$('dropzone').classList.add('dragging');});
for(const type of ['dragleave','drop'])$('dropzone').addEventListener(type,e=>{e.preventDefault();$('dropzone').classList.remove('dragging');});
$('dropzone').addEventListener('drop',e=>selectFiles(e.dataTransfer.files));
function clearSelection(){invalidateAnalysis();activeHistoryId=null;selected=[];if(previewUrl)URL.revokeObjectURL(previewUrl);previewUrl=null;$('preview-image').removeAttribute('src');$('preview').hidden=true;$('replace-image').hidden=true;$('dropzone').hidden=false;$('file-info').hidden=true;$('file-name').textContent='';$('file-size').textContent='';$('file-list').textContent='';$('image-input').value='';resetResult();error('');}
$('remove-image').addEventListener('click',()=>{if((busy&&!analysisRequest)||historyWorking)return;clearSelection();renderHistory();updateButton();});
$('replace-image').addEventListener('click',()=>$('image-input').click());

function assessmentFor(r){
  if(['ood_guard_v1','ood_guard_v2'].includes(r.assessment?.policy)&&['classified','review_required'].includes(r.assessment.status))return r.assessment;
  return {status:'review_required',title:'Сенімді бағалау мүмкін емес',message:'Бағалау күйі алынбады. Санат бойынша қорытынды берілмейді. Сайтты жаңартып, қайта талдаңыз.',notice:'Төмендегі модель баллдары тек техникалық нәтиже.'};
}
const needsReview=r=>assessmentFor(r).status==='review_required';
const resultLabel=r=>needsReview(r)?assessmentFor(r).title:r.model_class;

function explainResult(r){
  const assessment=assessmentFor(r),review=needsReview(r);
  const positive=r.pneumonia_score>=r.threshold;
  const label=positive?'PNEUMONIA':'NORMAL';
  const relation=positive?'осы шекке тең немесе одан жоғары':'осы шектен төмен';
  const comparison=`${decimal(r.pneumonia_score,6)} ${positive?'≥':'<'} ${decimal(r.threshold,6)}`;
  return {
    title:review?'Неге санат бойынша қорытынды берілмеді?':positive?'Неге PNEUMONIA деп жіктеді?':'Неге NORMAL деп жіктеді?',
    reason:review?assessment.message:`Модель суретке ${decimal(r.pneumonia_score,6)} балл берді. Сақталған шешім шегі — ${decimal(r.threshold,6)}. Балл ${relation} (${comparison}). Сондықтан жіктеу ережесі ${label} санатын таңдады. Салыстыру толық дәлдіктегі баллмен жасалады.`,
    method:'MobileNetV2 осы суретті 224 × 224 өлшемінде өңдеп, үйрету кезінде NORMAL және PNEUMONIA суреттерінен үйренген белгілер бойынша бір балл есептейді. Балл validation жиынында таңдалған шекпен салыстырылады.',
    meaning:review?assessment.notice:positive?'Бұл модельдің осы суретті пневмония санатына жатқызғанын білдіреді. Нәтиже адамның пневмониямен ауырғанын өздігінен растамайды.':'Бұл модельдің осы суретке берген баллы пневмония санатын таңдау шегіне жетпегенін білдіреді. Нәтиже пневмонияның жоқтығын немесе адамның толық сау екенін дәлелдемейді.',
    limit:'Grad-CAM таңдалған санатқа оң ықпал еткен белгілердің жуық аймақтарын көрсетеді. Карта 7 × 7 белгілерден үлкейтіледі, сондықтан дәл шекара немесе ауру ошағын көрсетпейді. Ол нақты медициналық белгілердің атауын анықтамайды. Суреттің медициналық түсіндірмесін маман береді.'
  };
}

function showResult(r){
  if(r.preview){$('preview-image').src=r.preview;$('preview-image').hidden=false;$('dicom-placeholder').hidden=true;}
  const pn=r.model_class==='PNEUMONIA',review=needsReview(r);$('empty-result').hidden=true;$('result-content').hidden=false;$('result-tag').textContent=review?'БАҒАЛАУ ШЕКТЕЛГЕН':'ДАЙЫН';$('result-category').textContent=review?'ҚОРЫТЫНДЫ БЕРІЛМЕДІ':r.model_class;$('result-category').classList.toggle('pneumonia',pn||review);$('result-title').textContent=review?assessmentFor(r).title:pn?'Пневмония санатына жіктелді':'Қалыпты санатқа жіктелді';$('result-explanation').textContent=explainResult(r).reason;$('assessment-note').hidden=!review;$('assessment-note').textContent=review?assessmentFor(r).notice:'';$('score-value').textContent=decimal(r.pneumonia_score);$('calibration-note').textContent=r.calibration?`Validation-да калибрленген балл (T=${decimal(r.calibration.T)}). Клиникалық ықтималдық ретінде тексерілмеген.`:'Калибрлеу қолданылмаған (T=1).';$('score-fill').style.width=(r.pneumonia_score*100)+'%';$('score-fill').style.background=pn||review?'#ba9651':'var(--green)';$('threshold-marker').style.left=(r.threshold*100)+'%';$('threshold-label').textContent='Шек: '+decimal(r.threshold);$('result-time').textContent=`${r.width} × ${r.height} px · ${r.elapsed_ms} мс`;$('result-model').textContent=r.domain_routing?.available?'MobileNetV2 · Routed':'MobileNetV2';$('routing-note').hidden=!r.domain_routing?.available;$('routing-note').textContent=r.domain_routing?.available?`Таңдалған тармақ: ${r.domain_routing.selected_expert==='rsna_adapted'?'RSNA деректерімен бейімделген':'бастапқы Kaggle'}. Бұл дереккөзге ұқсастық таңдауы, диагноздың дұрыстығын растамайды.`:'';showAttention(r);const container=$('batch-results');container.replaceChildren();if(results.length>1)for(const item of results){const row=document.createElement('div'),name=document.createElement('span'),label=document.createElement('strong');name.textContent=item.filename;label.textContent=resultLabel(item)+' · '+decimal(item.pneumonia_score)+(item.quality?.status==='warning'?' · Сапа ескертуі':!item.quality?.available?' · Сапа тексерілмеді':'');row.append(name,label);container.append(row);}}
function showAttention(r){
  showQuality(r);
  const check=r.xray_check;
  $('xray-warning').hidden=false;
  $('xray-warning').classList.toggle('flagged',needsReview(r));
  $('xray-warning').textContent=check?.available?((check.is_xray===false?'Сурет белгілері reference үлестірімінен ерекшеленеді (OOD). Басқа дереккөзден алынған нақты рентген де осылай белгіленуі мүмкін.':'OOD: reference үлестірімінен айқын ауытқу табылмады. Бұл рентген екенін немесе болжамның дұрыстығын растамайды.')+(Number.isFinite(check.distance)&&Number.isFinite(check.threshold)?` Қашықтық: ${decimal(check.distance,2)}; OOD шегі: ${decimal(check.threshold,2)}.`:'')):(check?.message||'OOD тексеруі қолжетімсіз.');
  showComparison(r);
  const explanation=r.explanation;
  $('attention-tools').hidden=!explanation?.available;
  $('attention-unavailable').hidden=!!explanation?.available;
  if(explanation?.available){
    $('attention-caption').textContent=(needsReview(r)?'Қорытынды берілмеді. Карта тек техникалық санатқа есептелген. ':'')+`Grad-CAM (MobileNetV2): ${explanation.target_class} санатына оң ықпал еткен жуық аймақтар. Бұл ауру ошағы немесе оның шекарасы емес.`+(r.domain_routing?.available?' Карта таңдалған тармаққа қатысты; тармақ таңдауының өзін түсіндірмейді.':'');
    setAttentionView(!needsReview(r));
  }else{
    $('attention-unavailable').textContent=explanation?.message||'Жылу картасы қолжетімсіз.';
    $('preview-image').src=r.preview||previewUrl;
  }
}
function qualityLines(r){
  const q=r.quality;
  if(!q?.available)return ['Сурет сапасын тексеру орындалмады.'];
  return [q.summary,...q.warnings.map(w=>`${w.title}. ${w.action}`),q.notice,'Сапа тексеруі модель баллын өзгертпейді.'];
}
function showQuality(r){
  const q=r.quality,warning=!q?.available||q.status==='warning';
  $('quality-panel').hidden=false;
  $('quality-panel').classList.toggle('flagged',warning);
  $('quality-tag').textContent=!q?.available?'ТЕКСЕРІЛМЕДІ':warning?'НАЗАР АУДАРЫҢЫЗ':'ЕСКЕРТУ ЖОҚ';
  $('quality-summary').textContent=q?.available?q.summary:'Сурет сапасын тексеру орындалмады.';
  const list=$('quality-warnings');list.replaceChildren();
  for(const item of q?.warnings||[]){
    const row=document.createElement('li'),title=document.createElement('strong'),action=document.createElement('span');
    title.textContent=item.title;action.textContent=item.action;row.append(title,action);list.append(row);
  }
  $('quality-notice').textContent=(q?.notice||'Техникалық сапаны бөлек тексеріңіз; кеуденің толық көрінуі расталмаған.')+' Сапа тексеруі модель баллын өзгертпейді.';
}
function setAttentionView(show){
  const r=results[0];
  if(!r?.explanation?.available)return;
  $('preview-image').src=show?r.explanation.overlay:(r.preview||previewUrl);
  $('preview-image').alt=show?`${r.model_class} санатына арналған Grad-CAM жылу картасы`:'Жүктелген рентген суреті';
  $('view-original').setAttribute('aria-pressed',String(!show));
  $('view-attention').setAttribute('aria-pressed',String(show));
}
$('view-original').addEventListener('click',()=>setAttentionView(false));
$('view-attention').addEventListener('click',()=>setAttentionView(true));
$('download-attention').addEventListener('click',()=>{
  const r=results[0];if(!r?.explanation?.available)return;
  const binary=atob(r.explanation.overlay.split(',')[1]);
  const bytes=Uint8Array.from(binary,c=>c.charCodeAt(0));
  const url=URL.createObjectURL(new Blob([bytes],{type:'image/png'}));
  const a=document.createElement('a');a.href=url;a.download='tynys-gradcam.png';document.body.append(a);a.click();a.remove();
  setTimeout(()=>URL.revokeObjectURL(url),1000);
});
async function analyzeSelected(){
  if(busy||historyWorking||!historyReady||!selected.length)return;
  const epoch=privacyEpoch,files=[...selected];
  const request={revision:selectionRevision,controller:new AbortController()};
  analysisRequest=request;retryRevision=null;$('retry-analysis').hidden=true;
  const current=()=>analysisRequest===request&&request.revision===selectionRevision&&epoch===privacyEpoch;
  busy=true;document.body.classList.add('busy');error('');resetResult();analysisProgress(true);updateButton();$('result-tag').textContent='ТАЛДАУДА';
  try{
    for(let i=0;i<files.length;i++){
      if(!current())return;
      analysisProgress(true,files.length>1?`Талдауда: ${i+1} / ${files.length}`:'Сурет талдануда…');
      const form=new FormData();form.append('file',files[i]);
      const signal=AbortSignal.any([request.controller.signal,AbortSignal.timeout(190000)]);
      const r=await api('/api/predict',{method:'POST',body:form,signal});
      if(!current())return;
      results.push(r);await rememberResult(files[i],r);
      if(!current())return;
      showResult(results[0]);
    }
  }catch(e){if(current()){
    error(e.message);if(results.length)showResult(results[0]);$('result-tag').textContent=results.length?'ІШІНАРА ДАЙЫН':'ҚАТЕ';
    retryRevision=e.retryable?selectionRevision:null;$('retry-analysis').hidden=retryRevision===null;
  }}finally{if(current()){analysisRequest=null;busy=false;document.body.classList.remove('busy');analysisProgress(false);updateButton();}}
}
$('analyze').addEventListener('click',analyzeSelected);
$('retry-analysis').addEventListener('click',()=>{if(retryRevision===selectionRevision)void analyzeSelected();});
$('download-result').addEventListener('click',()=>{
  if(!results.length)return;
  const lines=['TYNYS — РЕНТГЕНДІ КЛАССИФИКАЦИЯЛАУ',new Date().toLocaleString('kk-KZ',{timeZone:'Asia/Qyzylorda'}),''];
  for(const r of results){
    const explanation=explainResult(r);
    if(r.domain_routing?.available)lines.push(`Модель тармағы: ${r.domain_routing.selected_expert}`,r.domain_routing.notice,'');
    lines.push(`Файл: ${r.filename}`,`Нәтиже: ${resultLabel(r)}`,`${needsReview(r)?'Техникалық модель санаты (қорытынды емес)':'Модель санаты'}: ${r.model_class}`,`Пневмония санатының баллы: ${r.pneumonia_score}`,`Шешім шегі: ${r.threshold}`,`Модель ID: ${r.model_id}`,`Калибрлеу: ${r.calibration?'T='+r.calibration.T:'қолданылмаған'}`,`OOD ережесі: ${r.assessment?.policy||'белгісіз'}`,`OOD параметр ID: ${r.xray_check?.parameter_id||'сақталмаған'}`,`OOD қашықтығы: ${r.xray_check?.distance??'есептелмеді'}; OOD шегі: ${r.xray_check?.threshold??'жоқ'}`,assessmentFor(r).message,'',
      explanation.title,explanation.reason,'',
      'Модель баллды қалай есептейді?',explanation.method,'',
      'Бұл нәтиже нені білдіреді?',explanation.meaning,'',
      'Сурет сапасы — бөлек техникалық тексеру',...qualityLines(r),'',
      'Суреттегі модель назары',r.explanation?.available?`Grad-CAM картасы ${r.explanation.target_class} санатына есептелді. Қызыл-сары аймақтар осы санатқа оң ықпал еткен жуық белсенділікті көрсетеді. Белгіленген суретті сайттағы бөлек батырмамен жүктеуге болады.`:(r.explanation?.message||'Жылу картасы қолжетімсіз.'),'',
      'Түсіндірудің шектеуі',explanation.limit,'',
      needsReview(r)?'Техникалық баллдар — қорытынды емес':'Модельдердің осы суреттегі болжамдары',...(r.comparison?.available?r.comparison.models.map(m=>`${m.name}: ${m.model_class}; балл: ${m.pneumonia_score}; шешім шегі: ${m.threshold}`):[r.comparison?.message||'Салыстыру қолжетімсіз.']),
      r.comparison?.available?(r.comparison.agreement?'Модельдер бір санатты таңдады.':'Модельдер әртүрлі санат таңдады.'):'','Модель баллдары калибрленген клиникалық сенімділік емес. Бірінің баллы жоғары болуы оның дұрыс екенін дәлелдемейді.','',
      '------------------------------------------------------------','');
  }
  lines.push('Бұл — оқу моделінің болжамы, медициналық диагноз емес.','Балл диагноздың сенімділік пайызы емес.');
  const url=URL.createObjectURL(new Blob([lines.join('\n')],{type:'text/plain;charset=utf-8'}));
  const a=document.createElement('a');a.href=url;a.download='tynys-result.txt';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
});
$('download-pdf').addEventListener('click',async()=>{
  if(busy||historyWorking||!status?.ready||!selected.length||!results.length)return;
  busy=true;updateButton();error('');
  const epoch=privacyEpoch;
  const button=$('download-pdf');button.disabled=true;
  const label=button.firstChild;
  const oldLabel=label.textContent;label.textContent='PDF дайындалуда… ';
  try{
    const form=new FormData();for(const file of selected)form.append('files',file);
    const response=await apiResponse('/api/report',{method:'POST',body:form});
    const blob=await response.blob();
    if(epoch!==privacyEpoch)return;
    if(!blob.type.includes('application/pdf'))throw Error('Сервер PDF файл қайтармады.');
    const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download='tynys-report.pdf';document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);
  }catch(e){if(epoch===privacyEpoch)error(e.message);}finally{busy=false;button.disabled=false;label.textContent=oldLabel;updateButton();}
});
$('model-details').addEventListener('click',()=>{const box=$('model-info');box.replaceChildren();const rows=[['Модель','MobileNetV2'],['Күйі',status?.ready?'Дайын':'Қолжетімсіз'],['TensorFlow',status?.tensorflow_version||'—'],['Өлшем',status?.config?.image_size?.join(' × ')||'—'],['Шешім шегі',status?.config?decimal(status.config.threshold):'—'],['Кезең',status?.config?.selected_stage||'—'],['Модель ID',status?.model_id||'—']];for(const [key,value] of rows){const row=document.createElement('div');row.className='info-row';const label=document.createElement('span'),v=document.createElement('strong');label.textContent=key;v.textContent=value;row.append(label,v);box.append(row);}$('model-dialog').showModal();});
$('close-dialog').addEventListener('click',()=>$('model-dialog').close());
$('model-dialog').addEventListener('click',e=>{if(e.target===$('model-dialog')){const r=e.target.getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)e.target.close();}});
loadStatus();

function showComparison(r){
  const comparison=r.comparison;
  $('comparison-panel').hidden=false;
  const body=$('comparison-table');body.replaceChildren();
  const verdict=$('comparison-verdict');
  if(!comparison?.available){verdict.textContent='Қолжетімсіз';verdict.className='different';$('comparison-note').textContent=comparison?.message||'Модельдерді салыстыру қолжетімсіз.';return;}
  const cnn=comparison.models.find(m=>m.name==='CNN');
  const mobile=comparison.models.find(m=>m.name==='MobileNetV2');
  const ensemble=comparison.models.find(m=>m.name==='Ensemble');
  const review=needsReview(r);
  for(const [label,key] of [['Болжам санаты','model_class'],['Пневмония санатының баллы','pneumonia_score'],['Өз шешім шегі','threshold']]){
    const row=document.createElement('tr');
    const title=document.createElement('th');title.scope='row';title.textContent=review&&key==='model_class'?'Техникалық санат (қорытынды емес)':label;row.append(title);
    for(const model of [cnn,mobile,ensemble]){const cell=document.createElement('td');if(!model){cell.textContent='Қолжетімсіз';}else if(key==='model_class'){const tag=document.createElement('span');tag.className='comparison-category '+(model[key]==='PNEUMONIA'?'pneumonia':'normal');tag.textContent=model[key];cell.append(tag);}else{cell.textContent=decimal(model[key],6);}row.append(cell);}body.append(row);
  }
  verdict.textContent=review?'Қорытынды берілмеді':comparison.agreement?'Болжамдар сәйкес':'Болжамдар әртүрлі';verdict.className=!review&&comparison.agreement?'agree':'different';
  $('comparison-note').textContent=(review?'Төменде тек техникалық санаттар көрсетілген. Модельдердің келісуі OOD ескертуін жоймайды. ':'')+(comparison.agreement?'Модельдер осы суретке бір санат таңдады.':'Модельдер осы суретке әртүрлі санат таңдады.')+(comparison.ensemble?.available?' Ensemble салмақтары validation-да таңдалған.':(' '+(comparison.ensemble?.message||'')))+' Модель баллдары клиникалық сенімділік пайызы емес. Бірінің баллы жоғары болуы оның дұрыс екенін дәлелдемейді.';
}


function renderVaultState(){
  $('protect-history').hidden=historyProtected;
  $('unlock-history').hidden=!historyProtected||!historyLocked;
  $('lock-history').hidden=!historyProtected||historyLocked;
  $('vault-badge').textContent=historyProtected?(historyLocked?'ҚҰЛЫПТАУЛЫ':'ШИФРЛАНҒАН'):'ПАРОЛЬСІЗ';
  $('vault-badge').classList.toggle('protected',historyProtected);
  $('vault-status').textContent=historyProtected?(historyLocked?`Қойма құлыптаулы: ${storedHistoryCount} сақталған талдау. Көру үшін пароль енгізіңіз.`:'Осы браузердегі тарих шифрланған. 5 минут әрекетсіздіктен кейін құлыпталады. Құлыптау ашық сурет пен нәтижені экраннан алып тастайды.'):(historyEnabled?'Сақталған тарих парольмен қорғалмаған. Осы браузер профилін пайдаланатын басқа адам оны аша алады.':'Сақтау өшірулі. Қаласаңыз, сақталатын тарихты жеке парольмен қорғаңыз.');
  if(!TynysHistory.secureAvailable()&&!historyProtected)$('vault-status').textContent+=' Бұл браузерде шифрлау қолжетімсіз: HTTPS немесе localhost қажет.';
}
function lockHistory(broadcast=true){
  privacyEpoch++;historyStore.lock();
  if(historyProtected)historyLocked=true;
  releaseEntries(analysisHistory);analysisHistory=[];clearSelection();
  for(const id of ['file-name','file-size','result-title','result-category','result-explanation','score-value','result-time','routing-note','history-feedback','comparison-note','quality-summary','attention-caption'])$(id).textContent='';
  for(const id of ['quality-warnings','comparison-table','batch-results'])$(id).replaceChildren();
  if($('history-vault-dialog').open)$('history-vault-dialog').close();
  $('vault-password').value='';$('vault-confirm').value='';
  announceHistory('Тарих құлыпталды. Ашық суреттер мен нәтижелер экраннан алынды; сақталған деректер өшірілмеді.');
  if(broadcast)broadcastHistory('vault-lock');
  pendingHistorySync=true;renderHistory();updateHistoryMode();updateButton();
}
function openVaultDialog(mode){
  if(busy||historyWorking)return;
  vaultMode=mode;lastHistoryActivity=Date.now();
  $('history-vault-form').reset();$('vault-dialog-error').hidden=true;
  const protect=mode==='protect';
  $('vault-dialog-title').textContent=protect?'Тарихты парольмен қорғау':'Қорғалған тарихты ашу';
  $('vault-dialog-description').textContent=protect?'Қорғауды қоссаңыз, осы беттегі және сақталған тарих шифрланып, келесі талдаулар да осы браузерде сақталады. 20 талдау, 80 МБ, 7 күн шегі сақталады. Басқа ескі қойындыларды жабыңыз немесе жаңартыңыз. Парольді ұмытсаңыз, тарих ашылмайды.':'Пароль тек осы браузерде кілт алу үшін қолданылады. Ол серверге жіберілмейді. Қате пароль сақталған деректерді өшірмейді.';
  $('vault-confirm-field').hidden=!protect;$('vault-confirm').disabled=!protect;
  $('vault-password').autocomplete=protect?'new-password':'current-password';
  $('vault-submit').textContent=protect?'Шифрлауды қосу':'Тарихты ашу';
  $('history-vault-dialog').showModal();$('vault-password').focus();
}
$('protect-history').addEventListener('click',()=>openVaultDialog('protect'));
$('unlock-history').addEventListener('click',()=>openVaultDialog('unlock'));
$('lock-history').addEventListener('click',()=>lockHistory());
$('close-vault').addEventListener('click',()=>{if(!historyWorking){$('vault-password').value='';$('vault-confirm').value='';$('history-vault-dialog').close();}});
$('history-vault-dialog').addEventListener('cancel',event=>{if(historyWorking)event.preventDefault();else{$('vault-password').value='';$('vault-confirm').value='';}});
$('history-vault-dialog').addEventListener('close',()=>{$('vault-password').value='';$('vault-confirm').value='';});
$('history-vault-form').addEventListener('submit',async event=>{
  event.preventDefault();if(busy||historyWorking)return;
  const protect=vaultMode==='protect';
  if(protect&&$('vault-password').value!==$('vault-confirm').value){$('vault-dialog-error').textContent='Парольдер сәйкес емес.';$('vault-dialog-error').hidden=false;return;}
  historyWorking=true;updateButton();$('vault-submit').disabled=true;$('close-vault').disabled=true;
  $('vault-dialog-error').hidden=true;$('vault-submit').textContent=protect?'Шифрлануда…':'Ашылуда…';
  let success=false;
  try{
    if(protect)await historyStore.protect($('vault-password').value,analysisHistory.map(TynysHistory.recordFor));
    else await historyStore.unlock($('vault-password').value);
    success=true;$('history-vault-dialog').close();broadcastHistory();lastHistoryActivity=Date.now();
  }catch(error){
    $('vault-dialog-error').textContent=error instanceof TynysHistory.VaultError||error instanceof TynysHistory.HistoryLocked?error.message:'Қорғау әрекеті орындалмады. Бұрынғы деректер өзгертілмеді; басқа қойындыны не сақтау орнын тексеріңіз.';
    $('vault-dialog-error').hidden=false;
  }finally{
    $('vault-password').value='';$('vault-confirm').value='';historyWorking=false;
    $('vault-submit').disabled=false;$('close-vault').disabled=false;$('vault-submit').textContent=protect?'Шифрлауды қосу':'Тарихты ашу';
    updateButton();
  }
  if(success)await syncHistory();
});
for(const type of ['pointerdown','keydown','input'])document.addEventListener(type,()=>{lastHistoryActivity=Date.now();},{passive:true});
setInterval(()=>{if(historyProtected&&!historyLocked&&Date.now()-lastHistoryActivity>=HISTORY_IDLE_MS)lockHistory();},5000);

// History stays in memory unless the user explicitly opts into this browser's store.
function announceHistory(message){$('history-storage-status').textContent=message;}
function broadcastHistory(type='changed'){historyChannel?.postMessage({type});}
function releaseEntries(entries){for(const entry of entries)if(entry.thumbnail?.startsWith('blob:'))URL.revokeObjectURL(entry.thumbnail);}
function updateHistoryMode(){
  $('save-history').checked=historyEnabled;
  renderVaultState();
  $('history-description').textContent=historyEnabled?'Соңғы 20 талдау. Осы браузерде 7 күнге дейін, барлығы 80 МБ-қа дейін сақталады.':'Соңғы 20 талдау. Әдепкіде тарих тек осы бет ашық тұрғанда сақталады.';
}
async function syncHistory(){
  const epoch=privacyEpoch;
  if(busy||historyWorking){pendingHistorySync=true;return;}
  historyWorking=true;updateButton();
  try{
    const state=await historyStore.read();
    if(epoch!==privacyEpoch)return;
    historyProtected=state.protected===true;historyLocked=state.locked===true;storedHistoryCount=state.storedCount??state.records.length;
    // A disabled consent from another tab must not be silently re-enabled here.
    if(!state.enabled&&historyEnabled){releaseEntries(analysisHistory);analysisHistory=[];clearSelection();}
    const session=analysisHistory.filter(entry=>!entry.saved);
    const previousSaved=analysisHistory.filter(entry=>entry.saved);
    releaseEntries(previousSaved);
    const restored=state.records.map(TynysHistory.restore);
    const merged=[...restored,...session].sort((a,b)=>b.time-a.time);
    analysisHistory=merged.slice(0,TynysHistory.LIMIT);releaseEntries(merged.slice(TynysHistory.LIMIT));
    // A saved entry deleted/expired elsewhere must not remain as the active result.
    if(activeHistoryId&&!analysisHistory.some(entry=>entry.id===activeHistoryId))clearSelection();
    historyEnabled=state.enabled;historyGeneration=state.generation;historyDeletionPending=false;
    announceHistory(historyLocked?'Қорғалған тарих құлыптаулы. Жаңа талдау жасауға болады; ашқанша жаңа нәтижелер осы бетте ғана қалады.':historyEnabled?`Сақтау қосулы. Браузерде ${state.records.length} талдау сақталған. Бұл сақтық көшірме емес: браузер деректерді тазалауы мүмкін.`:'Сақтау өшірулі. Тарих тек осы бетте қалады.');
  }catch{
    if(historyProtected){historyStore.lock();historyLocked=true;releaseEntries(analysisHistory);analysisHistory=[];clearSelection();}
    announceHistory('Браузердегі тарихты оқу мүмкін болмады. Жаңа талдаулар осы бетте ғана қалады; сақталған деректерді жоюды қайта көріңіз.');
    historyEnabled=false;historyGeneration=null;
    historyDeletionPending=true;
  }finally{historyReady=true;historyWorking=false;updateHistoryMode();renderHistory();updateButton();}
}
async function rememberResult(file,result){
  const entry={id:crypto.randomUUID(),file,result,time:new Date(),saved:false,thumbnail:result.preview||URL.createObjectURL(file)};
  analysisHistory.unshift(entry);
  if(results[0]===result)activeHistoryId=entry.id;
  releaseEntries(analysisHistory.splice(TynysHistory.LIMIT));
  $('history-feedback').textContent='';
  if(historyEnabled&&!historyLocked){
    try{
      const state=await historyStore.append(TynysHistory.recordFor(entry),historyGeneration);
      const ids=new Set(state.ids);
      for(const item of analysisHistory)item.saved=ids.has(item.id);
      announceHistory(`Сақталды. Браузерде ${ids.size} талдау бар. 7 күн және 80 МБ шегі қолданылады.`);
      broadcastHistory();
    }catch(error){
      if(error instanceof TynysHistory.HistoryChanged||error instanceof TynysHistory.HistoryLocked){pendingHistorySync=true;historyEnabled=false;}
      announceHistory('Бұл нәтиже браузерге сақталмады. Осы бетте ғана қолжетімді. Сақтау орны не басқа беттегі өзгерісті тексеріңіз.');
    }
  }
  updateHistoryMode();renderHistory();
}
function renderHistory(){
  const list=$('history-list');list.replaceChildren();
  $('history-count').textContent=analysisHistory.length;
  $('history-empty').hidden=analysisHistory.length>0;
  for(const entry of analysisHistory){
    const row=document.createElement('li');row.className='history-entry'+(entry.id===activeHistoryId?' active':'');
    const thumbnail=document.createElement('img');thumbnail.src=entry.thumbnail;thumbnail.alt='Рентгеннің шағын көрінісі';thumbnail.loading='lazy';
    const content=document.createElement('div');content.className='history-item-content';
    const filename=document.createElement('strong');filename.textContent=entry.file.name;filename.title=entry.file.name;
    const details=document.createElement('span');details.className='history-item-details';
    details.textContent=`${resultLabel(entry.result)} · балл ${decimal(entry.result.pneumonia_score)} · ${entry.time.toLocaleString('kk-KZ',{timeZone:'Asia/Qyzylorda'})}`;
    const comparison=document.createElement('small');comparison.textContent=entry.result.comparison?.available?(entry.result.comparison.agreement?'Модельдер: болжамдар сәйкес':'Модельдер: болжамдар әртүрлі'):'MobileNetV2 нәтижесі';
    comparison.textContent+=' · Сапа: '+(!entry.result.quality?.available?'тексерілмеді':entry.result.quality.status==='warning'?`${entry.result.quality.warnings.length} ескерту`:'ескерту жоқ');
    const saved=document.createElement('small');saved.className='history-saved';saved.textContent=entry.saved?'Осы браузерде сақталған · 7 күнге дейін':'Тек осы бетте';
    content.append(filename,details,comparison,saved);
    const actions=document.createElement('div');actions.className='history-actions';
    const button=document.createElement('button');button.type='button';button.className='history-open';button.textContent=entry.id===activeHistoryId?'Ашық':'Қайта ашу';button.setAttribute('aria-label',`${entry.file.name} нәтижесін қайта ашу`);button.addEventListener('click',()=>openHistoryEntry(entry.id));
    const remove=document.createElement('button');remove.type='button';remove.className='history-delete';remove.textContent='Жою';remove.setAttribute('aria-label',`${entry.file.name} талдауын жою`);remove.addEventListener('click',()=>removeHistoryEntry(entry.id));
    actions.append(button,remove);row.append(thumbnail,content,actions);list.append(row);
  }
  updateButton();
}
function openHistoryEntry(id){
  if(busy||historyWorking)return;
  const entry=analysisHistory.find(item=>item.id===id);if(!entry)return;
  selectFiles([entry.file]);
  results=[entry.result];activeHistoryId=id;
  showResult(entry.result);renderHistory();updateButton();
  const changed=status?.model_id&&status.model_id!==entry.result.model_id;
  const policyChanged=status?.assessment_policy?.name&&status.assessment_policy.name!==entry.result.assessment?.policy;
  const parametersChanged=status?.xray_check?.parameter_id&&status.xray_check.parameter_id!==entry.result.xray_check?.parameter_id;
  $('history-feedback').textContent=`${entry.file.name}: бұрынғы нәтиже ашылды. Қайта есептеу жасалған жоқ.`+(changed?' Сервердегі модель өзгерген: бұл — бұрынғы модель нәтижесі.':'')+(policyChanged||parametersChanged?' OOD ережесі жаңарған: сақталған нәтиже бұрынғы ережеге қатысты. Қазіргі ережемен тексеру үшін «Суретті талдау» басыңыз.':'');
  $('analysis-title').scrollIntoView({behavior:'smooth',block:'start'});
}
async function removeHistoryEntry(id){
  if(busy||historyWorking)return;
  const entry=analysisHistory.find(item=>item.id===id);if(!entry)return;
  historyWorking=true;updateButton();
  try{
    if(entry.saved)await historyStore.remove(id,historyGeneration);
    releaseEntries([entry]);analysisHistory=analysisHistory.filter(item=>item.id!==id);
    if(activeHistoryId===id)clearSelection();
    $('history-feedback').textContent='Таңдалған талдау жойылды.';
    announceHistory(historyEnabled?'Таңдалған талдау браузердегі тарихтан жойылды.':'Таңдалған талдау осы беттегі тарихтан жойылды.');
    broadcastHistory();
  }catch{pendingHistorySync=true;announceHistory('Талдауды браузерден жою орындалмады. Қайта көріңіз.');}
  finally{historyWorking=false;renderHistory();updateButton();}
}
$('save-history').addEventListener('change',async()=>{
  const enabled=$('save-history').checked;
  if(busy||historyWorking){updateHistoryMode();return;}
  historyWorking=true;updateButton();
  try{
    if(enabled){
      const state=await historyStore.enable(analysisHistory.map(TynysHistory.recordFor));
      historyEnabled=true;historyGeneration=state.generation;
      const ids=new Set(state.ids);for(const entry of analysisHistory)entry.saved=ids.has(entry.id);
      announceHistory(`Сақтау қосылды: ${ids.size} талдау. Келесі талдаулар да осы браузерде сақталады.`);
    }else{
      await historyStore.clear();historyEnabled=false;historyGeneration=null;historyDeletionPending=false;historyProtected=false;historyLocked=false;storedHistoryCount=0;
      for(const entry of analysisHistory)entry.saved=false;
      announceHistory('Сақтау өшірілді, браузердегі көшірмелер жойылды. Қазіргі тарих осы бетте ғана қалады.');
    }
    broadcastHistory();
  }catch{announceHistory('Сақтау параметрін өзгерту орындалмады. Бұрынғы күй сақталды. Қайта көріңіз.');}
  finally{historyWorking=false;updateHistoryMode();renderHistory();updateButton();}
});
$('clear-history').addEventListener('click',async()=>{
  if(busy||historyWorking)return;
  historyWorking=true;updateButton();
  try{
    await historyStore.clear();
    releaseEntries(analysisHistory);analysisHistory=[];historyEnabled=false;historyGeneration=null;historyDeletionPending=false;historyProtected=false;historyLocked=false;storedHistoryCount=0;
    clearSelection();
    $('history-feedback').textContent='Талдау тарихы, ашық нәтиже және осы браузердегі сақталған көшірмелер жойылды. Сақтау өшірілді.';
    announceHistory('Браузердегі тарих тазаланды. Сақтау өшірулі.');
    broadcastHistory();
  }catch{
    releaseEntries(analysisHistory);analysisHistory=[];clearSelection();historyDeletionPending=true;
    announceHistory('Ашық беттегі тарих тазаланды. Браузердегі сақталған көшірмелерді жою расталмады; қайта көріңіз немесе браузердің осы сайтқа арналған деректерін тазалаңыз.');
  }
  finally{historyWorking=false;updateHistoryMode();renderHistory();updateButton();}
});
if(historyChannel)historyChannel.onmessage=event=>{if(event.data?.type==='vault-lock'){lockHistory(false);return;}pendingHistorySync=true;if(!busy&&!historyWorking){pendingHistorySync=false;void syncHistory();}};
setInterval(()=>{if(historyEnabled||pendingHistorySync){pendingHistorySync=false;void syncHistory();}},60000);
window.addEventListener('focus',()=>{if(historyReady)void syncHistory();});
window.addEventListener('pageshow',event=>{if(event.persisted)void syncHistory();});
window.addEventListener('pagehide',event=>{
  if(historyProtected||$('history-vault-dialog').open)lockHistory(false);
  else historyStore.lock();
  if(event.persisted)return;
  if(previewUrl)URL.revokeObjectURL(previewUrl);
  releaseEntries(analysisHistory);historyChannel?.close();
});
renderHistory();
void syncHistory();
