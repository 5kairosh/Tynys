'use strict';
const $=id=>document.getElementById(id);
let selected=[], results=[], status=null, busy=false, previewUrl=null;
let analysisHistory=[], activeHistoryId=null;
let historyReady=false,historyWorking=false,historyEnabled=false,historyGeneration=null,pendingHistorySync=false,historyDeletionPending=false;
const historyStore=new TynysHistory.Store();
let historyChannel=null;
try{if(typeof BroadcastChannel==='function')historyChannel=new BroadcastChannel('tynys-history-events-v1');}catch{}
let statusLoading=false, statusTimer=null, connectionError=false;
function scheduleStatusCheck(){
  clearTimeout(statusTimer);
  statusTimer=setTimeout(()=>loadStatus(),status?.ready?15000:3000);
}
const decimal=(n,d=3)=>Number(n).toFixed(d).replace('.',',');
const percent=n=>decimal(n*100,2)+'%';
function error(message){$('error-message').textContent=message;$('error-message').hidden=!message;}
function updateButton(){
  const locked=busy||historyWorking||!historyReady;
  $('analyze').disabled=locked||!selected.length||!status?.ready;
  $('image-input').disabled=locked;
  $('clear-history').disabled=locked||(!analysisHistory.length&&!historyEnabled&&!historyDeletionPending);
  $('save-history').disabled=locked;
  $('download-pdf').disabled=locked||!status?.ready;
  for(const button of $('history-list').querySelectorAll('button'))button.disabled=locked;
  if(pendingHistorySync&&!locked){pendingHistorySync=false;queueMicrotask(()=>void syncHistory());}
}
async function api(url,options){let r;try{r=await fetch(url,options);}catch{throw Error('Серверге қосылмады. start.bat файлын іске қосып, сервер терезесін ашық қалдырыңыз.');}let data;try{data=await r.json();}catch{throw Error('Серверден жауап алынбады. Сервердің іске қосылғанын тексеріңіз.');}if(!r.ok)throw Error(typeof data.detail==='string'?data.detail:'Сұрауды орындау мүмкін болмады.');return data;}
async function loadStatus(){
  if(statusLoading)return;
  if(busy){scheduleStatusCheck();return;}
  statusLoading=true;
  try{status=await api('/api/status',{signal:AbortSignal.timeout(8000)});
  renderSelectiveEvaluation(status.selective_evaluation);
  void renderGeneralizationEvaluation(status);
  if(connectionError&&status.ready){error('');connectionError=false;}
$('header-status').textContent=status.ready?'Модель дайын':'Модель қолжетімсіз';$('status-dot').classList.toggle('ready',status.ready);if(!status.ready)error(status.message||'Модель қосылмаған.');const m=status.metrics?.find(x=>x.model==='MobileNetV2');const cnn=status.metrics?.find(x=>x.model==='CNN');const ensemble=status.metrics?.find(x=>x.model==='Ensemble');if(m){$('metric-accuracy').textContent=percent(m.accuracy);$('metric-recall').textContent=percent(m.recall);$('metric-count').textContent=Math.round(m.TN+m.FP+m.FN+m.TP);const body=$('metrics-table');body.replaceChildren();for(const [key,label,type] of [['accuracy','Accuracy','percent'],['precision','Precision','percent'],['recall','Recall / sensitivity','percent'],['specificity','Specificity','percent'],['f1','F1-score','decimal'],['roc_auc','ROC-AUC','decimal']]){const tr=document.createElement('tr');for(const value of [label,cnn?cnn[key]:null,m[key],status.ensemble?.available?ensemble?.[key]:null]){const td=document.createElement('td');td.textContent=typeof value==='string'?value:value==null?'—':type==='percent'?percent(value):decimal(value,4);tr.append(td);}body.append(tr);}}const external=status.external_validation;$('external-note').textContent=external?.available?`Сыртқы тексеру (OpenI, ${external.metrics.count} сурет): accuracy ${percent(external.metrics.accuracy)}, recall ${percent(external.metrics.recall)}. Ішкі тесттен бөлек. Топтау есеп ID-іне сүйенеді.`:(external?.message||'Сыртқы бағалау қолжетімсіз.');const paired=status.mcnemar;$('mcnemar-note').textContent=paired?.available?`Ішкі test: McNemar p=${Number(paired.p_value).toExponential(3)}, α=0,05. ${paired.significant?'Айырма статистикалық маңызды.':'Статистикалық маңыздылық анықталмады.'} Бұл осы тестке қатысты.`:(paired?.message||'McNemar нәтижесі қолжетімсіз.');if(status.config)$('research-threshold').textContent=decimal(status.config.threshold);updateButton();}catch(e){
    status=null;connectionError=true;
    void renderGeneralizationEvaluation(null);
    renderSelectiveEvaluation({available:false,message:'Серверге қосылмады. Бағалау есебі сервер қосылғанда қайта тексеріледі.'});
    $('header-status').textContent='Серверге қосылмады';
    $('status-dot').classList.remove('ready');
    error('Серверге қосылмады. Сайт бумасындағы start.bat файлын іске қосып, сервер терезесін ашық қалдырыңыз. Қосылым автоматты түрде қайта тексеріледі.');
    updateButton();
  }finally{statusLoading=false;scheduleStatusCheck();}
}
window.addEventListener('online',()=>loadStatus());
window.addEventListener('focus',()=>loadStatus());
function resetResult(){$('empty-result').hidden=false;$('result-content').hidden=true;$('result-tag').textContent='КҮТУДЕ';$('attention-tools').hidden=true;$('attention-unavailable').hidden=true;$('comparison-panel').hidden=true;$('quality-panel').hidden=true;results=[];}
function selectFiles(files){if(busy||historyWorking||!historyReady)return;error('');const list=Array.from(files);if(!list.length)return;if(list.length>10){error('Бір ретте 10 суретке дейін таңдаңыз.');return;}for(const f of list){if(!/\.(jpe?g|png|dcm)$/i.test(f.name)){error('Тек JPG, JPEG, PNG немесе DICOM таңдаңыз.');return;}if(f.size>12*1024*1024){error('Әр суреттің көлемі 12 МБ-тан аспауы керек.');return;}if(!f.size){error('Бос файлды жүктеу мүмкін емес.');return;}}selected=list;activeHistoryId=null;renderHistory();resetResult();if(previewUrl)URL.revokeObjectURL(previewUrl);previewUrl=URL.createObjectURL(selected[0]);$('preview-image').src=previewUrl;const isDicom=/\.dcm$/i.test(selected[0].name);$('preview-image').hidden=isDicom;$('dicom-placeholder').hidden=!isDicom;$('preview').hidden=false;$('dropzone').hidden=true;$('file-info').hidden=false;$('file-name').textContent=selected[0].name;$('file-size').textContent=(selected[0].size/1024).toFixed(0)+' КБ'+(selected.length>1?` · ${selected.length} сурет таңдалды`:'');$('file-list').textContent=selected.length>1?selected.slice(1).map(f=>f.name).join(' · '):'';updateButton();}
$('image-input').addEventListener('change',e=>selectFiles(e.target.files));
$('dropzone').addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();$('image-input').click();}});
for(const type of ['dragenter','dragover'])$('dropzone').addEventListener(type,e=>{e.preventDefault();$('dropzone').classList.add('dragging');});
for(const type of ['dragleave','drop'])$('dropzone').addEventListener(type,e=>{e.preventDefault();$('dropzone').classList.remove('dragging');});
$('dropzone').addEventListener('drop',e=>selectFiles(e.dataTransfer.files));
function clearSelection(){activeHistoryId=null;selected=[];if(previewUrl)URL.revokeObjectURL(previewUrl);previewUrl=null;$('preview-image').removeAttribute('src');$('preview').hidden=true;$('dropzone').hidden=false;$('file-info').hidden=true;$('file-list').textContent='';$('image-input').value='';resetResult();error('');}
$('remove-image').addEventListener('click',()=>{if(busy||historyWorking)return;clearSelection();renderHistory();updateButton();});

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
$('analyze').addEventListener('click',async()=>{if(busy||!selected.length)return;busy=true;document.body.classList.add('busy');error('');results=[];updateButton();$('result-tag').textContent='ТАЛДАУДА';try{for(let i=0;i<selected.length;i++){$('analyze-label').textContent=selected.length>1?`Талдауда: ${i+1} / ${selected.length}`:'Сурет талдануда…';const form=new FormData();form.append('file',selected[i]);const r=await api('/api/predict',{method:'POST',body:form});results.push(r);await rememberResult(selected[i],r);showResult(results[0]);}}catch(e){error(e.message);if(results.length)showResult(results[0]);$('result-tag').textContent=results.length?'ІШІНАРА ДАЙЫН':'ҚАТЕ';}finally{busy=false;document.body.classList.remove('busy');$('analyze-label').textContent='Суретті талдау';updateButton();}});
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
  const button=$('download-pdf');button.disabled=true;
  const label=button.firstChild;
  const oldLabel=label.textContent;label.textContent='PDF дайындалуда… ';
  try{
    const form=new FormData();for(const file of selected)form.append('files',file);
    const response=await fetch('/api/report',{method:'POST',body:form});
    if(!response.ok){const data=await response.json().catch(()=>({}));throw Error(typeof data.detail==='string'?data.detail:'PDF есепті жасау мүмкін болмады.');}
    const blob=await response.blob();
    if(!blob.type.includes('application/pdf'))throw Error('Сервер PDF файл қайтармады.');
    const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download='tynys-report.pdf';document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);
  }catch(e){error(e.message);}finally{busy=false;button.disabled=false;label.textContent=oldLabel;updateButton();}
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


// History stays in memory unless the user explicitly opts into this browser's store.
function announceHistory(message){$('history-storage-status').textContent=message;}
function broadcastHistory(){historyChannel?.postMessage({type:'changed'});}
function releaseEntries(entries){for(const entry of entries)if(entry.thumbnail?.startsWith('blob:'))URL.revokeObjectURL(entry.thumbnail);}
function updateHistoryMode(){
  $('save-history').checked=historyEnabled;
  $('history-description').textContent=historyEnabled?'Соңғы 20 талдау. Осы браузерде 7 күнге дейін, барлығы 80 МБ-қа дейін сақталады.':'Соңғы 20 талдау. Әдепкіде тарих тек осы бет ашық тұрғанда сақталады.';
}
async function syncHistory(){
  if(busy||historyWorking){pendingHistorySync=true;return;}
  historyWorking=true;updateButton();
  try{
    const state=await historyStore.read();
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
    announceHistory(historyEnabled?`Сақтау қосулы. Браузерде ${state.records.length} талдау сақталған. Бұл сақтық көшірме емес: браузер деректерді тазалауы мүмкін.`:'Сақтау өшірулі. Тарих тек осы бетте қалады.');
  }catch{
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
  if(historyEnabled){
    try{
      const state=await historyStore.append(TynysHistory.recordFor(entry),historyGeneration);
      const ids=new Set(state.ids);
      for(const item of analysisHistory)item.saved=ids.has(item.id);
      announceHistory(`Сақталды. Браузерде ${ids.size} талдау бар. 7 күн және 80 МБ шегі қолданылады.`);
      broadcastHistory();
    }catch(error){
      if(error instanceof TynysHistory.HistoryChanged){pendingHistorySync=true;historyEnabled=false;}
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
      await historyStore.clear();historyEnabled=false;historyGeneration=null;historyDeletionPending=false;
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
    releaseEntries(analysisHistory);analysisHistory=[];historyEnabled=false;historyGeneration=null;historyDeletionPending=false;
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
if(historyChannel)historyChannel.onmessage=()=>{pendingHistorySync=true;if(!busy&&!historyWorking){pendingHistorySync=false;void syncHistory();}};
setInterval(()=>{if(historyEnabled||pendingHistorySync){pendingHistorySync=false;void syncHistory();}},60000);
window.addEventListener('focus',()=>{if(historyReady)void syncHistory();});
window.addEventListener('pageshow',event=>{if(event.persisted)void syncHistory();});
window.addEventListener('pagehide',event=>{
  if(event.persisted)return;
  if(previewUrl)URL.revokeObjectURL(previewUrl);
  releaseEntries(analysisHistory);historyChannel?.close();
});
renderHistory();
void syncHistory();
