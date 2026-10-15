'use strict';
const $=id=>document.getElementById(id);
let selected=[], results=[], status=null, busy=false, previewUrl=null;
let analysisHistory=[], nextHistoryId=1, activeHistoryId=null;
let statusLoading=false, statusTimer=null, connectionError=false;
function scheduleStatusCheck(){
  clearTimeout(statusTimer);
  statusTimer=setTimeout(()=>loadStatus(),status?.ready?15000:3000);
}
const decimal=(n,d=3)=>Number(n).toFixed(d).replace('.',',');
const percent=n=>decimal(n*100,2)+'%';
function error(message){$('error-message').textContent=message;$('error-message').hidden=!message;}
function updateButton(){ $('analyze').disabled=busy||!selected.length||!status?.ready; $('clear-history').disabled=busy||!analysisHistory.length; for(const button of $('history-list').querySelectorAll('button'))button.disabled=busy; }
async function api(url,options){let r;try{r=await fetch(url,options);}catch{throw Error('Серверге қосылмады. start.bat файлын іске қосып, сервер терезесін ашық қалдырыңыз.');}let data;try{data=await r.json();}catch{throw Error('Серверден жауап алынбады. Сервердің іске қосылғанын тексеріңіз.');}if(!r.ok)throw Error(typeof data.detail==='string'?data.detail:'Сұрауды орындау мүмкін болмады.');return data;}
async function loadStatus(){
  if(statusLoading)return;
  if(busy){scheduleStatusCheck();return;}
  statusLoading=true;
  try{status=await api('/api/status',{signal:AbortSignal.timeout(8000)});
  if(connectionError&&status.ready){error('');connectionError=false;}
$('header-status').textContent=status.ready?'Модель дайын':'Модель қолжетімсіз';$('status-dot').classList.toggle('ready',status.ready);if(!status.ready)error(status.message||'Модель қосылмаған.');const m=status.metrics?.find(x=>x.model==='MobileNetV2');const cnn=status.metrics?.find(x=>x.model==='CNN');if(m){$('metric-accuracy').textContent=percent(m.accuracy);$('metric-recall').textContent=percent(m.recall);$('metric-count').textContent=Math.round(m.TN+m.FP+m.FN+m.TP);const body=$('metrics-table');body.replaceChildren();for(const [key,label,type] of [['accuracy','Accuracy','percent'],['precision','Precision','percent'],['recall','Recall / sensitivity','percent'],['specificity','Specificity','percent'],['f1','F1-score','decimal'],['roc_auc','ROC-AUC','decimal']]){const tr=document.createElement('tr');for(const value of [label,cnn?cnn[key]:null,m[key]]){const td=document.createElement('td');td.textContent=typeof value==='string'?value:value==null?'—':type==='percent'?percent(value):decimal(value,4);tr.append(td);}body.append(tr);}}if(status.config)$('research-threshold').textContent=decimal(status.config.threshold);updateButton();}catch(e){
    status=null;connectionError=true;
    $('header-status').textContent='Серверге қосылмады';
    $('status-dot').classList.remove('ready');
    error('Серверге қосылмады. Сайт бумасындағы start.bat файлын іске қосып, сервер терезесін ашық қалдырыңыз. Қосылым автоматты түрде қайта тексеріледі.');
    updateButton();
  }finally{statusLoading=false;scheduleStatusCheck();}
}
window.addEventListener('online',()=>loadStatus());
window.addEventListener('focus',()=>loadStatus());
function resetResult(){$('empty-result').hidden=false;$('result-content').hidden=true;$('result-tag').textContent='КҮТУДЕ';$('attention-tools').hidden=true;$('attention-unavailable').hidden=true;$('comparison-panel').hidden=true;results=[];}
function selectFiles(files){if(busy)return;error('');const list=Array.from(files);if(!list.length)return;if(list.length>10){error('Бір ретте 10 суретке дейін таңдаңыз.');return;}for(const f of list){if(!/\.(jpe?g|png)$/i.test(f.name)){error('Тек JPG, JPEG немесе PNG суреттерін таңдаңыз.');return;}if(f.size>12*1024*1024){error('Әр суреттің көлемі 12 МБ-тан аспауы керек.');return;}if(!f.size){error('Бос файлды жүктеу мүмкін емес.');return;}}selected=list;activeHistoryId=null;renderHistory();resetResult();if(previewUrl)URL.revokeObjectURL(previewUrl);previewUrl=URL.createObjectURL(selected[0]);$('preview-image').src=previewUrl;$('preview').hidden=false;$('dropzone').hidden=true;$('file-info').hidden=false;$('file-name').textContent=selected[0].name;$('file-size').textContent=(selected[0].size/1024).toFixed(0)+' КБ'+(selected.length>1?` · ${selected.length} сурет таңдалды`:'');$('file-list').textContent=selected.length>1?selected.slice(1).map(f=>f.name).join(' · '):'';updateButton();}
$('image-input').addEventListener('change',e=>selectFiles(e.target.files));
$('dropzone').addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();$('image-input').click();}});
for(const type of ['dragenter','dragover'])$('dropzone').addEventListener(type,e=>{e.preventDefault();$('dropzone').classList.add('dragging');});
for(const type of ['dragleave','drop'])$('dropzone').addEventListener(type,e=>{e.preventDefault();$('dropzone').classList.remove('dragging');});
$('dropzone').addEventListener('drop',e=>selectFiles(e.dataTransfer.files));
$('remove-image').addEventListener('click',()=>{if(busy)return;activeHistoryId=null;renderHistory();selected=[];if(previewUrl)URL.revokeObjectURL(previewUrl);previewUrl=null;$('preview-image').removeAttribute('src');$('preview').hidden=true;$('dropzone').hidden=false;$('file-info').hidden=true;$('file-list').textContent='';$('image-input').value='';resetResult();error('');updateButton();});

function explainResult(r){
  const positive=r.pneumonia_score>=r.threshold;
  const label=positive?'PNEUMONIA':'NORMAL';
  const relation=positive?'осы шекке тең немесе одан жоғары':'осы шектен төмен';
  const comparison=`${decimal(r.pneumonia_score,6)} ${positive?'≥':'<'} ${decimal(r.threshold,6)}`;
  return {
    title:positive?'Неге PNEUMONIA деп жіктеді?':'Неге NORMAL деп жіктеді?',
    reason:`Модель суретке ${decimal(r.pneumonia_score,6)} балл берді. Сақталған шешім шегі — ${decimal(r.threshold,6)}. Балл ${relation} (${comparison}). Сондықтан жіктеу ережесі ${label} санатын таңдады. Салыстыру толық дәлдіктегі баллмен жасалады.`,
    method:'MobileNetV2 осы суретті 224 × 224 өлшемінде өңдеп, үйрету кезінде NORMAL және PNEUMONIA суреттерінен үйренген белгілер бойынша бір балл есептейді. Балл validation жиынында таңдалған шекпен салыстырылады.',
    meaning:positive?'Бұл модельдің осы суретті пневмония санатына жатқызғанын білдіреді. Нәтиже адамның пневмониямен ауырғанын өздігінен растамайды.':'Бұл модельдің осы суретке берген баллы пневмония санатын таңдау шегіне жетпегенін білдіреді. Нәтиже пневмонияның жоқтығын немесе адамның толық сау екенін дәлелдемейді.',
    limit:'Grad-CAM таңдалған санатқа оң ықпал еткен белгілердің жуық аймақтарын көрсетеді. Карта 7 × 7 белгілерден үлкейтіледі, сондықтан дәл шекара немесе ауру ошағын көрсетпейді. Ол нақты медициналық белгілердің атауын анықтамайды. Суреттің медициналық түсіндірмесін маман береді.'
  };
}

function showResult(r){const pn=r.model_class==='PNEUMONIA';$('empty-result').hidden=true;$('result-content').hidden=false;$('result-tag').textContent='ДАЙЫН';$('result-category').textContent=r.model_class;$('result-category').classList.toggle('pneumonia',pn);$('result-title').textContent=pn?'Пневмония санатына жіктелді':'Қалыпты санатқа жіктелді';$('result-explanation').textContent=explainResult(r).reason;$('score-value').textContent=decimal(r.pneumonia_score);$('calibration-note').textContent=r.calibration?`Validation-да калибрленген балл (T=${decimal(r.calibration.T)}). Клиникалық ықтималдық ретінде тексерілмеген.`:'Калибрлеу қолданылмаған (T=1).';$('score-fill').style.width=(r.pneumonia_score*100)+'%';$('score-fill').style.background=pn?'#ba9651':'var(--green)';$('threshold-marker').style.left=(r.threshold*100)+'%';$('threshold-label').textContent='Шек: '+decimal(r.threshold);$('result-time').textContent=`${r.width} × ${r.height} px · ${r.elapsed_ms} мс`;$('result-model').textContent='MobileNetV2';showAttention(r);const container=$('batch-results');container.replaceChildren();if(results.length>1)for(const item of results){const row=document.createElement('div'),name=document.createElement('span'),label=document.createElement('strong');name.textContent=item.filename;label.textContent=item.model_class+' · '+decimal(item.pneumonia_score);row.append(name,label);container.append(row);}}
function showAttention(r){
  showComparison(r);
  const explanation=r.explanation;
  $('attention-tools').hidden=!explanation?.available;
  $('attention-unavailable').hidden=!!explanation?.available;
  if(explanation?.available){
    $('attention-caption').textContent=`Grad-CAM (MobileNetV2): ${explanation.target_class} шешіміне оң ықпал еткен жуық аймақтар. Бұл ауру ошағы немесе оның шекарасы емес.`;
    setAttentionView(true);
  }else{
    $('attention-unavailable').textContent=explanation?.message||'Жылу картасы қолжетімсіз.';
    $('preview-image').src=previewUrl;
  }
}
function setAttentionView(show){
  const r=results[0];
  if(!r?.explanation?.available)return;
  $('preview-image').src=show?r.explanation.overlay:previewUrl;
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
$('analyze').addEventListener('click',async()=>{if(busy||!selected.length)return;busy=true;document.body.classList.add('busy');error('');results=[];updateButton();$('result-tag').textContent='ТАЛДАУДА';try{for(let i=0;i<selected.length;i++){$('analyze-label').textContent=selected.length>1?`Талдауда: ${i+1} / ${selected.length}`:'Сурет талдануда…';const form=new FormData();form.append('file',selected[i]);const r=await api('/api/predict',{method:'POST',body:form});results.push(r);rememberResult(selected[i],r);showResult(results[0]);}}catch(e){error(e.message);if(results.length)showResult(results[0]);$('result-tag').textContent=results.length?'ІШІНАРА ДАЙЫН':'ҚАТЕ';}finally{busy=false;document.body.classList.remove('busy');$('analyze-label').textContent='Суретті талдау';updateButton();}});
$('download-result').addEventListener('click',()=>{
  if(!results.length)return;
  const lines=['TYNYS — РЕНТГЕНДІ КЛАССИФИКАЦИЯЛАУ',new Date().toLocaleString('kk-KZ',{timeZone:'Asia/Qyzylorda'}),''];
  for(const r of results){
    const explanation=explainResult(r);
    lines.push(`Файл: ${r.filename}`,`Модель санаты: ${r.model_class}`,`Пневмония санатының баллы: ${r.pneumonia_score}`,`Шешім шегі: ${r.threshold}`,`Модель ID: ${r.model_id}`,'',
      explanation.title,explanation.reason,'',
      'Модель баллды қалай есептейді?',explanation.method,'',
      'Бұл нәтиже нені білдіреді?',explanation.meaning,'',
      'Суреттегі модель назары',r.explanation?.available?`Grad-CAM картасы ${r.explanation.target_class} санатына есептелді. Қызыл-сары аймақтар осы санатқа оң ықпал еткен жуық белсенділікті көрсетеді. Белгіленген суретті сайттағы бөлек батырмамен жүктеуге болады.`:(r.explanation?.message||'Жылу картасы қолжетімсіз.'),'',
      'Түсіндірудің шектеуі',explanation.limit,'',
      'Екі модельдің осы суреттегі болжамдары',...(r.comparison?.available?r.comparison.models.map(m=>`${m.name}: ${m.model_class}; балл: ${m.pneumonia_score}; шешім шегі: ${m.threshold}`):[r.comparison?.message||'Салыстыру қолжетімсіз.']),
      r.comparison?.available?(r.comparison.agreement?'Екі модель бір санатты таңдады.':'Екі модель әртүрлі санат таңдады.'):'','Модель баллдары калибрленген клиникалық сенімділік емес. Бірінің баллы жоғары болуы оның дұрыс екенін дәлелдемейді.','',
      '------------------------------------------------------------','');
  }
  lines.push('Бұл — оқу моделінің болжамы, медициналық диагноз емес.','Балл диагноздың сенімділік пайызы емес.');
  const url=URL.createObjectURL(new Blob([lines.join('\n')],{type:'text/plain;charset=utf-8'}));
  const a=document.createElement('a');a.href=url;a.download='tynys-result.txt';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
});
$('download-pdf').addEventListener('click',async()=>{
  if(busy||!selected.length||!results.length)return;
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
  if(!comparison?.available){verdict.textContent='Қолжетімсіз';verdict.className='different';$('comparison-note').textContent=comparison?.message||'Екі модельді салыстыру қолжетімсіз.';return;}
  const cnn=comparison.models.find(m=>m.name==='CNN');
  const mobile=comparison.models.find(m=>m.name==='MobileNetV2');
  for(const [label,key] of [['Болжам санаты','model_class'],['Пневмония санатының баллы','pneumonia_score'],['Өз шешім шегі','threshold']]){
    const row=document.createElement('tr');
    const title=document.createElement('th');title.scope='row';title.textContent=label;row.append(title);
    for(const model of [cnn,mobile]){const cell=document.createElement('td');if(key==='model_class'){const tag=document.createElement('span');tag.className='comparison-category '+(model[key]==='PNEUMONIA'?'pneumonia':'normal');tag.textContent=model[key];cell.append(tag);}else{cell.textContent=decimal(model[key],6);}row.append(cell);}body.append(row);
  }
  verdict.textContent=comparison.agreement?'Болжамдар сәйкес':'Болжамдар әртүрлі';verdict.className=comparison.agreement?'agree':'different';
  $('comparison-note').textContent=(comparison.agreement?'Екі модель осы суретке бір санат таңдады.':'Екі модель осы суретке әртүрлі санат таңдады.')+' Модель баллдары клиникалық сенімділік пайызы емес. Бірінің баллы жоғары болуы оның дұрыс екенін дәлелдемейді.';
}


// Session-only history. No localStorage, IndexedDB, cookies or server records.
function rememberResult(file,result){
  const entry={id:nextHistoryId++,file,result,time:new Date(),thumbnail:URL.createObjectURL(file)};
  analysisHistory.unshift(entry);
  if(results[0]===result)activeHistoryId=entry.id;
  while(analysisHistory.length>20){const removed=analysisHistory.pop();URL.revokeObjectURL(removed.thumbnail);}
  $('history-feedback').textContent='';
  renderHistory();
}
function renderHistory(){
  const list=$('history-list');list.replaceChildren();
  $('history-count').textContent=analysisHistory.length;
  $('history-empty').hidden=analysisHistory.length>0;
  $('clear-history').disabled=busy||!analysisHistory.length;
  for(const entry of analysisHistory){
    const row=document.createElement('li');row.className='history-entry'+(entry.id===activeHistoryId?' active':'');
    const thumbnail=document.createElement('img');thumbnail.src=entry.thumbnail;thumbnail.alt='Рентгеннің шағын көрінісі';thumbnail.loading='lazy';
    const content=document.createElement('div');content.className='history-item-content';
    const filename=document.createElement('strong');filename.textContent=entry.file.name;filename.title=entry.file.name;
    const details=document.createElement('span');details.className='history-item-details';
    details.textContent=`${entry.result.model_class} · балл ${decimal(entry.result.pneumonia_score)} · ${entry.time.toLocaleTimeString('kk-KZ',{timeZone:'Asia/Qyzylorda',hour:'2-digit',minute:'2-digit',second:'2-digit'})}`;
    const comparison=document.createElement('small');comparison.textContent=entry.result.comparison?.available?(entry.result.comparison.agreement?'Екі модель: болжамдар сәйкес':'Екі модель: болжамдар әртүрлі'):'MobileNetV2 нәтижесі';
    content.append(filename,details,comparison);
    const button=document.createElement('button');button.type='button';button.className='history-open';button.textContent=entry.id===activeHistoryId?'Ашық':'Қайта ашу';button.setAttribute('aria-label',`${entry.file.name} нәтижесін қайта ашу`);button.disabled=busy;button.addEventListener('click',()=>openHistoryEntry(entry.id));
    row.append(thumbnail,content,button);list.append(row);
  }
}
function openHistoryEntry(id){
  if(busy)return;
  const entry=analysisHistory.find(item=>item.id===id);if(!entry)return;
  // Restore the exact previous result and original File. No inference call.
  selectFiles([entry.file]);
  results=[entry.result];activeHistoryId=id;
  showResult(entry.result);renderHistory();updateButton();
  $('history-feedback').textContent=`${entry.file.name}: бұрынғы нәтиже ашылды. Қайта есептеу жасалған жоқ.`;
  $('analysis-title').scrollIntoView({behavior:'smooth',block:'start'});
}
$('clear-history').addEventListener('click',()=>{
  if(busy||!analysisHistory.length)return;
  for(const entry of analysisHistory)URL.revokeObjectURL(entry.thumbnail);
  analysisHistory=[];activeHistoryId=null;
  // Also release the currently displayed image and result.
  $('remove-image').click();renderHistory();
  $('history-feedback').textContent='Талдау тарихы мен ашық нәтиже тазаланды.';
});
window.addEventListener('pagehide',event=>{
  if(event.persisted)return;
  if(previewUrl)URL.revokeObjectURL(previewUrl);
  for(const entry of analysisHistory)URL.revokeObjectURL(entry.thumbnail);
});
renderHistory();
