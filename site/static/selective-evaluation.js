'use strict';
// Read-only aggregate evidence from /api/status. No inference or history writes.
let selectiveReport=null;
let selectiveViewKey=null;
function renderSelectiveEvaluation(report){
  const viewKey=JSON.stringify(report??null);
  if(viewKey===selectiveViewKey)return;
  selectiveViewKey=viewKey;
  const cards=document.getElementById('selective-cards');
  const state=document.getElementById('selective-status');
  const download=document.getElementById('selective-download');
  const finding=document.getElementById('selective-finding');
  const calibration=document.getElementById('ood-calibration-detail');
  calibration.hidden=true;
  cards.replaceChildren();selectiveReport=null;download.disabled=true;finding.hidden=true;
  if(!report?.available){state.textContent=report?.message||'Қамту/қателік есебі қолжетімсіз. Суретті талдау бөлек жұмыс істейді.';return;}
  const pct=value=>value===null?'Есептелмейді':(value*100).toFixed(2).replace('.',',')+'%';
  const element=(tag,content,className)=>{const el=document.createElement(tag);el.textContent=content;if(className)el.className=className;return el;};
  try{
    for(const [key,title,description] of [
      ['internal_test','Ішкі тест · Kaggle','Тест деректері. Топтар файл атынан алынған; ресми пациент ID емес.'],
      ['external_openi','Сыртқы тексеру · OpenI','Басқа дереккөз. Бір есеп ID-іне бір сурет; пациенттің қайталанбауы расталмаған.'],
      ...(report.cohorts.rsna_confirmation?[['rsna_confirmation','RSNA · жаңа 400 пациент','RSNA оң санаты — өкпе күңгірттенуі. Бұл клиникалық диагноз емес.']]:[])]){
      const value=report.cohorts[key],all=value.baseline,accepted=value.accepted,withheld=value.withheld;
      const card=element('article','','audit-card');card.dataset.cohort=key;
      card.append(element('h4',title),element('p',`${all.count} сурет · NORMAL ${all.normal} · ${key==='rsna_confirmation'?'RSNA_LUNG_OPACITY':'PNEUMONIA'} ${all.pneumonia}`,'audit-count'));
      const before=(report.comparison_to_stage5||report.comparison_to_stage4)?.cohorts?.[key];
      if(before)card.append(element('p',`Қамту: бұрын ${pct(before.coverage)} (${before.accepted.count}/${all.count}) → қазір ${pct(value.coverage)}.`,'audit-change'));
      const headline=element('div','','audit-coverage');headline.append(element('strong',pct(value.coverage)),element('span',`Қамту: ${accepted.count} / ${all.count} суретке қорытынды берілді`));card.append(headline);
      const bar=element('div','','audit-bar');bar.setAttribute('role','img');bar.setAttribute('aria-label',`Қорытынды берілді: ${accepted.count}; берілмеді: ${withheld.count}`);
      const filled=element('span','');filled.style.width=(value.coverage*100)+'%';bar.append(filled);card.append(bar);
      card.append(element('p',`Қорытынды берілмеді: ${withheld.count} / ${all.count} (${pct(value.withheld_rate)}).`,'audit-withheld'));
      const dl=element('dl','','audit-stats');
      for(const [label,text] of [
        ['Барлық сурет: ережеге дейінгі қателік',`${all.errors} / ${all.count} · ${pct(all.error_rate)}`],
        ['Қорытынды берілгендер: қателік',accepted.count?`${accepted.errors} / ${accepted.count} · ${pct(accepted.error_rate)}`:'Қорытынды берілген сурет жоқ'],
        ['Қорытынды берілгендер: accuracy',accepted.count?pct(accepted.accuracy):'Бағалау мүмкін емес']]){
        dl.append(element('dt',label),element('dd',text));
      }
      card.append(dl);
      const details=element('details','','audit-details');details.append(element('summary','Санаттар мен қателерді ашу'));
      const table=element('table','');table.append(element('caption','Тек қорытынды берілген суреттер'));
      const body=element('tbody','');
      for(const [label,text] of [
        ['NORMAL / PNEUMONIA саны',`${accepted.normal} / ${accepted.pneumonia}`],
        ['Sensitivity (пневмонияны табу)',accepted.pneumonia?`${accepted.TP} / ${accepted.pneumonia} · ${pct(accepted.sensitivity)}`:'Пневмония суреті жоқ; есептелмейді'],
        ['Specificity (қалыптыны тану)',accepted.normal?`${accepted.TN} / ${accepted.normal} · ${pct(accepted.specificity)}`:'NORMAL суреті жоқ; есептелмейді'],
        ['TN / FP / FN / TP',`${accepted.TN} / ${accepted.FP} / ${accepted.FN} / ${accepted.TP}`]]){
        const tr=element('tr','');tr.append(element('th',label),element('td',text));body.append(tr);
      }
      table.append(body);details.append(table,element('p',`Тоқтатылған ${withheld.count} суретке модельдің техникалық санатын қолданғанда ${withheld.errors} қате болар еді. Бұл суреттерге сайт қорытынды бермейді.`),element('p',description));card.append(details);cards.append(card);
    }
    state.textContent=`Бекітілген ереже: ${report.policy}. Жіктеу шегі ${Number(report.classification_threshold).toFixed(3).replace('.',',')}; шектер осы тесттерде таңдалған жоқ.`;
    if(report.ood_calibration){
      const c=report.ood_calibration;
      calibration.textContent=`OOD үлгісі: ${c.reference_count} train суреті. Шек: бөлек ${c.calibration_count} validation суреті, алдын ала бекітілген ${pct(c.target_acceptance)} қабылдау мақсаты. Бұл мақсат медициналық дәлдік немесе өзге дереккөздегі қамту кепілдігі емес. Test/OpenI бұрын да қаралған; бұл — сол жиындарды қайта тексеру.`;
      calibration.hidden=false;
    }
    const internal=report.cohorts.internal_test,external=report.cohorts.external_openi;
    if(internal.coverage<.1||external.accepted.count===0||internal.accepted.error_rate>=internal.baseline.error_rate){
      finding.textContent=`Шектеу анықталды: ішкі тестте ${internal.accepted.count} / ${internal.baseline.count}, сыртқы OpenI-де ${external.accepted.count} / ${external.baseline.count} суретке қорытынды берілді. Қамтудың артуы классификация дәлдігінің артуы емес. Қабылданған топ қатесін бөлек қараңыз; сыртқы деректердегі клиникалық сенімділік расталмаған.`;
      finding.hidden=false;
    }
    selectiveReport=report;download.disabled=false;
  }catch{
    cards.replaceChildren();state.textContent='Қамту/қателік есебін көрсету мүмкін болмады. Суретті талдау бөлек жұмыс істейді.';
  }
}
document.getElementById('selective-download').addEventListener('click',()=>{
  if(!selectiveReport)return;
  const url=URL.createObjectURL(new Blob([JSON.stringify(selectiveReport,null,2)],{type:'application/json'}));
  const link=document.createElement('a');link.href=url;link.download='tynys-selective-evaluation.json';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
});
