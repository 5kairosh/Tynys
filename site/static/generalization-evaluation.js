'use strict';
let generalizationPromise=null, generalizationViewKey=null, generalizationReport=null, generalizationEpoch=0;
async function renderGeneralizationEvaluation(status){
  const epoch=++generalizationEpoch;
  const panel=document.getElementById('generalization-study');
  if(!panel)return;
  const state=document.getElementById('generalization-status'),cards=document.getElementById('generalization-cards');
  const download=document.getElementById('generalization-download');
  try{
    if(!status?.ready){generalizationViewKey=null;generalizationReport=null;cards.replaceChildren();state.textContent='Сервер қосылғанда тәжірибе есебі тексеріледі.';download.disabled=true;return;}
    if(!generalizationPromise)generalizationPromise=fetch('/static/research/generalization-stage6.json?v=fb1f954b3524',{signal:AbortSignal.timeout(8000)}).then(r=>{if(!r.ok)throw Error('Есеп жүктелмеді.');return r.json();});
    const report=await generalizationPromise;
    if(epoch!==generalizationEpoch)return;
    const current=report.active_model;
    if(status.model_id!==current.id||status.config?.threshold!==current.threshold||status.calibration?.T!==current.temperature){
      generalizationViewKey=null;generalizationReport=null;state.textContent='Бұл тәжірибе есебі сервердегі модельге сәйкес емес. Ағымдағы сурет талдауы бөлек жұмыс істейді.';download.disabled=true;cards.replaceChildren();return;
    }
    const key=JSON.stringify([status.model_id,report]);if(key===generalizationViewKey)return;
    generalizationViewKey=key;generalizationReport=report;cards.replaceChildren();download.disabled=false;
    state.textContent='Жаңа нұсқа тексеру талаптарынан өтті. Төмендегі accuracy барлық суретке техникалық санат бергендегі нәтиже; OOD қамтуы бөлек көрсетілген.';
    const make=(tag,text,cls)=>{const el=document.createElement(tag);el.textContent=text;if(cls)el.className=cls;return el;};
    const pct=x=>(x*100).toFixed(2).replace('.',',')+'%';
    for(const [name,title,note] of [['rsna_confirmation','RSNA · жаңа тексеру','400 бөлек NIH пациенті · 200 қалыпты / 200 өкпе күңгірттенуі'],['external_openi','OpenI · қайта тексеру','200 сурет · әлсіз есеп белгілері · бұрын қаралған'],['internal_test','Kaggle · бақылау','618 сурет · бұрын қаралған ішкі тест']]){
      const value=report.cohorts[name],old=value.baseline,now=value.candidate;
      const card=make('article','','audit-card');card.dataset.cohort=name;
      card.append(make('h4',title),make('p',note,'audit-count'),make('p',`${pct(old.accuracy)} → ${pct(now.accuracy)}`,'study-comparison'));
      const stats=make('dl','','audit-stats');
      for(const [label,text] of [['Қате болжам',`${old.errors} → ${now.errors} / ${now.count}`],['Сезімталдық',pct(now.sensitivity)],['Қалыптыны тану',pct(now.specificity)],['Теңгерілген accuracy',pct(now.balanced_accuracy)]]){
        stats.append(make('dt',label),make('dd',text));
      }
      card.append(stats);
      if(value.fresh_holdout){const ci=value.paired_accuracy_bootstrap;card.append(make('p',`Дәлдік өсімінің 95% аралығы: +${(ci.lower_95*100).toFixed(1).replace('.',',')}…+${(ci.upper_95*100).toFixed(1).replace('.',',')} пайыздық тармақ. Бұл клиникалық сенімділік емес.`,'study-limit'));}
      else if(now.errors>old.errors)card.append(make('p','Бұл жиында қате азайған жоқ: бір қате артық.','study-tradeoff'));
      cards.append(card);
    }
  }catch{
    if(epoch!==generalizationEpoch)return;
    generalizationPromise=null;generalizationViewKey=null;generalizationReport=null;cards.replaceChildren();download.disabled=true;state.textContent='Сыртқы тәжірибе есебі қолжетімсіз. Суретті талдау бөлек жұмыс істейді.';
  }
}
document.getElementById('generalization-download')?.addEventListener('click',()=>{
  if(!generalizationReport)return;
  const url=URL.createObjectURL(new Blob([JSON.stringify(generalizationReport,null,2)],{type:'application/json'}));
  const link=document.createElement('a');link.href=url;link.download='tynys-generalization-stage6.json';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
});
