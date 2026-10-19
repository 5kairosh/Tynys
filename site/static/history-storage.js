'use strict';
// Explicit local consent only. This module never transmits data or uses cookies.
(() => {
  const DATABASE='tynys-history-v1', LIMIT=20, DAYS=7, MAX_BYTES=80*1024*1024;
  const ENTRY_LIMIT=12*1024*1024, AGE=DAYS*24*60*60*1000;
  class HistoryChanged extends Error {}
  const generation=()=>crypto.randomUUID();
  function validRecord(r,now){
    return r?.version===1&&typeof r.id==='string'&&r.id.length<=80&&
      r.file instanceof Blob&&r.file.size>0&&r.file.size<=ENTRY_LIMIT&&
      typeof r.filename==='string'&&r.filename.length<=1024&&/\.(jpe?g|png|dcm)$/i.test(r.filename)&&
      Number.isFinite(r.createdAt)&&r.createdAt<=now&&Number.isFinite(r.expiresAt)&&
      r.expiresAt>now&&r.expiresAt<=r.createdAt+AGE&&
      ['NORMAL','PNEUMONIA'].includes(r.result?.model_class)&&
      Number.isFinite(r.result.pneumonia_score)&&r.result.pneumonia_score>=0&&r.result.pneumonia_score<=1&&
      Number.isFinite(r.result.threshold)&&r.result.threshold>=0&&r.result.threshold<=1&&
      typeof r.result.filename==='string'&&typeof r.result.model_id==='string'&&
      Number.isFinite(r.result.width)&&r.result.width>0&&Number.isFinite(r.result.height)&&r.result.height>0&&
      Number.isFinite(r.result.elapsed_ms)&&
      (!r.result.calibration||Number.isFinite(r.result.calibration.T))&&
      (!r.result.quality?.available||(Array.isArray(r.result.quality.warnings)&&r.result.quality.warnings.every(w=>w&&typeof w.title==='string'&&typeof w.action==='string')))&&
      (!r.result.comparison?.available||(Array.isArray(r.result.comparison.models)&&r.result.comparison.models.every(m=>m&&['NORMAL','PNEUMONIA'].includes(m.model_class)&&typeof m.name==='string'&&Number.isFinite(m.pneumonia_score)&&Number.isFinite(m.threshold))))&&
      (!r.result.explanation?.available||typeof r.result.explanation.overlay==='string')&&
      (!r.result.preview||/^data:image\/png;base64,[A-Za-z0-9+/=]+$/.test(r.result.preview))&&
      (!r.result.explanation?.overlay||/^data:image\/png;base64,[A-Za-z0-9+/=]+$/.test(r.result.explanation.overlay));
  }
  function bytes(record){return record.file.size+new Blob([JSON.stringify(record.result)]).size+2048;}
  function retained(records,now){
    const seen=new Set(),keep=[];
    let used=0;
    for(const r of records.filter(r=>validRecord(r,now)).sort((a,b)=>b.createdAt-a.createdAt||b.id.localeCompare(a.id))){
      if(seen.has(r.id))continue;
      seen.add(r.id);
      const size=bytes(r);
      if(keep.length<LIMIT&&used+size<=MAX_BYTES){keep.push(r);used+=size;}
    }
    return keep;
  }
  function recordFor(entry){
    const createdAt=entry.time.getTime();
    return {version:1,id:entry.id,file:entry.file,filename:entry.file.name,type:entry.file.type,
      lastModified:entry.file.lastModified,createdAt,expiresAt:createdAt+AGE,result:entry.result};
  }
  function restore(record){
    const file=new File([record.file],record.filename,{type:record.type||'',lastModified:record.lastModified});
    return {id:record.id,file,result:record.result,time:new Date(record.createdAt),saved:true,
      thumbnail:record.result.preview||URL.createObjectURL(file)};
  }
  class Store {
    constructor(options={}){
      try{this.indexedDB='indexedDB' in options?options.indexedDB:globalThis.indexedDB;}catch{this.indexedDB=null;}
      this.now=options.now||(()=>Date.now());this.name=options.name||DATABASE;
    }
    open(){
      return new Promise((resolve,reject)=>{
        if(!this.indexedDB){reject(new Error('Browser storage unavailable'));return;}
        const request=this.indexedDB.open(this.name,1);
        let done=false;
        const timer=setTimeout(()=>{done=true;reject(new Error('Browser storage timed out'));},8000);
        request.onupgradeneeded=()=>{
          const db=request.result;
          if(!db.objectStoreNames.contains('entries'))db.createObjectStore('entries',{keyPath:'id'});
          if(!db.objectStoreNames.contains('settings'))db.createObjectStore('settings',{keyPath:'key'});
        };
        request.onsuccess=()=>{
          clearTimeout(timer);
          const db=request.result;
          if(done){db.close();return;}
          done=true;db.onversionchange=()=>db.close();resolve(db);
        };
        request.onerror=()=>{clearTimeout(timer);if(!done){done=true;reject(request.error);}};
        request.onblocked=()=>{clearTimeout(timer);if(!done){done=true;reject(new Error('Browser storage blocked'));}};
      });
    }
    async transaction(operation){
      const db=await this.open();
      try{
        return await new Promise((resolve,reject)=>{
          const tx=db.transaction(['entries','settings'],'readwrite');
          const entries=tx.objectStore('entries'),settings=tx.objectStore('settings');
          const prefs=settings.get('consent'),rows=entries.getAll();
          let value,failure;
          const apply=()=>{
            if(prefs.readyState!=='done'||rows.readyState!=='done')return;
            try{value=operation({entries,settings,prefs:prefs.result,rows:rows.result,now:this.now()});}
            catch(error){failure=error;tx.abort();}
          };
          prefs.onsuccess=apply;rows.onsuccess=apply;
          tx.oncomplete=()=>resolve(value);
          tx.onabort=()=>reject(failure||tx.error||new Error('Browser storage transaction aborted'));
          tx.onerror=()=>{}; // onabort is the authoritative failure, never claim saved early.
        });
      }finally{db.close();}
    }
    read(){
      return this.transaction(({entries,prefs,rows,now})=>{
        const enabled=prefs?.enabled===true;
        const keep=enabled?retained(rows,now):[];
        const ids=new Set(keep.map(r=>r.id));
        for(const row of rows)if(!ids.has(row.id))entries.delete(row.id);
        return {enabled,generation:prefs?.generation||null,records:keep};
      });
    }
    enable(records){
      return this.transaction(({entries,settings,rows,now})=>{
        const keep=retained([...records,...rows],now),token=generation();
        for(const row of records)if(bytes(row)>MAX_BYTES)throw new Error('History entry exceeds storage limit');
        entries.clear();for(const row of keep)entries.put(row);
        settings.put({key:'consent',enabled:true,generation:token});
        return {enabled:true,generation:token,ids:keep.map(r=>r.id)};
      });
    }
    append(record,token){
      return this.transaction(({entries,prefs,rows,now})=>{
        if(prefs?.enabled!==true||prefs.generation!==token)throw new HistoryChanged('History consent changed');
        if(!validRecord(record,now)||bytes(record)>MAX_BYTES)throw new Error('History entry is invalid or too large');
        const keep=retained([record,...rows],now),ids=new Set(keep.map(r=>r.id));
        for(const row of rows)if(!ids.has(row.id))entries.delete(row.id);
        if(ids.has(record.id))entries.put(record);
        return {ids:[...ids]};
      });
    }
    remove(id,token){
      return this.transaction(({entries,prefs})=>{
        if(prefs?.enabled!==true||prefs.generation!==token)throw new HistoryChanged('History consent changed');
        entries.delete(id);
      });
    }
    clear(){
      return this.transaction(({entries,settings})=>{
        entries.clear();settings.put({key:'consent',enabled:false,generation:generation()});
      });
    }
  }
  globalThis.TynysHistory=Object.freeze({Store,HistoryChanged,recordFor,restore,LIMIT,DAYS,MAX_BYTES});
})();
