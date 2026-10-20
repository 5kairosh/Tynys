'use strict';
// Explicit local consent only. This module never transmits data or uses cookies.
(() => {
  const DATABASE='tynys-history-v1', LIMIT=20, DAYS=7, MAX_BYTES=80*1024*1024;
  const ENTRY_LIMIT=12*1024*1024, AGE=DAYS*24*60*60*1000;
  class HistoryChanged extends Error {}
  class HistoryLocked extends Error {}
  class VaultError extends Error {}
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
        // v1 clients fail closed with VersionError instead of pruning v2 ciphertext.
        const request=this.indexedDB.open(this.name,2);
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
      return this.transaction(({entries,settings,prefs,rows,now})=>{
        if(prefs?.vault)throw new HistoryLocked('Protected history requires the vault reader');
        const enabled=prefs?.enabled===true;
        const keep=enabled?retained(rows,now):[];
        const ids=new Set(keep.map(r=>r.id));
        for(const row of rows)if(!ids.has(row.id))entries.delete(row.id);
        if(keep.length!==rows.length)settings.put({...prefs,key:'consent',revision:generation()});
        return {enabled,generation:prefs?.generation||null,revision:prefs?.revision||null,protected:false,locked:false,records:keep};
      });
    }
    enable(records){
      return this.transaction(({entries,settings,prefs,rows,now})=>{
        if(prefs?.vault)throw new HistoryLocked('Protected history cannot become plaintext');
        const keep=retained([...records,...rows],now),token=generation();
        for(const row of records)if(bytes(row)>MAX_BYTES)throw new Error('History entry exceeds storage limit');
        entries.clear();for(const row of keep)entries.put(row);
        settings.put({key:'consent',enabled:true,generation:token,revision:generation()});
        return {enabled:true,generation:token,ids:keep.map(r=>r.id)};
      });
    }
    append(record,token){
      return this.transaction(({entries,settings,prefs,rows,now})=>{
        if(prefs?.vault)throw new HistoryLocked('Protected history cannot become plaintext');
        if(prefs?.enabled!==true||prefs.generation!==token)throw new HistoryChanged('History consent changed');
        if(!validRecord(record,now)||bytes(record)>MAX_BYTES)throw new Error('History entry is invalid or too large');
        const keep=retained([record,...rows],now),ids=new Set(keep.map(r=>r.id));
        for(const row of rows)if(!ids.has(row.id))entries.delete(row.id);
        if(ids.has(record.id))entries.put(record);
        settings.put({...prefs,revision:generation()});
        return {ids:[...ids]};
      });
    }
    remove(id,token){
      return this.transaction(({entries,settings,prefs})=>{
        if(prefs?.vault)throw new HistoryLocked('Protected history is locked');
        if(prefs?.enabled!==true||prefs.generation!==token)throw new HistoryChanged('History consent changed');
        entries.delete(id);
        settings.put({...prefs,revision:generation()});
      });
    }
    clear(){
      return this.transaction(({entries,settings})=>{
        entries.clear();settings.put({key:'consent',enabled:false,generation:generation(),revision:generation()});
      });
    }
  }
  const ITERATIONS=600000, encoder=new TextEncoder(), decoder=new TextDecoder('utf-8',{fatal:true});
  const secureAvailable=()=>!!globalThis.crypto?.subtle&&globalThis.isSecureContext===true;
  const random=n=>crypto.getRandomValues(new Uint8Array(n));
  const aad=(vault,row)=>encoder.encode(JSON.stringify(['tynys-vault-v1',vault.id,...(row?[row.id,row.createdAt,row.expiresAt]:['verifier'])]));
  function validVault(v){
    return v?.schema===1&&typeof v.id==='string'&&v.id.length<=80&&v.iterations===ITERATIONS&&
      v.salt instanceof Uint8Array&&v.salt.byteLength===16&&
      v.iv instanceof Uint8Array&&v.iv.byteLength===12&&v.verifier instanceof ArrayBuffer&&v.verifier.byteLength===encoder.encode('Tynys history vault v1!').length+16;
  }
  async function derive(password,vault){
    if(!secureAvailable())throw new VaultError('Шифрлау үшін HTTPS немесе localhost және Web Crypto қолдауы қажет.');
    if(typeof password!=='string'||password.length<10||password.length>256||!password.trim())throw new VaultError('Пароль 10–256 таңба болсын; тек бос орыннан тұрмасын.');
    const material=await crypto.subtle.importKey('raw',encoder.encode(password),'PBKDF2',false,['deriveKey']);
    return crypto.subtle.deriveKey({name:'PBKDF2',hash:'SHA-256',salt:vault.salt,iterations:ITERATIONS},material,{name:'AES-GCM',length:256},false,['encrypt','decrypt']);
  }
  async function seal(record,key,vault){
    const metadata=encoder.encode(JSON.stringify({filename:record.filename,type:record.type,lastModified:record.lastModified,result:record.result}));
    const file=new Uint8Array(await record.file.arrayBuffer());
    if(metadata.length+file.length+4>MAX_BYTES)throw new VaultError('Жазба сақтау шегінен үлкен.');
    const frame=new Uint8Array(4+metadata.length+file.length);
    new DataView(frame.buffer).setUint32(0,metadata.length);
    frame.set(metadata,4);frame.set(file,4+metadata.length);
    const row={version:2,id:record.id,createdAt:record.createdAt,expiresAt:record.expiresAt,iv:random(12)};
    row.ciphertext=await crypto.subtle.encrypt({name:'AES-GCM',iv:row.iv,additionalData:aad(vault,row),tagLength:128},key,frame);
    return row;
  }
  function validEnvelope(row){
    return row?.version===2&&typeof row.id==='string'&&row.id.length<=80&&
      Number.isFinite(row.createdAt)&&Number.isFinite(row.expiresAt)&&row.expiresAt<=row.createdAt+AGE&&
      row.iv instanceof Uint8Array&&row.iv.byteLength===12&&row.ciphertext instanceof ArrayBuffer&&
      row.ciphertext.byteLength>20&&row.ciphertext.byteLength<=MAX_BYTES;
  }
  async function unseal(row,key,vault,now){
    if(!validEnvelope(row))throw new VaultError('Сақталған жазба форматы бүлінген.');
    try{
      const frame=await crypto.subtle.decrypt({name:'AES-GCM',iv:row.iv,additionalData:aad(vault,row),tagLength:128},key,row.ciphertext);
      const length=new DataView(frame).getUint32(0);
      if(length>frame.byteLength-4)throw new Error('Invalid frame');
      const metadata=JSON.parse(decoder.decode(new Uint8Array(frame,4,length)));
      const record={...metadata,version:1,id:row.id,createdAt:row.createdAt,expiresAt:row.expiresAt,file:new Blob([frame.slice(4+length)],{type:metadata.type||''})};
      if(!validRecord(record,now))throw new Error('Invalid record');
      return record;
    }catch{throw new VaultError('Шифрланған жазбаны оқу мүмкін болмады. Ол өшірілген жоқ.');}
  }
  function keptEnvelopes(rows,now){
    const seen=new Set(),keep=[];let size=0;
    for(const row of [...rows].sort((a,b)=>b.createdAt-a.createdAt||b.id.localeCompare(a.id))){
      if(!validEnvelope(row))throw new VaultError('Қоймадағы жазба бүлінген; деректер өзгертілмеді.');
      if(row.expiresAt<=now||row.createdAt>now||seen.has(row.id))continue;
      seen.add(row.id);const used=row.ciphertext.byteLength+2048;
      if(keep.length<LIMIT&&size+used<=MAX_BYTES){keep.push(row);size+=used;}
    }
    return keep;
  }
  class VaultStore extends Store{
    constructor(options={}){super(options);this.key=null;this.keyId=null;this.epoch=0;}
    lock(){this.key=null;this.keyId=null;this.epoch++;}
    snapshot(){return this.transaction(({prefs,rows})=>({prefs,rows}));}
    check(snapshot,prefs){
      if((snapshot.prefs?.generation||null)!==(prefs?.generation||null)||
         (snapshot.prefs?.revision||null)!==(prefs?.revision||null))throw new HistoryChanged('History changed during encryption');
    }
    unlocked(vault){
      if(!this.key||this.keyId!==vault?.id)throw new HistoryLocked('Тарих құлыптаулы. Парольмен ашыңыз.');
      return {key:this.key,epoch:this.epoch};
    }
    checkKey(vault,state){if(state.epoch!==this.epoch||this.keyId!==vault.id)throw new HistoryLocked('Тарих құлыпталды.');}
    async read(){
      let snapshot=await this.snapshot();
      if(!snapshot.prefs?.vault){this.lock();return super.read();}
      if(snapshot.prefs.enabled!==true){this.lock();throw new VaultError('Қойманың келісім күйі сәйкес емес. Деректер өзгертілмеді.');}
      const vault=snapshot.prefs.vault;
      if(!validVault(vault))throw new VaultError('Қойма параметрлері бүлінген; деректер өшірілген жоқ.');
      // Expired ciphertext can be removed without unlocking; timestamps remain
      // public metadata, and are also authenticated when a record is decrypted.
      snapshot=await this.transaction(({entries,settings,prefs,rows,now})=>{
        this.check(snapshot,prefs);
        const keep=rows.filter(row=>!(validEnvelope(row)&&row.expiresAt<=now));
        for(const row of rows)if(!keep.includes(row))entries.delete(row.id);
        if(keep.length!==rows.length){prefs={...prefs,revision:generation()};settings.put(prefs);}
        return {prefs,rows:keep};
      });
      const state={enabled:true,protected:true,locked:this.keyId!==vault.id||!this.key,generation:snapshot.prefs.generation,revision:snapshot.prefs.revision,storedCount:snapshot.rows.length,records:[]};
      if(state.locked){this.lock();return state;}
      const current=this.unlocked(vault),records=[];
      for(const row of snapshot.rows)records.push(await unseal(row,current.key,vault,this.now()));
      this.checkKey(vault,current);
      await this.transaction(({prefs})=>this.check(snapshot,prefs));
      state.records=retained(records,this.now());
      return state;
    }
    async protect(password,records=[]){
      const snapshot=await this.snapshot(),epoch=this.epoch;
      if(snapshot.prefs?.vault)throw new VaultError('Тарих қорғалған. Алдымен парольмен ашыңыз.');
      const vault={schema:1,id:generation(),iterations:ITERATIONS,salt:random(16),iv:random(12)};
      const key=await derive(password,vault);
      vault.verifier=await crypto.subtle.encrypt({name:'AES-GCM',iv:vault.iv,additionalData:aad(vault),tagLength:128},key,encoder.encode('Tynys history vault v1!'));
      const rows=[];
      for(const record of retained([...records,...snapshot.rows],this.now()))rows.push(await seal(record,key,vault));
      const outcome=await this.transaction(({entries,settings,prefs,now})=>{
        this.check(snapshot,prefs);
        if(epoch!==this.epoch)throw new HistoryLocked('Protection cancelled by lock');
        const keep=keptEnvelopes(rows,now),token=generation();
        entries.clear();for(const row of keep)entries.put(row);
        settings.put({key:'consent',enabled:true,generation:token,revision:generation(),vault});
        return {generation:token,ids:keep.map(row=>row.id)};
      });
      if(epoch===this.epoch){this.key=key;this.keyId=vault.id;}
      return outcome;
    }
    async unlock(password){
      this.lock();const epoch=this.epoch,snapshot=await this.snapshot(),vault=snapshot.prefs?.vault;
      if(!validVault(vault))throw new VaultError('Шифрланған қойма қолжетімсіз.');
      const key=await derive(password,vault);
      try{
        const value=await crypto.subtle.decrypt({name:'AES-GCM',iv:vault.iv,additionalData:aad(vault),tagLength:128},key,vault.verifier);
        if(decoder.decode(value)!=='Tynys history vault v1!')throw new Error('Verifier mismatch');
      }catch{throw new VaultError('Пароль дұрыс емес немесе қойма бүлінген. Деректер өшірілген жоқ.');}
      await this.transaction(({prefs})=>this.check(snapshot,prefs));
      if(epoch!==this.epoch)throw new HistoryLocked('Unlock cancelled');
      this.key=key;this.keyId=vault.id;
      try{return await this.read();}catch(error){this.lock();throw error;}
    }
    async saveEncrypted(records,token,enable=false){
      const snapshot=await this.snapshot(),vault=snapshot.prefs?.vault;
      if(!vault)return enable?super.enable(records):super.append(records[0],token);
      const current=this.unlocked(vault),sealed=[];
      if(!enable&&snapshot.prefs.generation!==token)throw new HistoryChanged('History consent changed');
      for(const record of records){
        if(!validRecord(record,this.now())||bytes(record)>MAX_BYTES)throw new VaultError('Жазба жарамсыз немесе тым үлкен.');
        sealed.push(await seal(record,current.key,vault));
      }
      return this.transaction(({entries,settings,prefs,now})=>{
        this.check(snapshot,prefs);this.checkKey(vault,current);
        const keep=keptEnvelopes([...sealed,...snapshot.rows],now),ids=new Set(keep.map(row=>row.id));
        for(const row of snapshot.rows)if(!ids.has(row.id))entries.delete(row.id);
        for(const row of sealed)if(ids.has(row.id))entries.put(row);
        const generationToken=enable?generation():prefs.generation;
        settings.put({...prefs,generation:generationToken,revision:generation()});
        return {enabled:true,generation:generationToken,ids:[...ids]};
      });
    }
    enable(records){return this.saveEncrypted(records,null,true);}
    append(record,token){return this.saveEncrypted([record],token);}
    remove(id,token){
      return this.transaction(({entries,settings,prefs})=>{
        if(prefs?.vault)this.unlocked(prefs.vault);
        if(prefs?.enabled!==true||prefs.generation!==token)throw new HistoryChanged('History consent changed');
        entries.delete(id);settings.put({...prefs,revision:generation()});
      });
    }
    async clear(){await super.clear();this.lock();}
  }
  globalThis.TynysHistory=Object.freeze({Store:VaultStore,HistoryChanged,HistoryLocked,VaultError,secureAvailable,recordFor,restore,LIMIT,DAYS,MAX_BYTES});
})();
