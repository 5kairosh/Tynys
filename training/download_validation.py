"""Fetch only the archived validation files using official Kaggle ZIP HTTP ranges.

Avoids downloading the entire 2.29 GiB dataset. Each file must match the saved SHA-256.
Signed redirect URL stays in memory. Source images stay in ignored training/cache.
"""
import csv
import json
import time
from types import SimpleNamespace
import hashlib
import io
from pathlib import Path
import struct
from concurrent.futures import ThreadPoolExecutor, as_completed
import zipfile
import zlib
import requests
import threading
THREAD = threading.local()
def session():
    if not hasattr(THREAD,"session"):
        THREAD.session=requests.Session()
    return THREAD.session

ROOT=Path(__file__).resolve().parents[1]


class RemoteZip(io.RawIOBase):
    def __init__(self):
        source='https://www.kaggle.com/api/v1/datasets/download/paultimothymooney/chest-xray-pneumonia?datasetVersionNumber=2'
        with requests.get(source,stream=True,timeout=(25,45)) as response:
            response.raise_for_status()
            self.url=response.url
            self.size=int(response.headers['Content-Length'])
        self.position=0

    def seekable(self):
        return True

    def seek(self,offset,whence=0):
        self.position=offset if whence==0 else self.position+offset if whence==1 else self.size+offset
        return self.position

    def tell(self):
        return self.position

    def read(self,n=-1):
        n=self.size-self.position if n<0 else min(n,self.size-self.position)
        data=self.range(self.position,n) if n else b''
        self.position+=len(data)
        return data

    def range(self,start,n):
        for attempt in range(3):
            try:
                with session().get(self.url,headers={'Range':f'bytes={start}-{min(self.size-1,start+n-1)}'},stream=True,timeout=(25,45)) as response:
                    if response.status_code!=206:
                        raise ValueError('Дереккөзі HTTP Range қолдамайды. Толық архивті қолмен жүктеңіз.')
                    expected=min(n,self.size-start)
                    data=bytearray()
                    began=time.monotonic()
                    for chunk in response.iter_content(64*1024):
                        data.extend(chunk)
                        if len(data)>expected or time.monotonic()-began>120:
                            raise requests.Timeout('Range transfer exceeded its bounded size/time.')
                    if len(data)!=expected:
                        raise requests.ConnectionError('Incomplete HTTP range.')
                    return bytes(data)
            except requests.RequestException:
                if attempt==2:
                    raise


def main():
    cache=ROOT/'training/cache/validation'
    cache.mkdir(parents=True,exist_ok=True)
    with (ROOT/'training/manifests/validation_manifest.csv').open() as f:
        rows=list(csv.DictReader(f))
    complete=[]
    for row in rows:
        target=cache/(row['sha256']+Path(row['path']).suffix)
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest()==row['sha256']:
            complete.append({**row,'path':str(target)})
    if len(complete)==len(rows):
        with (cache/'manifest.csv').open('w',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(complete[0]));writer.writeheader();writer.writerows(complete)
        print(f'Кэш толық тексерілді: {len(complete)}/{len(rows)}. Желі қажет емес.',flush=True)
        return
    remote=RemoteZip()
    index_path=ROOT/'training/cache/kaggle_zip_index_v2.json'
    if index_path.exists():
        raw_index=json.loads(index_path.read_text(encoding='utf-8'))
        assert raw_index['archive_size']==remote.size,'Kaggle archive size changed.'
        infos=[SimpleNamespace(**value) for value in raw_index['entries']]
    else:
        with zipfile.ZipFile(remote) as archive:
            infos=archive.infolist()
        raw_index={'archive_size':remote.size,'entries':[
            {key:getattr(info,key) for key in ['filename','header_offset','compress_size','compress_type','file_size','CRC']} for info in infos]}
        index_path.write_text(json.dumps(raw_index),encoding='utf-8')
    index={}
    for info in infos:
        index.setdefault(Path(info.filename).name,[]).append(info)
    def fetch(row):
        target=cache/(row['sha256']+Path(row['path']).suffix)
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest()==row['sha256']:
            return {**row,'path':str(target)}
        candidates=sorted(index.get(Path(row['path']).name,[]),key=lambda x:len(x.filename))
        for info in candidates:
            data=remote.range(info.header_offset,info.compress_size+65536)
            header=struct.unpack('<4s5H3I2H',data[:30])
            if header[0]!=b'PK\x03\x04':
                raise ValueError('ZIP тақырыбы жарамсыз.')
            offset=30+header[-2]+header[-1]
            compressed=data[offset:offset+info.compress_size]
            image=zlib.decompress(compressed,-15) if info.compress_type==8 else compressed
            if len(image)==info.file_size and zlib.crc32(image)==info.CRC and hashlib.sha256(image).hexdigest()==row['sha256']:
                target.write_bytes(image)
                return {**row,'path':str(target)}
        raise ValueError('Validation файлының SHA-256-сы сәйкес емес: '+Path(row['path']).name)
    resolved={}
    missing=[]
    for row in rows:
        target=cache/(row['sha256']+Path(row['path']).suffix)
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest()==row['sha256']:
            resolved[row['sha256']]={**row,'path':str(target)}
        else:
            missing.append(row)
    print(f'Кэштен тексерілді: {len(resolved)}/{len(rows)}; қалғаны: {len(missing)}',flush=True)
    errors=0
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures={pool.submit(fetch,row):row for row in missing}
        for future in as_completed(futures):
            try:
                row=future.result()
                resolved[row['sha256']]=row
                if len(resolved)%25==0 or len(resolved)==len(rows):
                    print(f'Validation: {len(resolved)}/{len(rows)}',flush=True)
            except requests.RequestException as exc:
                errors+=1
                print(f'Желі қатесі ({type(exc).__name__}), кэш сақталды.',flush=True)
                if errors>=8:
                    for pending in futures:
                        pending.cancel()
                    break
    if len(resolved)!=len(rows):
        raise requests.ConnectionError(f'Validation толық емес: {len(resolved)}/{len(rows)}. Кэштен қайта жалғастыруға болады.')
    resolved=[resolved[row['sha256']] for row in rows]
    with (cache/'manifest.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(resolved[0]));writer.writeheader();writer.writerows(resolved)


if __name__=='__main__':
    main()
