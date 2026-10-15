"""Exercise the actual Docker HTTP service; never writes uploaded images.

Run after docker run: python training/verify_docker_live.py --port 7861
Evidence contains HTTP status, model configuration and PDF checks, no image data.
"""
import argparse
import io
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image
from pypdf import PdfReader
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid
import requests

ROOT=Path(__file__).resolve().parents[1]


def dicom(modality):
    meta=FileMetaDataset();meta.TransferSyntaxUID=ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID=generate_uid();meta.MediaStorageSOPInstanceUID=generate_uid()
    ds=FileDataset(None,{},file_meta=meta,preamble=b'\0'*128)
    ds.Modality=modality;ds.Rows=64;ds.Columns=80;ds.SamplesPerPixel=1
    ds.PhotometricInterpretation='MONOCHROME2';ds.BitsAllocated=16;ds.BitsStored=16
    ds.HighBit=15;ds.PixelRepresentation=0
    ds.PixelData=np.arange(64*80,dtype=np.uint16).reshape(64,80).tobytes()
    stream=io.BytesIO();ds.save_as(stream,enforce_file_format=True);return stream.getvalue()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=7861)
    parser.add_argument('--image',type=Path,required=True);args=parser.parse_args()
    base=f'http://127.0.0.1:{args.port}'
    started=time.monotonic();status=None
    while time.monotonic()-started<120:
        try:
            response=requests.get(base+'/api/status',timeout=4)
            if response.status_code==200:
                status=response.json()
                if status.get('ready'):break
        except requests.RequestException:pass
        time.sleep(2)
    assert status and status.get('ready'),status
    for feature in ('dicom','xray_check','mcnemar','ensemble'):
        assert status[feature]['available'],(feature,status[feature])
    assert status['calibration']['T']!=1
    print('Container ready; calibration, DICOM, OOD, McNemar and ensemble available.',flush=True)
    png=io.BytesIO()
    with Image.open(args.image) as image:image.convert('RGB').save(png,format='PNG')
    data=png.getvalue();cr=dicom('CR');checks=[]
    for name,content,expected in [('sample.png',data,200),('sample_cr.dcm',cr,200),('sample_ct.dcm',dicom('CT'),400),('text.jpg',b'This is a text file, not an image.',400)]:
        response=requests.post(base+'/api/predict',files={'file':(name,content)},timeout=90)
        assert response.status_code==expected,(name,response.status_code,response.text[:2000])
        row={'file':name,'http_status':response.status_code}
        if expected==200:
            result=response.json()
            assert len(result['comparison']['models'])==3
            row.update(model_class=result['model_class'],raw_score=result['raw_score'],pneumonia_score=result['pneumonia_score'],threshold=result['threshold'],gradcam_available=result['explanation']['available'])
        checks.append(row);print(f'{name}: {response.status_code}',flush=True)
    response=requests.post(base+'/api/report',files=[('files',('sample.png',data,'image/png')),('files',('sample_cr.dcm',cr,'application/dicom'))],timeout=120)
    assert response.status_code==200,response.text[:2000]
    assert response.content.startswith(b'%PDF')
    pdf=PdfReader(io.BytesIO(response.content));assert len(pdf.pages)==2
    assert all('Ensemble' in page.extract_text() for page in pdf.pages)
    output=ROOT/'training/docker_evidence';output.mkdir(exist_ok=True)
    # Only a public demonstration image and synthetic DICOM enter the proof PDF.
    (output/'container_report.pdf').write_bytes(response.content)
    (output/'status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf-8')
    report={'base_url':base,'ready':True,'status':{k:status[k] for k in ('model_id','tensorflow_version','calibration','dicom','xray_check','mcnemar','ensemble')},'predict_checks':checks,'pdf':{'http_status':200,'signature':'%PDF','pages':len(pdf.pages),'ensemble_on_each_page':True},'total_verification_seconds':round(time.monotonic()-started,2),'synthetic_dicom':True,'public_png_source':args.image.name}
    (output/'live_result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('PDF: 200, %PDF, 2 pages; all live checks passed.',flush=True)


if __name__=='__main__':main()
