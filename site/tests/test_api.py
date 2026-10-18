import io
import numpy as np
from pypdf import PdfReader
from PIL import Image
from pydicom.dataset import FileDataset,FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian,generate_uid


def dicom(modality='CR'):
    meta=FileMetaDataset();meta.TransferSyntaxUID=ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID=generate_uid();meta.MediaStorageSOPInstanceUID=generate_uid()
    ds=FileDataset(None,{},file_meta=meta,preamble=b'\0'*128)
    ds.Modality=modality;ds.Rows=64;ds.Columns=80;ds.SamplesPerPixel=1
    ds.PhotometricInterpretation='MONOCHROME2';ds.BitsAllocated=16;ds.BitsStored=16;ds.HighBit=15;ds.PixelRepresentation=0
    ds.PixelData=np.arange(64*80,dtype=np.uint16).reshape(64,80).tobytes()
    stream=io.BytesIO();ds.save_as(stream,enforce_file_format=True);return stream.getvalue()


def test_protocol(client):
    status=client.get('/api/status').json()
    assert status['ready'] and status['comparison_ready'] and status['dicom']['available']
    assert 'calibration' in status and 'xray_check' in status and status['mcnemar']['available']
    assert client.post('/api/predict',files={'file':('bad.jpg',b'hello')}).status_code==400
    assert client.post('/api/predict',files={'file':('big.png',b'x'*(13*1024*1024))}).status_code==413
    assert client.post('/api/predict',files={'file':('empty.png',b'')}).status_code==400
    assert client.post('/api/predict',files={'file':('bad.dcm',b'broken')}).status_code==400
    assert client.post('/api/predict',files={'file':('ct.dcm',dicom('CT'))}).status_code==400
    image=io.BytesIO();Image.fromarray(np.random.default_rng(42).integers(0,256,(100,80,3),dtype=np.uint8)).save(image,format='PNG')
    raw=image.getvalue()
    prediction=client.post('/api/predict',files={'file':('sample.png',raw)})
    assert prediction.status_code==200,prediction.text
    r=prediction.json()
    assert 'xray_check' in r and r['comparison']['available']
    assert r['quality']['available'] and r['quality']['policy'] == 'image_quality_v1'
    from app import engine
    tf=engine.tf
    x=tf.io.decode_image(raw,channels=3,expand_animations=False)
    x=tf.image.resize_with_pad(tf.cast(x,tf.float32),224,224)
    expected=float(engine.model(x[None,...],training=False).numpy()[0,0])
    from calibration import calibrate_score
    assert r['pneumonia_score']==calibrate_score(r['raw_score'],engine.config.get('calibration'))
    assert abs(r['raw_score']-expected)<1e-6
    converted=client.post('/api/predict',files={'file':('sample.dcm',dicom())})
    assert converted.status_code==200,converted.text
    assert converted.json()['preview'].startswith('data:image/png;base64,')
    assert converted.json()['quality']['available']
    assert 'түрлендірілген PNG' in converted.json()['quality']['notice']
    report=client.post('/api/report',files=[('files',('a.png',raw)),('files',('b.dcm',dicom()))])
    assert report.status_code==200,report.text
    assert len(PdfReader(io.BytesIO(report.content)).pages)==2
    # Temperature changes the displayed score and class, but not the raw Grad-CAM reconstruction.
    old=engine.config.copy()
    try:
        engine.config={**old,'calibration':{'method':'temperature','T':2},'threshold':.5}
        calibrated=client.post('/api/predict',files={'file':('sample.png',raw)}).json()
        assert calibrated['raw_score']==r['raw_score']
        assert calibrated['pneumonia_score']==calibrate_score(calibrated['raw_score'],{'method':'temperature','T':2})
        if calibrated['explanation']['available']:
            assert abs(calibrated['explanation']['reconstructed_score']-r['raw_score'])<1e-5
    finally:
        engine.config=old


def test_memory_limit(client):
    from upload_memory import MAX_BODY
    # Reject by declared size before consuming the upload.
    response=client.post('/api/predict',headers={'content-length':str(MAX_BODY+1)},content=b'')
    assert response.status_code==413
