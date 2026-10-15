import io
import pytest
from PIL import Image
from pypdf import PdfReader
from pdf_report import make_report


def png():
    stream=io.BytesIO();Image.new('RGB',(60,90),(144,255,23)).save(stream,format='PNG');return stream.getvalue()


@pytest.mark.parametrize('count',[1,2,10])
@pytest.mark.parametrize('ood_warning',[False,True])
def test_pdf(count,ood_warning):
    r={'model_class':'PNEUMONIA','pneumonia_score':.8,'threshold':.74,'width':60,'height':90,'filename':'test.png','model_id':'mock-model-123',
       'explanation':{'available':False,'message':'Карта қолжетімсіз.'},
       'xray_check':{'available':True,'is_xray':not ood_warning},
       'calibration':{'method':'temperature','T':2.1},
       'comparison':{'available':True,'agreement':True,'models':[
           {'name':'CNN','model_class':'PNEUMONIA','pneumonia_score':.7,'threshold':.485},
           {'name':'MobileNetV2','model_class':'PNEUMONIA','pneumonia_score':.8,'threshold':.74}]}}
    data=make_report([(png(),r)]*count)
    assert data.startswith(b'%PDF')
    reader=PdfReader(io.BytesIO(data))
    assert len(reader.pages)==count
    assert 'mock-model-123' in reader.pages[0].extract_text()
    if ood_warning:
        assert 'рентгенге ұқсамайды' in reader.pages[0].extract_text()
