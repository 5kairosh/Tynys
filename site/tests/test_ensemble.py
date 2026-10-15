import copy
import math
from pathlib import Path
import pytest
from ensemble import ensemble_score, validate_ensemble


def config():
    return {'weights':{'CNN':.25,'MobileNetV2':.75},'threshold':.6}


def test_convex_score_and_threshold():
    value=config()
    assert ensemble_score(.2,.8,value)==pytest.approx(.65)
    assert ensemble_score(.2,.8,value)>=value['threshold']
    assert ensemble_score(0.,0.,value)==0
    assert ensemble_score(1.,1.,value)==1


@pytest.mark.parametrize('bad',[None,{}, {'weights':{'CNN':0,'MobileNetV2':1},'threshold':.5},
    {'weights':{'CNN':.5,'MobileNetV2':.6},'threshold':.5},
    {'weights':{'CNN':.5,'MobileNetV2':.5},'threshold':float('nan')},
    {'weights':{'CNN':True,'MobileNetV2':.5},'threshold':.5}])
def test_bad_config(bad):
    with pytest.raises(ValueError):validate_ensemble(bad)


def test_live_ensemble_and_optional_fallback(client):
    from app import engine
    import io
    from PIL import Image
    stream=io.BytesIO();Image.new('RGB',(100,120),(60,100,140)).save(stream,format='PNG')
    def predict():return client.post('/api/predict',files={'file':('x.png',stream.getvalue())})
    r=predict().json();models={m['name']:m for m in r['comparison']['models']}
    value=engine.config['ensemble']
    assert set(models)=={'CNN','MobileNetV2','Ensemble'}
    expected=ensemble_score(models['CNN']['pneumonia_score'],models['MobileNetV2']['pneumonia_score'],value)
    assert models['Ensemble']['pneumonia_score']==pytest.approx(expected,abs=1e-12)
    assert models['Ensemble']['model_class']==engine.config['classes'][int(expected>=value['threshold'])]
    old=copy.deepcopy(engine.config)
    try:
        engine.config.pop('ensemble')
        response=predict();assert response.status_code==200
        assert not response.json()['comparison']['ensemble']['available']
        engine.config['ensemble']={'weights':{},'threshold':.5}
        assert predict().status_code==200
        engine.config=copy.deepcopy(old)
        engine.config['ensemble']['model_ids']['CNN']='wrong'
        assert not engine.ensemble_status()['available']
    finally:engine.config=old


def test_ui_four_columns():
    from html.parser import HTMLParser
    class Headers(HTMLParser):
        def __init__(self):super().__init__();self.in_head=False;self.count=0;self.counts=[]
        def handle_starttag(self,tag,attrs):
            if tag=='thead':self.in_head=True;self.count=0
            if tag=='th' and self.in_head:self.count+=1
        def handle_endtag(self,tag):
            if tag=='thead':self.in_head=False;self.counts.append(self.count)
    parser=Headers();parser.feed((Path(__file__).parents[1]/'static/index.html').read_text(encoding='utf-8'))
    assert parser.counts==[4,4]
