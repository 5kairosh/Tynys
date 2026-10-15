"""Exact paired McNemar test, requiring row-aligned predictions."""
import csv
import hashlib
import json
from pathlib import Path
from scipy.stats import binomtest


def calculate_mcnemar(predictions, model_dir):
    with Path(predictions).open(encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    if not rows or len({r['sha256'] for r in rows}) != len(rows):
        raise ValueError('Болжамдар бос немесе суреттер қайталанған.')
    b = c = 0
    for row in rows:
        truth = int(row['label'])
        cnn, mobile = int(row['CNN_prediction']), int(row['MobileNetV2_prediction'])
        if any(x not in (0,1) for x in (truth,cnn,mobile)):
            raise ValueError('Белгі 0/1 болуы керек.')
        b += int(cnn == truth and mobile != truth)
        c += int(cnn != truth and mobile == truth)
    p = float(binomtest(b,b+c,.5,alternative='two-sided').pvalue) if b+c else 1.
    result = {'available':True,'method':'exact McNemar (two-sided binomial)',
              'n':len(rows),'cnn_only_correct':b,'mobilenet_only_correct':c,'p_value':p,
              'alpha':.05,'significant':p<.05,
              'notice':'Жұптық тест, осы ішкі test жиынына қатысты. Пациент суреттері тәуелсіз болмауы мүмкін.'}
    result['model_ids'] = {name:hashlib.sha256((Path(model_dir)/file).read_bytes()).hexdigest()[:12]
                           for name,file in [('CNN','cnn_best.keras'),('MobileNetV2','pneumonia_mobilenetv2.keras')]}
    result['config_sha256'] = hashlib.sha256((Path(model_dir)/'model_config.json').read_bytes()).hexdigest()
    return result


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    root=Path(__file__).resolve().parents[1]
    parser.add_argument('--predictions',type=Path,default=root/'training/manifests/test_predictions.csv')
    parser.add_argument('--model-dir',type=Path,default=root/'site/model')
    args=parser.parse_args()
    result=calculate_mcnemar(args.predictions,args.model_dir)
    (args.model_dir/'mcnemar.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))
