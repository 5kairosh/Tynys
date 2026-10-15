"""Scientific plots from current scored predictions, with no fabricated values."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay,RocCurveDisplay,PrecisionRecallDisplay

ROOT=Path(__file__).resolve().parents[1]


def main():
    frame=pd.read_csv(ROOT/'training/manifests/test_predictions.csv')
    for name in ('CNN','MobileNetV2','Ensemble'):
        if name+'_score' not in frame:
            continue
        fig,axes=plt.subplots(1,3,figsize=(15,4))
        ConfusionMatrixDisplay.from_predictions(frame.label,frame[name+'_prediction'],labels=[0,1],display_labels=['NORMAL','PNEUMONIA'],cmap='GnBu',colorbar=False,ax=axes[0])
        RocCurveDisplay.from_predictions(frame.label,frame[name+'_score'],ax=axes[1])
        PrecisionRecallDisplay.from_predictions(frame.label,frame[name+'_score'],ax=axes[2])
        axes[0].set(xlabel='Болжам',ylabel='Нақты белгі')
        axes[1].set(xlabel='Жалған оң үлесі',ylabel='Сезімталдық')
        axes[2].set(xlabel='Recall',ylabel='Precision')
        fig.suptitle(name+' — қазіргі test нәтижесі')
        fig.tight_layout()
        fig.savefig(ROOT/f'site/static/graphs/{name}_test_metrics.png',dpi=140)
        plt.close(fig)


if __name__=='__main__':
    main()
