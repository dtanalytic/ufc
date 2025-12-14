import pandas as pd
import numpy as np

import os
import time

import click
import joblib
import json

from ruamel.yaml import YAML

import seaborn as sns
import matplotlib.pyplot as plt
sns.set()

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay

import sys
sys.path.append('.')
from src.constants import ActivityLogger


@click.command()
def main():

    try:

        logger = ActivityLogger().get_logger(__file__)
        conf = YAML().load(open('params.yaml'))
  
        start = time.time()

        DN = conf['train_eval']['report_dn']

        df = pd.read_csv(conf['feat_shortlist_sel']['feat_short_fn'])
        nfeats = ['event', 'fighter', 'opponent', 'event_day', 'target', 'split']
        feats = [it for it in df.columns if not it in nfeats]

        # model = GaussianNB()
        if conf['train_eval']['model']=='logreg':
            # model = LogisticRegression(random_state=conf['seed'])
            model = Pipeline(steps=[('sc', RobustScaler()), ('clf', LogisticRegression(solver='liblinear', random_state=conf['seed']))])

        elif conf['train_eval']['model']=='dummy':
            model = DummyClassifier(strategy='prior', random_state=conf['seed'])

        
        # калибруем
        model = CalibratedClassifierCV(estimator=model, method='sigmoid', cv=5, n_jobs=-1, ensemble=False)
        model.fit(df.loc[df.split=='tr', feats], df.loc[df.split=='tr', 'target'])
        
        joblib.dump(model, conf['train_eval']['model_fn'])

        # метрики сохраняем
        df['score1'] = model.predict_proba(df[feats])[:, 1]
        roc_d = {f'roc_auc_{sample}':roc_auc_score(df.loc[df.split==sample, 'target'], df.loc[df.split==sample, 'score1']) for sample in ['tr', 'val', 'ts']}
        pr_d = {f'avg_pr_{sample}':average_precision_score(df.loc[df.split==sample, 'target'], df.loc[df.split==sample, 'score1']) for sample in ['tr', 'val', 'ts']}
        with open(conf['train_eval']['eval_fn'], 'wt') as f:
            json.dump({**roc_d, **pr_d}, f)

        # сохраняем гистограмму
        df['score1'].plot.hist()
        plt.savefig(f'{DN}/hist.png')

        # сохраняем калибровочный рисунок
        CalibrationDisplay.from_predictions(df.loc[df.split=='val', 'target'], model.predict_proba(df.loc[df.split=='val', feats])[:,1])
        plt.savefig(f'{DN}/calibr.png')
        
        df.to_csv(conf['train_eval']['score_df_fn'], index=False)
        end = time.time()
        
        logger.info(f'Обучение и оценка модели заняли {(end-start)/60:.1f} минут')

        
        ActivityLogger().close_logger(logger)

    except Exception:
        logger.exception('ВОЗНИКЛО ИСКЛЮЧЕНИЕ!!!')
        ActivityLogger().close_logger(logger)

if __name__=='__main__':
        
    main()