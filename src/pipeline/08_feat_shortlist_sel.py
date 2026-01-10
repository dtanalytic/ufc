import pandas as pd
import numpy as np

from functools import reduce

from ruamel.yaml import YAML
import os
import time

import click
import joblib

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


import sys
sys.path.append('.')
from src.constants import ActivityLogger
from src.feature_selection_funcs import BFS2Thres


@click.command()
def main():

    try:

        logger = ActivityLogger().get_logger(__file__)
        conf = YAML().load(open('params.yaml'))

        feat_df = pd.read_csv(conf['feat_longlist_sel']['feat_long_fn'])
        feat_cols = joblib.load(conf['feat_longlist_sel']['sign_feat_long_fn'])
        
        nfeat_cols = ['event', 'fighter', 'opponent', 'event_day', 'target', 'split']
        
        
        feat_df = feat_df.dropna(subset=feat_cols).reset_index(drop=True)
        
        
        logger.info(f"Для отбора из короткого списка используем {conf['feat_shortlist_sel']['method']}")
        start = time.time()
        
        fake_split = lambda feat_df: [(feat_df[feat_df.split=='tr'].index.tolist(), feat_df[feat_df.split=='val'].index.tolist())]
                
        if conf['feat_shortlist_sel']['method'] in ['b', 'f', 's']:
            
            if conf['feat_shortlist_sel']['method']!='b':        
                feat_cols = feat_cols[::-1]
                feat_df = feat_df[nfeat_cols+feat_cols]
            
            maxiter = 1000
            selector = BFS2Thres(Pipeline(steps=[('sc', RobustScaler()), ('clf', GaussianNB())]), scoring='roc_auc', 
                                 cv=fake_split(feat_df), thresh=conf['feat_shortlist_sel']['thresh'], direction=conf['feat_shortlist_sel']['method'])

            selector.fit(feat_df.drop(columns=nfeat_cols), feat_df['target'], verbose=False)
            
            feat_bayes = selector.feat_ar[selector.mask]
            bayes_qual = selector.sc_pr
            
            selector = BFS2Thres(Pipeline(steps=[('sc', RobustScaler()), ('clf', LogisticRegression(max_iter=maxiter))]), scoring='roc_auc', cv = fake_split(feat_df), thresh = conf['feat_shortlist_sel']['thresh'], direction=conf['feat_shortlist_sel']['method'])
            selector.fit(feat_df.drop(columns=nfeat_cols), feat_df['target'], verbose=False)
             
            feat_logreg = selector.feat_ar[selector.mask]
            
            logreg_qual = selector.sc_pr
            
            selector = BFS2Thres(HistGradientBoostingClassifier(), scoring='roc_auc', cv = fake_split(feat_df), thresh = conf['feat_shortlist_sel']['thresh'], direction=conf['feat_shortlist_sel']['method'])
            selector.fit(feat_df.drop(columns=nfeat_cols), feat_df['target'], verbose=False)
             
            feat_boost = selector.feat_ar[selector.mask]
            boost_qual = selector.sc_pr
            
            selector = BFS2Thres(Pipeline(steps=[('sc', RobustScaler()), ('clf', LinearSVC(max_iter=maxiter*10))]), scoring='roc_auc', cv = fake_split(feat_df), thresh = conf['feat_shortlist_sel']['thresh'], direction=conf['feat_shortlist_sel']['method'])

            selector.fit(feat_df.drop(columns=nfeat_cols), feat_df['target'], verbose=False)
             
            feat_svc = selector.feat_ar[selector.mask]
            svc_qual = selector.sc_pr


            logger.info(f"Качество байеса - {bayes_qual}, качество логрега - {logreg_qual}, качество svc - {svc_qual}, качество бустинга - {boost_qual}")


            res_df = pd.DataFrame([[it in l for it in feat_cols] for l in [feat_svc, feat_boost, feat_logreg, feat_bayes]], 
             columns=feat_cols, index=['svc', 'boost', 'logreg', 'bayes'])

            
            res_df = res_df.sum(axis=0).sort_values(ascending=False)
            res_df.to_frame('feat').to_csv(conf['feat_shortlist_sel']['sign_feat_short_all_fn'])

            feats = res_df.loc[lambda x: x>=1].index.tolist()
            
            logger.info(f"Итоговые признаки - {feats}")

            joblib.dump(feats, conf['feat_shortlist_sel']['sign_feat_short_fn'])

        
        feat_df = feat_df[nfeat_cols+feats]

        clip_val = conf['feat_shortlist_sel']['quantile_clip']

        if clip_val:

            train_df = feat_df.query('split=="tr"')
            upper_d = train_df[feats].quantile(1-clip_val/2).to_dict()
            lower_d = train_df[feats].quantile(clip_val/2).to_dict()
        
            cond = reduce(lambda x,y: x & y, [(train_df[colnm]<=upper_d[colnm]) & (train_df[colnm]>=lower_d[colnm]) for colnm in feats])
        
            outlier_idx = train_df.loc[~cond].index.tolist()
        
            logger.info(f'До фильтрации выбросов размер выборки train - {len(train_df)}')
            logger.info(f'После фильтрации выбросов размер выборки train - {len(train_df) - len(outlier_idx)}, уменьшение на {len(outlier_idx)} ({len(outlier_idx)/len(train_df):.2%})')
        
            feat_df.loc[feat_df.index[outlier_idx], 'split'] = None
        
            outlier_df = feat_df[feat_df.split.isna()]
            # outlier_df = feat_df.query('split=="tr"')
            
            for i, colnm in enumerate(feats):
                ser_colnm = ((outlier_df[colnm]>upper_d[colnm]) | (outlier_df[colnm]<lower_d[colnm])).replace({False:'', True:colnm}).map(lambda x: [x] if len(x)>0 else [])
                if i==0:
                    ser = ser_colnm
                else:
                    ser = ser + ser_colnm
        
            outlier_df.assign(outlier_cols = ser).to_csv(conf['feat_shortlist_sel']['feat_anom_fn'], index=False)
        
        feat_df.loc[feat_df.split.notna(), nfeat_cols+feats].reset_index(drop=True).to_csv(conf['feat_shortlist_sel']['feat_short_fn'], index=False)
        
        end = time.time()
        
        logger.info(f'Отбор по короткому списку занял {(end-start)/60:.1f} минут')

        ActivityLogger().close_logger(logger)

    except Exception:
        logger.exception('ВОЗНИКЛО ИСКЛЮЧЕНИЕ!!!')
        ActivityLogger().close_logger(logger)

if __name__=='__main__':
        
    main()