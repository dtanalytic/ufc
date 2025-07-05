import pandas as pd
import numpy as np

from itertools import chain

from ruamel.yaml import YAML
import os
import time

import click

import phik

from functools import partial
from fastcore.basics import chunked
from fastcore.parallel import parallel

import sys
sys.path.append('.')
from src.constants import ActivityLogger

def calc_pair_phik(feat_cols, df, target_nm):

    res_d = {}
    for colnm in feat_cols:
        res_d[colnm] = df[[colnm, target_nm]].phik_matrix().iloc[0,1]
    
    return res_d

@click.command()
def main():

    try:

        logger = ActivityLogger().get_logger(__file__)
        conf = YAML().load(open('params.yaml'))
  
        feat_df = pd.read_csv(conf['split_data']['feat_full_fn'])
        # feat_df = feat_df.assign(event_date=pd.to_datetime(feat_df['event_date'], format='%Y-%m-%d'))

        nfeat_cols = ['fighter', 'opponent', 'event_date', 'target', 'split']
        feat_cols = [it for it in feat_df.columns if not it in nfeat_cols]

        train_df = feat_df.query('split=="tr"')

        logger.info(f"Для отбора из длинного списка используем {conf['feat_longlist_sel']['method']}")
        start = time.time()
        if conf['feat_longlist_sel']['method']=='phik':
            
            feat_cols_chunks = list(chunked(feat_cols, n_chunks=os.cpu_count()))
            res_l = parallel(calc_pair_phik, feat_cols_chunks, df=train_df, target_nm='target', n_workers=os.cpu_count())
            
            ser = pd.Series(dict(chain(*[item.items() for item in res_l]))).sort_values(ascending=False)
            ser.to_frame().to_csv(conf['feat_longlist_sel']['sign_feat_long_fn'], index=True)
            
            assert conf['feat_longlist_sel']['thresh'] or conf['feat_longlist_sel']['topn_feats'], 'use thresh or topn_feats'
            
            if conf['feat_longlist_sel']['topn_feats']:
                fin_feats = ser.head(conf['feat_longlist_sel']['topn_feats']).index.tolist()
            elif conf['feat_longlist_sel']['thresh']:
                fin_feats = ser.loc[lambda x: x>=conf['feat_longlist_sel']['thresh']].index.tolist()
                
        end = time.time()
        
        logger.info(f'Отбор по длинному списку занял {(end-start)/60:.1f} минут')
        
        feat_df[nfeat_cols+fin_feats].to_csv(conf['feat_longlist_sel']['feat_long_fn'], index=False)
        
        ActivityLogger().close_logger(logger)

    except Exception:
        logger.exception('ВОЗНИКЛО ИСКЛЮЧЕНИЕ!!!')
        ActivityLogger().close_logger(logger)

if __name__=='__main__':
        
    main()