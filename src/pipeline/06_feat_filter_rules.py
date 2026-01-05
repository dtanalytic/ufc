import pandas as pd
import numpy as np

from ruamel.yaml import YAML

import click

import joblib


import sys
sys.path.append('.')
from src.constants import ActivityLogger

@click.command()
def main():

    try:

        logger = ActivityLogger().get_logger(__file__)
        conf = YAML().load(open('params.yaml'))
        
        feat_df = pd.read_csv(conf['split_data']['feat_full_fn'])

        nfeat_cols = ['event', 'fighter', 'opponent', 'event_day', 'target', 'split']
        feat_cols = [it for it in feat_df.columns if not it in nfeat_cols]
        
        psi_thresh = conf['feat_filter_rules']['drop_highpsi_thresh']
        if psi_thresh!=0:
            
            df = pd.concat([feat_df.query('split=="tr"').assign(split_col=0), feat_df.query('split=="val"').assign(split_col=1)])
            
            from feature_engine.selection import DropHighPSIFeatures
            
            psi = DropHighPSIFeatures(split_col='split_col', split_distinct=True, missing_values='ignore',
                                      variables=feat_cols, strategy='equal_frequency', bins=5)
            
            
            psi.fit(df)
        
            highpsi_feat_cols = [(k, v) for (k, v) in psi.psi_values_.items() if v>psi_thresh]
            
            joblib.dump(highpsi_feat_cols, conf['feat_filter_rules']['dropped_feat_fn'])
            
            prev_len = len(feat_cols)
            
            feat_cols = [it for it in feat_cols if not it in [k for (k,v) in highpsi_feat_cols]]
            
            cur_len = len(feat_cols)
            logger.info(f'После удаления признаков с высоким psi, общее количество изменилось с {prev_len} до {cur_len}, всего удалено - {prev_len-cur_len}')
        
        joblib.dump(feat_cols, conf['feat_filter_rules']['feat_fn'])
         

        
        ActivityLogger().close_logger(logger)

    except Exception:
        logger.exception('ВОЗНИКЛО ИСКЛЮЧЕНИЕ!!!')
        ActivityLogger().close_logger(logger)

if __name__=='__main__':
        
    main()