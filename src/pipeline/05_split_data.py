import pandas as pd
import numpy as np

from functools import reduce
from ruamel.yaml import YAML

import click

from sklearn.preprocessing import OneHotEncoder
import joblib


import sys
sys.path.append('.')
from src.constants import ActivityLogger

@click.command()
def main():

    try:

        logger = ActivityLogger().get_logger(__file__)
        conf = YAML().load(open('params.yaml'))
  

        feat_df = pd.read_csv(conf['fights_feat_gen']['feat_fn'])
        feat_df = feat_df.assign(event_day=pd.to_datetime(feat_df['event_day'], format='%Y-%m-%d'))
        feat_df = feat_df.sort_values(by='event_day', ascending=True)
        
        feat_df = feat_df.reset_index(drop=True)
        
        train_prop = conf['split_data']['train_prop']
        N = feat_df.shape[0]
        tr_idx, val_idx, ts_idx = np.split(range(N), [int((train_prop)*N), int((train_prop)*N) + int((1-train_prop)*N/2)])
        
        feat_df['split'] = 'tr'
        feat_df.loc[feat_df.index[val_idx], 'split'] = 'val'
        feat_df.loc[feat_df.index[ts_idx], 'split'] = 'ts'
        
        # для train выборки
        
        nfeat_cols = ['event', 'fighter', 'opponent', 'event_day', 'target', 'split']


        ohe = OneHotEncoder(handle_unknown='error', sparse_output=False, drop='first')
        ohe.fit(feat_df.loc[feat_df.index[tr_idx],['fighter_stance_custom_feat', 'opponent_stance_custom_feat']])
        
        joblib.dump(ohe, conf['split_data']['ohe_fn'])
        
        feat_df = feat_df.drop(columns=['fighter_stance_custom_feat', 'opponent_stance_custom_feat'])\
            .merge(pd.DataFrame(ohe.transform(feat_df[['fighter_stance_custom_feat', 'opponent_stance_custom_feat']]), 
                         index=feat_df.index, columns=ohe.get_feature_names_out()),
            left_index=True, right_index=True)
        
        feat_cols = [it for it in feat_df.columns if not it in nfeat_cols]

        feat_df[nfeat_cols+feat_cols].reset_index(drop=True).to_csv(conf['split_data']['feat_full_fn'], index=False)


        
        ActivityLogger().close_logger(logger)

    except Exception:
        logger.exception('ВОЗНИКЛО ИСКЛЮЧЕНИЕ!!!')
        ActivityLogger().close_logger(logger)

if __name__=='__main__':
        
    main()