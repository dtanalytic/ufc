import pandas as pd
import numpy as np

from functools import reduce
from ruamel.yaml import YAML

import click

import sys
sys.path.append('.')
from src.constants import ActivityLogger

@click.command()
def main():

    try:

        logger = ActivityLogger().get_logger(__file__)
        conf = YAML().load(open('params.yaml'))
  
        feat_df = pd.read_csv(conf['fights_feat_gen']['feat_fn'])
        feat_df = feat_df.assign(event_date=pd.to_datetime(feat_df['event_date'], format='%Y-%m-%d'))
        feat_df = feat_df.sort_values(by='event_date', ascending=True)
        
        feat_df = feat_df.reset_index(drop=True)

        train_prop = conf['split_data']['train_prop']
        N = feat_df.shape[0]
        tr_idx, val_idx, ts_idx = np.split(range(N), [int((train_prop)*N), int((train_prop)*N) + int((1-train_prop)*N/2)])

        feat_df['split'] = 'tr'
        feat_df.loc[feat_df.index[val_idx], 'split'] = 'val'
        feat_df.loc[feat_df.index[ts_idx], 'split'] = 'ts'
        
        # для train выборки
        clip_val = conf['split_data']['quantile_clip']
        nfeat_cols = ['fighter', 'opponent', 'event_date', 'target', 'split']
        feat_cols = [it for it in feat_df.columns if not it in nfeat_cols]
        
        train_df = feat_df.loc[feat_df.index[tr_idx], feat_cols]
        
        if clip_val:
            upper_d = train_df.quantile(1-clip_val/2).to_dict()
            lower_d = train_df.quantile(clip_val/2).to_dict()
        
            cond = reduce(lambda x,y: x & y, [(train_df[colnm]<=upper_d[colnm]) & (train_df[colnm]>=lower_d[colnm]) for colnm in train_df.columns])
        
            outlier_idx = train_df.loc[~cond].index.tolist()
            
            logger.info(f'До фильтрации выбросов размер выборки train - {len(tr_idx)}')
            logger.info(f'После фильтрации выбросов размер выборки train - {len(tr_idx) - len(outlier_idx)}, уменьшение на {len(outlier_idx)} ({len(outlier_idx)/len(tr_idx):.2%})')

            feat_df.loc[feat_df.index[outlier_idx], 'split'] = None

            outlier_df = feat_df[feat_df.split.isna()]
            # outlier_df = feat_df.query('split=="tr"')
            
            for i, colnm in enumerate(feat_cols):
                ser_colnm = ((outlier_df[colnm]>upper_d[colnm]) | (outlier_df[colnm]<lower_d[colnm])).replace({False:'', True:colnm}).map(lambda x: [x] if len(x)>0 else [])
                if i==0:
                    ser = ser_colnm
                else:
                    ser = ser + ser_colnm

            outlier_df.assign(outlier_cols = ser).to_csv(conf['split_data']['feat_anom_fn'], index=False)

        feat_df.loc[feat_df.split.notna(), nfeat_cols+feat_cols].reset_index(drop=True).to_csv(conf['split_data']['feat_full_fn'], index=False)
        ActivityLogger().close_logger(logger)

    except Exception:
        logger.exception('ВОЗНИКЛО ИСКЛЮЧЕНИЕ!!!')
        ActivityLogger().close_logger(logger)

if __name__=='__main__':
        
    main()