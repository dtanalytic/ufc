import pandas as pd
from ruamel.yaml import YAML
import numpy as np
import click
import time 

import os
import re

from functools import partial
from fastcore.basics import chunked

import sys
sys.path.append('.')

from src.stat_funcs import get_stat_feat, last_el
from src.constants import ActivityLogger

from pandarallel import pandarallel

pandarallel.initialize(progress_bar=False, nb_workers=os.cpu_count())


@click.command()
def main():

    try:

        logger = ActivityLogger().get_logger(__file__)
        start = time.time()
    
        
        conf = YAML().load(open('params.yaml'))
        
        fights_df = pd.read_csv(conf['filter']['filt_fights_fn'])
        
        fights_df.sort_values(by=['Fighter', 'event_date'], inplace=True)
        # stat_cols = [it for it in fights_df.columns if re.search('_stat', it)]
        stat_cols = [it for it in fights_df.columns if '_stat' in it]
        
        # stat_cols = [it for it in fights_df.columns if re.search('_stat', it) and all([not col in it for col in ['dob_stat', 'height_stat', 'reach_stat']])]

        # aggs = ['sum', 'mean', 'max', 'min', 'std']
        # cust_aggs = [('max_min_d', lambda x: x.max()-x.min())] + [(f'q_{l:.1f}', lambda x, l=l: x.quantile(l)) for l in np.arange(0.1, 1,0.5)]

        aggs = eval(conf['stat_feat_gen']['aggs'])
        cust_aggs = eval(conf['stat_feat_gen']['cust_aggs'])




        fighters_chunks = list(chunked(fights_df['Fighter'].unique(), n_chunks=os.cpu_count()))
        for i in range(len(fighters_chunks)):
            fights_df.loc[fights_df.Fighter.isin(fighters_chunks[i]), 'chunk'] = i
        
        
        def get_stat_feat_chunk(df, fn, aggs, cust_aggs, stat_cols, n_shift, conf):
            
            return df.groupby('Fighter').apply(lambda x: fn(x, aggs = aggs, cust_aggs=cust_aggs, cols = stat_cols,
                                                        min_per_num=conf['stat_feat_gen']['min_fights_num'],
                                                        rol_window_size=conf['stat_feat_gen']['rol_window_size'], n_shift=n_shift))
        
        visible_get_stat_feat_chunk = partial(get_stat_feat_chunk, fn=get_stat_feat, aggs=aggs, cust_aggs=cust_aggs, stat_cols=stat_cols, n_shift=1, conf=conf)

        if os.cpu_count()==1:
            t_dfs = []
            for ch, gr in fights_df.groupby('chunk'):
                t_dfs.append(visible_get_stat_feat_chunk(gr).assign(chunk=ch).reset_index().set_index(['chunk', 'Fighter', 'level_1']))
        
            fights_stat_df = pd.concat(t_dfs, ignore_index=False)
        
        else:
            fights_stat_df = fights_df.groupby('chunk').parallel_apply(visible_get_stat_feat_chunk)
            

        fights0_df = fights_df[['Fighter', 'event_date', 'chunk']].copy()

        fights0_df = fights0_df.assign(event_date = pd.to_datetime(fights0_df['event_date'], format='%Y-%m-%d', errors='coerce').dt.to_period(freq='D'))
        
        fights0_df['days_nofight_stat'] = fights0_df.groupby('Fighter', as_index=False)['event_date'].diff().map(lambda x: x.n, na_action='ignore')
        fights0_df['days_nofight_stat'] = fights0_df['days_nofight_stat'].replace({pd.NaT: np.nan}).astype('float32')
        
        # visible_get_stat_feat_chunk = partial(get_stat_feat_chunk, fn=get_stat_feat, aggs=aggs, cust_aggs=cust_aggs+[('last', last_el)], stat_cols=['days_nofight_stat'], n_shift=0, conf=conf)
        visible_get_stat_feat_chunk = partial(get_stat_feat_chunk, fn=get_stat_feat, aggs=aggs, cust_aggs=cust_aggs, stat_cols=['days_nofight_stat'], n_shift=0, conf=conf)

        if os.cpu_count()==1:
            t_dfs = []
            for ch, gr in fights0_df.groupby('chunk'):
                t_dfs.append(visible_get_stat_feat_chunk(gr).assign(chunk=ch).reset_index().set_index(['chunk', 'Fighter', 'level_1']))
        
            fights_stat0_df = pd.concat(t_dfs, ignore_index=False)
        
        else:
            fights_stat0_df = fights0_df.groupby('chunk').parallel_apply(visible_get_stat_feat_chunk)
        
        
        # fights_stat0_df = fights_stat0_df.assign(days_nofight_stat_custom=lambda x: x['days_nofight_stat_rol_last'])\
        #                                     .drop(columns=['days_nofight_stat_rol_last', 'days_nofight_stat_exp_last'])
        fights_stat0_df = fights_stat0_df.assign(index_col = fights_stat0_df.index.map(lambda x: x[2])).merge(fights0_df[['days_nofight_stat']], left_on='index_col', right_index=True, how='left')\
                .drop(columns='index_col')

        
        
        fights_stat = fights_stat0_df.merge(fights_stat_df, left_index=True, right_index=True)
        
        fights_stat = fights_stat.reset_index(level=0, drop=True)
        
        fighters_df = pd.read_csv(conf['preprocess']['prep_fighters_fn'])
        fighters_df = fighters_df.assign(dob = lambda x: pd.to_datetime(x['dob'], format='%Y-%m-%d').dt.to_period(freq='D'))
        
        fighters_df['dob_diff_stat_custom'] = fighters_df['dob'].map(lambda x: x.start_time.timestamp()/(30*24*3600) if not pd.isnull(x) else np.nan)
        
        fighters_cols = [it for it in fighters_df.columns]
        
        stat_df = fights_stat.reset_index().rename(columns={'Fighter':'full_name', 'level_1':'index'}).merge(fighters_df, on='full_name', how='left').set_index('index')[fighters_cols+fights_stat.columns.tolist()]
        
        # переименовать колонки
        stat_df = stat_df.rename(columns={'height_stat':'height_stat_custom', 'reach_stat':'reach_stat_custom'})
        
        full_df = fights_df[['Event', 'event_date', 'Fighter', 'Opponent', 'Result', 'left_corner_stat']]\
                .merge(stat_df, left_index=True, right_index=True, suffixes=['','_feat'], how='outer')\
                .drop(columns='full_name')
        
        full_df.to_csv(conf['stat_feat_gen']['feat_fn'], index=False)

        end = time.time()
        
        logger.info(f'Закончили подсчет признаков, который занял {(end-start)/60:.1f} минут')
        # так намного дольше
        # df[df.Fighter.isin(fighters_chunks[0][:10]+fighters_chunks[1][:10]+fighters_chunks[2][:10]+fighters_chunks[4][:10])]\
        # .groupby('Fighter').apply(lambda x: get_stat_feat(x, aggs = aggs, cust_aggs=cust_aggs, cols = stat_cols,
        #                                             min_per_num=conf['stat_feat_gen']['min_fights_num'],
        #                                             rol_window_size=conf['stat_feat_gen']['rol_window_size']))

        ActivityLogger().close_logger(logger)

    except Exception:
        logger.exception('ВОЗНИКЛО ИСКЛЮЧЕНИЕ!!!')
        ActivityLogger().close_logger(logger)

if __name__=='__main__':
        
    main()