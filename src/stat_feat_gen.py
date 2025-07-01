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

from src.funcs import get_stat_feat

from pandarallel import pandarallel

pandarallel.initialize(progress_bar=False, nb_workers=os.cpu_count())


@click.command()
def main():

    start = time.time()

    
    conf = YAML().load(open('params.yaml'))
    
    df = pd.read_csv(conf['filter']['filt_fn'])
    
    df.sort_values(by=['Fighter', 'Date'], inplace=True)
    stat_cols = [it for it in df.columns if re.search('_stat', it)]

    # aggs = ['sum', 'mean', 'max', 'min', 'std']
    # cust_aggs = [('max_min_d', lambda x: x.max()-x.min())] + [(f'q_{l:.1f}', lambda x, l=l: x.quantile(l)) for l in np.arange(0.1, 1,0.5)]

    aggs = eval(conf['stat_feat_gen']['aggs'])
    cust_aggs = eval(conf['stat_feat_gen']['cust_aggs'])
    fighters_chunks = list(chunked(df['Fighter'].unique(), n_chunks=os.cpu_count()))
    
    for i in range(len(fighters_chunks)):
        df.loc[df.Fighter.isin(fighters_chunks[i]), 'chunk'] = i

    
    def get_stat_feat_chunk(df, fn, aggs, cust_aggs, stat_cols, conf):
        
        return df.groupby('Fighter').apply(lambda x: fn(x, aggs = aggs, cust_aggs=cust_aggs, cols = stat_cols,
                                                    min_per_num=conf['stat_feat_gen']['min_fights_num'],
                                                    rol_window_size=conf['stat_feat_gen']['rol_window_size']))
    
        
        
    visible_get_stat_feat_chunk = partial(get_stat_feat_chunk, fn=get_stat_feat, aggs=aggs, cust_aggs=cust_aggs, stat_cols=stat_cols, conf=conf)
    
    
    df_stat = df.groupby('chunk').parallel_apply(visible_get_stat_feat_chunk)
    
    # chunks убираем, эта инфа нам не нужна больше
    df_stat = df_stat.reset_index(level=0, drop=True)
    
    

    
    df_full = df.merge(df_stat.reset_index(level=0), left_index=True, right_index=True, suffixes=['','_feat'], how='outer')

    
    df_full.to_csv(conf['stat_feat_gen']['feat_fn'], index=False)

    end = time.time()
    print(f'подсчет статистики занял {(end-start)/60} минут')

    # так намного дольше
    # df[df.Fighter.isin(fighters_chunks[0][:10]+fighters_chunks[1][:10]+fighters_chunks[2][:10]+fighters_chunks[4][:10])]\
    # .groupby('Fighter').apply(lambda x: get_stat_feat(x, aggs = aggs, cust_aggs=cust_aggs, cols = stat_cols,
    #                                             min_per_num=conf['stat_feat_gen']['min_fights_num'],
    #                                             rol_window_size=conf['stat_feat_gen']['rol_window_size']))
if __name__=='__main__':
        
    main()