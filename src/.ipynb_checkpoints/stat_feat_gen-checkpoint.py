import pandas as pd
from ruamel.yaml import YAML
import numpy as np
import click

import re
import sys
sys.path.append('.')

from src.funcs import get_rolling_stats


@click.command()
def main():
    
    conf = YAML().load(open('params.yaml'))
    
    df = pd.read_csv(conf['filter']['filt_fn'])
    
    df.sort_values(by=['Fighter', 'Date'], inplace=True)
    stat_cols = [it for it in df.columns if re.search('_stat', it)]
        
    op_d = {col:'median' for col in stat_cols if re.search('str',col)}
    op_d.update({col:'mean' for col in stat_cols if not re.search('str',col)})
    
    df_stat = df.groupby('Fighter').apply(lambda df: get_rolling_stats(df,op_d, conf['stat_feat_gen']['min_fights_num']))

    df_full = df.merge(df_stat.reset_index(level=0), left_index=True, right_index=True, suffixes=['','_rol'], how='outer')

    df_full = df_full[df_full['kd_stat_rol'].notnull()]
    # draw maybe divide on four win/lose of one, win/lose of other
    df_full = df_full[df_full.Result.isin(['W', 'L'])]
    feat_cols = [it for it in df_full.columns if re.search('_stat_rol', it)]

    df1 = df_full.set_index(['Fighter', 'Opponent', 'Date'])[feat_cols]
    df2 = df_full.set_index([ 'Opponent','Fighter', 'Date'])[feat_cols]
    df2.index.set_names(df1.index.names, inplace=True)
    
    res = df1-df2
    res = res[res.kd_stat_rol.notnull()]

    df_feat = df_full.merge(res.reset_index(), on = ['Fighter','Opponent','Date'], suffixes=['','_sub'])
    df_feat['Result'] = np.where(df_feat['Result']=='W',1,0)

    
    df_feat.to_csv(conf['stat_feat_gen']['feat_fn'], index=False)
    
if __name__=='__main__':
        
    main()