import pandas as pd
from functools import reduce

from ruamel.yaml import YAML

import numpy as np

import click

import sys
sys.path.append('.')
from src.constants import ActivityLogger

@click.command()
def main():

    try:

        logger = ActivityLogger().get_logger(__file__)
    
        conf = YAML().load(open('params.yaml'))
        
        stat_df = pd.read_csv(conf['stat_feat_gen']['feat_fn'])

        # Вылетают из обзора бои, где у одного из бойцов было недостаточно данных для подсета статистик
        # stat_df = stat_df[stat_df['kd_stat_rol_sum'].notnull()]
        stat_df = stat_df[stat_df.Result.isin(['W', 'L'])]
        
        feat_cols = [it for it in stat_df.columns if '_stat_rol' in it or '_stat_exp' in it]
    
        # у нас сатистика относительно fighter, мы разбиваем на 2 таблицы, во второй - замененные местами бойцы
        # к боям с оригинальным порядком из первой таблицы подтянуться из второй таблицы фичи с оппонентом, который фигурирует как fighter из таблицы 2
        # к дублям боев с помененным порядком бойцов из первой таблицы подтянуться бои с помененными местами бойцами, то есть статистика первого бойца из оригинала 
        df1 = stat_df.set_index(['Fighter', 'Opponent', 'Date'])[feat_cols]
        df2 = stat_df.set_index([ 'Opponent','Fighter', 'Date'])[feat_cols]
        df2.index.set_names(df1.index.names, inplace=True)
        
        res = df1-df2

        feat_df = stat_df.merge(res.reset_index(), on = ['Fighter','Opponent','Date'], suffixes=['','_sub'])

        feat_df = feat_df[['event_date', 'Fighter', 'Opponent', 'Result', 'left_corner_stat']+[it for it in feat_df.columns if '_sub' in it]]
        
        feat_df.columns = [it.lower() for it in feat_df.columns]
        
        feat_df = feat_df[feat_df.kd_stat_rol_sum_sub.notnull()]
        
        feat_df['target'] = np.where(feat_df['result']=='W',1,0)

        feat_df = feat_df.assign(fighters = feat_df.apply(lambda x: ' '.join(sorted([x['fighter'], x['opponent']])) , axis=1))
        

        feat_df = feat_df.drop_duplicates(subset=['fighters', 'event_date'])
        feat_df = feat_df[['fighter', 'opponent', 'event_date', 'target', 'left_corner_stat']+[it for it in feat_df.columns if '_sub' in it]]

        feat_df.reset_index(drop=True).to_csv(conf['fights_feat_gen']['feat_fn'], index=False)
        logger.info(f'Закончили создание таблицы боев')


        ActivityLogger().close_logger(logger)

    except Exception:
        logger.exception('ВОЗНИКЛО ИСКЛЮЧЕНИЕ!!!')
        ActivityLogger().close_logger(logger)

if __name__=='__main__':
        
    main()