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

        stat_df = stat_df.assign(event_day=pd.to_datetime(stat_df['event_date'], format='%Y-%m-%d', errors='coerce').dt.to_period(freq='D'))\
                        .assign(dob=pd.to_datetime(stat_df['dob'], format='%Y-%m-%d', errors='coerce').dt.to_period(freq='D'))\
                        .rename(columns={'Fighter':'fighter', 'Opponent':'opponent'})
  
        feat_cols = [it for it in stat_df.columns if '_stat_rol' in it or '_stat_exp' in it or '_stat_custom' in it]
    
        # у нас сатистика относительно fighter, мы разбиваем на 2 таблицы, во второй - замененные местами бойцы
        # к боям с оригинальным порядком из первой таблицы подтянуться из второй таблицы фичи с оппонентом, который фигурирует как fighter из таблицы 2
        # к дублям боев с помененным порядком бойцов из первой таблицы подтянуться бои с помененными местами бойцами, то есть статистика первого бойца из оригинала 
        
        df1 = stat_df.set_index(['fighter', 'opponent', 'event_day'])[feat_cols]
        df2 = stat_df.set_index([ 'opponent','fighter', 'event_day'])[feat_cols]
        df2.index.set_names(df1.index.names, inplace=True)
        
        res_df = df1-df2
        res_df.columns=[f"{it.replace('_stat', '')}_feat" for it in res_df.columns]
        
        feat_df = stat_df[['Event', 'event_day', 'fighter', 'opponent', 'Result', 'left_corner_stat', 'stance',	'dob', 'days_nofight_stat']]\
                            .merge(res_df.reset_index(), on=['fighter', 'opponent', 'event_day'], how='right')

        feat_df = feat_df.assign(fighter_dob_custom_feat = lambda x: (x['event_day'] - x['dob']).map(lambda y: y.n/30 if not pd.isnull(y) else np.nan))\
                .rename(columns={'stance':'fighter_stance_custom_feat', 'days_nofight_stat':'fighter_days_nofight_custom_feat'})\
                .drop(columns=['dob'])

        
        
        # feat_df = feat_df.merge(stat_df[['fighter', 'opponent', 'event_day', 'stance', 'days_nofight_stat']]\
        #                         .rename(columns={'fighter':'f', 'opponent':'op'})\
        #                         .rename(columns={'f':'opponent', 'op':'fighter'}),
        #                         on=['fighter', 'opponent', 'event_day'], how='left')\
        #                 .rename(columns={'stance':'opponent_stance_custom_feat', 'days_nofight_stat':'opponent_days_nofight_custom_feat'})


        feat_df = feat_df.merge(stat_df[['fighter', 'opponent', 'event_day', 'stance', 'days_nofight_stat', 'height_stat_custom', 'reach_stat_custom']]\
    .rename(columns={'fighter':'f', 'opponent':'op'})\
                        .rename(columns={'f':'opponent', 'op':'fighter', 'height_stat_custom':'opponent_height_custom_feat', 
                                         'reach_stat_custom':'opponent_reach_custom_feat'}),
                        on=['fighter', 'opponent', 'event_day'], how='left')\
                .rename(columns={'stance':'opponent_stance_custom_feat', 'days_nofight_stat':'opponent_days_nofight_custom_feat',
                                'height_custom_feat':'height_diff_custom_feat', 'reach_custom_feat':'reach_diff_custom_feat'})
        
        feat_df = feat_df.rename(columns={'left_corner_stat': 'left_corner_custom_feat'})
        
        feat_df.columns = [it.lower() for it in feat_df.columns]

        fighter_feats = ['left_corner_custom_feat', 'height_diff_custom_feat', 'reach_diff_custom_feat', 'dob_diff_custom_feat', 
         'fighter_stance_custom_feat', 'opponent_stance_custom_feat', 'fighter_dob_custom_feat', 'fighter_days_nofight_custom_feat', 'opponent_days_nofight_custom_feat', 'opponent_height_custom_feat', 'opponent_reach_custom_feat']


        # feat_df = feat_df[(feat_df.kd_rol_sum_feat.notnull())]

        
        feat_df['target'] = np.where(feat_df['result']=='W',1,0)

        # удаление дублей комментим, так как нам важно, что каждый бой с несимметричными признаками оценивался правильно
        # прогноз будем строить, как среднее арифметиеское 2 прогнозов с признаками относительно одного бойца выигрыша, и относительно проигрыша другого
        # feat_df = feat_df.assign(fighters = feat_df.apply(lambda x: ' '.join(sorted([x['fighter'], x['opponent']])) , axis=1))
        # feat_df = feat_df.drop_duplicates(subset=['fighters', 'event_day'])
        
        feat_df = feat_df[['event', 'fighter', 'opponent', 'event_day', 'target']+[it for it in feat_df.columns if '_feat' in it]]

        # единый порядок с inference
        
        other_feats = sorted([it for it in feat_df.columns if not it in fighter_feats and not it in ['event', 'fighter', 'opponent', 'event_day', 'target']])
        
        feat_df = feat_df[['event', 'fighter', 'opponent', 'event_day', 'target'] + fighter_feats + other_feats]
        
        feat_df.reset_index(drop=True).to_csv(conf['fights_feat_gen']['feat_fn'], index=False)
        logger.info(f'Закончили создание таблицы боев')


        ActivityLogger().close_logger(logger)

    except Exception:
        logger.exception('ВОЗНИКЛО ИСКЛЮЧЕНИЕ!!!')
        ActivityLogger().close_logger(logger)

if __name__=='__main__':
        
    main()