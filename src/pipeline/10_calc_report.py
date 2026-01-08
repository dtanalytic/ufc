import pandas as pd
import numpy as np

import os
import time

import click
import joblib
import json

from ruamel.yaml import YAML

import seaborn as sns
import matplotlib.pyplot as plt
sns.set()

import sys
sys.path.append('.')
from src.constants import ActivityLogger
from src.inference import place_bet, calc_profit, calc_time_profit
from src.inference import add_bet_info_cols


@click.command()
def main():

    try:

        logger = ActivityLogger().get_logger(__file__)
        conf = YAML().load(open('params.yaml'))
  
        start = time.time()
        
        DN = conf['train_eval']['report_dn']
        
        feat_df = pd.read_csv(conf['train_eval']['score_df_fn'])
        
        coef_df = pd.read_excel(conf['inference']['coef_fn'])
        vocab_df = pd.read_csv(conf['inference']['vocab_fn'])
        
        coef_df = coef_df.merge(vocab_df[['name_rus','fighters_name', 'fights_name', 'ranks_name']].rename(columns={**{'name_rus':'fighter1'}, **{it:f'{it}1' for it in ['fighters_name', 'fights_name', 'ranks_name']}})
                            , on='fighter1', how='left')\
            .merge(vocab_df[['name_rus','fighters_name', 'fights_name', 'ranks_name']].rename(columns={**{'name_rus':'fighter2'}, **{it:f'{it}2' for it in ['fighters_name', 'fights_name', 'ranks_name']}}), on='fighter2', how='left')
        
        feat_df = feat_df.merge(coef_df[['fights_name1', 'fights_name2', 'coef1', 'coef2']].drop_duplicates()\
                              .rename(columns={'fights_name1':'fighter', 'fights_name2':'opponent'})
                           , on=['fighter', 'opponent'])
        
        nfeat_cols = ['event', 'fighter', 'opponent', 'event_day', 'target', 'coef1', 'coef2']
        feat_cols = [it for it in feat_df.columns if not it in nfeat_cols]
        
        feat_df = feat_df[nfeat_cols+feat_cols]

        feat_df = feat_df.assign(event_day = pd.to_datetime(feat_df['event_day'], format='%Y-%m-%d').dt.to_period(freq='D'))

        # income by time
        # income_time_df, income_res_l = calc_time_profit(placebet_df=feat_df, strategy_selection=None, alpha=conf['calc_report']['alpha'])

        if conf['calc_report']['strategy']=='all':
            val_sel=pd.Series([])
            ts_sel=pd.Series([])
            all_sel=pd.Series([])
        elif conf['calc_report']['strategy']=='score_diff':
            df = add_bet_info_cols(feat_df.copy())
            diff_thresh = df.query('split=="val"')['diff'].abs().quantile(conf['calc_report']['diff_q'])            
            score_thresh = df.query('split=="val"')['score1'].quantile(conf['calc_report']['score_q'])
            val_sel = (df.query('split=="val"')['diff'].abs()>diff_thresh)&(df.query('split=="val"')['score']>score_thresh)
            ts_sel = (df.query('split=="ts"')['diff'].abs()>diff_thresh)&(df.query('split=="ts"')['score']>score_thresh)
            all_sel = (df['diff'].abs()>diff_thresh)&(df['score']>score_thresh)
            
        income_time_val_df, income_res_val_l = calc_time_profit(placebet_df=feat_df.query('split=="val"'), strategy_selection=val_sel, alpha=conf['calc_report']['alpha'])
        income_time_ts_df, income_res_ts_l = calc_time_profit(placebet_df=feat_df.query('split=="ts"'), strategy_selection=ts_sel, alpha=conf['calc_report']['alpha'])
        
        income_time_val_df.to_csv(f'{DN}/income_time_val_df.csv', index=False)
        joblib.dump(income_res_val_l, f'{DN}/income_res_val_l.pkl')
        
        income_time_ts_df.to_csv(f'{DN}/income_time_ts_df.csv', index=False)
        joblib.dump(income_res_ts_l, f'{DN}/income_res_ts_l.pkl')
        
        # график во времени
        plt.figure(figsize=(10, 8))
        income_time_val_df.set_index('event_day')['income'].sort_index(ascending=False).plot.barh()
        plt.tight_layout()
        plt.savefig(f'{DN}/income_time_val.png')

        
        # график во времени
        plt.figure(figsize=(10, 8))
        income_time_ts_df.set_index('event_day')['income'].sort_index(ascending=False).plot.barh()
        plt.tight_layout()
        plt.savefig(f'{DN}/income_time_ts.png')

        income, df = calc_profit(placebet_df=feat_df, strategy_selection=all_sel, alpha=conf['calc_report']['alpha'])
        
        # метрики дохода записываем
        d = {'income_time_val_mean':income_time_val_df['income'].mean(), 'income_time_ts_mean':income_time_ts_df['income'].mean(), 'income_all': income}
        with open(conf['calc_report']['report_metrics_fn'], 'wt') as f:
            json.dump(d, f)
            
        end = time.time()
        
        logger.info(f'Подготовка отчета заняла {(end-start)/60:.1f} минут\n{d}')

        
        ActivityLogger().close_logger(logger)

    except Exception:
        logger.exception('ВОЗНИКЛО ИСКЛЮЧЕНИЕ!!!')
        ActivityLogger().close_logger(logger)

if __name__=='__main__':
        
    main()