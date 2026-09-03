import unittest

from ruamel.yaml import YAML
import pandas as pd
import numpy as np

import sys
sys.path.append('.')
from src.inference import place_bet, calc_profit, calc_time_profit, add_bet_info_cols

class ReportTest(unittest.TestCase):
    
    conf = YAML().load(open('params.yaml'))
    fake_df = pd.DataFrame({'score1':[0.2, 0.5, 0.1, 0.8, 0.4, 0.1], 'target':[1,0,0,0,1,1], 
              'event_day':[pd.Period(it, freq='D') for it in ['2025-11-01', '2025-11-10', '2025-11-01', '2025-11-01', '2025-11-10', '2025-11-10']], 
              'coef1':[1.2,3,3.3,2.5,2.6, 2.25], 'coef2':[2.1, 1.21, 1.1,1.51,1.7,1.4]})

    real_df = pd.read_parquet('tests/data/real_ex.parquet')
    # DN = conf['train_eval']['report_dn']
    # feat_df = pd.read_csv(conf['train_eval']['score_df_fn'])
    
    # coef_df = pd.read_excel(conf['inference']['coef_fn'])
    # vocab_df = pd.read_csv(conf['inference']['vocab_fn'])
    
    # coef_df = coef_df.merge(vocab_df[['name_rus','fighters_name', 'fights_name', 'ranks_name']].rename(columns={**{'name_rus':'fighter1'}, **{it:f'{it}1' for it in ['fighters_name', 'fights_name', 'ranks_name']}})
    #                     , on='fighter1', how='left')\
    #     .merge(vocab_df[['name_rus','fighters_name', 'fights_name', 'ranks_name']].rename(columns={**{'name_rus':'fighter2'}, **{it:f'{it}2' for it in ['fighters_name', 'fights_name', 'ranks_name']}}), on='fighter2', how='left')
    
    # feat_df = feat_df.merge(coef_df[['fights_name1', 'fights_name2', 'coef1', 'coef2']].drop_duplicates()\
    #                       .rename(columns={'fights_name1':'fighter', 'fights_name2':'opponent'})
    #                    , on=['fighter', 'opponent'])
    
    # nfeat_cols = ['event', 'split', 'fighter', 'opponent', 'event_day', 'target', 'coef1', 'coef2', 'score1']
    # feat_cols = [it for it in feat_df.columns if not it in nfeat_cols]
    
    # feat_df = feat_df[nfeat_cols+feat_cols]
    
    # feat_df = feat_df.assign(event_day = pd.to_datetime(feat_df['event_day'], format='%Y-%m-%d').dt.to_period(freq='D'))
    # feat_df.to_parquet('tests/data/real_ex.parquet')
    
    def test_income_time_calcs(self):
        
        
        self.conf['calc_report']['strategy_income']='diff'
        self.conf['calc_report']['alpha']=1
        income_time_df, res_l = calc_time_profit(self.fake_df, strategy_selection=pd.Series([]), conf=self.conf)
        
        t_df = self.fake_df[self.fake_df.event_day=='2025-11-01']
        t_bet_df = place_bet(placebet_df=t_df, conf=self.conf, strategy_selection=pd.Series([]))
        row = t_bet_df.iloc[1]
        
        self.assertTrue((row['diff']/t_bet_df['diff'].sum()).round(2) == 0.15)
        self.assertTrue((t_bet_df['bet'].to_numpy() * np.array([2.1, 1.1, 2.5]))[1] == income_time_df['income'].iloc[0])
        
        
        
        t_df = self.fake_df[self.fake_df.event_day=='2025-11-10']
        t_bet_df = place_bet(placebet_df=t_df, conf=self.conf, strategy_selection=pd.Series([]))
        self.assertTrue(income_time_df['income'].iloc[1] == 0)
        

    def test_real_ex1(self):
        conf = self.conf.copy()
        # где nan как раз не ставим по strategy_selection (там False)
        conf['calc_report']['alpha'] = 1
        conf['calc_report']['strategy_income'] = 'diff'
        conf['calc_report']['strategy_betnum'] = 'score_diff'
        conf['calc_report']['score_q'] = 0.7
        conf['calc_report']['diff_q'] = 0.1

        df = add_bet_info_cols(self.real_df.copy(), conf)
        diff_thresh = df.query('split=="val"')['diff'].abs().quantile(conf['calc_report']['diff_q'])            
        score_thresh = df.query('split=="val"')['score1'].quantile(conf['calc_report']['score_q'])
        val_sel = (df.query('split=="val"')['diff'].abs()>diff_thresh)&(df.query('split=="val"')['score']>score_thresh)
        ts_sel = (df.query('split=="ts"')['diff'].abs()>diff_thresh)&(df.query('split=="ts"')['score']>score_thresh)
        all_sel = (df['diff'].abs()>diff_thresh)&(df['score']>score_thresh)

        placebet_df = self.real_df.query('split=="val"')
        income_time_val_df, income_res_val_l = calc_time_profit(placebet_df=placebet_df, strategy_selection=val_sel, conf=conf)
        
        event_day = '2022-07-30'
        test_res = [it for it in income_res_val_l if it[0]==pd.Period(event_day, freq='D')][0]
        
        
        self.assertTrue(test_res[2].round(2)==2.07)
        
        self.assertTrue(test_res[3]['income'].dropna().round(2).tolist()==[1.35, 0.08, 0.65])
        
        # sel = (placebet_df.event_day==event_day)
        # t_df = placebet_df.loc[sel]
        # income, df = calc_profit(placebet_df=t_df, strategy_selection=val_sel[sel], conf=conf)
        # bet_df = place_bet(placebet_df=t_df, conf=conf, strategy_selection = val_sel[sel])
        
        # test_res[3][['fighter', 'opponent', 'coef1', 'coef2', 'score1', 'score2', 'proba1', 'proba2', 
        #              'diff1', 'diff2', 'selector', 'diff', 'score', 'betwin', 'bet', 'income']]
    
    def test_real_ex2(self):
        conf = self.conf.copy()
        # где nan как раз не ставим по strategy_selection (там False)
        conf['calc_report']['alpha'] = 0.8
        conf['calc_report']['strategy_income'] = 'kelly'
        conf['calc_report']['strategy_betnum'] = 'score_diff'
        conf['calc_report']['score_q'] = 0
        conf['calc_report']['diff_q'] = 0
        
        df = add_bet_info_cols(self.real_df.copy(), conf)
        diff_thresh = df.query('split=="val"')['diff'].abs().quantile(conf['calc_report']['diff_q'])            
        score_thresh = df.query('split=="val"')['score1'].quantile(conf['calc_report']['score_q'])
        val_sel = (df.query('split=="val"')['diff'].abs()>diff_thresh)&(df.query('split=="val"')['score']>score_thresh)
        ts_sel = (df.query('split=="ts"')['diff'].abs()>diff_thresh)&(df.query('split=="ts"')['score']>score_thresh)
        all_sel = (df['diff'].abs()>diff_thresh)&(df['score']>score_thresh)
        
        placebet_df = self.real_df.query('split=="val"')
        income_time_val_df, income_res_val_l = calc_time_profit(placebet_df=placebet_df, strategy_selection=val_sel, conf=conf)
        
        event_day = '2022-07-23'
        test_res = [it for it in income_res_val_l if it[0]==pd.Period(event_day, freq='D')][0]
        
        
        
        self.assertTrue(test_res[3]['income'].dropna().round(2).tolist()==[0.0, 0.0, 0.41, 0.0, 0.16, 0.17, 0.0, 0.0, 0.64, 0.0])

    def test_bet_calcs(self):

        alpha = 0.6
        self.conf['calc_report']['strategy_income']='diff'
        self.conf['calc_report']['alpha']=alpha
        
        bet_df = place_bet(placebet_df=self.fake_df, conf=self.conf, strategy_selection=pd.Series([]))
        # взяли недооцененного фаворита
        row1 = bet_df.iloc[0]
        diff = row1['proba2'] - row1['score2']
        self.assertTrue(diff.round(2) == -0.44)
        self.assertTrue(row1['betwin'] == (diff<0))
        
        # с таким же betwin, суммируем diff
        sum_diffs = bet_df.loc[bet_df.betwin==row1['betwin'], 'diff'].abs().sum()
        self.assertTrue((-diff/sum_diffs*alpha).round(2) == row1['bet'].round(2))
        
        # взяли недооцененного аутсайдера
        row2 = bet_df.iloc[1]
        
        diff = row2['proba1'] - row2['score1']
        self.assertTrue(diff == row2['diff'])
        
        # с таким же betwin, суммируем diff
        sum_diffs = bet_df.loc[bet_df.betwin==row2['betwin'], 'diff'].abs().sum()
        self.assertTrue(((-diff/sum_diffs)*(1-alpha)).round(2) == row2['bet'].round(2))
        
                        

        
if __name__=='__main__':
    unittest.main()