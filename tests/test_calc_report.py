import unittest

from ruamel.yaml import YAML
import pandas as pd
import numpy as np

import sys
sys.path.append('.')
from src.inference import place_bet, calc_profit, calc_time_profit

class ReportTest(unittest.TestCase):
    
    conf = YAML().load(open('params.yaml'))
    alpha = 0.6
    fake_df = pd.DataFrame({'score1':[0.2, 0.5, 0.1, 0.8, 0.4, 0.1], 'target':[1,0,0,0,1,1], 
              'event_day':[pd.Period(it, freq='D') for it in ['2025-11-01', '2025-11-10', '2025-11-01', '2025-11-01', '2025-11-10', '2025-11-10']], 
              'coef1':[1.2,3,3.3,2.5,2.6, 2.25], 'coef2':[2.1, 1.21, 1.1,1.51,1.7,1.4]})

    def test_income_time_calcs(self):
        
        income_time_df, res_l = calc_time_profit(self.fake_df, strategy_selection=None, alpha=self.alpha)
        
        t_df = self.fake_df[self.fake_df.event_day=='2025-11-01']
        t_bet_df = place_bet(placebet_df=t_df, strategy_selection=None, alpha=self.alpha)
        row = t_bet_df.iloc[1]
        
        assert (row['diff']/t_bet_df['diff'].sum()).round(2) == 0.15
        assert (t_bet_df['bet'].to_numpy() * np.array([2.1, 1.1, 2.5]))[1] == income_time_df['income'].iloc[0]
        
        
        
        t_df = self.fake_df[self.fake_df.event_day=='2025-11-10']
        t_bet_df = place_bet(placebet_df=t_df, strategy_selection=None, alpha=self.alpha)
        assert income_time_df['income'].iloc[1] == t_bet_df['bet'].iloc[1]*t_bet_df['coef1'].iloc[1]
        

    def test_bet_calcs(self):

        alpha = self.alpha
        bet_df = place_bet(placebet_df=self.fake_df, strategy_selection=None, alpha=alpha)
        
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