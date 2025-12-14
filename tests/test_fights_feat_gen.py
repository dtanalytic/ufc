import unittest

from ruamel.yaml import YAML
import pandas as pd

class FightsFeatGenTest(unittest.TestCase):
    
    conf = YAML().load(open('params.yaml'))
    stat_df = pd.read_csv(conf['stat_feat_gen']['feat_fn'])
    feat_df = pd.read_csv(conf['fights_feat_gen']['feat_fn'])

    def stat_diff(self, fighter1, fighter2, feat_nm):
        stat = self.feat_df.loc[(self.feat_df.fighter==fighter1) & (self.feat_df.opponent==fighter2), feat_nm].iloc[0]
        stat1 = self.stat_df.loc[(self.stat_df.Fighter==fighter1) & (self.stat_df.Opponent==fighter2), feat_nm.replace('_sub', '')].iloc[0]
        stat2 = self.stat_df.loc[(self.stat_df.Fighter==fighter2) & (self.stat_df.Opponent==fighter1), feat_nm.replace('_sub', '')].iloc[0]
        self.assertEqual(stat.round(2), (stat1-stat2).round(2))
    
    def test_sub_stats(self):

        feat_nm = 'sub_att_stat_rol_sum_sub'
        
        if feat_nm in self.feat_df.columns:
            self.stat_diff(fighter1='Alexander Volkanovski', fighter2 = 'Ilia Topuria', feat_nm=feat_nm)

        feat_nm = 'ctrl_dam_stat_exp_max_min_d_sub'
        
        if feat_nm in self.feat_df.columns:
            self.stat_diff(fighter1='Ilia Topuria', fighter2 = 'Josh Emmett', feat_nm=feat_nm)

    def test_draws_num(self):
        
        self.assertTrue(self.stat_df[~self.stat_df.Result.isin(['W', 'L'])].shape[0]/self.stat_df.shape[0] < 0.02)

        
    def test_all_except_draws(self):
        
        stat_df_len = self.stat_df[self.stat_df.Result.isin(['W', 'L'])].shape[0]
        self.assertEqual(stat_df_len/2, self.feat_df.shape[0])

if __name__=='__main__':
    unittest.main()