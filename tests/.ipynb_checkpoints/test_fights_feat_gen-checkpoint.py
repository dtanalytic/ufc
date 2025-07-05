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
            
    
    def test_na_rolling_stats(self):
        
        self.assertTrue(self.feat_df.isna().sum().loc[lambda x: x>0].shape[0]==0)
        
if __name__=='__main__':
    unittest.main()