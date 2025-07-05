import unittest

from ruamel.yaml import YAML
import pandas as pd

class StatFeatGenTest(unittest.TestCase):
    
    conf = YAML().load(open('params.yaml'))
    stat_df = pd.read_csv(conf['stat_feat_gen']['feat_fn'])

    
    def test_rolling_stats(self):

        
        N = 6
        self.assertTrue(self.stat_df.query('Fighter=="Conor McGregor"')['ctrl_stat_exp_max_min_d'].iloc[N].round(2)==0.57)

        if self.conf['stat_feat_gen']['rol_window_size']==3:
            N = 13
            self.assertTrue(self.stat_df.query('Fighter=="Conor McGregor"')['lose_stat_rol_sum'].iloc[N]==2)
    
            N = 3
            self.assertTrue(self.stat_df.query('Fighter=="Conor McGregor"')['kd_stat_rol_sum'].iloc[N].round(2)==1.14)
    
            # под номером 3, ноу нас сдвиг, тут еще должно быть mean от признака
            # self.assertEqual(self.df_stat.loc[self.df_stat.Fighter=='Khabib Nurmagomedov', 'sig_str_stat'].iloc[4].round(2), 2.28)

    
    def test_na_rolling_stats(self):
        
        self.assertTrue(self.stat_df.groupby('Fighter').apply(lambda x: min(x.shape[0], self.conf['stat_feat_gen']['rol_window_size'])).sum()==self.stat_df['kd_stat_rol_sum'].isna().sum())
        
if __name__=='__main__':
    unittest.main()