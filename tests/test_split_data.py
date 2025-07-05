import unittest

from ruamel.yaml import YAML
import pandas as pd

class SplitDataTest(unittest.TestCase):
    
    conf = YAML().load(open('params.yaml'))
    feat_df = pd.read_csv(conf['split_data']['feat_full_fn'])

    def test_dates_order(self):
        
        self.assertTrue(self.feat_df.loc[self.feat_df.split=='tr', 'event_date'].max()<=self.feat_df.loc[self.feat_df.split=='val', 'event_date'].min())

        self.assertTrue(self.feat_df.loc[self.feat_df.split=='val', 'event_date'].max()<=self.feat_df.loc[self.feat_df.split=='ts', 'event_date'].min())
        
    def test_outlier_df(self):

        outlier_df = pd.read_csv(self.conf['split_data']['feat_anom_fn'])

        self.assertTrue(outlier_df['outlier_cols'].str.len().loc[lambda x: x==0].shape[0]==0)
        
if __name__=='__main__':
    unittest.main()