import unittest

from ruamel.yaml import YAML
import pandas as pd


class FeatShortListTest(unittest.TestCase):
    
    conf = YAML().load(open('params.yaml'))
    feat_df = pd.read_csv(conf['split_data']['feat_full_fn'])


            
    def test_t(self):
        ...
    # def test_na_rolling_stats(self):

    #     self.assertTrue(self.feat_df.isna().sum().loc[lambda x: x>0].shape[0]==0)
# feat_df.isna().sum().drop(index=[it for it in feat_df.columns if any([x in it for x in ['height', 'dob', 'reach', 'stance'] ])])\
#                                  .loc[lambda x: x>0]


        
        
if __name__=='__main__':
    unittest.main()