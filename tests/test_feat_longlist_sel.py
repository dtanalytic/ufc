import unittest

from ruamel.yaml import YAML
import pandas as pd
import phik

class FeatLongListTest(unittest.TestCase):
    
    conf = YAML().load(open('params.yaml'))
    feat_df = pd.read_csv(conf['split_data']['feat_full_fn'])


    def test_phik_values(self):


        if self.conf['feat_longlist_sel']['method']=='phik':
            ser = pd.read_csv(self.conf['feat_longlist_sel']['sign_feat_long_fn'], index_col=0)[['0']]
            self.assertTrue(self.feat_df.query('split=="tr"')[['target', 'left_corner_stat']].phik_matrix().iloc[0,1].round(2)==ser.loc['left_corner_stat'].iloc[0].round(2))

        
        
if __name__=='__main__':
    unittest.main()