import unittest

from ruamel.yaml import YAML
import pandas as pd
import joblib

class ModelTest(unittest.TestCase):
    
    conf = YAML().load(open('params.yaml'))
    feat_df = pd.read_csv(conf['split_data']['feat_full_fn'])

    def test_inference_train_predicts(self):

        name = 'Tony Ferguson'
        
        ferg_df = self.feat_df[(self.feat_df.fighter.str.contains(name))|(self.feat_df.opponent.str.contains(name))].sort_values(by='event_day')


        
        self.assertTrue(
        
        
    
if __name__=='__main__':
    unittest.main()