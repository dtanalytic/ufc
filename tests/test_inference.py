import unittest

from ruamel.yaml import YAML
import pandas as pd
import numpy as np

import sys
sys.path.append('.')
from src.inference import get_features_event

class InferenceTest(unittest.TestCase):
    
    conf = YAML().load(open('params.yaml'))
    # feat_df = pd.read_csv(conf['fights_feat_gen']['feat_fn'])
    feat_df = pd.read_csv(conf['split_data']['feat_full_fn']).drop(columns=['split'])
    coef_df = pd.read_excel(conf['inference']['coef_fn'])
    coef_df = coef_df.assign(event_day=lambda x:pd.to_datetime(x['date'], format='%d.%m.%Y'))
        
    
    def test_inference_stats(self):

        ev_nm = 'UFC Fight Night: Уиттакер - Де Риддер'

        feat_pred_df = get_features_event(ev_nm=ev_nm, 
                                    fights_fn=self.conf['inference']['fights_fn'], 
                                    fighters_fn=self.conf['inference']['fighters_fn'], 
                                    coef_fn=self.conf['inference']['coef_fn'], 
                                    vocab_fn=self.conf['inference']['vocab_fn'], 
                                    ufc_rankings_fn=self.conf['inference']['rankings_fn'], 
                                    rankings_big_fn=self.conf['inference']['rankings_big_fn'], conf=self.conf).drop(columns=['event', 'coef1', 'coef2'])



        # import pdb;pdb.set_trace()
        feat_row = self.feat_df.query('fighter=="Reinier de Ridder" and opponent=="Robert Whittaker"')
        
        pred_row = feat_pred_df.query('fighter=="Robert Whittaker" and opponent=="Reinier de Ridder"')

        
        self.assertTrue(set([it for it in feat_row.columns if not it in pred_row.columns])==set(['event', 'target']))
        
        self.assertTrue(len([it for it in pred_row.columns if not it in feat_row.columns])==0)
        
        self.assertTrue(np.abs(feat_row['dob_diff_custom_feat'].iloc[0].round(2)) == np.abs(pred_row['dob_diff_custom_feat'].iloc[0].round(2)))
                
        self.assertTrue(np.abs(feat_row['kd_rol_sum_feat'].iloc[0].round(2)) == np.abs(pred_row['kd_rol_sum_feat'].iloc[0].round(2)))

        self.assertTrue(pred_row.isna().sum().loc[lambda x: x>0].shape[0]==0)
        self.assertTrue(feat_row.isna().sum().loc[lambda x: x>0].shape[0]==0)
                        

        
if __name__=='__main__':
    unittest.main()