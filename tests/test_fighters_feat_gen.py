import unittest

from ruamel.yaml import YAML
import pandas as pd

class FightersFeatGenTest(unittest.TestCase):
    
    conf = YAML().load(open('params.yaml'))
    feat_df = pd.read_csv(conf['split_data']['feat_full_fn'])

    def test_fighter_feats(self):

        name = 'Tony Ferguson'
        
        ferg_df = self.feat_df[(self.feat_df.fighter.str.contains(name))|(self.feat_df.opponent.str.contains(name))].sort_values(by='event_day')
        
        
        # с этими бойцами в конце ufc карьеры Фергюсон выходил в левом углу
        l = ferg_df.loc[ferg_df['left_corner_custom_feat']==0, 'fighter'].tolist()
        self.assertTrue(all([it in l for it in ['Bobby Green', 'Paddy Pimblett', 'Michael Chiesa']]))
        
        # для Фергюсона 12*5 + 11 height
        self.assertTrue(ferg_df.loc[ferg_df.fighter=='Michael Chiesa', 'opponent_height_custom_feat'].iloc[0]==71)
        
        # для green-а нет данных, поэтому разность есть nan
        self.assertTrue(ferg_df.loc[ferg_df.fighter=='Bobby Green', 'reach_diff_custom_feat'].isna().sum()==1)
        
        # разница в днюхах в месяцах
        self.assertTrue(ferg_df.loc[ferg_df.fighter=='Michael Chiesa', 'dob_diff_custom_feat'].iloc[0].round()==46)
        
        # количество месяцев (считаем по 30 дней) (pd.Period('2011-12-03', freq='D') - pd.Period('1984-02-12', freq='D')).n/30
        self.assertTrue(ferg_df.loc[(ferg_df.fighter==name) & (ferg_df.event_day=='2011-12-03'), 'fighter_dob_custom_feat'].iloc[0].round()==339)
        
        self.assertTrue(ferg_df.loc[(ferg_df.fighter==name) & (ferg_df.event_day=='2011-12-03'), 'fighter_days_nofight_custom_feat'].iloc[0] == (pd.Period('2011-12-03', freq='D') - pd.Period('2011-09-24', freq='D')).n)

        
        
    
if __name__=='__main__':
    unittest.main()