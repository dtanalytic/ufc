import unittest

from ruamel.yaml import YAML
import pandas as pd

class PrepTest(unittest.TestCase):
    
    conf = YAML().load(open('params.yaml'))
    prep_df = pd.read_csv(conf['preprocess']['prep_fn'])
    
    def test_judge(self):
        # парсинг конкретного судьи
        self.assertTrue('Tony Weeks' in self.prep_df.query('Fighter=="Dusko Todorovic" and Opponent=="Zachary Reese" and Event=="UFC Fight Night: Blanchfield vs. Barber"')['judges'].iloc[0])
        # общее количество nan-ов
        self.assertTrue(self.prep_df['judges'].isna().mean()<0.05)
        # коротких скисков, так как в Details: судей нет, а виды побед
        self.assertTrue((self.prep_df['judges'].str.len()<10).mean()<0.6)

    def test_end_methods(self):

        self.assertTrue(self.prep_df['Method:'].nunique()==10)
        self.assertTrue(self.prep_df['method_type'].nunique()==4)

        self.assertEqual(self.prep_df.loc[self.prep_df.Fighter=='Khabib Nurmagomedov', 'wrest_w_stat'].sum(), 5)
        self.assertEqual(self.prep_df.loc[self.prep_df.Fighter=='Khabib Nurmagomedov', 'dec_w_stat'].sum(), 6)
        self.assertEqual(self.prep_df.loc[self.prep_df.Fighter=='Khabib Nurmagomedov', 'KO_w_stat'].sum(), 2)

        self.assertEqual(self.prep_df.loc[self.prep_df.Fighter=='Conor McGregor', 'KO_l_stat'].sum(), 2)
        self.assertEqual(self.prep_df.loc[self.prep_df.Fighter=='Conor McGregor', 'wrest_l_stat'].sum(), 2)

    def test_several_stats(self):
        
        # нокдаун показатель в бою Конора и Альдо
        self.assertTrue(self.prep_df.loc[(self.prep_df.Fighter=='Conor McGregor') & (self.prep_df.Opponent=='Jose Aldo'), 'kd_stat'].iloc[0].round(2)==4.62)
        # нокдаун показатель в бою Масвидаля
        self.assertTrue(self.prep_df.loc[self.prep_df.kd_stat>10, 'Fighter'].iloc[0] == 'Jorge Masvidal')

        # быстрый сабмишн у Тактарова
        self.assertTrue(self.prep_df.loc[self.prep_df.sub_att_stat>6, 'sub_att_stat'].iloc[0].round(2)==6.67)
        
        # Joshua Van - среднее количество значимых ударов за минуту соответствует
        self.assertTrue(self.prep_df.loc[(self.prep_df.Fighter=='Joshua Van') & (self.prep_df.event_date<='2025-03-08'), 'sig_str_stat'].mean().round(2)==8.06)

        self.assertTrue(self.prep_df.loc[self.prep_df.sig_str_gr_perc_stat>10, 'sig_str_gr_perc_stat'].iloc[0]==12)

    
    def test_strikes(self):

        # sig_str_stat
        bat_ser = self.prep_df.query('Fighter=="Ketlen Vieira" and Opponent=="Macy Chiasson" and Event=="UFC Fight Night: Blanchfield vs. Barber"').iloc[0]
        self.assertTrue(bat_ser['sig_str_stat'].round(2)==1.87)

        # sig_str_h_dam_stat, sig_str_h_stat
        bat_ser = self.prep_df.query('Fighter=="Muslim Salikhov" and Opponent=="Song Kenan" and Event=="UFC Fight Night: Yan vs. Figueiredo"').iloc[0]
        self.assertTrue(bat_ser['sig_str_h_dam_stat'].round(2), 1.05)
        self.assertTrue(bat_ser['sig_str_h_stat'].round(2), 1.83)

        # sig_str_l_dam_stat, sig_str_l_stat
        bat_ser = self.prep_df.query('Fighter=="Rose Namajunas" and Opponent=="Amanda Ribas" and Event=="UFC Fight Night: Ribas vs. Namajunas"').iloc[0]
        self.assertTrue(bat_ser['sig_str_l_dam_stat'].round(2)==1)
        self.assertTrue(bat_ser['sig_str_l_stat'].round(2)==0.16)

        bat_ser = self.prep_df.query('Fighter=="Miles Johns" and Opponent=="Cody Gibson" and Event=="UFC Fight Night: Ribas vs. Namajunas"').iloc[0]
        self.assertTrue(bat_ser['sig_str_cl_perc_stat'].round(2)==0.04)

        bat_ser = self.prep_df.query('Fighter=="Poliana Botelho" and Opponent=="Luana Carolina" and Event=="UFC Fight Night: Reyes vs. Prochazka"').iloc[0]
        self.assertTrue(bat_ser['sig_td_perc_stat'].round(2), 0.02)

        # время контроля в минутах деленное на все время боя
        self.assertTrue(bat_ser['ctrl_stat'] == 0.31)
        self.assertTrue(bat_ser['ctrl_dam_stat'].round(2) == 0.36)
        
if __name__=='__main__':
    unittest.main()