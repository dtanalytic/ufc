import pandas as pd

from ruamel.yaml import YAML
import re
import numpy as np

import click
import sys
sys.path.append('.')
from src.constants import ActivityLogger


def suf_change(col_names, suf_out, suf_dam):
    cols = []
    for it in col_names:
        if re.search(r'{suf}$'.format(suf=suf_dam), it):
            cols.append(re.sub(r'{suf}$'.format(suf=suf_dam), r'_d', it))
        elif re.search(r'{suf}((_)|$)'.format(suf=suf_out), it):
            cols.append(re.sub(r'{suf}((_)|$)'.format(suf=suf_out), r'\1', it))
        elif re.search(r'{suf}(_)'.format(suf=suf_dam), it):
            cols.append(re.sub(r'{suf}(_)'.format(suf=suf_dam), r'_d\1', it))
        else:
            cols.append(it)
    return cols
    
def insert_strikes_cols(df_in, colnm, suf):

    df = df_in.copy()
    df[f'sig_{suf}_stat'] =  df[colnm].str.extract(r'(\d+)\sof', expand=False)
    df[f'sig_{suf}_stat'] = df[f'sig_{suf}_stat'].astype(float)
    
    df[f'sig_{suf}_of_all_stat'] = df[colnm].str.extract(r'of\s(\d+)', expand=False)
    df[f'sig_{suf}_of_all_stat'] = df[f'sig_{suf}_of_all_stat'].astype(float)
    
    df[f'sig_{suf}_perc_stat'] = np.where(df[f'sig_{suf}_of_all_stat']!=0, df[f'sig_{suf}_stat']/df[f'sig_{suf}_of_all_stat'], 0)

    return df


@click.command()
def main():

    try:

        logger = ActivityLogger().get_logger(__file__)

        
        conf = YAML().load(open('params.yaml'))
    
        df = pd.read_csv(conf['preprocess']['source_fn'])
        df = df.drop_duplicates()
        df = df.assign(event_date1=pd.to_datetime(df['Date'], format='%Y-%m-%d', errors='coerce'))\
    .assign(event_date2=pd.to_datetime(df['Date'], format='%Y-%m-%d 00:00:00', errors='coerce'))\
    .assign(event_date=lambda x: x['event_date1'].mask(x['event_date1'].isna(), x['event_date2']))\
    .drop(columns=['event_date1', 'event_date2'])
        
        df = df.assign(notna=lambda x: x.notna().sum(axis=1)).sort_values(by='notna')\
            .drop_duplicates(['Event','Fighter_left', 'Fighter_right', 'Date'], keep='last')\
        
        df = df.sort_values(by='Date')
    
        df['Details:'] = df['Details:'].str.replace(r'\s{2,}', r'_', regex=True)
        df['Details:'] = df['Details:'].str.replace(r'\._', r'.', regex=True)
        df['Details:'] = df['Details:'].str.replace(r'\.;', r'.', regex=True)
        
        df['judges'] = df['Details:'].str.findall(r"(?P<judge>[A-Za-z'\s]+)[_:]{1}(?P<f1>\d+)\s*-\s*(?P<f2>\d+)")
        
        df.loc[df['judges'].str.len()>0, 'judge1'] = df.loc[df['judges'].str.len()>0, 'judges'].str[0]
        df.loc[df['judges'].str.len()>0, 'judge2'] = df.loc[df['judges'].str.len()>0, 'judges'].str[1]
        df.loc[df['judges'].str.len()>0, 'judge3'] = df.loc[df['judges'].str.len()>0, 'judges'].str[2]
    
        df_add = df.copy()
        df = df.rename(columns={'Fighter_left':'Fighter', 'Fighter_right':'Opponent', 'Win_lose_left':'Result'}).drop('Win_lose_right', axis=1)
        df.columns = suf_change(df.columns, suf_out='_l', suf_dam = '_r')
    
        df_add = df_add.rename(columns={'Fighter_right':'Fighter', 'Fighter_left':'Opponent', 'Win_lose_right':'Result'}).drop('Win_lose_left', axis=1)
        df_add.columns = suf_change(df_add.columns, suf_out='_r', suf_dam = '_l')
        
        df_all = pd.concat([df,df_add], ignore_index=True)
    
        del df, df_add
    
        #getting fighter params, adding to all suffix _stat
        #already float cols
        df_all.rename(columns={'KD':'kd_stat', 'Rev.':'rev_stat', 'Sub. att':'sub_att_stat', 
                               'KD_d':'kd_dam_stat', 'Rev._d':'rev_dam_stat', 'Sub. att_d':'sub_att_dam_stat'}, inplace=True)
    
        df_all = insert_strikes_cols(df_all, 'Sig. str', 'str')
        df_all = insert_strikes_cols(df_all, 'Sig. str_d', 'str_dam')
    
        df_all = insert_strikes_cols(df_all, 'Head', 'str_h')
        df_all = insert_strikes_cols(df_all, 'Head_d', 'str_h_dam')
    
        df_all = insert_strikes_cols(df_all, 'Leg', 'str_l')
        df_all = insert_strikes_cols(df_all, 'Leg_d', 'str_l_dam')
    
        df_all = insert_strikes_cols(df_all, 'Body', 'str_b')
        df_all = insert_strikes_cols(df_all, 'Body_d', 'str_b_dam')
    
        df_all = insert_strikes_cols(df_all, 'Distance', 'str_d')
        df_all = insert_strikes_cols(df_all, 'Distance_d', 'str_d_dam')
        
        df_all = insert_strikes_cols(df_all, 'Clinch', 'str_cl')
        df_all = insert_strikes_cols(df_all, 'Clinch_d', 'str_cl_dam')
    
        df_all = insert_strikes_cols(df_all, 'Ground', 'str_gr')
        df_all = insert_strikes_cols(df_all, 'Ground_d', 'str_gr_dam')
    
        df_all = insert_strikes_cols(df_all, 'Td', 'td')
        df_all = insert_strikes_cols(df_all, 'Td_d', 'td_dam')
    
    
        # ctrl_stat, time_stat in minutes
        df_all['ctrl_stat'] = df_all['Ctrl'].str.extract(r'(\d:\d{2})', expand=False)
        df_all.loc[df_all['ctrl_stat'].notnull(),'ctrl_stat'] = df_all.loc[df_all['ctrl_stat'].notnull(),'ctrl_stat'].str.split(':')\
                                                                .map(lambda x: int(x[0]) + int(x[1])/60)
        df_all['ctrl_stat'].fillna(0, inplace=True)
        
        
        df_all['ctrl_dam_stat'] = df_all['Ctrl_d'].str.extract(r'(\d:\d{2})', expand=False)
        df_all.loc[df_all['ctrl_dam_stat'].notnull(),'ctrl_dam_stat'] = df_all.loc[df_all['ctrl_dam_stat'].notnull(),'ctrl_dam_stat'].str.split(':')\
                                                                .map(lambda x: int(x[0]) + int(x[1])/60)
        df_all['ctrl_dam_stat'].fillna(0, inplace=True)
        
        
        df_all['time'] = (df_all['Round:']-1)*5 + df_all['Time:'].str.split(':').map(lambda x: int(x[0])+int(x[1])/60)
        
        # normalise stat cols stat to minutes 
        stat_cols = [it for it in df_all.columns if re.search('_stat', it)]
        for col in stat_cols:
            df_all[col] = df_all[col]/df_all['time']
        
        df_all['win_stat'] = (df_all['Result']=='W').astype(np.int8)
        df_all['lose_stat'] = (df_all['Result']=='L').astype(np.int8)
    
        # type of win stat
        met_d = {'Decision - Unanimous':'decision', "TKO - Doctor's Stoppage":'KO/TKO', 'KO/TKO':'KO/TKO', 'Submission':'wrestling', 'Decision - Split':'decision',
                'Decision - Majority':'decision', 'Could Not Continue':'KO/TKO', 'Overturned':'other', 'Other':'other', 'DQ':'other'} 
        
        df_all['method_type'] = df_all['Method:'].map(met_d)
        
        df_all['wrest_w_stat'] = ((df_all['method_type']=='wrestling') & (df_all['win_stat'])).astype(np.int8)
        df_all['wrest_l_stat'] = ((df_all['method_type']=='wrestling') & (df_all['lose_stat'])).astype(np.int8)
        
        df_all['KO_w_stat'] = ((df_all['method_type']=='KO/TKO')&(df_all['win_stat'])).astype(np.int8)
        df_all['KO_l_stat'] = ((df_all['method_type']=='KO/TKO') &(df_all['lose_stat'])).astype(np.int8)
        
        df_all['dec_w_stat' ] = ((df_all['method_type']=='decision') &(df_all['win_stat'])).astype(np.int8)
        df_all['dec_l_stat' ] = ((df_all['method_type']=='decision') &(df_all['lose_stat'])).astype(np.int8)
    
        df_all.to_csv(conf['preprocess']['prep_fn'], index=False)
    
        logger.info(f'Закончили препроцессинг стат файла')

        # коэффициенты
        coef_df = pd.read_excel(conf['preprocess']['source_coef_fn'])
        
        # чистка на наличие сущностей
        coef_df = coef_df[coef_df.fighter2.notna() & coef_df.fighter1.notna() & coef_df.coef1.notna() & coef_df.coef2.notna()]
        coef_df = coef_df[(coef_df.coef1!=0) & (coef_df.coef2!=0)]
        
        coef_df = coef_df.reset_index(drop=True)
        
        coef_df = coef_df.assign(fighters = coef_df.apply(lambda x: ' '.join(sorted([x['fighter1'], x['fighter2']])) , axis=1))\
                        .assign(event_day=lambda x:pd.to_datetime(x['date'], format='%d.%m.%Y'))
        
        # keep=last последние по порядку не помечаются как дубли и сохраняются в итоге (пример - .loc[lambda x: x['fighters']=='Дастин Порье Макс Холлоуэй'])
        duples_idx = coef_df[coef_df.fighters.duplicated(keep='last')].sort_values(by='fighters').index
        fighers_close_idx = coef_df.sort_values(by='fighters').groupby('fighters').apply(lambda x: np.min(np.abs(np.diff([-100]+x.index.tolist())))).loc[lambda x: x<50].index
        # тут и дубли по участикам и не далеко от друг друга
        dupl_df = coef_df[(coef_df.index.isin(duples_idx)) & (coef_df.fighters.isin(fighers_close_idx))].sort_values(by='fighters')
        
        dupl_alt_idx1 = coef_df[coef_df.duplicated(subset=['fighter1', 'fighter2', 'coef1', 'coef2'], keep='last')].index
        dupl_alt_idx2 = coef_df[coef_df.duplicated(subset=['fighter1', 'fighter2', 'event_day'], keep='last')].index
        
        # когда коэффициенты прям равны, это одни и те же бои (с очень близкой датой)
        coef_df = coef_df[~((coef_df.index.isin(dupl_alt_idx1))&(coef_df.index.isin(dupl_df.index)))]
        # когда одинаковая дата события (или nan) и участники одни, бои почти всегда рядом, их можно считать дублями 
        coef_df = coef_df[~((coef_df.index.isin(dupl_alt_idx2))&(coef_df.index.isin(dupl_df.index)))]
        
        # тут бои с одной датой nan и не близкие, но переносы (кроме Долидзе) 
        coef_df = coef_df[~((coef_df.index.isin(dupl_alt_idx2))&(~coef_df.index.isin(dupl_df.index) & (coef_df['fighters']!='Марвин Веттори Роман Долидзе')))]
        # убираем остаток от 2 критериев дублей, которые не залетают в dupl_df
        coef_df = coef_df[~((~coef_df.index.isin(dupl_alt_idx1))&(~coef_df.index.isin(dupl_alt_idx2))&(coef_df.index.isin(dupl_df.index)))]
    
        coef_df['susp'] = 0
        coef_df.loc[coef_df['name'].isin(["UFC Fight Night: Усман - Бакли", "UFC Fight Night: Хилл - Раунтри"]), 'susp'] = 1
        coef_df.to_csv(conf['preprocess']['prep_coef_fn'], index=False)

        logger.info(f'Закончили препроцессинг файла со ставками')

        ActivityLogger().close_logger(logger)

    except Exception:
        logger.exception('ВОЗНИКЛО ИСКЛЮЧЕНИЕ!!!')
        ActivityLogger().close_logger(logger)
        
if __name__=='__main__':
        
    main()