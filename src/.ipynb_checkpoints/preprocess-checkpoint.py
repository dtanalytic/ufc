import pandas as pd

from ruamel.yaml import YAML
import re
import numpy as np

import click
import sys
sys.path.append('.')
from src.constants import ActivityLogger
from src.prep_funcs import prep_strikes_fn


@click.command()
def main():

    try:

        logger = ActivityLogger().get_logger(__file__)
        
        conf = YAML().load(open('params.yaml'))

        prep_strikes_fn(fights_in_fn = conf['preprocess']['source_fn'], fights_out_fn= conf['preprocess']['prep_fn'], fighters_stat_fn=conf['preprocess']['source_fighters_fn'], fighters=[])
    
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