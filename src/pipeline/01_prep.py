import pandas as pd

from ruamel.yaml import YAML
import re
import numpy as np

import click
import sys
sys.path.append('.')
from src.constants import ActivityLogger
from src.prep_funcs import prep_strikes_fn, prep_fighters_fn, prep_coefs_fn


@click.command()
def main():

    try:

        logger = ActivityLogger().get_logger(__file__)
        
        conf = YAML().load(open('params.yaml'))

        fights_df = prep_strikes_fn(fights_in_fn = conf['preprocess']['source_fights_fn'], fighters=[])
        fights_df.to_csv(conf['preprocess']['prep_fights_fn'], index=False)
        
        logger.info(f'Закончили препроцессинг файлов с боями и бойцами')

        # fighters_df=prep_fighters_fn(fighters_fn=conf['preprocess']['source_fighters_fn'], fighters=[])
        fighters_df = prep_fighters_fn(fighters_fn=conf['preprocess']['source_fighters_fn'], fighters=[]).drop_duplicates(subset='full_name')

        fighters_df.to_csv(conf['preprocess']['prep_fighters_fn'], index=False)
        
    
        coef_df = prep_coefs_fn(conf['preprocess']['source_coef_fn'])
        coef_df.to_csv(conf['preprocess']['prep_coef_fn'], index=False)
        

        logger.info(f'Закончили препроцессинг файла со ставками')

        ActivityLogger().close_logger(logger)

    except Exception:
        logger.exception('ВОЗНИКЛО ИСКЛЮЧЕНИЕ!!!')
        ActivityLogger().close_logger(logger)
        
if __name__=='__main__':
        
    main()