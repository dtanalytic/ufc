import pandas as pd

from ruamel.yaml import YAML

import numpy as np

import click
import sys
sys.path.append('.')
from src.constants import ActivityLogger


@click.command()
def main():

    try:

        logger = ActivityLogger().get_logger(__file__)
        conf = YAML().load(open('params.yaml'))
        
        df = pd.read_csv(conf['preprocess']['prep_fn'])
        df = df[df['Sig. str'].notnull()]
        
        # drop fighters duples, есть еще Phil Hawes
        df = df[(df['Fighter']!='Phillip Hawes') & (df['Opponent']!='Phillip Hawes')]
        
        df.to_csv(conf['filter']['filt_fn'], index=False)

        logger.info(f'Закончили фильтрацию стат файла')

        ActivityLogger().close_logger(logger)

    except Exception:
        logger.exception('ВОЗНИКЛО ИСКЛЮЧЕНИЕ!!!')
        ActivityLogger().close_logger(logger)

if __name__=='__main__':
        
    main()