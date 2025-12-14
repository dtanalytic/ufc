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
        
        df = pd.read_csv(conf['preprocess']['prep_fights_fn'])
        df = df[df['Sig. str'].notnull()]
        
        # drop fighters duples, есть еще Phil Hawes
        df = df[(df['Fighter']!='Phillip Hawes') & (df['Opponent']!='Phillip Hawes')]
        fighters_duples = ['Bruno Silva', 'Jean Silva', 'Joey Gomez', 'Michael McDonald', 'Mike Davis']

        df = df[~(df.Fighter.isin(fighters_duples) | df.Opponent.isin(fighters_duples))]

        df.to_csv(conf['filter']['filt_fights_fn'], index=False)

        logger.info(f'Закончили фильтрацию стат файла')

        ActivityLogger().close_logger(logger)

    except Exception:
        logger.exception('ВОЗНИКЛО ИСКЛЮЧЕНИЕ!!!')
        ActivityLogger().close_logger(logger)

if __name__=='__main__':
        
    main()