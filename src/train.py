import pandas as pd

from ruamel.yaml import YAML

import numpy as np

import click


@click.command()
def main():
    
    conf = YAML().load(open('params.yaml'))
    
    df = pd.read_csv(conf['stat_feat_gen']['feat_fn'])
    

    

if __name__=='__main__':
        
    main()