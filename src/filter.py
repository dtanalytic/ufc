import pandas as pd

from ruamel.yaml import YAML

import numpy as np

import click


@click.command()
def main():
    
    conf = YAML().load(open('params.yaml'))
    
    df = pd.read_csv(conf['preprocess']['prep_fn'])
    df = df[df['Sig. str'].notnull()]    
    
    # drop fighters duples 
    df = df[df['Fighter']!='Phillip Hawes']
    
    df.to_csv(conf['filter']['filt_fn'], index=False)

    

if __name__=='__main__':
        
    main()