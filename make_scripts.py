import os
import sys
import click
import pickle
import sncosmo
import numpy as np
import pandas as pd
from astropy.table import Table

SCRIPT_DIR = 'scripts'
if not os.path.isdir(SCRIPT_DIR):
    os.makedirs(SCRIPT_DIR)

CURR_DIR = os.path.abspath('./')
DATA_DIR = os.path.join(CURR_DIR, 'data')
DS_NAMES = ['jla', 'csp', 'des', 'foundation', 'ps1']

TEMPLATE = """#!/bin/bash
#$ -N {dataset}_{start}_{end}
#$ -e {curr_dir}/logs
#$ -o {curr_dir}/logs

/home/samdixon/anaconda3/bin/python {curr_dir}/fit_lcs.py {dataset} {start} {end} {no_mc}
"""

@click.command()
@click.argument('dataset', type=click.Choice(DS_NAMES))
@click.argument('njobs', type=int)
@click.option('--no_mc', is_flag=True)
def main(dataset, njobs, no_mc):
    data_path = os.path.join(DATA_DIR, '{}_lcs.pkl'.format(dataset))
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    n_sne = len(data)

    if not no_mc:
        submit_fname = 'submit_{}_mcmc.sh'.format(dataset)
    else:
        submit_fname = 'submit_{}.sh'.format(dataset)
    submit_script_path = os.path.join(SCRIPT_DIR, submit_fname)
    with open(submit_script_path, 'w') as subf:
        for script_id in range(njobs):
            start = int(n_sne / njobs * script_id)
            end = int(n_sne / njobs * (script_id + 1))
            if not no_mc:
                script_fname = '{}_{}_{}_mcmc.sh'.format(dataset, start, end)
            else:
                script_fname = '{}_{}_{}.sh'.format(dataset, start, end)
            script_path = os.path.join(SCRIPT_DIR, script_fname)
            with open(script_path, 'w') as f:
                f.write(TEMPLATE.format(dataset=dataset,
                                        start=start,
                                        end=end,
                                        no_mc='--no_mc' if no_mc else ''))
            os.chmod(script_path, 0o755)
            subf.write('qsub {}\n'.format(os.path.abspath(script_path)))
    os.chmod(submit_script_path, 0o755)


if __name__ == '__main__':
    main()