import os
import pickle
import sncosmo
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm

DS_NAMES = ['csp', 'des', 'foundation', 'jla', 'ps1']
FIT_DIR = '/home/samdixon/sncosmo_lc_fits/fits'
DATA_DIR = '/home/samdixon/sncosmo_lc_fits/data'

def calc_mb(**param_dict):
    """Calculates the apparent magnitude in the B-band based on the 
    SALT2 parameters in the param_dict"""
    model = sncosmo.Model(source='salt2',
                          effects=[sncosmo.CCM89Dust()],
                          effect_names=['mw'],
                          effect_frames=['obs'])
    model.set(**param_dict)
    try:
        mb = model.bandmag(band='bessellb', time=param_dict['t0'], magsys='ab')
    except:
        mb = np.nan
    return mb

all_data = {}
for dataset in DS_NAMES:
    data_path = os.path.join(DATA_DIR, '{}_lcs.pkl'.format(dataset))
    data = pickle.load(open(data_path, 'rb'))
    result_dir = os.path.join(FIT_DIR, dataset)
    names = sorted(data.keys())
    for name in tqdm(names):
        result_path = os.path.join(result_dir, '{}.pkl'.format(name))
        try:
            with open(result_path, 'rb') as f:
                fit_result = pickle.load(f)
        except FileNotFoundError:
            logging.warning('{} has no fit file'.format(name))
            continue
        param_dict = dict(zip(fit_result['param_names'], fit_result['parameters']))
        all_data[name] = param_dict
        all_data[name]['mb'] = calc_mb(**param_dict)
        all_data[name]['x1_err'] = fit_result['errors']['x1']
        all_data[name]['c_err'] = fit_result['errors']['c']
        all_data[name]['survey'] = data[name].meta['survey']
all_data = pd.DataFrame.from_dict(all_data).T
all_data.to_csv('collected_lc_fit_results.csv')
