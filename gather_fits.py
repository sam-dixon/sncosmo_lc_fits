import os
import copy
import pickle
import sncosmo
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm

DS_NAMES = ['csp', 'des', 'foundation', 'jla', 'ps1']
FIT_DIR = '/home/samdixon/sncosmo_lc_fits/fits'
DATA_DIR = '/home/samdixon/sncosmo_lc_fits/data'

Z_CUT = {'sdss': 0.15,
         'snls': 0.6,
         'foundation': 0.035,
         'des': 0.4,
         'csp': 0.04,
         'ps1': 0.3}

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

def parse_cov(cov):
    rows = []
    for line in cov.split('['):
        if line == '':
            continue
        col = [float(x) for x in line.split(']')[0].split()]
        rows.append(col)
    return np.array(rows)


def gather_all(save_path='collected_lc_fit_results.csv'):
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
            all_data[name] = copy.copy(param_dict)
            all_data[name]['z_fit'] = all_data[name].pop('z')
            for k, v in data[name].meta.items():
                if k in ['name', 't0']:
                    continue
                all_data[name][k] = v
            mb = calc_mb(**param_dict)
            fit_cov = fit_result['covariance']
            cov = np.zeros((3, 3))
            cov[0, 0] = fit_cov[2, 2] * (-2.5/(np.log(10)*mb))**2
            cov[1:, 0] = fit_cov[3:, 2] * (-2.5/(np.log(10)*mb)) # off-diagonal, m_b x c_i
            cov[0, 1:] = fit_cov[2, 3:] * (-2.5/(np.log(10)*mb)) # off-diagonal, c_i x m_b
            cov[1:, 1:] = fit_cov[3:, 3:] # c_i x c_j
            all_data[name]['mb'] = mb
            all_data[name]['mb_err'] = cov[0, 0]
            all_data[name]['x1_err'] = fit_result['errors']['x1']
            all_data[name]['c_err'] = fit_result['errors']['c']
            all_data[name]['cov'] = cov
            all_data[name]['survey'] = data[name].meta['survey']
    all_data = pd.DataFrame.from_dict(all_data).T
    all_data.to_csv(save_path)
    return all_data


def cut_and_prep_for_unity(data_path='collected_lc_fit_results.csv',
                           save_path='lc_fit_results_unity.pkl'):
    try:
        df = pd.read_csv(data_path, index_col=0)
        df['cov'] = df['cov'].apply(parse_cov)
    except FileNotFoundError:
        df = gather_all(data_path)
    df['survey'] = df.survey.str.lower()
    df = df[df.survey.isin(Z_CUT.keys())]
    ctype = pd.api.types.CategoricalDtype(categories=['foundation',
                                                      'csp',
                                                      'sdss',
                                                      'ps1',
                                                      'des',
                                                      'snls'],
                                          ordered=True)
    df['survey'] = df.survey.astype(ctype)
    df['z_cutoff'] = df.survey.map(Z_CUT).astype(float)

    cut = np.abs(df.x1) < 3
    cut &= np.abs(df.c) < 0.3
    cut &= df.x1_err < 1
    cut &= df.c_err < 1
    cut &= df.z_helio < df.z_cutoff
        
    df = df[cut]
    
    stan_data = {'n_sne': len(df),
                 'names': df.index.values,
                 'n_props': 3,
                 'n_non_gaus_props': 0,
                 'n_sn_set': len(df.survey.unique()),
                 'sn_set_inds': (df.survey.cat.codes.astype(int)+1).astype(int),
                 'z_helio': df.z_helio.values.astype(float),
                 'z_CMB': df.z_cmb.values.astype(float),
                 'obs_mBx1c': df[['mb', 'x1', 'c']].values,
                 'obs_mBx1c_cov': df['cov'].values,
                 'n_age_mix': 0,
                 'age_gaus_mean': np.array([]).reshape(0, len(df), 0),
                 'age_gaus_std': np.array([]).reshape(0, len(df), 0),
                 'age_gaus_A': np.array([]).reshape(0, len(df), 0),
                 'do_fullDint': 0,
                 'outl_frac_prior_lnmean': -4.6,
                 'outl_frac_prior_lnwidth': 1.,
                 'lognormal_intr_prior': 0,
                 'allow_alpha_S_N': 0}

    with open(save_path, 'wb') as f:
        pickle.dump(stan_data, f)

if __name__ == '__main__':
    cut_and_prep_for_unity()
