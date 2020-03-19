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


def radectoxyz(RAdeg, DECdeg):
    x = np.cos(DECdeg/(180./np.pi))*np.cos(RAdeg/(180./np.pi))
    y = np.cos(DECdeg/(180./np.pi))*np.sin(RAdeg/(180./np.pi))
    z = np.sin(DECdeg/(180./np.pi))
    return np.array([x, y, z], dtype=np.float64)


def get_dz(RAdeg, DECdeg):
    dzCMB = 371.e3/299792458. # NED
    #http://arxiv.org/pdf/astro-ph/9609034
    #CMBcoordsRA = 167.98750000 # J2000 Lineweaver
    #CMBcoordsDEC = -7.22000000
    CMBcoordsRA = 168.01190437 # NED
    CMBcoordsDEC = -6.98296811
    CMBxyz = radectoxyz(CMBcoordsRA, CMBcoordsDEC)
    inputxyz = radectoxyz(RAdeg, DECdeg)
    dz = dzCMB*np.dot(CMBxyz, inputxyz)
    return dz


def get_zCMB(RAdeg, DECdeg, z_helio):
    dz = -get_dz(RAdeg, DECdeg)
    one_plus_z_pec = np.sqrt((1. + dz)/(1. - dz))
    one_plus_z_CMB = (1 + z_helio)/one_plus_z_pec
    return one_plus_z_CMB - 1.


def get_zhelio(RAdeg, DECdeg, z_CMB):
    dz = -get_dz(RAdeg, DECdeg)
    one_plus_z_pec = np.sqrt((1. + dz)/(1. - dz))
    one_plus_z_helio = (1 + z_CMB)*one_plus_z_pec
    return one_plus_z_helio - 1.


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
            all_data[name] = param_dict
            mb = calc_mb(**param_dict)
            all_data[name]['mb'] = mb
            all_data[name]['mb_err'] = 2.5 / (np.log(10) * mb)
            all_data[name]['x1_err'] = fit_result['errors']['x1']
            all_data[name]['c_err'] = fit_result['errors']['c']
            all_data[name]['cov'] = fit_result['cov']
            all_data[name]['survey'] = data[name].meta['survey']
    all_data = pd.DataFrame.from_dict(all_data).T
    all_data.to_csv(save_path)


def cut_and_prep_for_unity(save_path='lc_fit_results_unity.csv'):
    #
    # stan_data = {'n_sne': len(data),
    #              'names': data.index.values,
    #              'n_props': n_props,
    #              'n_non_gaus_props': 0,
    #              'n_sn_set': len(data.set.unique()),
    #              'sn_set_inds': (data.set.values.astype(int)-1).astype(int),
    #              'z_helio': data.zhel.values.astype(float),
    #              'z_CMB': data.zcmb.values.astype(float),
    #              'obs_mBx1c': obs_data,
    #              'obs_mBx1c_cov': obs_cov,
    #              'n_age_mix': 0,
    #              'age_gaus_mean': np.array([]).reshape(0, len(data), 0),
    #              'age_gaus_std': np.array([]).reshape(0, len(data), 0),
    #              'age_gaus_A': np.array([]).reshape(0, len(data), 0),
    #              'do_fullDint': 0,
    #              'outl_frac_prior_lnmean': -4.6,
    #              'outl_frac_prior_lnwidth': 1.,
    #              'lognormal_intr_prior': 0,
    #              'allow_alpha_S_N': 0}
    pass


if __name__ == '__main__':
    gather_all()
