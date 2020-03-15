import os
import pickle
import sncosmo
import numpy as np
import pandas as pd
from tqdm import tqdm
from sfdmap import SFDMap
from astropy.io import ascii
from astropy.table import Table
from collections import defaultdict


# The goal here is to get all of the disparate data sets into a single
# format (pickled dictionary of name: Table, where the Table is an astropy
# table suitable for use in sncosmo.fit_lc. Table also has a metadata attribute
# containing details about the redshift, MW extinction, etc


DATA_DIR = '/Users/samdixon/data/'
OUT_DIR = './data/'

CSP_DATA_DIR = os.path.join(DATA_DIR, 'CSP_photometry_DR3')
FOUNDATION_DATA_DIR = os.path.join(DATA_DIR, 'foundation_photometry')
DES_DATA_DIR = os.path.join(DATA_DIR, 'DES_DR1/02-DATA_PHOTOMETRY/')
DES_DATA_DIR = os.path.join(DES_DATA_DIR, 'DES-SN3YR_DES/')
PS_DATA_DIR = os.path.join(DATA_DIR, 'PS1_lightcurves')
JLA_DATA_DIR = os.path.join(DATA_DIR, 'jla_light_curves')

CSP_FILT_MAP = {'u': 'cspu',
                'g': 'cspg',
                'r': 'cspr',
                'i': 'cspi',
                'B': 'cspb',
                'V0': 'cspv3014',
                'V1': 'cspv3009',
                'V': 'cspv9844',
                'Y': 'cspys',
                'H': 'csphs',
                'J': 'cspjs',
                'Jrc2': 'cspjs',
                'Ydw': 'cspyd',
                'Jdw': 'cspjd',
                'Hdw': 'csphd'}
CSP_MAGSYS = sncosmo.get_magsystem('csp')

PS_FILTS = ascii.read(os.path.join(DATA_DIR, 'PSfilters.txt'))

def read_and_register_ps_filts(name):
    name = name.lower()
    wave = PS_FILTS['Wave']*10
    trans = PS_FILTS[name]
    band = sncosmo.Bandpass(wave=wave, trans=trans, name=name)
    sncosmo.register(band, force=True)

for filt in ['gp1', 'rp1', 'ip1', 'zp1', 'yp1', 'wp1']:
    read_and_register_ps_filts(filt)

MWDUSTMAP = SFDMap(os.path.join(DATA_DIR, 'sfd_data/'))

def parse_csp():
    csp = {}
    for fname in tqdm(os.listdir(CSP_DATA_DIR)):
        if '_snpy.txt' not in fname:
            continue
        path = os.path.join(CSP_DATA_DIR, fname)
        meta = {}
        lc = defaultdict(list)
        current_filt = None
        with open(path) as f:
            for l in f.readlines():
                if l.split()[0][:2] == 'SN':
                    name, z, ra, dec = l.split()
                    meta['name'] = name.strip()
                    meta['survey'] = 'csp'
                    meta['z'] = float(z.strip())
                    meta['mwebv'] = MWDUSTMAP.ebv(float(ra), float(dec))
                    meta['t0'] = np.nan
                    continue
                if l.split()[0] == 'filter':
                    current_filt = CSP_FILT_MAP[l.split()[-1]]
                else:
                    time, mag, mag_err = [float(x.strip()) for x in l.split()]
                    flux = CSP_MAGSYS.band_mag_to_flux(mag, current_filt)
                    flux_err = mag_err * flux * np.log(10) / 2.5
                    zp = 2.5 * np.log10(CSP_MAGSYS.zpbandflux(current_filt))
                    lc['time'].append(53000 + time)
                    lc['flux'].append(flux)
                    lc['flux_err'].append(flux_err)
                    lc['zp'].append(zp)
                    lc['zpsys'].append('csp')
                    lc['band'].append(current_filt)
        csp[meta['name']] = Table(lc, meta=meta)
    return csp


def parse_jla():
    jla = {}
    for f in tqdm(os.listdir(JLA_DATA_DIR)):
        if f[:2] != 'lc':
            continue
        lc = sncosmo.read_lc(os.path.join(JLA_DATA_DIR, f),
                             format='salt2', expand_bands=True,
                             read_covmat=True)
        name = lc.meta['SN']
        try:
            t0 = float(lc.meta['DayMax'].split()[0])
        except KeyError:
            t0 = np.nan
        try:
            survey = lc.meta['SURVEY']
        except KeyError:
            survey = 'hst'
        lc.meta = {'name': name,
                   'survey': survey,
                   'z': lc.meta['Z_HELIO'],
                   'mwebv': lc.meta['MWEBV'],
                   't0': t0}
        jla[name] = lc
    return jla


def parse_des():
    des = {}
    for fname in tqdm(os.listdir(DES_DATA_DIR)):
        if fname[-3:] != 'dat':
            continue
        path = os.path.join(DES_DATA_DIR, fname)
        meta, obs_list = sncosmo.read_snana_ascii(path, default_tablename='OBS')
        name = fname.split('_')[-1].split('.')[0]
        obs_list['OBS'].meta = meta
        lc = obs_list['OBS']
        lc['BAND'].name = 'band'
        lc['FLUXCAL'].name = 'flux'
        lc['FLUXCALERR'].name = 'flux_err'
        lc['ZPFLUX'].name = 'zp'
        lc['zpsys'] = ['ab' for _ in lc]
        lc['band'] = ['des'+band_name for band_name in lc['band']]
        lc.meta = {'name': name,
                   'survey': 'des',
                   'z': lc.meta['REDSHIFT_FINAL'],
                   'mwebv': lc.meta['MWEBV'],
                   't0': lc.meta['PEAKMJD']}
        des[name] = lc
    return des


def parse_ps():
    ps = {}
    for fname in tqdm(os.listdir(PS_DATA_DIR)):
        path = os.path.join(PS_DATA_DIR, fname)
        meta, obs_list = sncosmo.read_snana_ascii(path, default_tablename='OBS')
        name = fname.split('.')[0]
        obs_list['OBS'].meta = meta
        lc = obs_list['OBS']
        lc['FLT'].name = 'band'
        lc['FLUXCAL'].name = 'flux'
        lc['FLUXCALERR'].name = 'flux_err'
        lc['zpsys'] = ['ab' for _ in lc]
        lc['zp'] = [27.5 for _ in lc]
        lc['band'] = [sncosmo.get_bandpass(band + 'p1') for band in lc['band']]
        ps[name] = lc
    return ps


def parse_foundation():
    data_path = os.path.join(FOUNDATION_DATA_DIR, 'foundation_photometry.dat')
    meta_path = os.path.join(FOUNDATION_DATA_DIR, 'foundation_lc_params.tex')

    data = pd.read_csv(data_path, delimiter=', ', engine='python')
    meta = ascii.read(meta_path, format='latex').to_pandas()
    data = data.set_index('SN')
    meta = meta.set_index('SN')

    foundation = {}
    for sn_name in tqdm(data.index.unique()):
        sn_data = data.loc[sn_name]
        try:
            sn_meta = meta.loc[sn_name]
        except KeyError:
            continue
        meta_out = {'name': sn_name,
                    'survey': 'foundation',
                    'z': float(sn_meta['z_helio'].split()[0]),
                    't0': float(sn_meta['Peak_MJD'].split()[0]),
                    'mwebv': np.nan}
        bands = [sncosmo.get_bandpass(band.lower())
                 for band in sn_data['Filter']]
        lc = {'time': sn_data['MJD'],
              'band': bands,
              'flux': sn_data['Flux'],
              'flux_err': sn_data['Flux_Uncertainty'],
              'zp': [27.5 for _ in range(len(sn_data))],
              'zpsys': ['ab' for _ in range(len(sn_data))]}
        foundation[sn_name] = Table(lc, meta=meta_out)
    return foundation


if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)
    print('Reading Foundation LCs')
    with open(os.path.join(OUT_DIR, 'foundation_lcs.pkl'), 'wb') as f:
        pickle.dump(parse_foundation(), f)

    print('Reading JLA LCs')
    with open(os.path.join(OUT_DIR, 'jla_lcs.pkl'), 'wb') as f:
        pickle.dump(parse_jla(), f)

    print('Reading CSP LCs')
    with open(os.path.join(OUT_DIR, 'csp_lcs.pkl'), 'wb') as f:
        pickle.dump(parse_csp(), f)

    print('Reading DES LCs')
    with open(os.path.join(OUT_DIR, 'des_lcs.pkl'), 'wb') as f:
        pickle.dump(parse_des(), f)

    print('Reading PS1 LCs')
    with open(os.path.join(OUT_DIR, 'ps1_lcs.pkl'), 'wb') as f:
        pickle.dump(parse_ps(), f)
