import os
import click
import pickle
import sncosmo
import logging
import numpy as np


logger = logging.getLogger()
logger = logger.setLevel('INFO')
DS_NAMES = ['jla', 'csp', 'des', 'foundation', 'ps1']
DATA_DIR = '/home/samdixon/sncosmo_lc_fits/data'
OUT_DIR = '/home/samdixon/sncosmo_lc_fits/fits'


def fit_lc_and_save(lc, model_name, save_dir, no_mc):
    name = lc.meta['name']
    model = sncosmo.Model(source=model_name,
                          effects=[sncosmo.CCM89Dust()],
                          effect_names=['mw'],
                          effect_frames=['obs'])
    z = lc.meta['z']
    if np.isnan(lc.meta['mwebv']):
        mwebv = 0
    else:
        mwebv = lc.meta['mwebv']
    bounds = {}
    if np.isnan(lc.meta['t0']):
        t0 = np.mean(lc['time'])
        bounds['t0'] = (min(lc['time'])-20, max(lc['time']))
    else:
        t0 = lc.meta['t0']
        bounds['t0'] = (t0 - 5, t0 + 5)
    bounds['z'] = ((1 - 1e-4) * z, (1 + 1e-4) * z)
    for param_name in model.source.param_names[1:]:
        bounds[param_name] = (-50, 50)
    modelcov = model_name=='salt2'  # model covariance only supported for SALT2
    model.set(z=z, t0=t0, mwebv=mwebv)
    phase_range = (-15, 45) if model_name=='salt2' else (-10, 40)
    wave_range = (3000, 7000) if model_name=='salt2' else None
    save_path = os.path.join(save_dir, '{}.pkl'.format(name))
    try:
        min_result, min_fit_model = sncosmo.fit_lc(lc, model,
                                                   model.param_names[:-2],
                                                   bounds=bounds,
                                                   phase_range=phase_range,
                                                   wave_range=wave_range,
                                                   warn=False,
                                                   modelcov=modelcov)
        if not no_mc:
            cut_lc = sncosmo.select_data(lc, min_result['data_mask'])
            mc_result, mc_fit_model = sncosmo.mcmc_lc(cut_lc,
                                                      min_fit_model,
                                                      model.param_names[:-2],
                                                      guess_t0=False,
                                                      bounds=bounds,
                                                      warn=False,
                                                      nwalkers=10,
                                                      modelcov=modelcov)
            pickle.dump(mc_result, open(save_path, 'wb'))
        else:
            pickle.dump(min_result, open(save_path, 'wb'))
    except:
        logging.warning('Fit to {} failed'.format(name))


@click.command()
@click.argument('dataset', type=click.Choice(DS_NAMES))
@click.argument('start', default=0)
@click.argument('end', default=-1)
@click.option('--outdir', default=OUT_DIR)
@click.option('--model', default='salt2')
@click.option('--no_mc', is_flag=True)
def main(dataset, start, end, outdir, model, no_mc):
    data_path = os.path.join(DATA_DIR, '{}_lcs.pkl'.format(dataset))
    save_dir = os.path.join(outdir, dataset)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    names = sorted([str(_) for _ in data.keys()])
    for sn_name in names[start:end]:
        logging.info('Fitting {}'.format(sn_name))
        fit_lc_and_save(data[sn_name], model, save_dir, no_mc)


if __name__ == '__main__':
    main()
