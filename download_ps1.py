import os
import requests
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

TARGET_PATH = '/Users/samdixon/data/PS1_lightcurves'
URL = 'https://archive.stsci.edu/hlsps/ps1cosmo/jones/lightcurves/'
CAT_URL = 'https://archive.stsci.edu/hlsps/ps1cosmo/jones/' \
          'hlsp_ps1cosmo_panstarrs_gpc1_all_multi_v1_snparams.txt'


def get_urls(base_url=URL, ext='.dat'):
    """Gets a list of sub URLs with the specified file extension.
    """
    r = requests.get(base_url)
    soup = BeautifulSoup(r.content, 'html.parser')
    urls = []
    for link in soup.find_all('a'):
        link_url = link.get('href')
        if link_url.endswith(ext):
            urls.append(os.path.join(URL, link_url))
    return urls


def download_to_file(dir, fname, url):
    path = os.path.join(dir, fname)
    r = requests.get(url)
    with open(path, 'w') as f:
        f.write(r.text)


def get_spec_conf_names(url=CAT_URL):
    cat = pd.read_csv(CAT_URL, delim_whitespace=True, skiprows=1,
                      names=['ID', 'RA', 'Dec', 'RAHost', 'DecHost', 'zHost',
                             'NormSep', 'TDR', 'zSource', 'zSN', 'tpk', 'e_tpk',
                             'x1', 'e_x1', 'c', 'e_c', 'mB', 'e_mB', 'logM',
                             'e_logM', 'distcorr', 'PIa_PSNID', 'PIa_NN',
                             'PIa_Fitprob', 'PIa_Galsnid'])
    spec_conf = cat.PIa_PSNID==1.0
    spec_conf &= cat.PIa_NN==1.0
    spec_conf &= cat.PIa_Fitprob==1.0
    spec_conf &= cat.PIa_Galsnid==1.0
    return cat[spec_conf].ID


def main():
    urls = get_urls()
    sn_names = [url.split('_')[4] for url in urls]
    urls = dict(zip(sn_names, urls))
    spec_conf_names = get_spec_conf_names()
    os.makedirs(TARGET_PATH, exist_ok=True)
    for name in tqdm(spec_conf_names):
        fname = '{}.dat'.format(name)
        download_to_file(TARGET_PATH, fname, urls[name])


if __name__ == '__main__':
    main()

