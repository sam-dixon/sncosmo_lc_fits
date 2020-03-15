import os
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

TARGET_PATH = '/Users/samdixon/data/PS1_lightcurves'
URL = 'https://archive.stsci.edu/hlsps/ps1cosmo/jones/lightcurves/'


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


def main():
    urls = get_urls()
    sn_names = [url.split('_')[4] for url in urls]
    os.makedirs(TARGET_PATH, exist_ok=True)
    for name, url in tqdm(zip(sn_names, urls)):
        fname = '{}.dat'.format(name)
        download_to_file(TARGET_PATH, fname, url)


if __name__ == '__main__':
    main()
