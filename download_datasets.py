#!/usr/bin/python3
import argparse
import os
import zipfile
from io import BytesIO
import numpy as np
import requests
import pandas as pd
from typing import Dict, Union, Tuple


def download_file(urls: str, filename: str) -> None:
    chunk_size = 8192
    with open(filename, 'wb') as f:
        for url in urls:
            zip_data = BytesIO()
            print(f'> downloading {url}')
            with requests.get(url=url, stream=True) as r:
                r.raise_for_status()
                content_length = r.headers.get('Content-Length')
                if content_length:
                    max_chunks = int(content_length) // chunk_size
                else:
                    max_chunks = None
                chunk_count = 0
                for chunk in r.iter_content(chunk_size=chunk_size):
                    zip_data.write(chunk)
                    chunk_count += 1
                    max_len = 100
                    if max_chunks:
                        bar_len = int(max_len * chunk_count / max_chunks)
                    else:
                        bar_len = chunk_count % max_len
                    bar = bar_len * '='
                    space = int(max_len - bar_len) * ' '
                    print(f'\r{chunk_count}/{max_chunks} [{bar}{space}]', end='', flush=True)
                bar = max_len * '='
                print(f'\r{chunk_count}/{max_chunks} [{bar}]')
                if zipfile.is_zipfile(filename=zip_data):
                    with zipfile.ZipFile(zip_data) as zip_file:
                        data = zip_file.open(zip_file.namelist()[0]).read()
                else:
                    data = zip_data.getbuffer()
                f.write(data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to generate different chaotic time series data sets.')
    parser.add_argument('-f', '--force', dest='force', action='store_true', default=False,
                        help='use the force to overwrite already existing datasets with the same name as the generated ones.')
    args = parser.parse_args()
    force = args.force

    systems: Dict[str, Tuple[float, float, Union[Tuple, str]]] = {
        'roessler': (0.12, 0.069, 'https://dataverse.harvard.edu/api/access/datafile/5677685'),
        #'lorenz': (0.01, 0.905, 'https://dataverse.harvard.edu/api/access/datafile/5677683'),
        #'lorenz96': (0.05, 1.67, ('https://dataverse.harvard.edu/api/access/datafile/5677686',
        #                          'https://dataverse.harvard.edu/api/access/datafile/5677687')),
        #'thomas': (0.1, 0.055, 'https://dataverse.harvard.edu/api/access/datafile/5677682'),
        #'mackeyglass': (1.0, 0.006, 'https://dataverse.harvard.edu/api/access/datafile/5677684'),
        #'hyperroessler': (0.1, 0.14, 'https://dataverse.harvard.edu/api/access/datafile/5677688')
                                                         }

    for system, (dt, lle, urls) in systems.items():
        input_steps = 150
        samples = 10000
        frac_output_steps = 1.0 / lle / dt
        output_steps = int(np.ceil(frac_output_steps))
        safety_factor = 5.0
        max_limit = int(samples * (input_steps + frac_output_steps) * safety_factor * dt)

        print(f'Downloading dataset {system} with an LLE of {lle} and a dt of {dt}:')
        print(f'> ... {int((input_steps + frac_output_steps) * samples * safety_factor)} states')
        print(f'> ... {samples} samples with a safety_factor of {safety_factor}')
        print(f'> ... {input_steps} input_steps')
        print(f'> ... {output_steps} output_steps')

        file_name = f'data/{system}_{dt}_{lle}.csv'
        if os.path.exists(file_name) and not force:
            print(f'[ERROR] {file_name} already exists. Please remove/rename it or provide -f/--force to overwrite it.')
            exit(1)
        elif os.path.exists(file_name) and force:
            print(f'[INFO] overwriting dataset in {file_name} with generated data.')

        with open(file_name, 'wb') as f:
            if type(urls) is str:
                urls = (urls,)
            download_file(urls=urls, filename=file_name)
            print(50 * '-')
