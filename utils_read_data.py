from typing import List

import h5py
import pandas as pd


# Open the HDF5 file and collect 1D-compatible datasets
def read_data_into_dataframe(file_path: str, fields_of_interest:List[str])->pd.DataFrame:
    '''
    Reads data from an HDF5 file and returns a DataFrame containing specified fields of interest.
    :param file_path: The path to the h5 file.
    :param fields_of_interest: Fields that should be read
    :return: 
    A pandas dataframe containing the specified fields of interest.
    '''
    data = {}
    with h5py.File(file_path, 'r') as f:
        for key in f.keys():
            if key not in fields_of_interest:
                continue
            key_data = f[key][()]
            try:
                # Check if it's a vector 1d vector
                if key_data.ndim == 1 or key_data.ndim == 0:
                    data[key] = key_data
                elif key_data.ndim == 2:
                    print(key_data.shape)
                    for i in range(key_data.shape[1]):
                        data[f"{key}_{i}"] = key_data[:, i]
                else:
                    raise Exception(f"Cannot process column {key}")
            except Exception as e:
                print(f"Skipping {key}: {e}")
    df = pd.DataFrame(data)
    return df

