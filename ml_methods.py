from torch.utils.data import Dataset
import glob
import pandas as pd
import json
import numpy as np
import torch

# split the lists of wavelengths and optical properties into rows of single values
def explode(df, lst_cols, fill_value=''):
    # source: https://stackoverflow.com/questions/45846765/efficient-way-to-unnest-explode-multiple-list-columns-in-a-pandas-dataframe
    # make sure `lst_cols` is a list
    if lst_cols and not isinstance(lst_cols, list):
        lst_cols = [lst_cols]
    # all columns except `lst_cols`
    idx_cols = df.columns.difference(lst_cols)

    # calculate lengths of lists
    lens = df[lst_cols[0]].str.len()

    if (lens > 0).all():
        # ALL lists in cells aren't empty
        return pd.DataFrame({
            col:np.repeat(df[col].values, df[lst_cols[0]].str.len())
            for col in idx_cols
        }).assign(**{col:np.concatenate(df[col].values) for col in lst_cols}) \
          .loc[:, df.columns]
    else:
        # at least one list in cells is empty
        return pd.DataFrame({
            col:np.repeat(df[col].values, df[lst_cols[0]].str.len())
            for col in idx_cols
        }).assign(**{col:np.concatenate(df[col].values) for col in lst_cols}) \
          .append(df.loc[lens==0, idx_cols]).fillna(fill_value) \
          .loc[:, df.columns]

# load JSON files into pandas dataframe
def make_from_JSON(num_layers, path, wavelength_range, interval):
    wavelength_l, wavelength_h = wavelength_range
    num_json = len(glob.glob1(path,'**'))
    cols = []
    filenames = []
    for i in range(1, num_layers+1):
        cols.append('mat_' + str(i))
    for i in range(1, num_layers+1):
        cols.append('t_' + str(i))
    cols.append('transmission')
    cols.append('reflection')
    cols.append('wavelength')
    df = pd.DataFrame(columns=cols, index = range(num_json))
    file_idx = 0
    ID = 100 + num_layers - 2
    for filename in glob.glob(path + str(ID) + '*'):
        data = json.load(open(str(filename)))
        for i in range(1, num_layers+1):
            df.loc[file_idx]['mat_' + str(i)] = data['geometry']['materials'][i-1]
            df.loc[file_idx]['t_' + str(i)] = data['geometry']['thickness'][i-1]
        df.loc[file_idx]['transmission'] = []
        df.loc[file_idx]['reflection'] = []
        df.loc[file_idx]['wavelength'] = []
        filenames.append(filename.split('\\')[-1])
        for i in range(len(data['opticalProp']['wavelength_nm'])):
            if (data['opticalProp']['wavelength_nm'][i] - wavelength_l)% interval == 0:
                df.loc[file_idx]['transmission'].append(data['opticalProp']['transmission'][i])
                df.loc[file_idx]['reflection'].append(data['opticalProp']['reflection'][i])
                df.loc[file_idx]['wavelength'].append(data['opticalProp']['wavelength_nm'][i])
        df.loc[file_idx]['transmission'] = np.array(df.loc[file_idx].transmission)
        df.loc[file_idx]['reflection'] = np.array(df.loc[file_idx].reflection)
        df.loc[file_idx]['wavelength'] = np.array(df.loc[file_idx].wavelength)
        file_idx += 1
    return df, filenames

# encode the materials combinations into one-hot vectors
def one_hot_encoding(num_layers, df):
    cols = []
    prefixes = []
    for i in range(1, num_layers+1):
        cols.append('mat_' + str(i))
        prefixes.append('mat' + str(i))
    dummies = pd.get_dummies(df, columns=cols, prefix=prefixes)
    return dummies

# rescale the features into [0, 1]
def scaler(num_layers, thickness_range, wavelength_range, df):
    thickness_l, thickness_h = thickness_range
    wavelength_l, wavelength_h = wavelength_range
    for i in range(1, num_layers+1):
        df['t_' + str(i)] = (df['t_' + str(i)] - thickness_l) / (thickness_h - thickness_l)
    df['wavelength'] = (df['wavelength'] - wavelength_l) / (wavelength_h - wavelength_l)
    return df

# build a dataframe with given properties
# returns the dataframe and a list of filenames
def build_dataset(num_layers, paths, thickness_range, wavelength_range, interval, rescale, split):
    df_all, filenames_all = [], []
    for path in paths:
        df, filenames = make_from_JSON(num_layers, path, wavelength_range, interval)
        df_all.append(df)
        filenames_all += filenames
    df_all = pd.concat(df_all, ignore_index=True)
    if split:
        df_all = explode(df_all, ['transmission', 'reflection', 'wavelength'])
    df_all = one_hot_encoding(num_layers, df_all)
    if rescale:
        df_all = scaler(num_layers, thickness_range, wavelength_range, df_all)
    return df_all, filenames_all

# build an iterrable dataset with given properties
class EntireSpectrumSplit(Dataset):
    def __init__(self, num_layers, root_dirs, thickness_range=(100,500), wavelength_range=(200,900), interval=5, rescale=True, split=True):
        self.split = split
        self.dataframe, self.filenames = build_dataset(num_layers, root_dirs, thickness_range, wavelength_range, interval, rescale, split)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.loc[idx]
        transmission = row['transmission']
        reflection = row['reflection']
        if self.split:
            row = torch.Tensor(row.drop(['transmission', 'reflection']))
        else:
            row = torch.Tensor(row.drop(['transmission', 'reflection', 'wavelength']))
        return {'features': row, 'transmission': transmission, 'reflection': reflection}