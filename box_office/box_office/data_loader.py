"""Loading data"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer
import torch


def load_tensor_data(fileloc):
    """
    Helper function to load the actors data, filter by criterias of 1 million
    min. revenue and actors in at least 20 movies. Returns actor matrix and
    logNormal revenue as torch tensors.
    """
    
    data_actors = pd.read_csv(fileloc, index_col=0)
    X = data_actors.iloc[:, 2:]
    X_data = torch.Tensor(X.to_numpy(dtype='float32'))
    transformer = FunctionTransformer(np.log1p, validate=True)
    data_actors["log_revenue"] = transformer.transform(
        data_actors["revenue"].values.reshape(-1, 1)
    )
    Y_data = torch.Tensor(
        data_actors["log_revenue"].to_numpy().reshape(X.shape[0], 1)
    )
    
    cols_keep = ['Judi Dench', 'Cobie Smulders']
    cols_20 = ['title_x', 'revenue', 'log_revenue']
    for col in data_actors.columns[2:-1]:
        
        if col in cols_keep: 
            continue
        elif np.sum(data_actors[col]) >= 20:
            cols_20.append(col)

    data_million = data_actors[cols_20+cols_keep]
    must_keep = data_million[(data_million["Judi Dench"]==1) | (data_million["Cobie Smulders"]==1)]
    data_million = data_million[data_million["revenue"] > 1000000]
    
#     X_all = data_million[
#         data_million.columns.difference(
#             ['title_x', 'revenue', 'log_revenue']
#         )
#     ].append(must_keep[must_keep.columns.difference(
#             ['title_x', 'revenue', 'log_revenue'])], ignore_index = True)
    
    X_all = data_million[
    data_million.columns.difference(
        ['revenue', 'log_revenue']
    )
].append(must_keep[must_keep.columns.difference(
        ['revenue', 'log_revenue'])], ignore_index = True)
    
    y_all = data_million['revenue'].append(must_keep['revenue'])
    
    x_train = X_all
    y_train = y_all
    x_train_tensors = torch.tensor(x_train.drop("title_x", axis= 1 ).to_numpy(dtype='float32'))
    y_train_tensors = torch.tensor(
        y_all.to_numpy(dtype='float32')
    )
    
    cols = list(x_train.columns)
    cols = [cols[-1]] + cols[:-1]
    x_train = x_train[cols]
    
    return x_train_tensors, y_train_tensors, x_train.columns, x_train
