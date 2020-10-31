import pickle
import pandas as pd
import numpy as np
from preprocess import preprocess_data


def load_train_data(data_path):
    print('Load train data...')

    x, y = load_train_csv(data_path) if data_path[-3:] == 'csv' else load_train_pickle(data_path)
    return x, y


def load_train_csv(csv_path):
    df = pd.read_csv(csv_path)

    raw_data_list = df['Page content']
    x = preprocess_data(raw_data_list)

    y = df['Popularity'].to_numpy()[:x.shape[0]].reshape(-1, 1)
    y[y == -1] = 0

    file = open('%s.pkl' % (csv_path[:-4]), 'wb')
    pickle.dump(np.concatenate((x, y), axis=1), file)

    return x, y


def load_train_pickle(pickle_path):
    file = open(pickle_path, 'rb')
    data = pickle.load(file)

    x = data[:, :-1]
    y = data[:, -1]

    return x, y


def load_test_data(data_path):
    print('Load test data...')
    x, ids = load_test_csv(data_path) if data_path[-3:] == 'csv' else load_test_pickle(data_path)
    return x, ids


def load_test_csv(csv_path):
    df = pd.read_csv(csv_path)

    raw_data_list = df['Page content']
    x = preprocess_data(raw_data_list)

    ids = df['Id'].to_numpy().reshape(-1, 1)

    file = open('%s.pkl' % (csv_path[:-4]), 'wb')
    pickle.dump(np.concatenate((x, ids), axis=1), file)

    return x, ids


def load_test_pickle(pickle_path):
    file = open(pickle_path, 'rb')
    data = pickle.load(file)

    x = data[:, :-1]
    ids = data[:, -1]

    return x, ids


def export_test_result(file_name, y_predict, ids):
    ids = ids.reshape(-1, 1)
    y_predict = y_predict.reshape(-1, 1)

    output = np.concatenate((ids, y_predict), axis=1)

    df = pd.DataFrame(output, columns=['Id', 'Popularity'])
    df.Id = df.Id.astype(int)
    df.to_csv(file_name, index=False, header=True)
