import random
import numpy as np

from argparse import ArgumentParser
from sklearn.model_selection import train_test_split

from data import load_train_data, load_test_data, export_test_result
from feature import get_feature
from model import *


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('-tr', '--train_data', type=str, default='data/train.pkl', help='train data path')
    parser.add_argument('-te', '--test_data', type=str, default='data/test.pkl', help='test data path')
    parser.add_argument('-out', '--output_data', type=str, default='result.csv', help='export result path')
    parser.add_argument('-m', '--model', type=str, default='gb', help='svm, lr, gb')
    parser.add_argument('-c', type=float, default=10, help='svm c')
    parser.add_argument('-k', '--kernel', type=str, default='rbf', help='svm rbf, linear, poly')
    parser.add_argument('-ro', '--remove_outlier', action='store_true', help='use local outlier factor')
    parser.add_argument('-sd', '--scalar_data', action='store_true', help='min max scalar data')
    parser.add_argument('-ss', '--show_statistics', action='store_true', help='show statistics of data')

    return parser.parse_args()


def show_configuration(args):
    models = {'svm': 'SVM', 'lr': 'Logistic Regression', 'gb': 'Gradient Boosting Classifier'}
    print('Train data path:', args.train_data)
    print('Test data path:', args.test_data)
    print('Output result csv path:', args.output_data)
    print('Classifier model:', models[args.model])
    print('Show statistics of data:', args.show_statistics)
    print('Min max scalar data:', args.scalar_data)
    print('Remove outliers of data:', args.remove_outlier)


def train(args):
    print('=' * 40)
    print('[Training]')

    x, y = load_train_data(args.train_data)
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2)

    x_train, x_val, y_train = get_feature(x_train, x_val, y_train, args)

    classifier = get_model(args)
    classifier.fit(x_train[:], y_train[:])

    y_predict = classifier.predict(x_val)
    accuracy = np.sum(y_predict == y_val.reshape(-1)) / y_val.shape[0]

    print('Validation accuracy = %.6f\n' % accuracy)


def test(args):
    print('=' * 40)
    print('[Testing]')

    x_train, y_train = load_train_data(args.train_data)
    x_test, ids = load_test_data(args.test_data)

    x_train, x_test, y_train = get_feature(x_train, x_test, y_train, args)

    print('Fitting model...', end='')
    classifier = get_model(args)
    classifier.fit(x_train[:], y_train[:])
    print('Done.')

    y_predict = classifier.predict_proba(x_test)[:, 1]

    print('Export result...', end='')
    export_test_result(file_name=args.output_data, ids=ids, y_predict=y_predict)
    print('Done.', 'Saved result in "%s"' % args.output_data)


if __name__ == '__main__':
    manual_seed = 100
    random.seed(manual_seed)
    np.random.seed(manual_seed)

    args = parse_arguments()
    show_configuration(args)

    train(args)
    test(args)
