import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import LocalOutlierFactor
from preprocess import TextData


NUMBER_THRESHOLD = {'title_num': 50, 'content_num': 5000, 'image_num': 10, 'video_num': 5, 'link_num': 10}

CHRISTMAS_DAY = [(25, 12, 2013), (25, 12, 2014), (25, 12, 2015)]
NEW_YEAR_DAY = [(1, 1, 2013), (1, 1, 2014), (1, 1, 2015)]
# THANKS_GIVING_DAY = [(11, 28, 2013), (11, 27, 2014), (11, 26, 2014)]
# MOTHERS_DAY = [(5, 12, 2013), (5, 11, 2014), (5, 10, 2015)]

HOLIDAY = CHRISTMAS_DAY + NEW_YEAR_DAY


def get_feature(x_train, x_test, y_train, args):
    print('Transform data to feature...')

    train_size = x_train.shape[0]
    x = np.concatenate((x_train, x_test), axis=0)

    if args.show_statistics:
        show_statistics_of_data(x)

    x = reset_data_by_threshold(x)
    x = add_holiday_feature(x)

    x_train, x_test = x[:train_size], x[train_size:]

    if args.remove_outlier:
        x_train, y_train = remove_outliers(x_train, y_train)

    if args.scalar_data:
        x_train, x_test = scalar_data(x_train, x_test)

    return x_train, x_test, y_train


def scalar_data(x_train, x_test):
    scalar_model = MinMaxScaler()
    x = scalar_model.fit_transform(np.concatenate((x_train, x_test), axis=0))
    x_train = x[:x_train.shape[0]]
    x_test = x[x_train.shape[0]:]

    return x_train, x_test


def remove_outliers(x_train, y_train) -> (np.ndarray, np.ndarray):
    outlier_model = LocalOutlierFactor(n_neighbors=5)

    indices = outlier_model.fit_predict(x_train) == 1
    x_train, y_train = x_train[indices], y_train[indices].reshape(-1)

    return x_train, y_train


def show_statistics_of_data(x):
    attrs = TextData().get_attr_names()

    for i in range(len(attrs)):
        print('%s: mean = %.2f, var = %.2f, max = %d, min = %d' %
              (attrs[i], x[:, i].mean(), x[:, i].var(), x[:, i].max(), x[:, i].min()))


def reset_data_by_threshold(x) -> np.ndarray:
    attrs = TextData().get_attr_names()

    for i, attr in enumerate(attrs):
        if attr in NUMBER_THRESHOLD:
            x_column = x[:, i].copy()

            x_column[x_column <= NUMBER_THRESHOLD[attr]] = 1
            x_column[x_column > NUMBER_THRESHOLD[attr]] = 0

            x[:, i] = x_column

    return x


def add_holiday_feature(x):
    holiday_feature = [int(is_in_holiday(x_vec)) for x_vec in x]
    holiday_feature = np.array(holiday_feature).reshape(-1, 1)

    x = np.concatenate((x, holiday_feature), axis=1)
    return x


def is_in_holiday(data_vec: np.ndarray) -> bool:
    assert data_vec.ndim == 1
    tmp = TextData()

    day = data_vec[tmp.data_index('day')]
    month = data_vec[tmp.data_index('month')]
    year = data_vec[tmp.data_index('year')]

    return (day, month, year) in HOLIDAY
