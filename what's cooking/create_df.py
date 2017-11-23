import json
import pandas as pd
import numpy as np


def drop_na(df):
    return df.dropna()


def one_hot_to_real(one_index):
    sorted_ingred = [u'brazilian', u'british', u'cajun_creole', u'chinese', u'filipino', u'french', u'greek', u'indian',
                     u'irish', u'italian', u'jamaican', u'japanese', u'korean', u'mexican', u'moroccan', u'russian',
                     u'southern_us', u'spanish', u'thai', u'vietnamese']

    return sorted_ingred[one_index]


def make_train_test_df(file_path):
    with open(file_path) as data_file:
        data = json.load(data_file)

    ingred_list = set()
    for item in data:
        ingred_list |= set(item['ingredients'])

    ingred_list = list(ingred_list)
    print len(ingred_list)

    ingred_dict = dict()
    for i, ingredient in enumerate(ingred_list):
        ingred_dict[ingredient] = i

    feature_df_np = np.zeros((len(data), len(ingred_list)), dtype=np.int8)

    label_df_np = []
    for i, item in enumerate(data):
        label_df_np.append(item['cuisine'])
        for ingredient in item['ingredients']:
            feature_df_np[i][ingred_dict[ingredient]] = 1

    df_pd = pd.DataFrame(feature_df_np, columns=ingred_list)
    df_pd['cuisine'] = label_df_np

    df_pd = df_pd.sample(frac=1, random_state=1)

    # Drop NAN
    # print '[INFO] Before dropna, Shape:', df_pd.shape
    # df_pd = drop_na(df_pd)
    # print '[INFO] Ater dropna, Shape:', df_pd.shape

    # sorted_ingred = sorted(list(set(df_pd['cuisine'].tolist())))

    one_hot = pd.get_dummies(df_pd['cuisine'])
    one_hot = one_hot.values.tolist()

    df_pd = df_pd.drop('cuisine', axis=1)
    df_pd['labels'] = one_hot

    # print len(df_pd['labels'][0])

    # Calculate ones
    # summms = df_pd.sum(axis=1) + 6715
    # print summms.sum(axis=0)

    # df_pd.to_csv('feature_sheet.csv')
    # print 'Feature sheet generated'

    rows = df_pd.shape[0]

    split = 0.95
    train_df = df_pd.iloc[:int(split*rows) + 1, :]
    test_df = df_pd.iloc[int(split*rows) + 1:]

    # print train_df.shape
    # print test_df.shape
    # print train_df.head()
    train_features = list(np.array(train_df.iloc[:, :-1]))
    train_labels = list(np.array(train_df.iloc[:, -1]))
    test_features = list(np.array(test_df.iloc[:, :-1]))
    test_labels = list(np.array(test_df.iloc[:, -1]))
    print 'Size of train set: {} \n Size of test set: {}'.format(train_df.shape[0], test_df.shape[0])

    return train_features, train_labels, test_features, test_labels

def init_df():
    with open('train.json') as data_file:
        data = json.load(data_file)

    ingred_list = set()
    for item in data:
        ingred_list |= set(item['ingredients'])

    ingred_list = list(ingred_list)
    print len(ingred_list)

    ingred_dict = dict()
    for i, ingredient in enumerate(ingred_list):
        ingred_dict[ingredient] = i

    feature_df_np = np.zeros((len(data), len(ingred_list)), dtype=np.int8)

    df_pd = pd.DataFrame(feature_df_np, columns=ingred_list)
    return ingred_list, ingred_dict


def make_df(filename):
    with open(filename) as data_file:
        data = json.load(data_file)

    ingred_list, ingred_dict = init_df()
    feature_df_np = np.zeros((len(data), len(ingred_list)), dtype=np.int8)

    df_pd = pd.DataFrame(feature_df_np, columns=ingred_list)

    if filename == 'train.json':
        label_df_np = []
        for i, item in enumerate(data):
            label_df_np.append(item['cuisine'])
            for ingredient in item['ingredients']:
                feature_df_np[i][ingred_dict[ingredient]] = 1
        df_pd = df_pd.sample(frac=1, random_state=1)
        df_pd['cuisine'] = label_df_np

        # Drop NAN
        # print '[INFO] Before dropna, Shape:', df_pd.shape
        # df_pd = drop_na(df_pd)
        # print '[INFO] Ater dropna, Shape:', df_pd.shape

        # sorted_ingred = sorted(list(set(df_pd['cuisine'].tolist())))

        one_hot = pd.get_dummies(df_pd['cuisine'])
        one_hot = one_hot.values.tolist()

        df_pd = df_pd.drop('cuisine', axis=1)
        df_pd['labels'] = one_hot

        features = list(np.array(df_pd.iloc[:, :-1]))
        labels = list(np.array(df_pd.iloc[:, -1]))
        return features, labels

    elif filename == 'test.json':
        id = []
        for i, item in enumerate(data):
            id.append(item['id'])
            for ingredient in item['ingredients']:
                try:
                    feature_df_np[i][ingred_dict[ingredient]] = 1
                except:
                    pass

        # df_pd['id'] = id
        # print np.sum(df_pd, axis=0)
        # print df_pd.head()
        features = list(np.array(df_pd))
        return features, id


if __name__ == "__main__":
    # tra_X, y = make_df('train.json')
    tst_X = make_df('test.json')
