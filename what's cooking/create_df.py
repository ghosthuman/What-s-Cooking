import json
import pandas as pd
import numpy as np



def drop_na(df):
    return df.dropna()


def make_df(file_path):
    with open(file_path) as data_file:
        data = json.load(data_file)

    ingred_list = set()
    for item in data:
        ingred_list |= set(item['ingredients'])

    ingred_list = list(ingred_list)

    ingred_dict = dict()
    for i, ingredient in enumerate(ingred_list):
        ingred_dict[ingredient] = i

    feature_df_np = np.ones((len(data), len(ingred_list)), dtype=np.int8) * -1

    label_df_np = []
    for i, item in enumerate(data):
        label_df_np.append(item['cuisine'])
        for ingredient in item['ingredients']:
            feature_df_np[i][ingred_dict[ingredient]] = 1

    df_pd = pd.DataFrame(feature_df_np, columns=ingred_list)
    df_pd['cuisine'] = label_df_np

    # Drop NAN
    # print '[INFO] Before dropna, Shape:', df_pd.shape
    # df_pd = drop_na(df_pd)
    # print '[INFO] Ater dropna, Shape:', df_pd.shape

    one_hot = pd.get_dummies(df_pd['cuisine'])
    one_hot = one_hot.values.tolist()

    df_pd = df_pd.drop('cuisine', axis=1)
    df_pd['labels'] = one_hot

    # Calculate ones
    # summms = df_pd.sum(axis=1) + 6715
    # print summms.sum(axis=0)

    # df_pd.to_csv('feature_sheet.csv')
    print 'Feature sheet generated'

    print df_pd.shape
    rows = df_pd.shape[0]

    train_df = df_pd.iloc[:int(0.8*rows), :]
    test_df = df_pd.iloc[int(0.8*rows):]

    print train_df.shape
    print test_df.shape

    train_features = list(train_df.iloc[:, :-1])
    train_labels = list(train_df.iloc[:, -1])
    test_features = list(test_df.iloc[:, :-1])
    test_labels = list(test_df.iloc[:, -1])

    return train_features,train_labels,test_features,test_labels



