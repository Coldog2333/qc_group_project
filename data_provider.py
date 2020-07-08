import numpy as np
import pandas as pd
from sklearn import preprocessing

def load_titanic_pd(train_file, test_file):
    # get header
    with open(train_file) as f:
        train_header_names = f.readline().strip().split(",")
    with open(test_file) as f:
        test_header_names = f.readline().strip().split(",")

    df_train = pd.read_csv(train_file, header=0, names=train_header_names)
    df_test = pd.read_csv(test_file, header=0, names=test_header_names)

    df_train = remove_useless_feature(df_train)
    df_test = remove_useless_feature(df_test)

    df_train = df_train.fillna(value="<NULL>")
    df_test = df_test.fillna(value="<NULL>")

    df_train = numerical_df(df_train)
    df_test = numerical_df(df_test)


    df_train = fill_NULL_df(df_train)
    df_test = fill_NULL_df(df_test)

    print(df_train)
    y_train = df_train.Survived
    df_train = df_train.drop(["Survived"], axis=1)
    return df_train, y_train, df_test


def numerical_df(df):
    # Encode to number (numerical)
    nan_bool_map = df.isna()
    for col in df.columns:
        if col in ["Sex", "Embarked"]:
            numer_map = {}
            for i in range(len(df[col])):
                if nan_bool_map[col][i]:  # skip NULL
                    continue
                if df[col][i] not in numer_map.keys():
                    numer_map[df[col][i]] = len(numer_map)
                df[col][i] = numer_map[df[col][i]]
    return df

def remove_useless_feature(df):
    # They are not so related to the survived possibility
    df = df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1)
    return df


def get_sample_p(df_col, skip="<NULL>"):
    frequency = {}
    valid_num = 0
    for i in range(len(df_col)):
        if df_col[i] == skip:
            continue
        if df_col[i] not in frequency.keys():
            frequency[df_col[i]] = 1
        else:
            frequency[df_col[i]] += 1
        valid_num += 1

    pre = 0
    for k in frequency.keys():
        frequency[k] /= valid_num + pre
        pre = frequency[k]

    return frequency

def fill_NULL_df(df):
    for col in df.columns:
        if col == "Sex":
            frequency = get_sample_p(df[col])
            for i in range(len(df[col])):
                if df[col][i] == "<NULL>":
                    rand = np.random.rand()
                    if rand < frequency[0]:
                        df[col][i] = 0
                    else:
                        df[col][i] = 1

        elif col == "Embarked":
            frequency = get_sample_p(df[col])
            for i in range(len(df[col])):
                if df[col][i] == "<NULL>":
                    rand = np.random.rand()
                    if rand < frequency[0]:
                        df[col][i] = 0
                    elif rand < frequency[1]:
                        df[col][i] = 1
                    else:
                        df[col][i] = 2
        elif col == "Fare":
            value = np.mean(df[col][df[col] != "<NULL>"])
            for i in range(len(df[col])):
                if df[col][i] == "<NULL>":
                    df[col][i] = value
        else:
            value = np.round(np.mean(df[col][df[col] != "<NULL>"]))
            for i in range(len(df[col])):
                if df[col][i] == "<NULL>":
                    df[col][i] = value
    return df




# def load_csv(csv_file, header=True, names=""):
#     with open(csv_file) as f:
#         if header:
#             names = f.readline().strip().split(",")
#         data = [line.strip().split(",") for line in f.readlines()]
#     name2index_map = {}
#     index2name_map = {}
#     # you can do
#     # >> data[name2index_map["Age"]]
#     # with this.
#     for i in range(len(names)):
#         name2index_map[names[i]] = i
#         index2name_map[i] = names[i]
#     return data, name2index_map, index2name_map
#
#
# def load_titanic(train_file, test_file):
#     train_data, train_name2index_map, train_index2name_map = load_csv(train_file, header=True)
#     test_data, test_name2index_map, test_index2name_map = load_csv(test_file, header=True)
#
#     train_data = np.array(train_data)
#     test_data = np.array(test_data)
#
#     train_data, train_name2index_map, train_index2name_map = remove_useless_column(train_data, train_name2index_map, train_index2name_map)
#     test_data, test_name2index_map, test_index2name_map = remove_useless_column(test_data, test_name2index_map, test_index2name_map)
#
#     train_data = numerical(train_data)
#     test_data = numerical(test_data)
#
#     train_data = fill_NULL(train_data)
#     test_data = fill_NULL(test_data)
#
#
# def remove_useless_column(data, name2index_map, index2name_map):
#     # useless_column = ["PassengerId", "Name", "Ticket", "Cabin"]
#     useless_column = [0, 3, 8, 10]
#     new_name2index_map = {}
#     new_index2name_map = {}
#     column_list = []
#     for col in range(data.shape[1]):
#         if col not in useless_column:
#             column_list.append(data[:, col])
#             new_index2name_map[len(new_name2index_map)] = index2name_map[col]
#             new_name2index_map[index2name_map[col]] = len(new_name2index_map)
#     column_list = np.array(column_list)
#     return column_list, new_name2index_map, new_index2name_map
#
#
# def numerical(data):
#     for col in range(data.shape[1]):
#         if col in [2, 9]:
#             numer_map = {}
#             for row in range(data.shape[0]):
#                 if data[row, col] == "":  # skip NULL
#                     continue
#                 if data[row, col] not in numer_map.keys():
#                     numer_map[data[row, col]] = len(numer_map)
#                 data[row, col] = numer_map[data[row, col]]
#         else:
#             for row in range(data.shape[0]):
#                 if data[row, col] == "":  # skip NULL
#                     continue
#                 else:
#                     data[row, col] = eval(data[row, col])
#     return data
#
#
# def fill_NULL(data):
#     for col in range(data.shape[1]):
#         if isinstance(data[0, col], str) or isinstance(data[0, col], int):
#             print(col)
#             print(data[:, col])
#             value = np.argmax(np.bincount(data[:, col]))  # fill with mode
#         elif isinstance(data[0, col], float):
#             value = np.mean(data[:, col] * (data[:, col] != ""))
#         else:
#             raise NameError("Unknown column type: %s" % str(type(data[0, col])))
#
#         for row in range(data.shape[0]):
#             if data[row, col] == "":
#                 data[row, col] = value
#
#     return data



if __name__ == "__main__":
    train_file = "train.csv"
    test_file = "test.csv"

    df_train, y_train, df_test = load_titanic_pd(train_file, test_file)