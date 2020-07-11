import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

def encoder_3bits_1qubit(df_q):
    data = []

    for col_name in df_q.columns:
        num_cat = len(np.unique(df_q[col_name]))
        num_bit = int(np.ceil(np.log2(num_cat)))

        print("Number of categories: %d\nNumber of bits: %d" % (num_cat, num_bit))

        # Padding to x3
        if num_bit % 3 != 0:
            num_bit = num_bit + (3 - (num_bit % 3))

        num_qubit = num_bit // 3

        features = []
        for size in df_q[col_name]:
            # Get last num_bit digit and reverse
            # 11 --> 001  | 011 --> 110 | 100

            all_b_st = f"{size:010b}"[-num_bit:][::-1]

            var_list = []
            for i in range(num_qubit):
                b_st = all_b_st[i * 3: (i + 1) * 3]

                # b_st = b_1, b_2, b_3 = \sqrt{3}r_x, \sqrt{3}r_y, \sqrt{3}r_z

                if b_st[0] == '1':
                    theta = np.arccos(1 / np.sqrt(3))
                else:
                    theta = np.arccos(-1 / np.sqrt(3))

                if b_st[1] == '1' and b_st[2] == '1':
                    varphi = np.pi / 4

                if b_st[1] == '1' and b_st[2] == '0':
                    varphi = 3 * np.pi / 4

                if b_st[1] == '0' and b_st[2] == '0':
                    varphi = -3 * np.pi / 4

                if b_st[1] == '0' and b_st[2] == '1':
                    varphi = -np.pi / 4

                var_list += [theta, varphi]

            features.append(var_list)
        #         print(size, var_list)
        data.append(np.array(features))

    data = np.concatenate(data, axis=1)
    return data


def select_features(df_train, y_train, df_test=None, y_test=None, feat_num=2, modelname="RandomForestClassifier"):
    # model
    print("-----\nFull features:")
    if modelname == "RandomForestClassifier":
        model = RandomForestClassifier()
    else:
        raise NameError("Only support RandomForestClassifier.")
    model.fit(df_train, y_train)
    # Train score
    print("Final train score: %f" % model.score(df_train, y_train))
    # F1 score
    print("Final F1 score: %f" % f1_score(y_train, model.predict(df_train)))

    print("-----\nMajority")
    print("Final train acc: %f\nFinal train F1:%f" % (
    np.mean(np.zeros_like(y_train) == y_train), f1_score(np.zeros_like(y_train), y_train)))

    # select important features
    col_num = feat_num
    mvp_col = df_train.columns[sorted(range(len(model.feature_importances_)),
                                      key=lambda x: model.feature_importances_[x],
                                      reverse=True)].tolist()
    print("Feature rank based on importance")
    print(mvp_col)
    mvp_col = mvp_col[:col_num]
    print("Selected features: %s" % ",".join(mvp_col))
    return mvp_col