import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

np.random.seed(123123)

# Read stuff
df = pd.read_csv('breast-cancer.data', header=None,
                 names=['target',
                        'age',
                        'menopause',
                        'tumor-size',
                        'inv-nodes',
                        'node-caps',
                        'deg-malig',
                        'breast',
                        'breast-quad',
                        'irradiat'])

# Encode to number (numerical)
for col in df.columns:
    le = preprocessing.LabelEncoder().fit(df[col])
    df[col] = le.transform(df[col])

# Split data and target
df_data = df.drop(['target'], axis=1)
y = df.target

# Split train, test
df_train, df_test, y_train, y_test = train_test_split(df_data, y.values, test_size=0.2)

# model
print("-----\nFull features:")
model = RandomForestClassifier()
model.fit(df_train, y_train)
# Test score
print("Test score: %f" % model.score(df_test, y_test))
# F1 score
print("F1 score: %f" % f1_score(y_test, model.predict(df_test)))


print("-----\nPart of features:")
# Get most important col
# 2 columns
col_num = 2
mvp_col = df_train.columns[sorted(range(len(model.feature_importances_)),
                                  key=lambda x: model.feature_importances_[x],
                                  reverse=True)[:col_num]].tolist()
# mvp_col = ['tumor-size', 'breast-quad']
print("Selected features: %s" % ",".join(mvp_col))

# Get only MVP columns
df_train_mvp = df_train[mvp_col].values
df_test_mvp = df_test[mvp_col].values

# another model
model_mvp = RandomForestClassifier()
model_mvp.fit(df_train_mvp, y_train)
# Test score
print("Test score: %f" % model_mvp.score(df_test_mvp, y_test))
# F1 score
print("F1 score: %f" % f1_score(y_test, model_mvp.predict(df_test_mvp)))