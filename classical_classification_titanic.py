import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

from data_provider import load_titanic_pd
from utils import record_test_result_for_kaggle

np.random.seed(123123)

train_file = "train.csv"
test_file = "test.csv"

df_train, df_test, y_train, y_test = load_titanic_pd(train_file, test_file)

# model
print("-----\nFull features:")
model = RandomForestClassifier()
model.fit(df_train, y_train)
# Train score
print("Final train score: %f" % model.score(df_train, y_train))
# F1 score
print("Final F1 score: %f" % f1_score(y_train, model.predict(df_train)))

print("-----\nRecording test result...")
y_pred = model.predict(df_test)
print(y_pred)

record_test_result_for_kaggle(y_pred, submission_file="gender_submission.csv")


print("-----\nPart of features:")
# Get most important col
# 2 columns
col_num = 4
mvp_col = df_train.columns[sorted(range(len(model.feature_importances_)),
                                  key=lambda x: model.feature_importances_[x],
                                  reverse=True)[:col_num]].tolist()
print("Selected features: %s" % ",".join(mvp_col))

# Get only MVP columns
df_train_mvp = df_train[mvp_col].values
df_test_mvp = df_test[mvp_col].values

# another model
model_mvp = RandomForestClassifier()
model_mvp.fit(df_train_mvp, y_train)
# Train score
print("Final train score: %f" % model.score(df_train, y_train))
# F1 score
print("Final F1 score: %f" % f1_score(y_train, model.predict(df_train)))

print("-----\nRecording test result...")
y_pred = model.predict(df_test)
print(y_pred)

record_test_result_for_kaggle(y_pred, submission_file="gender_submission.csv")