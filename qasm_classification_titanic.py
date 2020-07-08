import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

from qiskit import IBMQ, BasicAer
from qiskit.circuit.library import ZZFeatureMap, TwoLocal
from qiskit.aqua import QuantumInstance
from qiskit.aqua.algorithms import VQC
from qiskit.aqua.components import variational_forms
from qiskit.aqua.components.optimizers import COBYLA, ADAM

from data_provider import load_titanic_pd

np.random.seed(123123)

train_file = "train.csv"
test_file = "test.csv"

df_train, y_train, df_test = load_titanic_pd(train_file, test_file)

# model
print("-----\nFull features:")
model = RandomForestClassifier()
model.fit(df_train, y_train)
# Train score
print("Final train score: %f" % model.score(df_train, y_train))
# F1 score
print("Final F1 score: %f" % f1_score(y_train, model.predict(df_train)))