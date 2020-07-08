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
from utils import record_test_result_for_kaggle

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

col_num = 2
mvp_col = df_train.columns[sorted(range(len(model.feature_importances_)),
                                  key=lambda x: model.feature_importances_[x],
                                  reverse=True)[:col_num]].tolist()

print("Selected features: %s" % ",".join(mvp_col))

df_train_q = df_train[mvp_col].values
df_test_q = df_test[mvp_col].values

# Choose balance 50 sample
# 25 pos, 25 neg

np.random.seed(777)

pos_sample = 100
neg_sample = 100

y_train = np.array(y_train)
pos_label = np.argwhere(y_train == 1).reshape([-1])
chosen_pos_label_idx = pos_label[np.random.permutation(len(pos_label))[:pos_sample]]

neg_label = np.argwhere(y_train == 0).reshape([-1])
chosen_neg_label_idx = neg_label[np.random.permutation(len(neg_label))[:neg_sample]]

print("Postive sample num: %d" % len(pos_label))
print("Negative sample num: %d" % len(neg_label))
# Construct dict to feed QSVM
training_input = {
    0: df_train_q[chosen_pos_label_idx],
    1: df_train_q[chosen_neg_label_idx]
}

test_input = df_test_q

###### data prepared
print("data prepared.")
###### building quantum dude
seed = 10598
# seed = 1024
var_form = variational_forms.RYRZ(2)
feature_map = ZZFeatureMap(feature_dimension=len(mvp_col), reps=2, entanglement='linear')
qsvm = VQC(ADAM(100), feature_map, var_form, training_input)
backend = BasicAer.get_backend('qasm_simulator')
quantum_instance = QuantumInstance(backend, shots=1024, seed_simulator=seed, seed_transpiler=seed, optimization_level=3)
result = qsvm.run(quantum_instance)

y_pred = qsvm.predict(df_train_q)[1]
print("Final train acc: %f\nFinal train F1:%f" % (np.mean(y_pred == y_train), f1_score(y_pred, y_train)))

y_pred = qsvm.predict(df_test_q)[1]
# print(y_pred)

record_test_result_for_kaggle(y_pred, submission_file="quantum_submission.csv")