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
from qiskit.aqua.components.optimizers import COBYLA

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

# Get most important col
# 2 columns
col_num = 2
mvp_col = df_train.columns[sorted(range(len(model.feature_importances_)),
                                  key=lambda x: model.feature_importances_[x],
                                  reverse=True)[:col_num]].tolist()
# mvp_col = ['tumor-size', 'breast-quad']
print("Selected features: %s" % ",".join(mvp_col))

# Get only MVP columns
df_train_q = df_train[mvp_col].values
df_test_q = df_test[mvp_col].values

# Choose balance 50 sample
# 25 pos, 25 neg

np.random.seed(777)

pos_sample = 30
neg_sample = 30

pos_label = np.argwhere(y_train == 1).reshape([-1])
chosen_pos_label_idx = pos_label[np.random.permutation(len(pos_label))[:pos_sample]]

neg_label = np.argwhere(y_train == 0).reshape([-1])
chosen_neg_label_idx = neg_label[np.random.permutation(len(neg_label))[:neg_sample]]

# Construct dict to feed QSVM
training_input = {
    0: df_train_q[y_train == 0],
    1: df_train_q[y_train == 1]
}

test_input = {
    0: df_test_q[y_test == 0],
    1: df_test_q[y_test == 1]
}
###### data prepared
print("data prepared.")
###### building quantum dude
# seed = 10598
seed = 1024
var_form = variational_forms.RYRZ(2)
feature_map = ZZFeatureMap(feature_dimension=len(mvp_col), reps=3, entanglement='linear')
qsvm = VQC(COBYLA(100), feature_map, var_form, training_input)
backend = BasicAer.get_backend('qasm_simulator')
quantum_instance = QuantumInstance(backend, shots=1024, seed_simulator=seed, seed_transpiler=seed,optimization_level=3)
result = qsvm.run(quantum_instance)

y_pred = qsvm.predict(df_test_q)[1]

print("Test acc: %f\nTest F1:%f" %
      (np.mean(y_pred == y_test), f1_score(y_test, y_pred)))