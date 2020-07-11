import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from qiskit import IBMQ, BasicAer
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.circuit.library import ZZFeatureMap, TwoLocal
from qiskit.aqua import QuantumInstance
from qiskit.aqua.algorithms import VQC
from qiskit.aqua.components import variational_forms
from qiskit.aqua.components.optimizers import COBYLA, ADAM, SPSA

# setup aqua logging
import logging
from qiskit.aqua import set_qiskit_aqua_logging
set_qiskit_aqua_logging(logging.DEBUG)  # choose INFO, DEBUG to see the log

from data_provider import load_titanic_pd
from utils import record_test_result_for_kaggle
from quantum_utils import select_features, encoder_3bits_1qubit


def sampling_dataset(df_train, y_train, df_test, y_test=None, pos_sample=None, neg_sample=None):
    np.random.seed(777)

    if pos_sample and neg_sample:
        y_train = np.array(y_train)
        pos_label = np.argwhere(y_train == 1).reshape([-1])
        chosen_pos_label_idx = pos_label[np.random.permutation(len(pos_label))[:pos_sample]]

        neg_label = np.argwhere(y_train == 0).reshape([-1])
        chosen_neg_label_idx = neg_label[np.random.permutation(len(neg_label))[:neg_sample]]

        print("Postive sample num: %d" % len(pos_label))
        print("Negative sample num: %d" % len(neg_label))

        # Construct dict to feed QSVM
        training_input = {
            0: df_train[chosen_pos_label_idx],
            1: df_train[chosen_neg_label_idx]
        }
    else:
        # Construct dict to feed QSVM
        training_input = {
            0: df_train[y_train == 0],
            1: df_train[y_train == 1]
        }
    # print(training_input)
    test_input = df_test
    return training_input, test_input


if __name__ == "__main__":
    USE_ENCODER = True
    FEAT_NUM = 4
    ITER = 100
    POS_SAMPLE = None
    NEG_SAMPLE = None
    OPTIMIZER = SPSA # SPSA
    VAR_FORM = variational_forms.RYRZ
    FEAT_MAP = "ZZMap"  # or "Ours"

    BACKEND = "simulator"  # or "real"

    np.random.seed(123123)

    train_file = "train.csv"
    test_file = "test.csv"

    df_train, y_train, df_test = load_titanic_pd(train_file, test_file)

    mvp_col = select_features(df_train, y_train, feat_num=FEAT_NUM, modelname="RandomForestClassifier")

    df_train_q, df_test_q = df_train[mvp_col], df_test[mvp_col]

    # encode with 3-1
    if USE_ENCODER:
        print("encoding...")
        df_train_q = encoder_3bits_1qubit(df_train_q)
        df_test_q = encoder_3bits_1qubit(df_test_q)
    else:
        df_train_q = df_train_q.values
        df_test_q = df_test_q.values

    # Choose balance 200 sample
    # 100 pos, 100 neg
    training_input, test_input = sampling_dataset(df_train_q, y_train, df_test_q,
                                                  pos_sample=POS_SAMPLE, neg_sample=NEG_SAMPLE)

    ###### data prepared
    print("data prepared.")
    ###### building quantum dude
    seed = 10598
    var_form = VAR_FORM(num_qubits=df_train_q.shape[1] // 2, depth=4)

    if FEAT_MAP == "ZZMap":
        feature_map = ZZFeatureMap(feature_dimension=len(mvp_col), reps=3, entanglement='linear')
    elif FEAT_MAP == "Ours":
        X = [Parameter(f'x[{i}]') for i in range(df_train_q.shape[1])]

        var_form = VAR_FORM(df_train_q.shape[1] // 2)

        qc = QuantumCircuit(df_train_q.shape[1] // 2)

        for i in range(df_train_q.shape[1] // 2):
            qc.u3(X[2 * i], X[2 * i + 1], 0, i)  # Encoder

        feature_map = qc  # + tmp1 + tmp2
    else:
        raise NameError("Unknown feature map.")

    qsvm = VQC(OPTIMIZER(ITER), feature_map, var_form, training_input)

    if BACKEND == "simulator":
        backend = BasicAer.get_backend('qasm_simulator')
    elif BACKEND == "real":
        provider = IBMQ.get_provider()
        backend = provider.get_backend('ibmq_london')
    else:
        raise NameError("Plz choose simulator/real")

    quantum_instance = QuantumInstance(backend, shots=1024, seed_simulator=seed, seed_transpiler=seed, optimization_level=3)
    result = qsvm.run(quantum_instance)

    y_pred = qsvm.predict(df_train_q)[1]
    print("Final train acc: %f\nFinal train F1:%f" % (np.mean(y_pred == y_train), f1_score(y_pred, y_train)))

    y_pred = qsvm.predict(df_test_q)[1]
    print(y_pred)

    record_test_result_for_kaggle(y_pred, submission_file="quantum_submission.csv")