import itertools
import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import numpy as np
from qiskit.aqua.components import variational_forms
from qiskit.aqua.algorithms import QSVM, VQC
from qiskit.aqua.components.optimizers import COBYLA, SPSA
from qiskit.providers.aer import QasmSimulator
from qiskit.aqua import QuantumInstance
import logging
from qiskit.aqua import set_qiskit_aqua_logging
set_qiskit_aqua_logging(logging.DEBUG)  # choose INFO, DEBUG to see the log

from quantum_utils import CustomFeatureMap

parser = argparse.ArgumentParser(description='Choose METHOD')
parser.add_argument('--method', dest='method', type=str, default='qrac_map')
parser.add_argument('--dup', dest='dup', type=int, default=1)
parser.add_argument('--bit', dest='bit', type=int, default=3)
parser.add_argument('--seed', dest='seed', type=int, default=111)
args = parser.parse_args()

METHODS = ['qrac', 'qrac_map', 'naive', 'conv']

assert args.method in METHODS, f"The method {args.method} not exist"
assert args.bit % 3 == 0, f"number of bit should be x3"

METHOD = args.method
BIT = args.bit
DUP = args.dup



mapper = {
    '000':'000',
    '011':'001',
    '101':'010',
    '110':'011',
    '001':'100',
    '010':'101',
    '100':'110',
    '111':'111'
}



def main():
    num_bits = BIT

    x_train = []
    y_train = []

    for comb in itertools.product('01', repeat=num_bits):
        comb = [int(x) for x in comb]
        x_train.append(comb)
        y_train.append(sum(comb) % 2)

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    if METHOD == 'qrac_map':
        q_maps = [mapper for _ in range(DUP * num_bits // 3)]
    else:
        q_maps = None

    # Encode information
    if METHOD in ['qrac', 'qrac_map']:
        num_qubit = num_bits // 3 * DUP
        x_st = []
        for x in x_train:
            x_st.append(''.join(x.astype(str)) * DUP)

        x_st = np.array(x_st)

    elif METHOD == 'conv':
        num_qubit = (num_bits - 2) * DUP
        x_st = []
        for x in x_train:
            st = ''
            for i in range(num_bits - 2):
                st += ''.join(x[i:i+3].astype(str)) * DUP
            x_st.append(st)

        x_st = np.array(x_st)

    seed = 10598

    vqc_ordinal_log = []

    def loss_history_callback(_, __, loss, ___):
        vqc_ordinal_log.append(loss)

    feature_map = CustomFeatureMap('ALL3in1', 1, num_qubit, q_maps)
    var_form = variational_forms.RYRZ(num_qubit, depth=4)

    training_input = {
        0: x_st[y_train == 0],
        1: x_st[y_train == 1]
    }

    qsvm = VQC(SPSA(100), feature_map, var_form, training_input, callback=loss_history_callback)

    backend_options = {"method": "statevector_gpu"}
    backend = QasmSimulator(backend_options)

    quantum_instance = QuantumInstance(backend, shots=1024, seed_simulator=seed, seed_transpiler=seed,optimization_level=3)


    result = qsvm.run(quantum_instance)

    y_pred_train = qsvm.predict(x_st)[1]

    # F1 score
    acc = np.mean(y_pred_train == y_train)

    qsvm.save_model(f'models/Parity_check_{METHOD}_{BIT}_{DUP}')

    import pickle

    with open(f'results/Parity_check_{METHOD}_{BIT}_{DUP}', 'wb') as f:
        pickle.dump([vqc_ordinal_log, acc], f)

if __name__ == "__main__":
    main()

