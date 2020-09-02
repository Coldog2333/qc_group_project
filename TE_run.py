import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from qiskit import BasicAer, QuantumCircuit
from qiskit.ml.datasets import *
from qiskit.circuit.library import ZZFeatureMap
from qiskit.aqua.components.feature_maps import SecondOrderExpansion
from qiskit.aqua.utils import split_dataset_to_data_and_labels, map_label_to_class_name
from qiskit.aqua import QuantumInstance
from qiskit.aqua.algorithms import QSVM, VQC
from qiskit.aqua.components import variational_forms
from qiskit.aqua.components.optimizers import COBYLA, SPSA
import logging
from qiskit.aqua import set_qiskit_aqua_logging
set_qiskit_aqua_logging(logging.DEBUG)  # choose INFO, DEBUG to see the log


from embed_utils import MyVQC
from var_utils import MyRYRZ

from bc_utils_ver2 import *

import os
# Train VQC
def train_vqc(feature_map, \
              var_form, \
              backend, \
              optimizer, \
              seed, \
              X_train, X_test, y_train, y_test, \
              fold_path, \
              positivedata_duplicate_ratio=1, \
              shots=1024,
              randomizer="standard_normal",
              lamb=None):

    # Input preparation
    # Input dict
    training_input, test_input = get_input_dict_for_VQC(X_train, X_test, y_train, y_test, positivedata_duplicate_ratio)
    # Quantum instance
    quantum_instance = QuantumInstance(backend, shots=shots, seed_simulator=seed, seed_transpiler=seed,optimization_level=3)

    temp_model_name = os.path.join(fold_path, 'temp.npz')
    final_model_filename = os.path.join(fold_path, 'final_model.npz')

    # Callback function for collecting models' parameters and losses along the way
    training_loss_list, validation_loss_list = [], []
    training_acc, validation_acc = [], []

    # Simulator for save val loss/acc
    sim_backend = QasmSimulator({"method": "statevector_gpu"})
    sim_quantum_instance = QuantumInstance(sim_backend, shots=shots, seed_simulator=seed, seed_transpiler=seed,optimization_level=3)
    def callback_collector(eval_count, model_params, loss, ___, train_acc):
        # Collect training loss
        training_loss_list.append(loss)
        training_acc.append(train_acc)

        # Save a temp model
        temp_model_filename = os.path.join(fold_path, f'_evalcount{eval_count+1}.npz')
        np.savez(temp_model_filename, opt_params = model_params)
#         zip_obj.write(temp_model_filename, compress_type=ZIP_DEFLATED)
        # Load the temp model
        vqc_val = MyVQC(optimizer, feature_map, var_form, training_input, test_input)
        vqc_val.load_model(temp_model_filename)
#         os.remove(temp_model_filename)
        # Collect validation loss
        y_test_prob, y_pred = vqc_val.predict(X_test, sim_quantum_instance)
        val_loss = -np.mean(y_test*np.log(y_test_prob[:,1]) + (1 - y_test)*np.log(y_test_prob[:,0]))
        validation_acc.append(np.mean(y_pred == y_test))
        validation_loss_list.append(val_loss)

    # Run VQC
    vqc = MyVQC(optimizer, feature_map, var_form, training_input, test_input, callback=callback_collector, randomizer=randomizer, lamb=lamb)
    vqc.random.seed(seed)
    result = vqc.run(quantum_instance)
    clear_output()
    print('Trained successfully!')
    vqc.save_model(final_model_filename)

    # Evaluate a final model
    y_train_pred, y_test_pred = vqc.predict(X_train, quantum_instance)[1], vqc.predict(X_test,  quantum_instance)[1]
    acc_train, f1_train = np.mean(y_train_pred==y_train), f1_score(y_train, y_train_pred)
    acc_test, f1_test = np.mean(y_test_pred==y_test), f1_score(y_test, y_test_pred)
    clear_output()
    print(f'Final accuracy (test set): {acc_test:.2%} | Final accuracy (training set): {acc_train:.2%}')
    print(f'Final F1 score (test set): {f1_test:.2%} | Final F1 score (training set): {f1_train:.2%}')
    print(f'Final model is saved at {final_model_filename}.\nTemp models are saved at {temp_model_name}.')

    result['Training losses'], result['Validation losses'] = np.array(training_loss_list), np.array(validation_loss_list)
    result['Training accuracy logs'], result['Validation accuracy logs'] = np.array(training_acc), np.array(validation_acc)
    result['Training F1 score'], result['Training accuracy'] = f1_train, acc_train
    result['Test F1 score'], result['Test accuracy'] = f1_test, acc_test

    return result

def kfold_vqc(feature_map, \
              var_form, \
              backend, \
              optimizer_generator, \
              seed, \
              X, y, \
              model_foldername, \
              result_filename, \
              k=5, \
              positivedata_duplicate_ratio=1, \
              shots=1024, \
              seed_kfold=123123,
              randomizer="standard_normal",
              lamb=None):

    print('='*100)
    print(f'{k}-fold VQC Classification')
    print(f"Model is saved at {model_foldername}")
    if not os.path.exists(model_foldername):
        os.makedirs(model_foldername)

    # Final result initialization (dict)
    params_to_collect = ['Training losses', 'Validation losses', \
                         'Training accuracy', 'Test accuracy', \
                         'Training F1 score', 'Test F1 score', \
                         'Training accuracy logs', 'Validation accuracy logs']
    result = {key:[] for key in params_to_collect}
    # result['Default test accuracies'] = [] # Uncomment for validating the predicted accuracy
    np.random.seed(seed_kfold)
    kf = KFold(n_splits=k, shuffle=True)
    kf_id = list(kf.split(X))
    for (fold, (train_id, test_id)) in enumerate(kf_id, start=1):
        fold_path = os.path.join(model_foldername, f"fold_{fold}")
        if not os.path.exists(fold_path):
            os.makedirs(fold_path)
        print('='*100 + f'\nFold number {fold}\n' + '='*100)
        # Split the data
        X_train, X_test, y_train, y_test = X[train_id], X[test_id], y[train_id], y[test_id]
        # Train a model
        optimizer = optimizer_generator()
        result_onefold = train_vqc(feature_map, \
                                var_form, \
                                backend, \
                                optimizer, \
                                seed, \
                                X_train, X_test, y_train, y_test, \
                                fold_path, \
                                positivedata_duplicate_ratio, \
                                shots,
                                  randomizer,
                                  lamb)
        # Save the trained model to the final zip file
        # Final model
        final_model_filename_fold = os.path.join(fold_path, f"final.npz")
        # Collect results
        for key in params_to_collect:
            result[key].append(result_onefold[key])

        with open(result_filename + f'_fold{fold}', 'wb') as f:
            pickle.dump(result_onefold, f)


    # Average accuracies and f1 scores
    dict_items_without_meanvalues = list(result.items())
    for key, value in dict_items_without_meanvalues:
        result[key + ' (mean)'] = np.mean(value, axis=0)
    # Convert to numpy arrays
    for key, value in result.items():
        if type(value)==list:
            result[key] = np.array(value)
    # Save final results
    with open(result_filename, 'wb') as f:
        pickle.dump(result, f)
    clear_output()
    print('='*100)
    print('='*35 + f' {k}-fold VQC Classification ' + '='*35)
    print(f"Training accuracy (mean): {result['Training accuracy (mean)']:.2%} | Test accuracy (mean): {result['Test accuracy (mean)']:.2%}")
    print(f"Training F1 score (mean): {result['Training F1 score (mean)']:.2%} | Test F1 score (mean): {result['Test F1 score (mean)']:.2%}")
    print(f'All models are saved at {model_foldername}.\nResults are saved at {result_filename}.')
    print('='*100)

    return result


X_df, y_df = get_breast_cancer_data()
X, y = X_df.values, y_df.values

selected_features_num = 4
_, __, feature_importances_list = kfold_randomforest(X, y)

# Feature selection from feature importances
selected_features = X_df.columns[sorted(range(X.shape[1]), key=lambda i: np.mean(feature_importances_list, axis=0)[i])[:-selected_features_num-1:-1]]

# Preparing data
X, y = X_df[selected_features].values, y_df.values
X_binary_encoded = binary_encoder(X)

# Preparing inputs for feeding VQC
num_qubit = len(X_binary_encoded[0])//3
# Custom Feature Map
feature_map = QuantumCircuit(num_qubit)
var_form = MyRYRZ(num_qubit, 4)

from qiskit import IBMQ
IBMQ.load_account()
provider = IBMQ.get_provider(hub='ibm-q-utokyo', project='qc2020s')
backend = provider.get_backend('ibmq_toronto')

#backend = QasmSimulator({"method": "statevector_gpu"})

# Test Run VQC (CustomFeatureMap)
seed, epoch = 666, 100
optimizer = lambda: SPSA(epoch)
result_bc_depth4_reg = kfold_vqc(feature_map, \
                    var_form, \
                    backend, \
                    optimizer, \
                    seed, \
                    X_binary_encoded, y, \
                    'models/BC_self_learn_encoder_standard_4fold_depth4_lamb0.02_real_device_model', \
                    'results/BC_self_learn_encoder_standard_4fold_depth4_lamb0.02_real_device_results.pkl', \
                    k=4,
                    randomizer="standard_normal",
                    lamb=0.02)
