import numpy as np, pandas as pd, matplotlib.pyplot as plt, os
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import f1_score
from IPython.display import clear_output
from zipfile import ZipFile, ZIP_DEFLATED
import pickle

from qiskit import BasicAer
from qiskit.providers.aer import QasmSimulator
from qiskit.ml.datasets import *
from qiskit.circuit.library import ZZFeatureMap
from qiskit.aqua.components.feature_maps import SecondOrderExpansion
from qiskit.aqua.utils import split_dataset_to_data_and_labels, map_label_to_class_name
from qiskit.aqua import QuantumInstance
from qiskit.aqua.algorithms import QSVM, VQC
from qiskit.aqua.components import variational_forms
from qiskit.aqua.components.optimizers import COBYLA, SPSA
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score

# Make feature map with encoder
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.circuit.library import TwoLocal
from quantum_utils import CustomFeatureMap

# # Self learning feature map
# from bc_utils import MyVQC

# Data preprocessing
def get_breast_cancer_data():
    # Read the data
    df = pd.read_csv('breast-cancer.data', \
                     header=None, \
                     names=['target', \
                            'age', \
                            'menopause', \
                            'tumor-size', \
                            'inv-nodes', \
                            'node-caps', \
                            'deg-malig', \
                            'breast', \
                            'breast-quad', \
                            'irradiat'])
    # Ordinal Encoding
    for col in df.columns:
        df[col] = LabelEncoder().fit_transform(df[col])

    return df.drop(['target'], axis=1), df.target

# K-fold Random Forest Classification
def kfold_randomforest(X, y, k=5):
    print('='*50)
    print(f'{k}-fold Random Forest Classification')
    acc_list, f1_list, feature_importances_list = [], [], []
    for train_id, test_id in KFold(n_splits=k, shuffle=True).split(X):
        # Split the data
        X_train, X_test, y_train, y_test = X[train_id], X[test_id], y[train_id], y[test_id]
        # Train the model
        rf_classifier = RandomForestClassifier()
        rf_classifier.fit(X_train, y_train)
        # Get scores and feature importances
        acc_list.append(rf_classifier.score(X_test, y_test))
        f1_list.append(f1_score(y_test, rf_classifier.predict(X_test)))
        feature_importances_list.append(rf_classifier.feature_importances_)

    # Average accuracy and f1 score
    print(f'Mean Accuracy: {np.mean(acc_list):.2%}')
    print(f'Mean F1 score: {np.mean(f1_list):.2%}')
    print('='*50)

    return acc_list, f1_list, feature_importances_list

# Dictionary to feed VQC
def get_input_dict_for_VQC(X_train, X_test, y_train, y_test, positivedata_duplicate_ratio):
    X_duplicate_shape = tuple([0] + list(X_train.shape)[1:])
    X_duplicate = np.empty(X_duplicate_shape)
    if positivedata_duplicate_ratio >= 0:
        for i in range(int(positivedata_duplicate_ratio)):
            X_duplicate = np.concatenate((X_duplicate, X_train[y_train==1]), axis=0)
        X_duplicate = np.concatenate((X_duplicate, X_train[y_train == 1][:int((positivedata_duplicate_ratio - int(positivedata_duplicate_ratio))*X_train[y_train==1].shape[0])]), axis=0)
    else:
        raise ValueError('Please enter nonnegative real number')
    training_input = { 0: X_train[y_train == 0],
                       1: np.concatenate((X_train[y_train == 1], X_duplicate), axis=0) }
    test_input = { 0: X_test[y_test == 0],
                   1: X_test[y_test == 1] }
    return training_input, test_input

# Train VQC
def train_vqc(feature_map, \
              var_form, \
              backend, \
              optimizer, \
              seed, \
              X_train, X_test, y_train, y_test, \
              model_filename, \
              positivedata_duplicate_ratio=1, \
              shots=1024):

    # Input preparation
    # Input dict
    training_input, test_input = get_input_dict_for_VQC(X_train, X_test, y_train, y_test, positivedata_duplicate_ratio)
    # Quantum instance
    quantum_instance = QuantumInstance(backend, shots=shots, seed_simulator=seed, seed_transpiler=seed,optimization_level=3)
    # Final zip file for temp models and its working directory
    wdir = '/'.join(model_filename.split('/')[:-1])
    print('='*100 + f'\nWorking directory: {wdir}\n' + '='*100)
#     os.chdir(wdir)
    temp_model_zip_filename = model_filename.split('.')[0] + '_temp.zip'
    final_model_filename = model_filename.split('.')[0] + '_final.npz'
    zip_obj = ZipFile(temp_model_zip_filename, 'w')

    # Callback function for collecting models' parameters and losses along the way
    training_loss_list, validation_loss_list = [], []
    def callback_collector(eval_count, model_params, loss, ___):
        # Collect training loss
        training_loss_list.append(loss)
        # Save a temp model
        temp_model_filename = model_filename.split('.')[0] + f'_evalcount{eval_count+1}.npz'
        np.savez(temp_model_filename, opt_params = model_params)
        zip_obj.write(temp_model_filename, compress_type=ZIP_DEFLATED)
        # Load the temp model
        vqc_val = VQC(optimizer, feature_map, var_form, training_input, test_input)
        vqc_val.load_model(temp_model_filename)
        os.remove(temp_model_filename)
        # Collect validation loss
        y_test_prob = vqc_val.predict(X_test, quantum_instance)[0]
        val_loss = -np.mean(y_test*np.log(y_test_prob[:,1]) + (1 - y_test)*np.log(y_test_prob[:,0]))
        validation_loss_list.append(val_loss)

    # Run VQC
    vqc = VQC(optimizer, feature_map, var_form, training_input, test_input, callback=callback_collector)
    result = vqc.run(quantum_instance)
    clear_output()
    print('Trained successfully!')
    vqc.save_model(final_model_filename)
    zip_obj.close()

    # Evaluate a final model
    y_train_pred, y_test_pred = vqc.predict(X_train, quantum_instance)[1], vqc.predict(X_test,  quantum_instance)[1]
    acc_train, f1_train = np.mean(y_train_pred==y_train), f1_score(y_train, y_train_pred)
    acc_test, f1_test = np.mean(y_test_pred==y_test), f1_score(y_test, y_test_pred)
    clear_output()
    print(f'Final accuracy (test set): {acc_test:.2%} | Final accuracy (training set): {acc_train:.2%}')
    print(f'Final F1 score (test set): {f1_test:.2%} | Final F1 score (training set): {f1_train:.2%}')
    print(f'Final model is saved at {final_model_filename}.\nTemp models are saved at {temp_model_zip_filename}.')

    result['Training losses'], result['Validation losses'] = np.array(training_loss_list), np.array(validation_loss_list)
    result['Training F1 score'], result['Training accuracy'] = f1_train, acc_train
    result['Test F1 score'], result['Test accuracy'] = f1_test, acc_test

    return result

def kfold_vqc(feature_map, \
              var_form, \
              backend, \
              optimizer_generator, \
              seed, \
              X, y, \
              model_filename, \
              result_filename, \
              k=5, \
              positivedata_duplicate_ratio=1, \
              shots=1024, \
              seed_kfold=123123, \
              double_positive_data=True,
              one_third_positive_data=False):

    print('='*100)
    print(f'{k}-fold VQC Classification')
    # Final zip file for saving and its directory
    wdir = '/'.join(model_filename.split('/')[:-1])
    print('='*100 + f'\nWorking directory: {wdir}\n' + '='*100)
#     os.chdir(wdir)
    zip_filename = model_filename.split('.')[0] + '.zip'
    print(zip_filename)
    zip_obj = ZipFile(zip_filename, 'w')
    # Final result initialization (dict)
    params_to_collect = ['Training losses', 'Validation losses', \
                         'Training accuracy', 'Test accuracy', \
                         'Training F1 score', 'Test F1 score']
    result = {key:[] for key in params_to_collect}
    # result['Default test accuracies'] = [] # Uncomment for validating the predicted accuracy
    np.random.seed(seed_kfold)
    kf = KFold(n_splits=k, shuffle=True)
    kf_id = list(kf.split(X))
    for (fold, (train_id, test_id)) in enumerate(kf_id, start=1):
        print('='*100 + f'\nFold number {fold}\n' + '='*100)
        # Split the data
        X_train, X_test, y_train, y_test = X[train_id], X[test_id], y[train_id], y[test_id]
        # Double positive data
        if double_positive_data:
            X_train, y_train = np.concatenate((X_train, X_train[y_train==1]), axis=0), np.hstack((y_train, np.ones(np.sum(y_train==1))))
        elif one_third_positive_data:
            X_train, y_train = np.concatenate([X_train, X_train[:len(X_train)//3]], axis=0), np.hstack((y_train, np.ones(len(X_train)//3)))
        # Train a model
        model_filename_fold = model_filename.split('.')[0] + f'_foldnumber{fold}.npz'
        optimizer = optimizer_generator()
        result_onefold = train_vqc(feature_map, \
                                var_form, \
                                backend, \
                                optimizer, \
                                seed, \
                                X_train, X_test, y_train, y_test, \
                                model_filename_fold, \
                                positivedata_duplicate_ratio, \
                                shots)
        # Save the trained model to the final zip file
        # Final model
        final_model_filename_fold = model_filename_fold.split('.')[0] + '_final.npz'
        zip_obj.write(final_model_filename_fold, compress_type=ZIP_DEFLATED)
        os.remove(final_model_filename_fold)
        # Temp model
        temp_model_zip_filename_fold = model_filename_fold.split('.')[0] + '_temp.zip'
        zip_obj.write(temp_model_zip_filename_fold, compress_type=ZIP_DEFLATED)
        os.remove(temp_model_zip_filename_fold)
        # Collect results
        for key in params_to_collect:
            result[key].append(result_onefold[key])

    # Average accuracies and f1 scores
    zip_obj.close()
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
    print(f'All models are saved at {zip_filename}.\nResults are saved at {result_filename}.')
    print('='*100)

    return result

# Convert a 3-bit string into inputs for U3 gate
def convert_to_angle(b_st):
    if b_st=='111':
        return [np.arccos(1/np.sqrt(3)), np.pi/4]
    if b_st=='110':
        return [np.arccos(1/np.sqrt(3)), 3*np.pi/4]
    if b_st=='101':
        return [np.arccos(1/np.sqrt(3)), -np.pi/4]
    if b_st=='100':
        return [np.arccos(1/np.sqrt(3)), -3*np.pi/4]
    if b_st=='011':
        return [np.arccos(-1/np.sqrt(3)), np.pi/4]
    if b_st=='010':
        return [np.arccos(-1/np.sqrt(3)), 3*np.pi/4]
    if b_st=='001':
        return [np.arccos(-1/np.sqrt(3)), -np.pi/4]
    if b_st=='000':
        return [np.arccos(-1/np.sqrt(3)), -3*np.pi/4]

# Binary Encoder
def binary_encoder(X):
    # The number of necessary bits in each column
    bit_each_col = [int(np.ceil(np.log2(len(np.unique(X[:,col]))))) for col in range(X.shape[1])]
    # Padding check in order to make an input string into quantum circuit divisible by three
    if sum(bit_each_col)%3 != 0:
        pad = 3 - (sum(bit_each_col)%3)
    else:
        pad = 0
    # Encode X into a binary string
    X_binary_encoded = []
    for sample in X:
        bit_string = ''
        for value, num_bit in zip(sample, bit_each_col):
            bit_string += f'{value:010b}'[-num_bit:]
        bit_string += pad*'0'
        X_binary_encoded.append(bit_string)
    return np.array(X_binary_encoded)

# U3gate Input Encoder
def U3gate_input_encoder(X):
    X_binary_encoded = binary_encoder(X)
    if len(X_binary_encoded[0]) % 3 != 0:
        raise ValueError('The input string is not divisible by three')
    else:
        X_U3gate_input = []
        for bitstring in X_binary_encoded:
            U3gate_input = []
            for qubit in range(len(X_binary_encoded[0])//3):
                U3gate_input.extend(convert_to_angle(bitstring[qubit*3: (qubit+1)*3]))
            X_U3gate_input.append(U3gate_input)
        return np.array(X_U3gate_input)
