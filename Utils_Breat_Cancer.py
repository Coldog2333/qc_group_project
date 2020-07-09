# Utility Functions
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

# Get data
def get_breast_cancer_data():
    df = pd.read_csv('breast-cancer.data', header=None, 
                     names=['target','age', 'menopause', \
                            'tumor-size','inv-nodes','node-caps',\
                            'deg-malig','breast','breast-quad', 'irradiat'])    
    df['node-caps'] = df['node-caps'].replace('?','no')
    # Encode to number
    for col in df.columns:
        le = preprocessing.LabelEncoder().fit(df[col])
        df[col] = le.transform(df[col])
    # Split data and target
    df_data = df.drop(['target'], axis=1)
    y = df.target    
    return df_data, y

# Dictionary to feed VQC
def double(x):
    return np.concatenate([x, x], axis=0)

def get_input_dict_for_VQC(x_train, x_test, y_train, y_test):
    training_input = { 0: x_train[y_train == 0],
                       1: x_train[y_train == 1] }
    test_input = { 0: x_test[y_test == 0],
                   1: x_test[y_test == 1] }
    return training_input, test_input

# VQC related
# Function converting any 3-bit binary string into (theta, varphi) of a single pure qubit state
def angles_from_threebitstring(threebit):
    rx, ry, rz = np.sqrt(1/3)*np.array([(-1) ** float(i) for i in threebit])
    theta = np.arccos(rz)
    if rx > 0:
        if ry < 0:
            varphi = np.arctan(ry/rx) + 2*np.pi
        else:
            varphi = np.arctan(ry/rx)
    elif rx < 0:
        varphi = np.arctan(ry/rx) + np.pi
    elif rx == 0:
        if ry > 0:
            varphi = np.pi/2
        elif ry < 0:
            varphi = 3*np.pi/2
        else:
            varphi = 0.
    return theta, varphi

# Encoding function
def qubit_encoding(dataframe):
    data, bit_each_col = [], []
    # Check number of different category in df_all
    for col in dataframe.columns:
        bit_each_col.append(int(np.ceil(np.log2(len(dataframe[col].unique())))))        
    # Count required bit
    num_bit = sum(bit_each_col)
    num_qubit = int(np.ceil(num_bit / 3))
    print(f'Bits required (before padded): {num_bit}')
    print(f'Qubits required: {num_qubit}')
    pad = 0
    # If necessary, pad the binary string with '0' to make divisible by three
    if num_bit % 3 != 0:
        pad = 3 - (num_bit % 3)    
    # Encode train
    for row in dataframe.values:
        bstring = ''
        for v, num_bit in zip(row, bit_each_col):
            bstring += f"{v:010b}"[-num_bit:]
        all_b_st = bstring + (pad * '0')        
        var_list = []
        for i in range(num_qubit):
            var_list += angles_from_threebitstring(all_b_st[i * 3: (i+1) * 3])
        data.append(var_list)    
    data = np.array(data)
    print(f"Data's shape: {data.shape}")
    return data

def get_RYRZvarform(encoded_data):
    return variational_forms.RYRZ(encoded_data.shape[1] // 2, depth=4)

def get_feature_map(encoded_data):
    X = [Parameter(f'x[{i}]') for i in range(encoded_data.shape[1])]
    feature_map = QuantumCircuit(encoded_data.shape[1] // 2)    
    # Encoder
    for i in range(encoded_data.shape[1] // 2):
        feature_map.u3(X[2*i], X[2*i+1], 0, i) 
    return feature_map

def run_vqc(feature_map, var_form, backend, seed, training_input, test_input=None, optimizer=SPSA(100), shots=1024):
    # Collecting loss
    vqc_loss = []
    def loss_history_callback(_, __, loss, ___):
        vqc_loss.append(loss)
    
    vqc = VQC(optimizer, feature_map, var_form, training_input, test_input, callback=loss_history_callback)
    quantum_instance = QuantumInstance(backend, shots=shots, seed_simulator=seed, seed_transpiler=seed,optimization_level=3)
    result = vqc.run(quantum_instance)
    return vqc, result, vqc_loss

def get_acc_and_f1(trained_model, X, target):
    return np.mean(trained_model.predict(X)[1] == target), f1_score(target, trained_model.predict(X)[1])