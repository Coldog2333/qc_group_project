# qc_group_project
### Topic
**Classification with Variational Quantum Classifier**
### Introduction
This is the repository of the group project of Quantum Computing. 

We performed classification on the [Breast-Cancer dataset](https://archive.ics.uci.edu/ml/datasets/breast+cancer) 
predicted the survival of the Titanic passengers, whose dataset is provided at [kaggle](https://www.kaggle.com/c/titanic).

To reduce the number of involved qubits, 
we implement a (3,1)-QRAC to encode 3 bits category feature into 1 qubit.
In this way, we can also obtain a performance gain.

### Methods
#### Breast-Cancer

##### Modeling
+ We trained a random forest classifier on Breast-Cancer dataset.
and selected four features according to the feature importance as `"tumor-size" "node-caps", "deg-malig", "menopause"`.
+ Then we encoded these category features with qubits, using QRAC.
+ Besides, we built the corresponding circuit as FeatureMap.
+ Finally, we trained a VQC with QRAC FeatureMap over all of the data. 
All of the parameters were optimized by SPSA optimizer with 100 iterations.
And our model was evaluated on test set. 

#### Titanic

##### Preprocess
Before the training step, we did some preprocessing jobs on Titanic dataset.
+ For simplicity, we removed `"PassengerId", "Name", "Ticket", "Cabin"` features 
because they are not so related to the survival or hard to process.
+ Since there are some values missing in the Titanic dataset, 
firstly we masked them as `<NULL>` and numericalized the rest of existing values.
As for `"Sex", "Embarked"`, we numericalized them into ordinal features.
As for others, we just kept their original values since they are integer or float numbers.
+ Secondly, we filled the empty values according to some rules.
As for `"Sex", "Embarked"`, we randomly gave them values according to the distribution of the existing data.
As for the rest of features, we fixed the missing values with the average of the existing data.

##### Modeling
+ Similarly, we also trained a random forest classifier on Titanic dataset.
and selected four features according to the feature importance as `"Sex", "Age", "Pclass", "Fare"`.
+ Then we encoded these category features with qubits, using QRAC.
+ Besides, we built the corresponding circuit as FeatureMap.
+ Finally, we trained a VQC with QRAC FeatureMap over all of the data. 
All of the parameters were optimized by SPSA optimizer with 100 iterations.
And our model was evaluated on test set. 

### Experiment Result
Dataset | Breast-Cancer | | | | Titanic | | | |
:-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
Model | Train Acc | Train F1 | Test Acc | Test F1 | Train Acc | Train F1 | Test Acc | Test F1 | 
:-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
Random Forest       | | | 0.7759 | 0.3810 | 0.9798 | 0.9734 | 0.7344 | |
VQC                 | | | | | | 0.6734 | 0.5727 | 0.6292 | |
VQC (real)          | | | | | | | | |
VQC + QRAC          | | | 0.7241 | 0.4667 | 0.7868 | 0.7104 | 0.7656 | |
VQC + QRAC (real)   | | | | | | | | |

### Usage
```python
# VQC
python3 main.py --encoder --featmap ZZMap --optimizer SPSA --backend simulator --dataset titanic
# VQC + QRAC
python3 main.py --encoder --featmap QRAC --optimizer SPSA --backend simulator --dataset titanic
```

### Environment
+ Python 3.7
+ qiskit 0.19.2
+ numpy 1.18.5
+ sklearn 0.23.1

### References
\[1\] Yano, H., Suzuki, Y., Raymond, R., & Yamamoto, N. (2020). Efficient Discrete Feature Encoding for Variational Quantum Classifier. arXiv preprint arXiv:2005.14382.

### TODO
+ @Zhihong stated that we can fix the missing values of `"Sex"` according to the name ("Mr.", "Ms.", etc.).
+ We plan to evaluate our model on the titanic dataset on real devices.





