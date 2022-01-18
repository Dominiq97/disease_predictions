import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from collections import Counter
from sklearn.tree import DecisionTreeClassifier
import math

def precise_disease(newdata):
    dftest = pd.read_csv(r'C:\Users\gigel\Desktop\symptom-checker\sc_scripts\Testing.csv')
    dftrain = pd.read_csv(r'C:\Users\gigel\Desktop\symptom-checker\sc_scripts\Training.csv')

    null_columns = dftrain.columns[dftrain.isnull().any()]
    null_columns=dftest.columns[dftest.isnull().any()]
    columns = list(dftrain.columns)

    X_train = dftrain.iloc[:, :-1].values # the training attributes
    y_train = dftrain.iloc[:, 132].values # the training labels
    X_test = dftest.iloc[:, :-1].values # the testing attributes
    y_test = dftest.iloc[:, 132].values # the testing labels

    classifierDT = DecisionTreeClassifier(splitter='best', criterion='entropy', min_samples_leaf=2)
    classifierDT.fit(X_train, y_train)

    y_predDT = classifierDT.predict(X_test)
    imp = classifierDT.feature_importances_

    columns = columns[:132]
    column_names = ['symptom', 'importance']
    df3 = np.vstack((columns, imp)).T
    df3 = pd.DataFrame(df3, columns = column_names)

    coefficients = classifierDT.feature_importances_

    # set a minimum threshold for feature importance
    importance_threshold = np.quantile(coefficients, q = 0.75)

    import numpy
    # identify features with feature importance values below the minimum threshold
    low_importance_features = numpy.array(df3.symptom[np.abs(coefficients) <= importance_threshold])
    columns = list(low_importance_features)

    for i in columns :
        dftrain.drop(i, axis=1, inplace=True)
        dftest.drop(i, axis=1, inplace=True)

    # split dataset into attributes and labels
    X_train = dftrain.iloc[:, :-1].values # the training attributes
    y_train = dftrain.iloc[:, 33].values # the training labels
    X_test = dftest.iloc[:, :-1].values # the testing attributes
    y_test = dftest.iloc[:, 33].values # the testing labels

    # using DT based on information gain
    classifierDT = DecisionTreeClassifier(splitter='best', criterion='entropy', min_samples_leaf=2)
    classifierDT.fit(X_train, y_train)

    # for DT model
    y_predDT = classifierDT.predict(X_test)

    # using accuracy performance metric
    print("Train Accuracy: ", accuracy_score(y_train, classifierDT.predict(X_train)))
    print("Test Accuracy: ", accuracy_score(y_test, y_predDT))

    # compute probabilities of assigning to each of the classes of prognosis
    probaDT = classifierDT.predict_proba(newdata)
    probaDT.round(9) # round probabilities to four decimal places, if applicable

    predDT = classifierDT.predict(newdata)
    print(predDT[0])
    return predDT[0], accuracy_score(y_test, y_predDT), columns

newdata = [[1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1,
                0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1]]
precise_disease(newdata)

