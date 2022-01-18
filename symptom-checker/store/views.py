from django.shortcuts import render, redirect
from django.views import generic
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from .forms import SymptomFormset
from .models import Symptom
from flashtext import KeywordProcessor
from spellchecker import SpellChecker
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from collections import Counter
from sklearn.tree import DecisionTreeClassifier

val=None

def create_diagnosis(request):
    global val
    template_name = 'store/create_diagnosis.html'
    heading_message = 'Write your symptoms'
    if request.method == 'GET':
        formset = SymptomFormset(request.GET or None)
    elif request.method == 'POST':
        formset = SymptomFormset(request.POST)
        if formset.is_valid():
            for form in formset:
                try:
                    name = form.cleaned_data.get('name')
                except AttributeError:
                    name = "no matches"
                if name:
                    Symptom(name=name).save()
            symp = []
            for i in formset.cleaned_data:
                spell = SpellChecker()
                try:
                    symp.append(spell.correction(i['name']))
                except KeyError:
                    symp.append('No Matches')
            def val():
                return symp
            return redirect('store:result')

    return render(request, template_name, {
        'formset': formset,
        'heading': heading_message,
    })

def result(request):
    symptoms = pd.read_csv(r'C:\Users\gigel\Desktop\symptom-checker\sc_scripts\Testing.csv', nrows=1)
    symptoms = symptoms.keys().tolist()

    keyword_processor = KeywordProcessor()
    keyword_processor.add_keywords_from_list(list(symptoms))
    res = val()
    matched_keyword = []
    for i in res:
        matched_keyword.append(keyword_processor.extract_keywords(i))
    resulted_match = []
    for i in matched_keyword:
        if not i:
            matched_keyword.remove(i)
        else:
            resulted_match.append(i[0])
# ------------------------------------------------
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
#-------------------------------------------------
    for i in columns:
        print(i)
    print(resulted_match)
    if len(matched_keyword) == 0:
        return "No Matches"
    else:
        lista = []
        listin = []
        for i in dftest.keys().tolist():
            if i in resulted_match:
                listin.append(1)
            else:
                listin.append(0)

    listin.pop()
    lista.append(listin)
    print(lista)
    probaDT = classifierDT.predict_proba(lista)
    probaDT.round(4)
    predDT = classifierDT.predict(lista)
    result = predDT[0]


    template_name='store/result.html'
    return render(request, template_name, {'result':result})


class SymptomListView(generic.ListView):
    model = Symptom
    context_object_name = 'symptoms'
    template_name = 'store/list.html'



