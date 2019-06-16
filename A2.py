#The following script is open source

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix  
from sklearn import metrics
from sklearn.metrics import mean_squared_error, r2_score
from collections import Counter
from sklearn.linear_model import LinearRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC

from sklearn.metrics import roc_curve  
from sklearn.metrics import roc_auc_score  

import warnings
warnings.filterwarnings("ignore")

secom = ''
secomLabel = ''
secom_train = ''
secom_test = ''
secomLabel_train = ''
secomLabel_test = ''

def keys():
    print()
    print('Please select process to complete:')
    print('KEYS:\t ''I'' to display information on the dataset.')
    print('\t ''E'' to extract data into a CSV file.')
    print('\t ''M'' to display details of features with missing values in the file. Accepts a number as an argument as the minimum percentage to be shown.')
    print('\t ''F'' to display the top features of that dataset determined using Random Forest.')
    print('\t ''D'' to display a rebalanced datasample using the SMOTE algorithm.')
    print('\t ============== Model Keys ==============')
    print('\t ''K'' to run the dataset with the KNN model. Accepts a number as an argument for K value. Default is 5.')
    print('\t ''R'' to run the dataset with the Random Forest model. Accepts a number as an argument for N value. Default is 20.')
    print('\t ''L'' to run the dataset with the Linear Regression model.')
    print('\t ''X'' to run the dataset with the XGBoost model. Accepts a number as an argument for N value. Default is 5.')
    print('\t ''C'' to run ALL models for comparison.')
    print('\t ========================================')
    print('\t HELP to display keys.')  
    print('\t ''Q'' to quit.')

def loadFile():
    global secom
    global secomLabel
    r = open('data/secom.data','r')
    secom = pd.read_table(r, header=None, delim_whitespace=True)
    r = open('data/secom_labels.data','r')
    secomLabel = pd.read_table(r, header=None, usecols=[0], squeeze=True, delim_whitespace=True)

def extractor():
    print('Creating CSV file...')
    dataDirectory = 'data/'
    secomFeatures = 'secom.data'
    filename = dataDirectory + secomFeatures
    sf = open(filename, 'r')
    featureOutput = sf.read()
    featureRows = featureOutput.split('\n')
    frLength = len(featureRows)
    secomId = 'secom_labels.data'
    filename = dataDirectory + secomId
    si = open(filename, 'r')
    idOutput = si.read()
    idRows = idOutput.split('\n')
    irLength = len(idRows)
    w = open('data/data.csv','w')
    if frLength == irLength:
        print('File rowcount equal. Beginning aggregation...')
        counter = 0
        while counter < frLength:
            entry = ''
            idValue = idRows[counter].split(' ')
            entry = idValue[1] + ' ' + idValue[2] + ',' + idValue[0] + ','
            featureValue = featureRows[counter].replace(' ', ',')
            entry = entry + featureValue
            w.write(entry + '\n')
            counter = counter + 1
        print('File saved as data.csv in the data folder.')
    else:
        print('Files are not equal cannot proceed. Please verify files.')
        print('sl: ', str(frLength))
        print('s: ', str(irLength))
    w.close()

def dataInfo():
    print('The dataset has {} observations/rows and {} variables/columns.'.format(secom.shape[0], secom.shape[1]))
    print('The majority class has {} observations, minority class {}.'.format(secomLabel[secomLabel == -1].size, secomLabel[secomLabel == 1].size))
    print('The dataset is imbalanced. The ratio of majority class to minority class is {}:1.'.format(int(secomLabel[secomLabel == -1].size/secomLabel[secomLabel == 1].size)))

def missingValues(display, margin):
    global secom
    totalVals = secom.shape[0]
    calc = 100/totalVals
    missingVals = secom.isnull().sum()
    count = 1
    values = []
    for value in missingVals:
        percentage = (calc * value)
        if margin <= percentage:
            percentage = "{:.{}f}".format(percentage, 2)
            item = [0,0]
            item[0] = count
            item[1] = percentage
            values.append(item)
        count = count + 1
    sortedList = sorted(values,key=lambda x: x[1], reverse = True)
    remove = []
    for value in sortedList:
        if display:
            print('Feature' + str(value[0]) + '\t' + str(value[1]) + '%')
        remove.append(value[0])
    if display:
        print('Total of ' + str(len(sortedList)) + ' features with more than ' + str(margin) + '% missing values')
        missingGraph(values)
    else:
        secom.drop(secom.columns[remove], axis=1, inplace=True)

def missingGraph(values):
    x = []
    y = []
    x = list(range(0,len(values)))
    ind = np.arange(len(x)) 
    for value in values:
        percent = float(value[1])
        y.append(percent)
    width = 0.5       # the width of the bars
    tick_marks = np.arange(len(x))
    plt.bar(tick_marks, y)
    plt.xticks(tick_marks, x, rotation=45)
    plt.show()
        
def featureSelection(display):
    global secomLabel
    global secom
    secomLabel = secomLabel.fillna(secomLabel.mean())
    secom = secom.fillna(secom.mean())
    rf = RandomForestClassifier(n_estimators=100, random_state=7)
    rf.fit(secom, secomLabel)
    importance = rf.feature_importances_
    ranked_indices = np.argsort(importance)[::-1]
    if display:
        print("Feature Rank:")
        for i in range(15):
            print("{0:3d} column  {1:3d}  {2:6.4f}".format(i+1, ranked_indices[i], importance[ranked_indices[i]]))
        print("\n")
        for i in range(len(importance)-5,len(importance)):
            print("{0:3d} column  {1:3d}  {2:6.4f}".format(i+1, ranked_indices[i], importance[ranked_indices[i]]))
        navg = 0
        for i in range(len(importance)):    
            if importance[ranked_indices[i]] > np.average(rf.feature_importances_):
                navg = navg+1
        print('The number of features better than average is: {}'.format(navg))
        loadFile()
    else:
        temp = []
        for i in range(15):
            number = ranked_indices[i]
            temp.append(number)
        secom = secom.iloc[: , temp]

def dataBalancing(display):
    global secomLabel
    global secom

    secomLabel = secomLabel.fillna(secomLabel.mean())
    secom = secom.fillna(secom.mean())

    secom_resampled, secomLabel_resampled = SMOTE().fit_resample(secom, secomLabel)
    if display:
        print('The new dataset size is ' + str(len(secomLabel_resampled)))
        print(sorted(Counter(secomLabel_resampled).items()))
        loadFile()
    else:
        secom = secom_resampled
        secomLabel = secomLabel_resampled

def mean():
    global secomLabel
    global secom

    secomLabel = secomLabel.fillna(secomLabel.mean())
    secom = secom.fillna(secom.mean())

def knn(k):
    global secomLabel
    global secom
    if k==0:
        k = 5
    secom_train, secom_test, secomLabel_train, secomLabel_test = train_test_split(secom, secomLabel, test_size=0.20)
    classifier = KNeighborsClassifier(n_neighbors=k)  
    classifier.fit(secom_train, secomLabel_train)  
    secomLabel_pred = classifier.predict(secom_test) 
    print('Confusion Matrix:')
    print(confusion_matrix(secomLabel_test, secomLabel_pred))  
    print('Classification Report:')
    print(classification_report(secomLabel_test, secomLabel_pred))  
    auc = roc_auc_score(secomLabel_test, secomLabel_pred)  
    print('AUC: %.2f' % auc)  
    fpr, tpr, thresholds = roc_curve(secomLabel_test, secomLabel_pred)  
    plot_roc_curve(fpr, tpr, 'ROC Curve title for KNN') 
    return auc

def randomForest(n):
    global secomLabel
    global secom
    if n == 0:
        n = 20
    secom_train, secom_test, secomLabel_train, secomLabel_test = train_test_split(secom, secomLabel, test_size=0.20)  
    clf=RandomForestClassifier(n_estimators=n)
    clf.fit(secom_train,secomLabel_train)
    y_pred=clf.predict(secom_test)
    print('Confusion Matrix:')
    print(confusion_matrix(secomLabel_test, y_pred))  
    print('Classification Report:')
    print(classification_report(secomLabel_test, y_pred)) 
    auc = roc_auc_score(secomLabel_test, y_pred)  
    print('AUC: %.2f' % auc)
    fpr, tpr, thresholds = roc_curve(secomLabel_test, y_pred)  
    plot_roc_curve(fpr, tpr, 'ROC Curve title for Random Forest') 
    return auc   

def linearRegression():
    global secomLabel
    global secom
    secom_train, secom_test, secomLabel_train, secomLabel_test = train_test_split(secom, secomLabel, test_size=0.20)  
    lm = LinearRegression()
    lm.fit(secom_train,secomLabel_train)
    y_pred = lm.predict(secom_test)
    plt.scatter(secomLabel_test,y_pred)
    co = lm.coef_
    intercept = lm.intercept_
    rmse = mean_squared_error(secomLabel_test, y_pred)
    r2 = r2_score(secomLabel_test, y_pred)
    print('Coefficient: ' + str(co))
    print('Intercept: ' + str(intercept))
    print('RMSE: ' + str(rmse))
    print('R2: ' + str(r2))

def xgboost(n):
    global secomLabel
    global secom
    if n == 0:
        n = 5
    secom_train, secom_test, secomLabel_train, secomLabel_test = train_test_split(secom, secomLabel, test_size=0.20) 
    model = XGBClassifier(n_estimators=n)
    model.fit(secom_train, secomLabel_train) 
    y_pred = model.predict(secom_test)
    print('Confusion Matrix:')
    print(confusion_matrix(secomLabel_test, y_pred))  
    print('Classification Report:')
    print(classification_report(secomLabel_test, y_pred))  
    auc = roc_auc_score(secomLabel_test, y_pred)  
    print('AUC: %.2f' % auc)  
    fpr, tpr, thresholds = roc_curve(secomLabel_test, y_pred)  
    plot_roc_curve(fpr, tpr, 'ROC Curve title for XGBoost') 
    return auc

def plot_roc_curve(fpr, tpr, title):  
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend()
    plt.show()

def compare(label, values):
    index = np.arange(len(label))
    barList = plt.bar(index, values)
    plt.xlabel('Model', fontsize=10)
    plt.ylabel('Performance accuracy', fontsize=10)
    plt.xticks(index, label, fontsize=7, rotation=30)
    plt.title('Comparison of model performance accuracy')
    plt.show()

#------------------------------------------------------------------------------------------------------------------------------
print('============================================Data Mining Assignment 2==================================================')
print('This program was designed to extract data and produce a machine learning model for the Secom Dataset')
print('Author: Sanath Samarasekara')
print()
print('Loading Dataset...')
loadFile()
print('Loading complete.')
keys()

while True:
    value = input('Enter your input:')
    value = value.strip()
    value = value.split(' ')
    if value[0] == 'I' or value[0] == 'i':
        dataInfo()
    elif value[0] == 'E' or value[0] == 'e':
        extractor()
    elif value[0] == 'M' or value[0] == 'm':
        if len(value) == 2:
            try:
                margin = int(value[1])
                if margin < 0 or margin > 100:
                    print('Margin must be between 0 and 100')
                else:
                    print('Printing missing values with a percentage value higher than ' + str(margin) + '%...')
                    missingValues(True,margin)
            except ValueError:
                print('Please enter a valid number for margin estimation...')
        elif len(value) == 1:
            print('Printing all missing values...')
            missingValues(True,0)
        elif len(value) > 2:
            print('Too many arguments provided...')
        else:
            print('Printing all results...')
            missingValues(True,0)
    elif value[0] == 'F' or value[0] == 'f':
        featureSelection(True)
    elif value[0] == 'D' or value[0] == 'd':
        missingValues(False,40)
        featureSelection(False)
        dataBalancing(True)
        loadFile()
    elif value[0] == 'K' or value[0] == 'k':
        if len(value) == 2:
            try:                
                margin = int(value[1])
                if margin < 0:
                    print('K value must be greater than 0')
                else:
                    missingValues(False,40)
                    featureSelection(False)
                    dataBalancing(False)
                    knn(margin)
                    loadFile()
            except ValueError:
                print('Please enter a valid number for K...')
        elif len(value) == 1:
            missingValues(False,40)
            featureSelection(False)
            dataBalancing(False)
            knn(0)
            loadFile()
        elif len(value) > 2:
            print('Too many arguments provided...')
        else:
            missingValues(False,40)
            featureSelection(False)
            dataBalancing(False)
            knn(0)
            loadFile()
    elif value[0] == 'R' or value[0] == 'r':
        if len(value) == 2:
            try:                
                margin = int(value[1])
                if margin < 0:
                    print('N value must be greater than 0')
                else:
                    missingValues(False,40)
                    featureSelection(False)
                    dataBalancing(False)
                    randomForest(margin)
                    loadFile()
            except ValueError:
                print('Please enter a valid number for N...')
        elif len(value) == 1:
            missingValues(False,40)
            featureSelection(False)
            dataBalancing(False)
            randomForest(0)
            loadFile()
        elif len(value) > 2:
            print('Too many arguments provided...')
        else:
            missingValues(False,40)
            featureSelection(False)
            dataBalancing(False)
            randomForest(0)
            loadFile()
    elif value[0] == 'L' or value[0] == 'l':
        missingValues(False,40)
        featureSelection(False)
        dataBalancing(False)
        linearRegression()
        loadFile()
    elif value[0] == 'X' or value[0] == 'x':
        if len(value) == 2:
            try:                
                margin = int(value[1])
                if margin < 0:
                    print('N value must be greater than 0')
                else:
                    missingValues(False,40)
                    featureSelection(False)
                    dataBalancing(False)
                    xgboost(margin)
                    loadFile()
            except ValueError:
                print('Please enter a valid number for N...')
        elif len(value) == 1:
            missingValues(False,40)
            featureSelection(False)
            dataBalancing(False)
            xgboost(0)
            loadFile()
        elif len(value) > 2:
            print('Too many arguments provided...')
        else:
            missingValues(False,40)
            featureSelection(False)
            dataBalancing(False)
            xgboost(0)
            loadFile()
    elif value[0] == 'HELP' or value[0] == 'help':
        keys()
    elif value[0] == 'Q' or value[0] == 'q':
        break
    elif value[0] == 'C' or value[0] == 'c':
        labels = ['KNN No Pre-processing', 'KNN', 'Random Forest No Pre-processing', 'Random Forest', 'XGBoost No Pre-processing', 'XGBoost']
        aucs = []

        mean()
        auc = knn(5)
        aucs.append(auc)
        loadFile()

        missingValues(False,40)
        featureSelection(False)
        dataBalancing(False)
        auc = knn(5)
        aucs.append(auc)
        loadFile()  

        mean()
        auc = randomForest(1000)
        aucs.append(auc)
        loadFile() 

        missingValues(False,40)
        featureSelection(False)
        dataBalancing(False)
        auc = randomForest(1000)
        aucs.append(auc)
        loadFile()  

        mean()
        auc = xgboost(2000)
        aucs.append(auc)
        loadFile() 

        missingValues(False,40)
        featureSelection(False)
        dataBalancing(False)
        auc = xgboost(2000)
        aucs.append(auc)
        loadFile() 

        compare(labels, aucs)
    elif value[0] == 'test':
        labels = ['KNN No Pre-processing', 'KNN', 'Random Forest No Pre-processing', 'Random Forest', 'XGBoost No Pre-processing', 'XGBoost']
        aucs = [0.52,0.89,0.50,0.94,0.5,0.95]
        compare(labels, aucs)
    else:
        print('Please enter a valid key. Enter ''help'' to view keys')
    print()
