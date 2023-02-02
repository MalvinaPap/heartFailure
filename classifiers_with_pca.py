import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import matthews_corrcoef, f1_score, accuracy_score, confusion_matrix, roc_auc_score
from sklearn.metrics import average_precision_score, precision_recall_curve, auc
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC

def createTrainTestSets(df):
    #use numpy to convert to arrays
    #target array
    labels = np.array(df['DEATH_EVENT'])
    # Remove the labels from the features
    # axis 1 refers to the columns
    features = df.drop('DEATH_EVENT', axis = 1)
    # Saving feature names for later use
    feature_list = list(features.columns)
    # Convert to numpy array
    features = np.array(features)
    Xtrain, Xtest, ytrain, ytest = train_test_split(features, labels, test_size = 0.3, random_state = 32)

    #xtrain & xtest for using classifiers with top 2 principal components
    pca = PCA(n_components=2)
    Xtrain1 = pca.fit_transform(Xtrain)
    Xtest1 = pca.transform(Xtest)

    #xtrain & xtest for using classifiers with top 10 principal components
    pca = PCA(n_components=10)
    Xtrain2 = pca.fit_transform(Xtrain)
    Xtest2 = pca.transform(Xtest)

    return Xtrain, Xtest, ytrain, ytest, Xtrain1, Xtest1, Xtrain2, Xtest2

def results(ytest,predictions):
    # Mean Absolute Error
    ytest=ytest.astype(int)
    errors = abs(predictions - ytest)
    print("Mean Absolue Error: ", round(np.mean(errors),3) )
    #Matthews correlation coefficient 
    mcc=matthews_corrcoef(ytest, predictions)
    print("MCC Score : ", round(mcc,3) )
    #F1 Score
    f1 = f1_score(ytest,predictions)
    print("F1 Score: ", round(f1,3) )
    #Accuracy Score
    accuracy = accuracy_score(ytest,predictions)
    print("Accuracy: ", round(accuracy,3) )
    #TP Rate (sensitivity) & TN Rate (specificity)
    cm=confusion_matrix(ytest, predictions)
    tp, fp, tn, fn =cm[0,0], cm[1,0], cm[1,1], cm[0,1]
    sensitivity = tp/(tp+fn)
    specificity = tn/(tn+fp)
    print("TP Rate: ", round(sensitivity,3) )
    print("TN Rate: ", round(specificity,3) )
    #PR AUC
    precision, recall, thresholds = precision_recall_curve(ytest, predictions)
    auc_pr = auc(recall, precision)
    print("PR AUC: ", round(auc_pr,3) )
    #ROC AUC
    roc = roc_auc_score(ytest, predictions)
    print("ROC AUC: ", round(roc,3) )

def rf_2pc(df):
    Xtrain, Xtest, ytrain, ytest, Xtrain1, Xtest1, Xtrain2, Xtest2 = createTrainTestSets(df)
    rf = RandomForestClassifier(max_depth=10, random_state=32)
    rf.fit(Xtrain1, ytrain)
    # Use the forest's predict method on the test data
    print('---------------------------------------------------')
    print("Classification Results: Random Forest with 2 Principal Components:\n ")
    predictions = rf.predict(Xtest1)
    results(ytest,predictions)
    print('---------------------------------------------------')

def rf_10pc(df):
    Xtrain, Xtest, ytrain, ytest, Xtrain1, Xtest1, Xtrain2, Xtest2 = createTrainTestSets(df)
    rf = RandomForestClassifier(max_depth=10, random_state=32)
    rf.fit(Xtrain2, ytrain)
    # Use the forest's predict method on the test data
    print('---------------------------------------------------')
    print("Classification Results: Random Forest with 10 Principal Components:\n ")
    predictions = rf.predict(Xtest2)
    results(ytest,predictions)
    print('---------------------------------------------------')
    
def gb_2pc(df):
    Xtrain, Xtest, ytrain, ytest, Xtrain1, Xtest1, Xtrain2, Xtest2 = createTrainTestSets(df)
    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0).fit(Xtrain1, ytrain)
    gb.fit(Xtrain1, ytrain)
    # Use the gb predict method on the test data
    print('---------------------------------------------------')
    print("Classification Results: Gradient Boosting with 2 Principal Components:\n ")
    predictions = gb.predict(Xtest1)
    results(ytest,predictions)
    print('---------------------------------------------------')

def gb_10pc(df):
    Xtrain, Xtest, ytrain, ytest, Xtrain1, Xtest1, Xtrain2, Xtest2 = createTrainTestSets(df)
    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0).fit(Xtrain1, ytrain)
    gb.fit(Xtrain2, ytrain)
    # Use the gb predict method on the test data
    print('---------------------------------------------------')
    print("Classification Results: Gradient Boosting with 10 Principal Components:\n ")
    predictions = gb.predict(Xtest2)
    results(ytest,predictions)
    print('---------------------------------------------------')


def svmr_2pc(df):
    Xtrain, Xtest, ytrain, ytest, Xtrain1, Xtest1, Xtrain2, Xtest2 = createTrainTestSets(df)
    svm = SVC(C=1.0, gamma=0.1)
    svm.fit(Xtrain1, ytrain)
    # Use the SVC predict method on the test data
    print('---------------------------------------------------')
    print("Classification Results: SVM Radial with 2 Principal Components:\n ")
    predictions = svm.predict(Xtest1)
    results(ytest,predictions)
    print('---------------------------------------------------')


def svmr_10pc(df):
    Xtrain, Xtest, ytrain, ytest, Xtrain1, Xtest1, Xtrain2, Xtest2 = createTrainTestSets(df)
    svm = SVC(C=1.0, gamma=0.1)
    svm.fit(Xtrain2, ytrain)
    # Use the SVC predict method on the test data
    print('---------------------------------------------------')
    print("Classification Results: SVM Radial with 10 Principal Components:\n ")
    predictions = svm.predict(Xtest2)
    results(ytest,predictions)
    print('---------------------------------------------------')
