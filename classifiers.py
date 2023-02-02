import numpy as np
import pandas as pd
import random
from sklearn.metrics import matthews_corrcoef, f1_score, accuracy_score, confusion_matrix, roc_auc_score
from sklearn.metrics import average_precision_score, precision_recall_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


def knn(df):
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

    #Repeat the training and predictions 100 times to report the mean for each assessment metric
    knn_metrics = {
    "mae": [],
    "mcc": [],
    "f1": [],
    "accuracy": [],
    "tprate": [],
    "tnrate": [],
    "pr_auc": [],
    "roc_auc": []
    }

    for i in range(100):  
        #split dataset to train & test 
        r_state=random.randint(40, 100) 
        Xtrain, Xtest, ytrain, ytest = train_test_split(features, labels, test_size = 0.3, random_state = r_state)

        #hyper parameter optimization with grid search
        knn = KNeighborsClassifier()
        grid_params = { 'n_neighbors' : [2,3,4,5,6,7,8,9,10,11,12,13,14,15],
                        'weights' : ['uniform','distance'],
                        'metric' : ['minkowski','euclidean','manhattan']}
        gs = GridSearchCV(knn, grid_params)
        knn = gs.fit(Xtrain, ytrain)
        #best_knn.best_params_

        predictions=knn.predict(Xtest)

        # Mean Absolute Error
        errors = abs(predictions - ytest)
        knn_metrics["mae"].append(round(np.mean(errors),3))

        #Matthews correlation coefficient 
        mcc=matthews_corrcoef(ytest, predictions)
        knn_metrics["mcc"].append(round(mcc,3))

        #F1 Score
        f1 = f1_score(ytest,predictions)
        knn_metrics["f1"].append(round(f1,3))

        #Accuracy Score
        accuracy = accuracy_score(ytest,predictions)
        knn_metrics["accuracy"].append(round(accuracy,3))

        #TP Rate (sensitivity) & TN Rate (specificity)
        cm=confusion_matrix(ytest, predictions)
        tp, fp, tn, fn =cm[0,0], cm[1,0], cm[1,1], cm[0,1]

        sensitivity = tp/(tp+fn)
        knn_metrics["tprate"].append(round(sensitivity,3))

        specificity = tn/(tn+fp)
        knn_metrics["tnrate"].append(round(specificity,3))

        #PR AUC
        precision, recall, thresholds = precision_recall_curve(ytest, predictions)
        auc_pr = auc(recall, precision)
        knn_metrics["pr_auc"].append(round(auc_pr,3))

        #ROC AUC
        roc = roc_auc_score(ytest, predictions)
        knn_metrics["roc_auc"].append(round(roc,3))

    #final mean metrics after looping
    print('---------------------------------------------------')
    print("KNN Mean results after 100 loops\n")
    print("Mean Absolue Error: ", round(np.mean(knn_metrics["mae"]),3) )
    print("MCC Score : ", round(np.mean(knn_metrics["mcc"]),3) )
    print("F1 Score: ", round(np.mean(knn_metrics["f1"]),3) )
    print("Accuracy: ", round(np.mean(knn_metrics["accuracy"]),3) )
    print("TP Rate: ", round(np.mean(knn_metrics["tprate"]),3) )
    print("TN Rate: ", round(np.mean(knn_metrics["tnrate"]),3) )
    print("PR AUC: ", round(np.mean(knn_metrics["pr_auc"]),3) )
    print("ROC AUC: ", round(np.mean(knn_metrics["roc_auc"]),3) )
    print('---------------------------------------------------')






def random_forest(df):
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

    #Repeat the training and predictions 100 times to report the mean for each assessment metric
    random_forest_metrics = {
    "mae": [],
    "mcc": [],
    "f1": [],
    "accuracy": [],
    "tprate": [],
    "tnrate": [],
    "pr_auc": [],
    "roc_auc": []
    }   

    for i in range(100):
        #split dataset to train & test 
        r_state=random.randint(40, 100) 
        #random state different in each split so that splitting is random
        Xtrain, Xtest, ytrain, ytest = train_test_split(features, labels, test_size = 0.2, random_state = r_state)
        
        # Instantiate model 
        rf = RandomForestClassifier(max_depth=10, random_state=r_state)

        # Train the model on training data
        rf.fit(Xtrain, ytrain)

        # Use the forest's predict method on the test data
        predictions = rf.predict(Xtest)
        #round predictions to have values 0/1 and convert predictions array from float to int
        predictions = np.round(predictions,decimals=0)
        predictions=predictions.astype(int)
        """
        print(predictions)
        print(ytest)
        """
        # Mean Absolute Error
        errors = abs(predictions - ytest)
        random_forest_metrics["mae"].append(round(np.mean(errors),3))

        #Matthews correlation coefficient 
        mcc=matthews_corrcoef(ytest, predictions)
        random_forest_metrics["mcc"].append(round(mcc,3))

        #F1 Score
        f1 = f1_score(ytest,predictions)
        random_forest_metrics["f1"].append(round(f1,3))

        #Accuracy Score
        accuracy = accuracy_score(ytest,predictions)
        random_forest_metrics["accuracy"].append(round(accuracy,3))

        #TP Rate (sensitivity) & TN Rate (specificity)
        cm=confusion_matrix(ytest, predictions)
        tp, fp, tn, fn =cm[0,0], cm[1,0], cm[1,1], cm[0,1]

        sensitivity = tp/(tp+fn)
        random_forest_metrics["tprate"].append(round(sensitivity,3))

        specificity = tn/(tn+fp)
        random_forest_metrics["tnrate"].append(round(specificity,3))

        #PR AUC
        precision, recall, thresholds = precision_recall_curve(ytest, predictions)
        auc_pr = auc(recall, precision)
        random_forest_metrics["pr_auc"].append(round(auc_pr,3))

        #ROC AUC
        roc = roc_auc_score(ytest, predictions)
        random_forest_metrics["roc_auc"].append(round(roc,3)) 

    #final mean metrics after looping
    print('---------------------------------------------------')
    print("Random Forest Mean results after 100 loops\n")
    print("Mean Absolue Error: ", round(np.mean(random_forest_metrics["mae"]),3) )
    print("MCC Score : ", round(np.mean(random_forest_metrics["mcc"]),3) )
    print("F1 Score: ", round(np.mean(random_forest_metrics["f1"]),3) )
    print("Accuracy: ", round(np.mean(random_forest_metrics["accuracy"]),3) )
    print("TP Rate: ", round(np.mean(random_forest_metrics["tprate"]),3) )
    print("TN Rate: ", round(np.mean(random_forest_metrics["tnrate"]),3) )
    print("PR AUC: ", round(np.mean(random_forest_metrics["pr_auc"]),3) )
    print("ROC AUC: ", round(np.mean(random_forest_metrics["roc_auc"]),3) )
    print('---------------------------------------------------')