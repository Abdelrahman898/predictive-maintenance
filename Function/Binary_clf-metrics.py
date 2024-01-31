
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler




def split_metric_data(df_train, df_test, features, target):
    """
    Args:
        df_train (dataframe): train dataframe
        df_test (dataframe): test dataframe 
        features (list): list of features
        target (str): target variable
                
    Return:
        X_train_scaled (dataframe): scaled train dataframe
        X_test_scaled (dataframe): scaled test dataframe
        y_train (dataframe): train target
        y_test (dataframe): test target
    """
    
    X_train = df_train[features]
    y_train = df_train[target]
    
    X_test = df_test[features]
    y_test = df_test[target]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test



def binary_clf(model, clf, df_train, df_test, features, target):
    """
    Args:
        model (str): model name
        clf (classifier object): classifier object from sklearn
        df_test (dataframe): test dataframe
        df_train (dataframe): train dataframe
        features (list): list of features
        target (str): target variable
        RANDOM_STATE (int): random state
        
    Return:
        df_predictions (dataframe): dataframe with predictions [y_pred] and probabilities [y_score]
        
    """
    # split the data
    X_train_scaled, X_test_scaled, y_train, y_test = split_metric_data(df_train, df_test, features, target)
    
    # fit the model
    model_fit = clf.fit(X_train_scaled, y_train)
    y_pred = model_fit.predict(X_test_scaled)
    
    if hasattr(model_fit, 'predict_proba'):
        y_score = model_fit.predict_proba(X_test_scaled)[:,1]
    elif hasattr(model_fit, 'decision_function'):
        y_score = model_fit.decision_function(X_test_scaled)
    else:
        y_score = y_pred
        
    # create dataframe with predictions and probabilities
    predictions = {'y_pred': y_pred, 'y_score': y_score}
    df_predictions = pd.DataFrame.from_dict(predictions)
    
    return df_predictions    
    
    



def binary_clf_metrics(model, y_test, y_score, y_pred, plot_out = True, print_out= True):
    
    """
    model (str): the model name
    y_test (array): the true labels
    y_score (array): the predicted probabilities
    y_pred (array): the predicted labels
    plot_out (bool): whether to plot the ROC curves
    print_out (bool): whether to print the metrics
    
    Return:
        df_metric (dataframe): the metrics of the model
        df_roc_thresh (dataframe): the metrics of the model for different thresholds of ROC
        df_prc_thresh (dataframe): the metrics of the model for different thresholds of Precision-Recall
    """
    
    binaryclf_metric = {
        " Accuracy": metrics.accuracy_score(y_test, y_pred),
        " Precision": metrics.precision_score(y_test, y_pred),
        " Recall": metrics.recall_score(y_test, y_pred),
        " F1": metrics.f1_score(y_test, y_pred),
        " ROC AUC": metrics.roc_auc_score(y_test, y_score)
    }
    
    df_metric = pd.DataFrame.from_dict(binaryclf_metric, orient='index')
    df_metric.columns = [model]
    
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_score)
    
    auc = metrics.auc(fpr, tpr)
    
    # let's get maintenance capacity for thresholds_roc
    engine_roc = []
    for thr in thresholds:
        """
        here we get the mean of every row that satisfiy the condition which to be class as 1.
        every row represent 100 engine we have in our dataset
        """
        engine_roc.append((y_score >= thr).mean())
    engine_roc = np.array(engine_roc)
    
    roc_thresh = {
        'thresholds_roc': thresholds,
        'TPR': tpr,
        'FPR': fpr,
        'maintenance_capacity_roc': engine_roc
    }
    
    df_roc_thresh = pd.DataFrame.from_dict(roc_thresh)
    
    """
    let's get ather classification mertrics
    - we know that from thruth data that positive calss = 25, and negative class = 75
    - positive class = 25 = TP + FN
    - negative class = 75 = FP + TN
    - we will try here to get TP, FP, TN, FN THREN GET TNR, FNR. we already know TPR and FPR
    
    - we know that 
        - TPR (Recall = sensitivity) = TP / (TP + FN)
        - FPR (1 - Specificity) = FP / (FP + TN)
        - TNR (Specificity) = TN / (FP + TN)
        - FNR (1 - Recall/Sensitivity) = FN / (FN + TP)
    
    """
    
    df_roc_thresh['TP'] = df_roc_thresh['TPR'] * 25
    df_roc_thresh['FP'] = df_roc_thresh['FPR'] * 75
    df_roc_thresh['TN'] = (1 - df_roc_thresh['FPR']) * 75
    df_roc_thresh['FN'] = (1 - df_roc_thresh['TPR']) * 25
    
    df_roc_thresh['TNR'] = df_roc_thresh['TN'] / (df_roc_thresh['FP'] + df_roc_thresh['TN'])
    df_roc_thresh['FNR'] = df_roc_thresh['FN'] / (df_roc_thresh['FN'] + df_roc_thresh['TP'])
    
    df_roc_thresh['Model'] = model
    
    precision, recall, thresh_prc = metrics.precision_recall_curve(y_test, y_score)
    
    """
    append value 1 to the end of threshold array cuz precision_recall_curve() does not provide threshold for last point on the curve which point is (1,0).
    by addind 1 you asscoiating this point with threshold 1.
    """
    thresh_prc = np.append(thresh_prc,1)

    # let's get maintenance capacity for thresh_prc
    engine_prc = []
    for thr in thresh_prc:
        """
        here we get the mean of every row that satisfiy the condition which to be class as 1.
        every row represent 100 engine we have in our dataset
        """
        engine_prc.append((y_score >= thr).mean())
    engine_prc = np.array(engine_prc)
    
    thr_prc = {
        'thresholds_prc': thresh_prc,
        'Precision': precision,
        'Recall': recall,
        'maintenance_capacity_prc': engine_prc
    }

    df_prc_thresh = pd.DataFrame.from_dict(thr_prc)
    
    if print_out:
        print(f"Model: {model}\n")
        print('Confusion Matrix:')
        print(metrics.confusion_matrix(y_test, y_pred))
        print('\nClassification Report:')
        print(metrics.classification_report(y_test, y_pred))
        print('\nMetrics:')
        print(df_metric)

        print('\nROC Thresholds:\n')
        print(df_roc_thresh[['thresholds_roc', 'TP', 'FP', 'TN', 'FN', 'TPR', 'FPR', 'TNR','FNR', 'maintenance_capacity_roc']])

        print('\nPrecision-Recall Thresholds:\n')
        print(df_prc_thresh[['thresholds_prc', 'Precision', 'Recall', 'maintenance_capacity_prc']])

    if plot_out:
        fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2, figsize=(20,13),sharex=False, sharey=False)
        fig.set_size_inches(20,13)

        ax1.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % auc)
        ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax1.set_xlim([-0.06, 1.0]) # -0.05 better for visualization
        ax1.set_ylim([0.0, 1.05])
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.legend(loc="lower right", fontsize='small')

        ax2.plot(recall, precision, color='blue', lw=2, label='Precision-Recall curve')
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.legend(loc="lower right", fontsize='small')

        ax3.plot(thresholds, fpr, color='red', lw=2, label='FPR')
        ax3.plot(thresholds, tpr, color='green', lw=2, label='TPR')
        ax3.plot(thresholds, engine_roc, color='blue', lw=2, label='Maintenance Capacity')
        ax3.set_ylim([0.0, 1.05])
        ax3.set_xlabel('Threshold_roc')
        ax3.set_ylabel('%')
        ax3.legend(loc="upper right", fontsize='small')
        
        ax4.plot(thresh_prc, precision, color='red', lw=2, label='Precision')  
        ax4.plot(thresh_prc, recall, color='green',label='Recall') 
        ax4.plot(thresh_prc, engine_prc, color='blue',label='Maintenance Capacity') 
        ax4.set_ylim([0.0, 1.05])
        ax4.set_xlabel('Threshold_prc')  
        ax4.set_ylabel('%')
        ax4.legend(loc='lower left', fontsize='small')

    return df_metric, df_roc_thresh, df_prc_thresh
    
