
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler



def split_data(df_train, df_test, features, target, scale=True):
    """
    Args:
        df_train (dataframe): train dataframe
        df_test (dataframe): test dataframe
        features (list): list of features
        target (str): target variable
        scale (bool): if True, scale data

    Return:
        X_train (dataframe): scaled train dataframe
        X_test_ (dataframe): scaled test dataframe

    """

    X_train = df_train[features]

    X_test = df_test[features]

    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    return X_train, X_test




def multiclass_classify(model, clf, df_train, df_test, features, target, params=None, score=None, OvR=True, prob='P'):

    """

    Args:
        model (str): The model name identifier
        clf (clssifier object): The classifier to be tuned
        df_train (dataframe): train dataframe
        df_test (dataframe): test dataframe
        features (list): The set of input features names
        target (str): target lable
        params (dict): Random Search parameters
        score (str): Random Search score
        OvR (bool): True if the classifier inherently support multiclass One-Vs-Rest
        prob (str): For getting classification scores: 'P' for predict_proba, 'D' for decision_function

    Returns:
        Tuned Calssifier object
        array: prediction values
        array: prediction scores
    """


    X_train, X_test = split_data(df_train, df_test, features, target, scale=True)


    random_search = RandomizedSearchCV(estimator=clf, param_distributions= params, cv=5, scoring=score, verbose=1)

    random_search.fit(X_train, y_train)
    y_pred = random_search.predict(X_test)

    if prob == 'P':
        y_score = random_search.predict_proba(X_test)
        if OvR:    # for algorithm that inherently support multiclass One-Vs-Rest
            y_score = [y_score[i][:,[1]] for i in range(len(y_score))]
            y_score = np.concatenate(y_score, axis=1)
    elif prob == 'D':
        y_score = random_search.decision_function(X_test)
    else:
        y_score = y_pred


    return random_search.best_estimator_, y_pred, y_score




def multiclass_metrics(model, y_test, y_pred, y_score, print_out=True, plot_out=True):

    """
    Args:
        model (str): The model name identifier
        y_test (series): Contains the test label values
        y_pred (series): Contains the predicted values
        y_score (series): Contains the predicted scores
        print_out (bool): Print the classification metrics and thresholds values
        plot_out (bool): Plot AUC ROC, Precision-Recall, and Threshold curves

    Returns:
        dataframe: The combined metrics in single dataframe
        dict: ROC thresholds
        dict: Precision-Recall thresholds
        Plot: AUC ROC
        plot: Precision-Recall
    """

    multiclass_metrics = {
                            'Accuracy' : metrics.accuracy_score(y_test, y_pred),
                            'weighted F1' : metrics.f1_score(y_test, y_pred, average='weighted'),
                            'micro F1' : metrics.f1_score(y_test, y_pred, average='micro'),
                            'weighted Precision' : metrics.precision_score(y_test, y_pred,  average='weighted'),
                            'micro Precision' : metrics.precision_score(y_test, y_pred,  average='micro'),
                            'weighted Recall' : metrics.recall_score(y_test, y_pred,  average='weighted'),
                            'micro Recall' : metrics.recall_score(y_test, y_pred,  average='micro'),
                            'weighted ROC AUC' : metrics.roc_auc_score(y_test, y_score, average='weighted'),
                            'micro ROC AUC' : metrics.roc_auc_score(y_test, y_score, average='micro')
                        }

    df_metrics = pd.DataFrame.from_dict(multiclass_metrics, orient='index')
    df_metrics.columns = [model]


    n_classes = y_train.shape[1]

    fpr = dict()
    tpr = dict()
    threshold_roc = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], threshold_roc[i] = metrics.roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])

    # use micro as an average value represent 3 classes
    fpr["micro"], tpr["micro"], threshold_roc["micro"] = metrics.roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])


    roc_threshold = {
                    'Threshold_ROC' : threshold_roc,
                    'TPR' : tpr,
                    'FPR' : fpr,
                    'AUC' : roc_auc
                }

    df_roc_threshold = pd.DataFrame.from_dict(roc_threshold)
    df_roc_threshold['Model'] = model
    df_roc_threshold['Class'] = df_roc_threshold.index



    precision = dict()
    recall = dict()
    threshold_prc = dict()
    average_precision = dict()

    for i in range(n_classes):
        precision[i], recall[i], threshold_prc[i] = metrics.precision_recall_curve(y_test[:, i], y_score[:, i])
        average_precision[i] = metrics.average_precision_score(y_test[:, i], y_score[:, i])

    precision["micro"], recall["micro"], threshold_prc["micro"] = metrics.precision_recall_curve(y_test.ravel(), y_score.ravel())
    average_precision["micro"] = metrics.average_precision_score(y_test, y_score, average="micro")

    prc_thresh = {
                    'Threshold_PRC' : threshold_prc,
                    'Precision' : precision,
                    'Recall' : recall,
                    'Avg Precision' : average_precision
                }

    df_prc_threshold = pd.DataFrame.from_dict(prc_thresh)
    df_prc_threshold['Model'] = model
    df_prc_threshold['Class'] = df_prc_threshold.index


    y_test_orig = lb.inverse_transform(y_test)
    y_pred_orig = lb.inverse_transform(y_pred)

    if print_out:
        print('*'*70)
        print(model, '\n')
        print('Confusion Matrix:')
        print(metrics.confusion_matrix(y_test_orig, y_pred_orig))
        print('\nClassification Report:')
        print(metrics.classification_report(y_test_orig, y_pred_orig))
        print('\nMetrics:')
        print(df_metrics)

    if plot_out:

        colors = ['red', 'blue', 'green']

        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12,6), sharex=False, sharey=False )
        fig.set_size_inches(12,6)

        for i, color in zip(range(n_classes), colors):
            ax1.plot(fpr[i], tpr[i], color=color, lw=1, label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))

        ax1.plot(fpr["micro"], tpr["micro"], color='darkmagenta', lw=3, label='micro-average ROC curve (area = {0:0.2f})'''.format(roc_auc["micro"]), linestyle='dashdot')
        ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax1.set_xlim([-0.06, 1.0])
        ax1.set_ylim([0.0, 1.05])
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.legend(loc="lower right", fontsize='small')


        for i, color in zip(range(n_classes), colors):
            ax2.plot(recall[i], precision[i], color=color, lw=1, label='Precision-recall curve of class {0} (area = {1:0.2f})'.format(i, average_precision[i]))

        ax2.plot(recall["micro"], precision["micro"], color='darkmagenta', lw=3, linestyle='dashdot', label='micro-average Precision-recall curve (area = {0:0.2f})'''.format(average_precision["micro"]))
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.legend(loc="lower left", fontsize='small')

    return df_metrics, df_prc_threshold, df_roc_threshold


