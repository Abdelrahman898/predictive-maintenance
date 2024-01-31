
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier



# split data function
def split_data(df, features, target, print_shapes=False, RANDOM_STATE = 42):
    """
    Args:
        df (dataframe): dataframe to split 
        features (list): list of features
        target (str): target variable
        print_shapes (bool): if True, print shapes of train and validation
        RANDOM_STATE (int): random state
        
    """
    

    X = df[features]
    y = df[target]
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    
    if print_shapes:
        print(f"X_train shape: {X_train.shape}\nX_val shape: {X_val.shape}\ny_train shape: {y_train.shape}\ny_val shape: {y_val.shape}")
    
    return X_train, X_val, y_train, y_val

    
    
# scale data function
def scale_data(X_train, X_val):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    return X_train_scaled, X_val_scaled


                    ####..................................................................####
                    ####..................................................................####


# DecisionTreeClassifier
def plot_decisionTree_roc(df, features, target, plot_roc = True, RANDOM_STATE = 42):
    
    """
    Args:
        df: dataframe
        features: list of features
        target: target variable
        plot_roc: boolean
    
    Return:
        fig: figures with ROC curve fro the DecisionTreeClassifier hyprerparameters min_samples_split and max_depth values.
        it's help to choose the best parameters to avoid overfitting.
        
    """
    
    X_train, X_val, y_train, y_val = split_data(df, features, target)
    X_train_scaled, X_val_scaled = scale_data(X_train, X_val)   
    
    min_samples_split_list = [2,10, 30, 50, 100, 200, 300, 700] 
    max_depth_list = [1,2, 3, 4, 8, 16, 32, 64, None] # None means that there is no depth limit.
    
    roc_minsamples_train = []
    roc_minsample_val = []
    roc_maxdepth_train = []
    roc_maxdepth_val = []
    

    for min_samples_split in min_samples_split_list:
        
        # You can fit the model at the same time you define it, because the fit function returns the fitted estimator.
        model = DecisionTreeClassifier(min_samples_split=min_samples_split,random_state=RANDOM_STATE).fit(X_train_scaled,y_train)        

        if hasattr(model, 'predict_proba'):
            predictions_train = model.predict_proba(X_train_scaled)[:,1] ## The predicted probabilities for the train dataset
            predictions_val = model.predict_proba(X_val_scaled)[:,1] ## The predicted probabilities for the validation dataset
        elif hasattr(model, 'decision_function'):
            predictions_train = model.decision_function(X_train_scaled) ## The predicted values for the train dataset
            predictions_val = model.decision_function(X_val_scaled) ## The predicted values for the test dataset
        else:
            predictions_train = model.predict(X_train_scaled) ## The predicted values for the train dataset
            predictions_val = model.predict(X_val_scaled) ## The predicted values for the test dataset
    
        
        accuracy_train = metrics.roc_auc_score(y_train, predictions_train)
        accuracy_val = metrics.roc_auc_score(y_val, predictions_val)
        roc_minsamples_train.append(accuracy_train)
        roc_minsample_val.append(accuracy_val)

    for max_depth in max_depth_list:
    
        # You can fit the model at the same time you define it, because the fit function returns the fitted estimator.
        model = DecisionTreeClassifier(max_depth = max_depth,random_state = RANDOM_STATE).fit(X_train_scaled,y_train) 
        
        if hasattr(model, 'predict_proba'):
            predictions_train = model.predict_proba(X_train_scaled)[:,1] ## The predicted probabilities for the train dataset
            predictions_val = model.predict_proba(X_val_scaled)[:,1] ## The predicted probabilities for the validation dataset
        elif hasattr(model, 'decision_function'):
            predictions_train = model.decision_function(X_train_scaled) ## The predicted values for the train dataset
            predictions_val = model.decision_function(X_val_scaled) ## The predicted values for the test dataset
        else:
            predictions_train = model.predict(X_train_scaled) ## The predicted values for the train dataset
            predictions_val = model.predict(X_val_scaled) ## The predicted values for the test dataset
        
        
        accuracy_train = metrics.roc_auc_score(y_train, predictions_train)
        accuracy_val = metrics.roc_auc_score(y_val, predictions_val)
        roc_maxdepth_train.append(accuracy_train)
        roc_maxdepth_val.append(accuracy_val)

    if plot_roc:
        fig, ((ax1,ax2)) = plt.subplots(1,2, figsize=(15,6),sharex=False, sharey=False)
        
        ax1.plot(roc_minsamples_train)
        ax1.plot(roc_minsample_val)
        ax1.set_xlabel("min_samples_split")
        ax1.set_ylabel('roc_auc_score')
        ax1.set_xticks(range(len(min_samples_split_list)),labels=min_samples_split_list)
        ax1.legend(['Train','Validation'])
        
        ax2.plot(roc_maxdepth_train)
        ax2.plot(roc_maxdepth_val)
        ax2.set_xlabel('max_depth')
        ax2.set_ylabel('roc_auc_score')
        ax2.set_xticks(ticks = range(len(max_depth_list )),labels=max_depth_list)
        ax2.legend(['Train','Validation'])
                
                
def plot_decisionTree_accuracy(df, features, target, plot_accuracy = True, RANDOM_STATE = 42):
    
    """
    Args:
        df: dataframe
        features: list of features
        target: target variable
        plot_accuracy: boolean
    
    Return:
        fig: figures with accuracy_score curves for the DecisionTreeClassifier hyprerparameters min_samples_split and max_depth values.
        it's help to choose the best parameters to avoid overfitting. as accuracy may be not the best metric to evaluate cua our imbalanced dataset.
        
    """
    
    X_train, X_val, y_train, y_val = split_data(df, features, target)
    X_train_scaled, X_val_scaled = scale_data(X_train, X_val)   
    
    min_samples_split_list = [2,10, 30, 50, 100, 200, 300, 700] 
    max_depth_list = [1,2, 3, 4, 8, 16, 32, 64, None] # None means that there is no depth limit.
    
    accuracy_minsamples_train = []
    accuracy_minsample_val = []
    accuracy_maxdepth_train = []
    accuracy_maxdepth_val = []
    

    for min_samples_split in min_samples_split_list:
        
        # You can fit the model at the same time you define it, because the fit function returns the fitted estimator.
        model = DecisionTreeClassifier(min_samples_split=min_samples_split,random_state=RANDOM_STATE).fit(X_train_scaled,y_train)        

        predictions_train = model.predict(X_train_scaled) ## The predicted values for the train dataset
        predictions_val = model.predict(X_val_scaled) ## The predicted values for the test dataset
    
        
        accuracy_train = metrics.accuracy_score(y_train, predictions_train)
        accuracy_val = metrics.accuracy_score(y_val, predictions_val)
        accuracy_minsamples_train.append(accuracy_train)
        accuracy_minsample_val.append(accuracy_val)

    for max_depth in max_depth_list:
    
        # You can fit the model at the same time you define it, because the fit function returns the fitted estimator.
        model = DecisionTreeClassifier(max_depth = max_depth,random_state = RANDOM_STATE).fit(X_train_scaled,y_train) 
        
        predictions_train = model.predict(X_train_scaled) ## The predicted values for the train dataset
        predictions_val = model.predict(X_val_scaled) ## The predicted values for the test dataset
        
        
        accuracy_train = metrics.accuracy_score(y_train, predictions_train)
        accuracy_val = metrics.accuracy_score(y_val, predictions_val)
        accuracy_maxdepth_train.append(accuracy_train)
        accuracy_maxdepth_val.append(accuracy_val)

    if plot_accuracy:
        fig, ((ax1,ax2)) = plt.subplots(1,2, figsize=(15,6),sharex=False, sharey=False)
        
        ax1.plot(accuracy_minsamples_train)
        ax1.plot(accuracy_minsample_val)
        ax1.set_xlabel("min_samples_split")
        ax1.set_ylabel('accuracy_score')
        ax1.set_xticks(range(len(min_samples_split_list)),labels=min_samples_split_list)
        ax1.legend(['Train','Validation'])
        
        ax2.plot(accuracy_maxdepth_train)
        ax2.plot(accuracy_maxdepth_val)
        ax2.set_xlabel('max_depth')
        ax2.set_ylabel('accuracy_score')
        ax2.set_xticks(ticks = range(len(max_depth_list )),labels=max_depth_list)
        ax2.legend(['Train','Validation'])
                
                
def plot_decisionTree_f1(df, features, target, plot_f1 = True, RANDOM_STATE = 42):
    
    """
    Args:
        df: dataframe
        features: list of features
        target: target variable
        plot_f1: boolean
    
    Return:
        fig: figures with f1_score curves for the DecisionTreeClassifier hyprerparameters min_samples_split and max_depth values.
        it's help to choose the best parameters to avoid overfitting. as accuracy may be not the best metric to evaluate cua our imbalanced dataset.
        
    """
    
    X_train, X_val, y_train, y_val = split_data(df, features, target)
    X_train_scaled, X_val_scaled = scale_data(X_train, X_val)   
    
    min_samples_split_list = [2,10, 30, 50, 100, 200, 300, 700] 
    max_depth_list = [1,2, 3, 4, 8, 16, 32, 64, None] # None means that there is no depth limit.
    
    f1_minsamples_train = []
    f1_minsample_val = []
    f1_maxdepth_train = []
    f1_maxdepth_val = []
    

    for min_samples_split in min_samples_split_list:
        
        # You can fit the model at the same time you define it, because the fit function returns the fitted estimator.
        model = DecisionTreeClassifier(min_samples_split=min_samples_split,random_state=RANDOM_STATE).fit(X_train_scaled,y_train)        

        predictions_train = model.predict(X_train_scaled) ## The predicted values for the train dataset
        predictions_val = model.predict(X_val_scaled) ## The predicted values for the test dataset
    
        
        accuracy_train = metrics.f1_score(y_train, predictions_train)
        accuracy_val = metrics.f1_score(y_val, predictions_val)
        f1_minsamples_train.append(accuracy_train)
        f1_minsample_val.append(accuracy_val)

    for max_depth in max_depth_list:
    
        # You can fit the model at the same time you define it, because the fit function returns the fitted estimator.
        model = DecisionTreeClassifier(max_depth = max_depth,random_state = RANDOM_STATE).fit(X_train_scaled,y_train) 
        
        predictions_train = model.predict(X_train_scaled) ## The predicted values for the train dataset
        predictions_val = model.predict(X_val_scaled) ## The predicted values for the test dataset
        
        
        accuracy_train = metrics.f1_score(y_train, predictions_train)
        accuracy_val = metrics.f1_score(y_val, predictions_val)
        f1_maxdepth_train.append(accuracy_train)
        f1_maxdepth_val.append(accuracy_val)

    if plot_f1:
        fig, ((ax1,ax2)) = plt.subplots(1,2, figsize=(15,6),sharex=False, sharey=False)
        
        ax1.plot(f1_minsamples_train)
        ax1.plot(f1_minsample_val)
        ax1.set_xlabel("min_samples_split")
        ax1.set_ylabel('f1_score')
        ax1.set_xticks(range(len(min_samples_split_list)),labels=min_samples_split_list)
        ax1.legend(['Train','Validation'])
        
        ax2.plot(f1_maxdepth_train)
        ax2.plot(f1_maxdepth_val)
        ax2.set_xlabel('max_depth')
        ax2.set_ylabel('f1_score')
        ax2.set_xticks(ticks = range(len(max_depth_list )),labels=max_depth_list)
        ax2.legend(['Train','Validation'])
                
                
                    ####..................................................................####
                    ####..................................................................####

# RandomForestClassifier

def plot_randomForest_roc(df, features, target, plot_roc = True, RANDOM_STATE = 42):
    
    """
    Args:
        df: dataframe
        features: list of features
        target: target variable
        plot_roc: boolean
    
    Return:
        fig: figures with ROC curve fro the RandomForestClassifier hyprerparameters min_samples_split, n_estimators, and max_depth values.
        it's help to choose the best parameters to avoid overfitting.
        
    """
    
    X_train, X_val, y_train, y_val = split_data(df, features, target)
    X_train_scaled, X_val_scaled = scale_data(X_train, X_val)   
    
    min_samples_split_list = [2,10, 30, 50, 100, 300, 500, 700]
    max_depth_list = [1,2, 3, 4, 7, 8, 16, 32, 64, None] # None means that there is no depth limit.
    n_estimators_list = [2,5,10,50,100,500,700]
    
    roc_minsamples_train = []
    roc_minsample_val = []
    roc_maxdepth_train = []
    roc_maxdepth_val = []
    roc_n_estimators_train = []
    roc_n_estimators_val = []
    

    for min_samples_split in min_samples_split_list:
        
        # You can fit the model at the same time you define it, because the fit function returns the fitted estimator.
        model = RandomForestClassifier(min_samples_split=min_samples_split,random_state=RANDOM_STATE).fit(X_train_scaled,y_train)        

        if hasattr(model, 'predict_proba'):
            predictions_train = model.predict_proba(X_train_scaled)[:,1] ## The predicted probabilities for the train dataset
            predictions_val = model.predict_proba(X_val_scaled)[:,1] ## The predicted probabilities for the validation dataset
        elif hasattr(model, 'decision_function'):
            predictions_train = model.decision_function(X_train_scaled) ## The predicted values for the train dataset
            predictions_val = model.decision_function(X_val_scaled) ## The predicted values for the test dataset
        else:
            predictions_train = model.predict(X_train_scaled) ## The predicted values for the train dataset
            predictions_val = model.predict(X_val_scaled) ## The predicted values for the test dataset
    
        
        accuracy_train = metrics.roc_auc_score(y_train, predictions_train)
        accuracy_val = metrics.roc_auc_score(y_val, predictions_val)
        roc_minsamples_train.append(accuracy_train)
        roc_minsample_val.append(accuracy_val)

    for max_depth in max_depth_list:
    
        # You can fit the model at the same time you define it, because the fit function returns the fitted estimator.
        model = RandomForestClassifier(max_depth = max_depth,random_state = RANDOM_STATE).fit(X_train_scaled,y_train) 
        
        if hasattr(model, 'predict_proba'):
            predictions_train1 = model.predict_proba(X_train_scaled)[:,1] ## The predicted probabilities for the train dataset
            predictions_val1 = model.predict_proba(X_val_scaled)[:,1] ## The predicted probabilities for the validation dataset
        elif hasattr(model, 'decision_function'):
            predictions_train1 = model.decision_function(X_train_scaled) ## The predicted values for the train dataset
            predictions_val1 = model.decision_function(X_val_scaled) ## The predicted values for the test dataset
        else:
            predictions_train1 = model.predict(X_train_scaled) ## The predicted values for the train dataset
            predictions_val1 = model.predict(X_val_scaled) ## The predicted values for the test dataset
        
        
        accuracy_train1 = metrics.roc_auc_score(y_train, predictions_train1)
        accuracy_val1 = metrics.roc_auc_score(y_val, predictions_val1)
        roc_maxdepth_train.append(accuracy_train1)
        roc_maxdepth_val.append(accuracy_val1)

    for n_estimator in n_estimators_list:
    
        # You can fit the model at the same time you define it, because the fit function returns the fitted estimator.
        model = RandomForestClassifier(n_estimators=n_estimator,random_state = RANDOM_STATE).fit(X_train_scaled,y_train) 
        
        if hasattr(model, 'predict_proba'):
            predictions_train2 = model.predict_proba(X_train_scaled)[:,1] ## The predicted probabilities for the train dataset
            predictions_val2 = model.predict_proba(X_val_scaled)[:,1] ## The predicted probabilities for the validation dataset
        elif hasattr(model, 'decision_function'):
            predictions_train2 = model.decision_function(X_train_scaled) ## The predicted values for the train dataset
            predictions_val2 = model.decision_function(X_val_scaled) ## The predicted values for the test dataset
        else:
            predictions_train2 = model.predict(X_train_scaled) ## The predicted values for the train dataset
            predictions_val2 = model.predict(X_val_scaled) ## The predicted values for the test dataset
        
        
        accuracy_train2 = metrics.roc_auc_score(y_train, predictions_train2)
        accuracy_val2 = metrics.roc_auc_score(y_val, predictions_val2)
        roc_n_estimators_train.append(accuracy_train2)
        roc_n_estimators_val.append(accuracy_val2)

    

    if plot_roc:
        fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2, figsize=(15,12),sharex=False, sharey=False)
        
        ax1.plot(roc_minsamples_train)
        ax1.plot(roc_minsample_val)
        ax1.set_xlabel("min_samples_split")
        ax1.set_ylabel('roc_auc_score')
        ax1.set_xticks(range(len(min_samples_split_list)),labels=min_samples_split_list)
        ax1.legend(['Train','Validation'])
        
        ax2.plot(roc_maxdepth_train)
        ax2.plot(roc_maxdepth_val)
        ax2.set_xlabel('max_depth')
        ax2.set_ylabel('roc_auc_score')
        ax2.set_xticks(ticks = range(len(max_depth_list )),labels=max_depth_list)
        ax2.legend(['Train','Validation'])
        
        ax3.plot(roc_n_estimators_train)
        ax3.plot(roc_n_estimators_val)
        ax3.set_xlabel('n_estimators')
        ax3.set_ylabel('roc_auc_score')
        ax3.set_xticks(ticks = range(len(n_estimators_list )),labels=n_estimators_list)
        ax3.legend(['Train','Validation'])

def plot_randomForest_accuracy(df, features, target, plot_accuracy = True, RANDOM_STATE = 42):
    
    """
    Args:
        df: dataframe
        features: list of features
        target: target variable
        plot_accuracy: boolean
    
    Return:
        fig: figures with accuracy_score curve fro the RandomForestClassifier hyprerparameters min_samples_split, n_estimators, and max_depth values.
        it's help to choose the best parameters to avoid overfitting.
        
    """
    
    X_train, X_val, y_train, y_val = split_data(df, features, target)
    X_train_scaled, X_val_scaled = scale_data(X_train, X_val)   
    
    min_samples_split_list = [2,10, 30, 50, 100, 300, 500, 700]
    max_depth_list = [1,2, 3, 4, 7, 8, 16, 32, 64, None] # None means that there is no depth limit.
    n_estimators_list = [2,5,10,50,100,500,700]
    
    accuracy_minsamples_train = []
    accuracy_minsample_val = []
    accuracy_maxdepth_train = []
    accuracy_maxdepth_val = []
    accuracy_n_estimators_train = []
    accuracy_n_estimators_val = []
    

    for min_samples_split in min_samples_split_list:
        
        # You can fit the model at the same time you define it, because the fit function returns the fitted estimator.
        model = RandomForestClassifier(min_samples_split=min_samples_split,random_state=RANDOM_STATE).fit(X_train_scaled,y_train)        

        predictions_train = model.predict(X_train_scaled) ## The predicted values for the train dataset
        predictions_val = model.predict(X_val_scaled) ## The predicted values for the test dataset
    
        
        accuracy_train = metrics.accuracy_score(y_train, predictions_train)
        accuracy_val = metrics.accuracy_score(y_val, predictions_val)
        accuracy_minsamples_train.append(accuracy_train)
        accuracy_minsample_val.append(accuracy_val)

    for max_depth in max_depth_list:
    
        # You can fit the model at the same time you define it, because the fit function returns the fitted estimator.
        model = RandomForestClassifier(max_depth = max_depth,random_state = RANDOM_STATE).fit(X_train_scaled,y_train) 
        
    
        predictions_train1 = model.predict(X_train_scaled) ## The predicted values for the train dataset
        predictions_val1 = model.predict(X_val_scaled) ## The predicted values for the test dataset
    
        
        accuracy_train1 = metrics.accuracy_score(y_train, predictions_train1)
        accuracy_val1 = metrics.accuracy_score(y_val, predictions_val1)
        accuracy_maxdepth_train.append(accuracy_train1)
        accuracy_maxdepth_val.append(accuracy_val1)

    for n_estimator in n_estimators_list:
    
        # You can fit the model at the same time you define it, because the fit function returns the fitted estimator.
        model = RandomForestClassifier(n_estimators=n_estimator,random_state = RANDOM_STATE).fit(X_train_scaled,y_train) 
        
    
        predictions_train2 = model.predict(X_train_scaled) ## The predicted values for the train dataset
        predictions_val2 = model.predict(X_val_scaled) ## The predicted values for the test dataset
    
        
        accuracy_train2 = metrics.accuracy_score(y_train, predictions_train2)
        accuracy_val2 = metrics.accuracy_score(y_val, predictions_val2)
        accuracy_n_estimators_train.append(accuracy_train2)
        accuracy_n_estimators_val.append(accuracy_val2)
    

    if plot_accuracy:
        fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2, figsize=(15,12),sharex=False, sharey=False)
        
        ax1.plot(accuracy_minsamples_train)
        ax1.plot(accuracy_minsample_val)
        ax1.set_xlabel("min_samples_split")
        ax1.set_ylabel('accuracy_score')
        ax1.set_xticks(range(len(min_samples_split_list)),labels=min_samples_split_list)
        ax1.legend(['Train','Validation'])
        
        ax2.plot(accuracy_maxdepth_train)
        ax2.plot(accuracy_maxdepth_val)
        ax2.set_xlabel('max_depth')
        ax2.set_ylabel('accuracy_score')
        ax2.set_xticks(ticks = range(len(max_depth_list )),labels=max_depth_list)
        ax2.legend(['Train','Validation'])
        
        ax3.plot(accuracy_n_estimators_train)
        ax3.plot(accuracy_n_estimators_val)
        ax3.set_xlabel('n_estimators')
        ax3.set_ylabel('accuracy_score')
        ax3.set_xticks(ticks = range(len(n_estimators_list )),labels=n_estimators_list)
        ax3.legend(['Train','Validation'])


def plot_randomForest_f1(df, features, target, plot_f1 = True, RANDOM_STATE = 42):
    
    """
    Args:
        df: dataframe
        features: list of features
        target: target variable
        plot_f1: boolean
    
    Return:
        fig: figures with f1_score curve fro the RandomForestClassifier hyprerparameters min_samples_split, n_estimators, and max_depth values.
        it's help to choose the best parameters to avoid overfitting.
        
    """
    
    X_train, X_val, y_train, y_val = split_data(df, features, target)
    X_train_scaled, X_val_scaled = scale_data(X_train, X_val)   
    
    min_samples_split_list = [2,10, 30, 50, 100, 300, 500, 700]
    max_depth_list = [1,2, 3, 4, 7, 8, 16, 32, 64, None] # None means that there is no depth limit.
    n_estimators_list = [2,5,10,50,100,500,700]
    
    f1_minsamples_train = []
    f1_minsample_val = []
    f1_maxdepth_train = []
    f1_maxdepth_val = []
    f1_n_estimators_train = []
    f1_n_estimators_val = []
    

    for min_samples_split in min_samples_split_list:
        
        # You can fit the model at the same time you define it, because the fit function returns the fitted estimator.
        model = RandomForestClassifier(min_samples_split=min_samples_split,random_state=RANDOM_STATE).fit(X_train_scaled,y_train)        

        predictions_train = model.predict(X_train_scaled) ## The predicted values for the train dataset
        predictions_val = model.predict(X_val_scaled) ## The predicted values for the test dataset
    
        
        accuracy_train = metrics.f1_score(y_train, predictions_train)
        accuracy_val = metrics.f1_score(y_val, predictions_val)
        f1_minsamples_train.append(accuracy_train)
        f1_minsample_val.append(accuracy_val)

    for max_depth in max_depth_list:
    
        # You can fit the model at the same time you define it, because the fit function returns the fitted estimator.
        model = RandomForestClassifier(max_depth = max_depth,random_state = RANDOM_STATE).fit(X_train_scaled,y_train) 
        
    
        predictions_train1 = model.predict(X_train_scaled) ## The predicted values for the train dataset
        predictions_val1 = model.predict(X_val_scaled) ## The predicted values for the test dataset
    
        
        accuracy_train1 = metrics.f1_score(y_train, predictions_train1)
        accuracy_val1 = metrics.f1_score(y_val, predictions_val1)
        f1_maxdepth_train.append(accuracy_train1)
        f1_maxdepth_val.append(accuracy_val1)

    for n_estimator in n_estimators_list:
    
        # You can fit the model at the same time you define it, because the fit function returns the fitted estimator.
        model = RandomForestClassifier(n_estimators=n_estimator,random_state = RANDOM_STATE).fit(X_train_scaled,y_train) 
        
    
        predictions_train2 = model.predict(X_train_scaled) ## The predicted values for the train dataset
        predictions_val2 = model.predict(X_val_scaled) ## The predicted values for the test dataset
    
        
        accuracy_train2 = metrics.f1_score(y_train, predictions_train2)
        accuracy_val2 = metrics.f1_score(y_val, predictions_val2)
        f1_n_estimators_train.append(accuracy_train2)
        f1_n_estimators_val.append(accuracy_val2)
    

    if plot_f1:
        fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2, figsize=(15,12),sharex=False, sharey=False)
        
        ax1.plot(f1_minsamples_train)
        ax1.plot(f1_minsample_val)
        ax1.set_xlabel("min_samples_split")
        ax1.set_ylabel('f1_score')
        ax1.set_xticks(range(len(min_samples_split_list)),labels=min_samples_split_list)
        ax1.legend(['Train','Validation'])
        
        ax2.plot(f1_maxdepth_train)
        ax2.plot(f1_maxdepth_val)
        ax2.set_xlabel('max_depth')
        ax2.set_ylabel('f1_score')
        ax2.set_xticks(ticks = range(len(max_depth_list )),labels=max_depth_list)
        ax2.legend(['Train','Validation'])
        
        ax3.plot(f1_n_estimators_train)
        ax3.plot(f1_n_estimators_val)
        ax3.set_xlabel('n_estimators')
        ax3.set_ylabel('f1_score')
        ax3.set_xticks(ticks = range(len(n_estimators_list )),labels=n_estimators_list)
        ax3.legend(['Train','Validation'])




                    ####..................................................................####
                    ####..................................................................####

# KNNClassifier

def plot_KNN_accuracy(df, features, target, plot_accuracy = True, RANDOM_STATE = 42):
    """
    Args:
        df (dataframe): train dataframe
        features (list): list of features
        target (str): target variable
        plot_accuracy (bool): plot accuracy curve
        RANDOM_STATE (int): random state
        
    Return:
        accuracy_score plot for train and validation for n_neighbors values from 1 to 21 ti help in finding best n_neighbors.
        
    """
    
    X_train, X_val, y_train, y_val = split_data(df, features, target)
    X_train_scaled, X_val_scaled = scale_data(X_train, X_val)
    
    accuracy_n_neighbors_train = []
    accuracy_n_neighbors_val = []
    
    for n_neighbors in range(1,22):
        model = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X_train_scaled,y_train)        

        predictions_train = model.predict(X_train_scaled)
        predictions_val = model.predict(X_val_scaled)

        accuracy_train = metrics.accuracy_score(y_train, predictions_train)
        accuracy_val = metrics.accuracy_score(y_val, predictions_val)
        accuracy_n_neighbors_train.append(accuracy_train)
        accuracy_n_neighbors_val.append(accuracy_val)    
    
    if plot_accuracy:
        fig, ax = plt.subplots(figsize=(15,6))
        ax.plot(accuracy_n_neighbors_train)
        ax.plot(accuracy_n_neighbors_val)
        ax.set_xlabel("n_neighbors")
        ax.set_ylabel('accuracy_score')
        ax.set_xticks(range(len(range(1,22))),labels=range(1,22))
        ax.legend(['Train','Validation'])



def plot_KNN_roc(df, features, target, plot_roc = True, RANDOM_STATE = 42):
    """
    Args:
        df (dataframe): train dataframe
        features (list): list of features
        target (str): target variable
        plot_roc (bool): plot roc curve
        RANDOM_STATE (int): random state
        
    Return:
        roc curve plot for train and validation for n_neighbors values from 1 to 50 ti help in finding best n_neighbors.  

    """
    
    X_train, X_val, y_train, y_val = split_data(df, features, target)
    X_train_scaled, X_val_scaled = scale_data(X_train, X_val)
    
    roc_n_neighbors_train = []
    roc_n_neighbors_val = []
    
    n_neighbors_list = [1,10,30,50,70,100]
    
    for n_neighbors in n_neighbors_list:
        model = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X_train_scaled,y_train)        

        if hasattr(model, 'predict_proba'):
            predictions_train = model.predict_proba(X_train_scaled)[:,1]
            predictions_val = model.predict_proba(X_val_scaled)[:,1]
        elif hasattr(model, 'decision_function'):
            predictions_train = model.decision_function(X_train_scaled)
            predictions_val = model.decision_function(X_val_scaled)
        else:
            predictions_train = model.predict(X_train_scaled)
            predictions_val = model.predict(X_val_scaled)
            
        
        roc_train = metrics.roc_auc_score(y_train, predictions_train)
        roc_val = metrics.roc_auc_score(y_val, predictions_val)
        roc_n_neighbors_train.append(roc_train)
        roc_n_neighbors_val.append(roc_val)    
    
    if plot_roc:
        fig, ax = plt.subplots(figsize=(15,6))
        ax.plot(roc_n_neighbors_train)
        ax.plot(roc_n_neighbors_val)
        ax.set_xlabel("n_neighbors")
        ax.set_ylabel('roc_auc_score')
        ax.set_xticks(range(len(n_neighbors_list)),labels=n_neighbors_list)
        ax.legend(['Train','Validation'])




def plot_KNN_f1(df, features, target, plot_f1 = True, RANDOM_STATE = 42):
    """
    Args:
        df (dataframe): train dataframe
        features (list): list of features
        target (str): target variable
        plot_f1 (bool): plot accuracy curve
        RANDOM_STATE (int): random state
        
    Return:
        f1_score plot for train and validation for n_neighbors values from 1 to 50 to help in finding best n_neighbors.
        
    """
    
    X_train, X_val, y_train, y_val = split_data(df, features, target)
    X_train_scaled, X_val_scaled = scale_data(X_train, X_val)
    
    f1_n_neighbors_train = []
    f1_n_neighbors_val = []
    
    for n_neighbors in range(1,51):
        model = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X_train_scaled,y_train)        

        predictions_train = model.predict(X_train_scaled)
        predictions_val = model.predict(X_val_scaled)

        accuracy_train = metrics.f1_score(y_train, predictions_train)
        accuracy_val = metrics.f1_score(y_val, predictions_val)
        f1_n_neighbors_train.append(accuracy_train)
        f1_n_neighbors_val.append(accuracy_val)    
    
    if plot_f1:
        fig, ax = plt.subplots(figsize=(15,6))
        ax.plot(f1_n_neighbors_train)
        ax.plot(f1_n_neighbors_val)
        ax.set_xlabel("n_neighbors")
        ax.set_ylabel('f1_score')
        ax.set_xticks(range(len(range(1,51))),labels=range(1,51))
        ax.legend(['Train','Validation'])




                    ####..................................................................####
                    ####..................................................................####


# LGBMClassifier

def plot_LightGBM_roc(df, features, target, plot_roc = True, RANDOM_STATE = 42):
    
    """
    Args:
        df: dataframe
        features: list of features
        target: target variable
        plot_roc: boolean
    
    Return:
        fig: figures with ROC curve fro the LGBMClassifier hyprerparameters n_estimator, learning_rate, and max_depth values.
        it's help to choose the best parameters to avoid overfitting.
        
    """
    
    X_train, X_val, y_train, y_val = split_data(df, features, target)
    X_train_scaled, X_val_scaled = scale_data(X_train, X_val)   
    
    learning_rate_list = [0.01, 0.1, 0.9, 1]
    max_depth_list = [2, 3, 4, 7, 8, 16, 32, 65] # None means that there is no depth limit.
    n_estimators_list = [2,5,10,50,100,500,700]
    
    roc_lreate_train = []
    roc_lreate_val = []
    roc_maxdepth_train = []
    roc_maxdepth_val = []
    roc_n_estimators_train = []
    roc_n_estimators_val = []
    

    for learning_rates in learning_rate_list:
        
        # You can fit the model at the same time you define it, because the fit function returns the fitted estimator.
        model = LGBMClassifier(learning_rate=learning_rates,random_state=RANDOM_STATE).fit(X_train_scaled,y_train)        

        if hasattr(model, 'predict_proba'):
            predictions_train = model.predict_proba(X_train_scaled)[:,1] ## The predicted probabilities for the train dataset
            predictions_val = model.predict_proba(X_val_scaled)[:,1] ## The predicted probabilities for the validation dataset
        elif hasattr(model, 'decision_function'):
            predictions_train = model.decision_function(X_train_scaled) ## The predicted values for the train dataset
            predictions_val = model.decision_function(X_val_scaled) ## The predicted values for the test dataset
        else:
            predictions_train = model.predict(X_train_scaled) ## The predicted values for the train dataset
            predictions_val = model.predict(X_val_scaled) ## The predicted values for the test dataset
    
        
        accuracy_train = metrics.roc_auc_score(y_train, predictions_train)
        accuracy_val = metrics.roc_auc_score(y_val, predictions_val)
        roc_lreate_train.append(accuracy_train)
        roc_lreate_val.append(accuracy_val)

    for max_depth in max_depth_list:
    
        # You can fit the model at the same time you define it, because the fit function returns the fitted estimator.
        model = LGBMClassifier(max_depth = max_depth,random_state = RANDOM_STATE).fit(X_train_scaled,y_train) 
        
        if hasattr(model, 'predict_proba'):
            predictions_train1 = model.predict_proba(X_train_scaled)[:,1] ## The predicted probabilities for the train dataset
            predictions_val1 = model.predict_proba(X_val_scaled)[:,1] ## The predicted probabilities for the validation dataset
        elif hasattr(model, 'decision_function'):
            predictions_train1 = model.decision_function(X_train_scaled) ## The predicted values for the train dataset
            predictions_val1 = model.decision_function(X_val_scaled) ## The predicted values for the test dataset
        else:
            predictions_train1 = model.predict(X_train_scaled) ## The predicted values for the train dataset
            predictions_val1 = model.predict(X_val_scaled) ## The predicted values for the test dataset
        
        
        accuracy_train1 = metrics.roc_auc_score(y_train, predictions_train1)
        accuracy_val1 = metrics.roc_auc_score(y_val, predictions_val1)
        roc_maxdepth_train.append(accuracy_train1)
        roc_maxdepth_val.append(accuracy_val1)

    for n_estimator in n_estimators_list:
    
        # You can fit the model at the same time you define it, because the fit function returns the fitted estimator.
        model = LGBMClassifier(n_estimators=n_estimator,random_state = RANDOM_STATE).fit(X_train_scaled,y_train) 
        
        if hasattr(model, 'predict_proba'):
            predictions_train2 = model.predict_proba(X_train_scaled)[:,1] ## The predicted probabilities for the train dataset
            predictions_val2 = model.predict_proba(X_val_scaled)[:,1] ## The predicted probabilities for the validation dataset
        elif hasattr(model, 'decision_function'):
            predictions_train2 = model.decision_function(X_train_scaled) ## The predicted values for the train dataset
            predictions_val2 = model.decision_function(X_val_scaled) ## The predicted values for the test dataset
        else:
            predictions_train2 = model.predict(X_train_scaled) ## The predicted values for the train dataset
            predictions_val2 = model.predict(X_val_scaled) ## The predicted values for the test dataset
        
        
        accuracy_train2 = metrics.roc_auc_score(y_train, predictions_train2)
        accuracy_val2 = metrics.roc_auc_score(y_val, predictions_val2)
        roc_n_estimators_train.append(accuracy_train2)
        roc_n_estimators_val.append(accuracy_val2)

    
    
    if plot_roc:
        fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2, figsize=(15,12),sharex=False, sharey=False)
        
        ax1.plot(roc_lreate_train)
        ax1.plot(roc_lreate_val)
        ax1.set_xlabel("learning_rate")
        ax1.set_ylabel('roc_auc_score')
        ax1.set_xticks(range(len(learning_rate_list)),labels=learning_rate_list)
        ax1.legend(['Train','Validation'])
        
        ax2.plot(roc_maxdepth_train)
        ax2.plot(roc_maxdepth_val)
        ax2.set_xlabel('max_depth')
        ax2.set_ylabel('roc_auc_score')
        ax2.set_xticks(ticks = range(len(max_depth_list)),labels=max_depth_list)
        ax2.legend(['Train','Validation'])
        
        ax3.plot(roc_n_estimators_train)
        ax3.plot(roc_n_estimators_val)
        ax3.set_xlabel('n_estimators')
        ax3.set_ylabel('roc_auc_score')
        ax3.set_xticks(ticks = range(len(n_estimators_list)),labels=n_estimators_list)
        ax3.legend(['Train','Validation'])



def plot_LightGBM_accuracy(df, features, target, plot_accuracy = True, RANDOM_STATE = 42):
    
    """
    Args:
        df: dataframe
        features: list of features
        target: target variable
        plot_accuracy: boolean
    
    Return:
        fig: figures with accuracy_score curve for the LGBMClassifier hyprerparameters n_estimator, learning_rate, and max_depth values.
        it's help to choose the best parameters to avoid overfitting.
        
    """
    
    X_train, X_val, y_train, y_val = split_data(df, features, target)
    X_train_scaled, X_val_scaled = scale_data(X_train, X_val)   
    
    learning_rate_list = [0.01, 0.1, 0.9, 1]
    max_depth_list = [2, 3, 4, 7, 8, 16, 32, 65] # None means that there is no depth limit.
    n_estimators_list = [2,5,10,50,100,500,700]
    
    accuracy_lreate_train = []
    accuracy_lreate_val = []
    accuracy_maxdepth_train = []
    accuracy_maxdepth_val = []
    accuracy_n_estimators_train = []
    accuracy_n_estimators_val = []
    

    for learning_rates in learning_rate_list:
        
        # You can fit the model at the same time you define it, because the fit function returns the fitted estimator.
        model = LGBMClassifier(learning_rate=learning_rates,random_state=RANDOM_STATE).fit(X_train_scaled,y_train)        

        predictions_train = model.predict(X_train_scaled) ## The predicted values for the train dataset
        predictions_val = model.predict(X_val_scaled) ## The predicted values for the test dataset

        
        accuracy_train = metrics.accuracy_score(y_train, predictions_train)
        accuracy_val = metrics.accuracy_score(y_val, predictions_val)
        accuracy_lreate_train.append(accuracy_train)
        accuracy_lreate_val.append(accuracy_val)

    for max_depth in max_depth_list:
    
        # You can fit the model at the same time you define it, because the fit function returns the fitted estimator.
        model = LGBMClassifier(max_depth = max_depth,random_state = RANDOM_STATE).fit(X_train_scaled,y_train) 
        
        predictions_train1 = model.predict(X_train_scaled) ## The predicted values for the train dataset
        predictions_val1 = model.predict(X_val_scaled) ## The predicted values for the test dataset
        
        
        accuracy_train1 = metrics.accuracy_score(y_train, predictions_train1)
        accuracy_val1 = metrics.accuracy_score(y_val, predictions_val1)
        accuracy_maxdepth_train.append(accuracy_train1)
        accuracy_maxdepth_val.append(accuracy_val1)

    for n_estimator in n_estimators_list:
    
        # You can fit the model at the same time you define it, because the fit function returns the fitted estimator.
        model = LGBMClassifier(n_estimators=n_estimator,random_state = RANDOM_STATE).fit(X_train_scaled,y_train) 
        
        predictions_train2 = model.predict(X_train_scaled) ## The predicted values for the train dataset
        predictions_val2 = model.predict(X_val_scaled) ## The predicted values for the test dataset
    
        
        accuracy_train2 = metrics.accuracy_score(y_train, predictions_train2)
        accuracy_val2 = metrics.accuracy_score(y_val, predictions_val2)
        accuracy_n_estimators_train.append(accuracy_train2)
        accuracy_n_estimators_val.append(accuracy_val2)

    
    if plot_accuracy:
        fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2, figsize=(15,12),sharex=False, sharey=False)
        
        ax1.plot(accuracy_lreate_train)
        ax1.plot(accuracy_lreate_val)
        ax1.set_xlabel("learning_rate")
        ax1.set_ylabel('accuracy_score')
        ax1.set_xticks(range(len(learning_rate_list)),labels=learning_rate_list)
        ax1.legend(['Train','Validation'])
        
        ax2.plot(accuracy_maxdepth_train)
        ax2.plot(accuracy_maxdepth_val)
        ax2.set_xlabel('max_depth')
        ax2.set_ylabel('accuracy_score')
        ax2.set_xticks(ticks = range(len(max_depth_list)),labels=max_depth_list)
        ax2.legend(['Train','Validation'])
        
        ax3.plot(accuracy_n_estimators_train)
        ax3.plot(accuracy_n_estimators_val)
        ax3.set_xlabel('n_estimators')
        ax3.set_ylabel('accuracy_score')
        ax3.set_xticks(ticks = range(len(n_estimators_list)),labels=n_estimators_list)
        ax3.legend(['Train','Validation'])



def plot_LightGBM_f1(df, features, target, plot_f1 = True, RANDOM_STATE = 42):
    
    """
    Args:
        df: dataframe
        features: list of features
        target: target variable
        plot_f1: boolean
    
    Return:
        fig: figures with f1_score curve for the LGBMClassifier hyprerparameters n_estimator, learning_rate, and max_depth values.
        it's help to choose the best parameters to avoid overfitting.
        
    """
    
    X_train, X_val, y_train, y_val = split_data(df, features, target)
    X_train_scaled, X_val_scaled = scale_data(X_train, X_val)   
    
    learning_rate_list = [0.01, 0.1, 0.9, 1]
    max_depth_list = [2, 3, 4, 7, 8, 16, 32, 65] # None means that there is no depth limit.
    n_estimators_list = [2,5,10,50,100,500,700]
    
    f1_lreate_train = []
    f1_lreate_val = []
    f1_maxdepth_train = []
    f1_maxdepth_val = []
    f1_n_estimators_train = []
    f1_n_estimators_val = []
    

    for learning_rates in learning_rate_list:
        
        # You can fit the model at the same time you define it, because the fit function returns the fitted estimator.
        model = LGBMClassifier(learning_rate=learning_rates,random_state=RANDOM_STATE).fit(X_train_scaled,y_train)        

        predictions_train = model.predict(X_train_scaled) ## The predicted values for the train dataset
        predictions_val = model.predict(X_val_scaled) ## The predicted values for the test dataset

        
        accuracy_train = metrics.f1_score(y_train, predictions_train)
        accuracy_val = metrics.f1_score(y_val, predictions_val)
        f1_lreate_train.append(accuracy_train)
        f1_lreate_val.append(accuracy_val)

    for max_depth in max_depth_list:
    
        # You can fit the model at the same time you define it, because the fit function returns the fitted estimator.
        model = LGBMClassifier(max_depth = max_depth,random_state = RANDOM_STATE).fit(X_train_scaled,y_train) 
        
        predictions_train1 = model.predict(X_train_scaled) ## The predicted values for the train dataset
        predictions_val1 = model.predict(X_val_scaled) ## The predicted values for the test dataset
        
        
        accuracy_train1 = metrics.f1_score(y_train, predictions_train1)
        accuracy_val1 = metrics.f1_score(y_val, predictions_val1)
        f1_maxdepth_train.append(accuracy_train1)
        f1_maxdepth_val.append(accuracy_val1)

    for n_estimator in n_estimators_list:
    
        # You can fit the model at the same time you define it, because the fit function returns the fitted estimator.
        model = LGBMClassifier(n_estimators=n_estimator,random_state = RANDOM_STATE).fit(X_train_scaled,y_train) 
        
        predictions_train2 = model.predict(X_train_scaled) ## The predicted values for the train dataset
        predictions_val2 = model.predict(X_val_scaled) ## The predicted values for the test dataset
    
        
        accuracy_train2 = metrics.f1_score(y_train, predictions_train2)
        accuracy_val2 = metrics.f1_score(y_val, predictions_val2)
        f1_n_estimators_train.append(accuracy_train2)
        f1_n_estimators_val.append(accuracy_val2)

    
    if plot_f1:
        fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2, figsize=(15,12),sharex=False, sharey=False)
        
        ax1.plot(f1_lreate_train)
        ax1.plot(f1_lreate_val)
        ax1.set_xlabel("learning_rate")
        ax1.set_ylabel('f1_score')
        ax1.set_xticks(range(len(learning_rate_list)),labels=learning_rate_list)
        ax1.legend(['Train','Validation'])
        
        ax2.plot(f1_maxdepth_train)
        ax2.plot(f1_maxdepth_val)
        ax2.set_xlabel('max_depth')
        ax2.set_ylabel('f1_score')
        ax2.set_xticks(ticks = range(len(max_depth_list)),labels=max_depth_list)
        ax2.legend(['Train','Validation'])
        
        ax3.plot(f1_n_estimators_train)
        ax3.plot(f1_n_estimators_val)
        ax3.set_xlabel('n_estimators')
        ax3.set_ylabel('f1_score')
        ax3.set_xticks(ticks = range(len(n_estimators_list)),labels=n_estimators_list)
        ax3.legend(['Train','Validation'])




                    ####..................................................................####
                    ####..................................................................####


# XGBClassifier

def plot_xgb_roc(df, features, target, plot_roc = True, RANDOM_STATE = 42):
    
    """
    Args:
        df: dataframe
        features: list of features
        target: target variable
        plot_roc: boolean
    
    Return:
        fig: figures with ROC curve fro the XGBClassifier hyprerparameters n_estimator, learning_rate, and max_depth values.
        it's help to choose the best parameters to avoid overfitting.
        
    """
    
    X_train, X_val, y_train, y_val = split_data(df, features, target)
    X_train_scaled, X_val_scaled = scale_data(X_train, X_val)   
    
    learning_rate_list = [0.01, 0.1, 0.9, 1]
    max_depth_list = [1,2, 3, 4, 7, 8, 16, 32, 65] # None means that there is no depth limit.
    n_estimators_list = [2,5,10,50,100,500,700]
    
    roc_lreate_train = []
    roc_lreate_val = []
    roc_maxdepth_train = []
    roc_maxdepth_val = []
    roc_n_estimators_train = []
    roc_n_estimators_val = []
    

    for learning_rates in learning_rate_list:
        
        # You can fit the model at the same time you define it, because the fit function returns the fitted estimator.
        model = XGBClassifier(learning_rate=learning_rates,random_state=RANDOM_STATE).fit(X_train_scaled,y_train)        

        if hasattr(model, 'predict_proba'):
            predictions_train = model.predict_proba(X_train_scaled)[:,1] ## The predicted probabilities for the train dataset
            predictions_val = model.predict_proba(X_val_scaled)[:,1] ## The predicted probabilities for the validation dataset
        elif hasattr(model, 'decision_function'):
            predictions_train = model.decision_function(X_train_scaled) ## The predicted values for the train dataset
            predictions_val = model.decision_function(X_val_scaled) ## The predicted values for the test dataset
        else:
            predictions_train = model.predict(X_train_scaled) ## The predicted values for the train dataset
            predictions_val = model.predict(X_val_scaled) ## The predicted values for the test dataset
    
        
        accuracy_train = metrics.roc_auc_score(y_train, predictions_train)
        accuracy_val = metrics.roc_auc_score(y_val, predictions_val)
        roc_lreate_train.append(accuracy_train)
        roc_lreate_val.append(accuracy_val)

    for max_depth in max_depth_list:
    
        # You can fit the model at the same time you define it, because the fit function returns the fitted estimator.
        model = XGBClassifier(max_depth = max_depth,random_state = RANDOM_STATE).fit(X_train_scaled,y_train) 
        
        if hasattr(model, 'predict_proba'):
            predictions_train1 = model.predict_proba(X_train_scaled)[:,1] ## The predicted probabilities for the train dataset
            predictions_val1 = model.predict_proba(X_val_scaled)[:,1] ## The predicted probabilities for the validation dataset
        elif hasattr(model, 'decision_function'):
            predictions_train1 = model.decision_function(X_train_scaled) ## The predicted values for the train dataset
            predictions_val1 = model.decision_function(X_val_scaled) ## The predicted values for the test dataset
        else:
            predictions_train1 = model.predict(X_train_scaled) ## The predicted values for the train dataset
            predictions_val1 = model.predict(X_val_scaled) ## The predicted values for the test dataset
        
        
        accuracy_train1 = metrics.roc_auc_score(y_train, predictions_train1)
        accuracy_val1 = metrics.roc_auc_score(y_val, predictions_val1)
        roc_maxdepth_train.append(accuracy_train1)
        roc_maxdepth_val.append(accuracy_val1)

    for n_estimator in n_estimators_list:
    
        # You can fit the model at the same time you define it, because the fit function returns the fitted estimator.
        model = XGBClassifier(n_estimators=n_estimator,random_state = RANDOM_STATE).fit(X_train_scaled,y_train) 
        
        if hasattr(model, 'predict_proba'):
            predictions_train2 = model.predict_proba(X_train_scaled)[:,1] ## The predicted probabilities for the train dataset
            predictions_val2 = model.predict_proba(X_val_scaled)[:,1] ## The predicted probabilities for the validation dataset
        elif hasattr(model, 'decision_function'):
            predictions_train2 = model.decision_function(X_train_scaled) ## The predicted values for the train dataset
            predictions_val2 = model.decision_function(X_val_scaled) ## The predicted values for the test dataset
        else:
            predictions_train2 = model.predict(X_train_scaled) ## The predicted values for the train dataset
            predictions_val2 = model.predict(X_val_scaled) ## The predicted values for the test dataset
        
        
        accuracy_train2 = metrics.roc_auc_score(y_train, predictions_train2)
        accuracy_val2 = metrics.roc_auc_score(y_val, predictions_val2)
        roc_n_estimators_train.append(accuracy_train2)
        roc_n_estimators_val.append(accuracy_val2)

    
    
    if plot_roc:
        fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2, figsize=(15,12),sharex=False, sharey=False)
        
        ax1.plot(roc_lreate_train)
        ax1.plot(roc_lreate_val)
        ax1.set_xlabel("learning_rate")
        ax1.set_ylabel('roc_auc_score')
        ax1.set_xticks(range(len(learning_rate_list)),labels=learning_rate_list)
        ax1.legend(['Train','Validation'])
        
        ax2.plot(roc_maxdepth_train)
        ax2.plot(roc_maxdepth_val)
        ax2.set_xlabel('max_depth')
        ax2.set_ylabel('roc_auc_score')
        ax2.set_xticks(ticks = range(len(max_depth_list)),labels=max_depth_list)
        ax2.legend(['Train','Validation'])
        
        ax3.plot(roc_n_estimators_train)
        ax3.plot(roc_n_estimators_val)
        ax3.set_xlabel('n_estimators')
        ax3.set_ylabel('roc_auc_score')
        ax3.set_xticks(ticks = range(len(n_estimators_list)),labels=n_estimators_list)
        ax3.legend(['Train','Validation'])



def plot_xgb_accuracy(df, features, target, plot_accuracy = True, RANDOM_STATE = 42):
    
    """
    Args:
        df: dataframe
        features: list of features
        target: target variable
        plot_accuracy: boolean
    
    Return:
        fig: figures with accuracy_score curve for the XGBClassifier hyprerparameters n_estimator, learning_rate, and max_depth values.
        it's help to choose the best parameters to avoid overfitting.
        
    """
    
    X_train, X_val, y_train, y_val = split_data(df, features, target)
    X_train_scaled, X_val_scaled = scale_data(X_train, X_val)   
    
    learning_rate_list = [0.01, 0.1, 0.9, 1]
    max_depth_list = [1,2, 3, 4, 7, 8, 16, 32, 65] # None means that there is no depth limit.
    n_estimators_list = [2,5,10,50,100,500,700]
    
    accuracy_lreate_train = []
    accuracy_lreate_val = []
    accuracy_maxdepth_train = []
    accuracy_maxdepth_val = []
    accuracy_n_estimators_train = []
    accuracy_n_estimators_val = []
    

    for learning_rates in learning_rate_list:
        
        # You can fit the model at the same time you define it, because the fit function returns the fitted estimator.
        model = XGBClassifier(learning_rate=learning_rates,random_state=RANDOM_STATE).fit(X_train_scaled,y_train)        

        predictions_train = model.predict(X_train_scaled) ## The predicted values for the train dataset
        predictions_val = model.predict(X_val_scaled) ## The predicted values for the test dataset

        
        accuracy_train = metrics.accuracy_score(y_train, predictions_train)
        accuracy_val = metrics.accuracy_score(y_val, predictions_val)
        accuracy_lreate_train.append(accuracy_train)
        accuracy_lreate_val.append(accuracy_val)

    for max_depth in max_depth_list:
    
        # You can fit the model at the same time you define it, because the fit function returns the fitted estimator.
        model = XGBClassifier(max_depth = max_depth,random_state = RANDOM_STATE).fit(X_train_scaled,y_train) 
        
        predictions_train1 = model.predict(X_train_scaled) ## The predicted values for the train dataset
        predictions_val1 = model.predict(X_val_scaled) ## The predicted values for the test dataset
        
        
        accuracy_train1 = metrics.accuracy_score(y_train, predictions_train1)
        accuracy_val1 = metrics.accuracy_score(y_val, predictions_val1)
        accuracy_maxdepth_train.append(accuracy_train1)
        accuracy_maxdepth_val.append(accuracy_val1)

    for n_estimator in n_estimators_list:
    
        # You can fit the model at the same time you define it, because the fit function returns the fitted estimator.
        model = XGBClassifier(n_estimators=n_estimator,random_state = RANDOM_STATE).fit(X_train_scaled,y_train) 
        
        predictions_train2 = model.predict(X_train_scaled) ## The predicted values for the train dataset
        predictions_val2 = model.predict(X_val_scaled) ## The predicted values for the test dataset
    
        
        accuracy_train2 = metrics.accuracy_score(y_train, predictions_train2)
        accuracy_val2 = metrics.accuracy_score(y_val, predictions_val2)
        accuracy_n_estimators_train.append(accuracy_train2)
        accuracy_n_estimators_val.append(accuracy_val2)

    
    if plot_accuracy:
        fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2, figsize=(15,12),sharex=False, sharey=False)
        
        ax1.plot(accuracy_lreate_train)
        ax1.plot(accuracy_lreate_val)
        ax1.set_xlabel("learning_rate")
        ax1.set_ylabel('accuracy_score')
        ax1.set_xticks(range(len(learning_rate_list)),labels=learning_rate_list)
        ax1.legend(['Train','Validation'])
        
        ax2.plot(accuracy_maxdepth_train)
        ax2.plot(accuracy_maxdepth_val)
        ax2.set_xlabel('max_depth')
        ax2.set_ylabel('accuracy_score')
        ax2.set_xticks(ticks = range(len(max_depth_list)),labels=max_depth_list)
        ax2.legend(['Train','Validation'])
        
        ax3.plot(accuracy_n_estimators_train)
        ax3.plot(accuracy_n_estimators_val)
        ax3.set_xlabel('n_estimators')
        ax3.set_ylabel('accuracy_score')
        ax3.set_xticks(ticks = range(len(n_estimators_list)),labels=n_estimators_list)
        ax3.legend(['Train','Validation'])



def plot_xgb_f1(df, features, target, plot_f1 = True, RANDOM_STATE = 42):
    
    """
    Args:
        df: dataframe
        features: list of features
        target: target variable
        plot_f1: boolean
    
    Return:
        fig: figures with f1_score curve for the XGBClassifier hyprerparameters n_estimator, learning_rate, and max_depth values.
        it's help to choose the best parameters to avoid overfitting.
        
    """
    
    X_train, X_val, y_train, y_val = split_data(df, features, target)
    X_train_scaled, X_val_scaled = scale_data(X_train, X_val)   
    
    learning_rate_list = [0.01, 0.1, 0.9, 1]
    max_depth_list = [2, 3, 4, 7, 8, 16, 32, 65] # None means that there is no depth limit.
    n_estimators_list = [2,5,10,50,100,500,700]
    
    f1_lreate_train = []
    f1_lreate_val = []
    f1_maxdepth_train = []
    f1_maxdepth_val = []
    f1_n_estimators_train = []
    f1_n_estimators_val = []
    

    for learning_rates in learning_rate_list:
        
        # You can fit the model at the same time you define it, because the fit function returns the fitted estimator.
        model = XGBClassifier(learning_rate=learning_rates,random_state=RANDOM_STATE).fit(X_train_scaled,y_train)        

        predictions_train = model.predict(X_train_scaled) ## The predicted values for the train dataset
        predictions_val = model.predict(X_val_scaled) ## The predicted values for the test dataset

        
        accuracy_train = metrics.f1_score(y_train, predictions_train)
        accuracy_val = metrics.f1_score(y_val, predictions_val)
        f1_lreate_train.append(accuracy_train)
        f1_lreate_val.append(accuracy_val)

    for max_depth in max_depth_list:
    
        # You can fit the model at the same time you define it, because the fit function returns the fitted estimator.
        model = XGBClassifier(max_depth = max_depth,random_state = RANDOM_STATE).fit(X_train_scaled,y_train) 
        
        predictions_train1 = model.predict(X_train_scaled) ## The predicted values for the train dataset
        predictions_val1 = model.predict(X_val_scaled) ## The predicted values for the test dataset
        
        
        accuracy_train1 = metrics.f1_score(y_train, predictions_train1)
        accuracy_val1 = metrics.f1_score(y_val, predictions_val1)
        f1_maxdepth_train.append(accuracy_train1)
        f1_maxdepth_val.append(accuracy_val1)

    for n_estimator in n_estimators_list:
    
        # You can fit the model at the same time you define it, because the fit function returns the fitted estimator.
        model = XGBClassifier(n_estimators=n_estimator,random_state = RANDOM_STATE).fit(X_train_scaled,y_train) 
        
        predictions_train2 = model.predict(X_train_scaled) ## The predicted values for the train dataset
        predictions_val2 = model.predict(X_val_scaled) ## The predicted values for the test dataset
    
        
        accuracy_train2 = metrics.f1_score(y_train, predictions_train2)
        accuracy_val2 = metrics.f1_score(y_val, predictions_val2)
        f1_n_estimators_train.append(accuracy_train2)
        f1_n_estimators_val.append(accuracy_val2)

    
    if plot_f1:
        fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2, figsize=(15,12),sharex=False, sharey=False)
        
        ax1.plot(f1_lreate_train)
        ax1.plot(f1_lreate_val)
        ax1.set_xlabel("learning_rate")
        ax1.set_ylabel('f1_score')
        ax1.set_xticks(range(len(learning_rate_list)),labels=learning_rate_list)
        ax1.legend(['Train','Validation'])
        
        ax2.plot(f1_maxdepth_train)
        ax2.plot(f1_maxdepth_val)
        ax2.set_xlabel('max_depth')
        ax2.set_ylabel('f1_score')
        ax2.set_xticks(ticks = range(len(max_depth_list)),labels=max_depth_list)
        ax2.legend(['Train','Validation'])
        
        ax3.plot(f1_n_estimators_train)
        ax3.plot(f1_n_estimators_val)
        ax3.set_xlabel('n_estimators')
        ax3.set_ylabel('f1_score')
        ax3.set_xticks(ticks = range(len(n_estimators_list)),labels=n_estimators_list)
        ax3.legend(['Train','Validation'])



                    ####..................................................................####
                    ####..................................................................####





