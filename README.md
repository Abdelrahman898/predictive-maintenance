# predictive-maintenance

### Problem

Failure prediction is a major topic in predictive maintenance in many industries. Airlines are particularly interested in predicting equipment failures in advance so that they can enhance operations, cut cost of time-based preventive maintenance, and reduce flight delays.

Observing engine's health and condition through sensors and telemetry data is assumed to facilitate this type of maintenance by predicting Time-To-Failure (TTF) or Remaining Useful Life (RUL) of in-service equipment. The project is trying to answer the following question: 
By using aircraft engine's sensors measurements, can we predict engine's TTF? 


### Client 

A hypothetical airline company in a case-study provided by Microsoft Cortana Intelligence platform. In a predictive maintenance scenario provided by Microsoft, A fictitious airline Company is trying to utilize historical data of equipment sensors to make data-driven decisions on its maintenance planning. Based on this analysis, the company will be able to estimate engine's  time-to-failure and  optimize  its maintenance operations accordingly. 

### Approach 

By  exploring  aircraft  engine’s  sensor  values  over  time,  machine  learning  algorithm  can  learn  the relationship  between  sensor  values  and  changes  in  sensor  values  to  the  historical  failures  in  order  to predict failures in the future. Supervised machine learning algorithms will be used to make the following predictions: 
* Use of Regression algorithms to predict engine TTF. 
* Use of Binary Classification algorithms to predict if the engine will fail in this period.
* Use of Multiclass Classification algorithms to predict the period an engine will fail.


### Data
Text files contain simulated aircraft engine run-to-failure events, operational settings, and 21 sensors measurements are provided by Microsoft. It is assumed that the engine progressing degradation pattern is reflected in its sensor measurements.

**Training Data:** The aircraft engine run-to-failure data. [Download trianing data](https://azuremlsamples.azureml.net/templatedata/PM_train.txt) 
**Test Data:** The aircraft engine operating data without failure events recorded. [Download test data](https://azuremlsamples.azureml.net/templatedata/PM_test.txt)
**Ground Truth Data:** The true remaining cycles which have TTF for each engine in the [Download Truth data](https://azuremlsamples.azureml.net/templatedata/PM_truth.txt)

**Labels:**
* `Regression:` Time-to-Failure (TTF), for each cycle/engine, is the number cycles between that cycle and last cycle of the engine in the training data.
* `Binary Classification:` if the remaining cycles (TTF) is less than specific number of cycles (e.g. 30) then the engine will fail in this period, otherwise the engine is fine.
* `Multiclass Classification:` segmenting TTF into cycle bands (e.g. 0-15, 16-30, 30+), in which band will the engine fail? How could we improve maintenance planning?

### Results

#### Regression

In accordance with our analysis in the data exploratory phase, non-linear regression models like Polynomial, Random Forest, LightGBM and XGBoost performed better than linear model. **xgboost** clearly outperformed other models scoring RMSE of 16.095827 cycles, i.e. the model predicts TTF within average error range of ±16 cycles.

#### Binary Classification

* All of the binary classifiers except Logistic Regression showed better performance metrics without the addition of new features.  

* Logistic Regression and Naive Bayes showed same performance before and after feature engineering with all metrics except AUC. 
 
* Naive Bayes scored better than other classifiers in Recall (Sensitivity) while others scored better in Precision. 

* XGBClassifier (before feature engineering) scored better than other classifier in Precision.

* Logistic Regression B\A, DecisionTree B, and RandomForest B showed same performance in Precision and F1-Score, which is the highest in all f1scores.

* The GaussianNB B algorithm has the highest AUC-ROC with 0.9877,  KNN B comes second with 0.9845.

* KNN A has precision-recall curve operating at threshold 0.8, giving  %100 precision and %60 recall, targeting %17 of the engines.

#### Multiclass Classification

- Random Forest A outperforms all other models in the two graphs above.
- Random Forest A and KNN B perform the best in terms of micro ROC AUC.
- KNN B outperforms Random Forest A in terms of precision and f1 but not in recall.
- KNN A outperforms rest of models in terms of micro ROC AUC.
- Highest F1 score is achieved by Decision Tree A, and acually ROC AUC is not bad.

### Expected Profit

Based on the book: [Data Science for Business](https://www.amazon.com/Data-Science-Business-Data-Analytic-Thinking/dp/1449361323), Expected Value is a method to compare different classification models by constructing cost-benefit matrix in line with the confusion matrix, and then convert model performance to a single monetary value by multiplying confusion matrix into the cost-benefit matrix.  

**Expected Profit = Prop(+ve) x [TPR x benefit(TP) + FNR x cost(FN)] + Prob(-ve) x [TNR x benefit(TN) + FPR x cost(FP)]**

Cost-benefit matrix should be designed by domain expert, Let us assume the following:  

- True Positive (TP) has benefit of USD 300K:
engines that need maintenance and correctly selected by the model.  
- True Negative (TN) has benefit of USD 0K:
engines that are OK and not selected by the model.
- False Positive (FP) has cost of USD -100K:
engines that are OK but selected by the model.
- False Negative (FN) has cost of USD -200K:
engines that need maintenance but not selected by the model.


**GaussianNB B** and **KNN B** has the best profit per engine (USD 69K per engine) if the company has the capacity to maintain **%31** of the engines per period.
