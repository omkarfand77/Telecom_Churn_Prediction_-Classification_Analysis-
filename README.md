# ***Telecom_Churn_Prediction_-Classification_Analysis***

Customer churn is a big problem for telecommunications companies. Indeed, their annual churn rates are usually higher than 10%.This is a classification project since the variable to be predicted is binary (churn or loyal customer). The goal here is to model churn probability, conditioned on the customer features.

## ***Exploratory Data Analysis***

## ***Feature Engineering and Feature Selection***

Unnecessary data coloumns like are Unnamed: 0, state, area.code are dropped from working data.
further data is scaled with `standard scaler` for better working in model of values.
The Dependant feature is churn cloumn and Independant feature are rest of coloumn in dataset.
for model evaluation and model working data set divided into `train_test` split with test size of 20%.

## ***Model Building***

The models are build using different classification algorithms Logistic Regression, `Random Forest classifier`,
Suppport Vector Machine, Decison Tree Classfier, K-Nearest Classifier, Naives Bayes classifier, Gausian Naive Bayes.

## ***Model Evaluation***

The models build with sklearn are evaluated with `classification report` where precision, recall, f1 score
and acuracy score are evaluated for all models builded.

## ***Model Selection***

Based on models `accuracy score` the higest score is of Random Forest Classifier model achiving `95% accuarcy`.
The model is selected for deployment and made pickle.

## ***Deployment***

[app_Link](https://telecommunication-churn-prediction-classificationanalysis.streamlit.app/)
The model is deloyed with framework streamlit for real time prediction.
