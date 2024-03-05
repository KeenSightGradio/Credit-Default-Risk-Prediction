import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd 
import pickle

import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
def load_dataset(dataset_path):
    data = pd.read_csv(dataset_path)
    
    data['person_home_ownership']=data['person_home_ownership'].map({'RENT':0,'OWN':1, 'MORTGAGE':2,'OTHER':3})
    data["loan_intent"]=data["loan_intent"].map({'PERSONAL':0, 'EDUCATION':1, 'MEDICAL':2, 'VENTURE':3, 'HOMEIMPROVEMENT':4,
       'DEBTCONSOLIDATION':5})
    data["loan_grade"]=data["loan_grade"].map({'A':0,'B':1,'C':2,'D':3,'E':4,'F':5,'G':6})
    data["cb_person_default_on_file"] = data["cb_person_default_on_file"].map({'Y':1,'N':0})
    
    y = data.loan_status
    X = data.drop(["loan_status"], axis=1)
    
    X_train, X_test, y_train, x_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, x_test
  
def train_model(train_data, train_label, n_estimators , learning_rate):
    
    model = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=1, gamma=0,
              learning_rate=learning_rate, max_delta_step=0, max_depth=7,
              min_child_weight=1, missing=np.NaN, n_estimators=n_estimators, n_jobs=-1,
              nthread=None, objective='binary:logistic', random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
              silent=None, subsample=1, verbosity=0)
    
    model.fit(train_data,train_label)
    
    with open("C:/Users/hp/Credit-Default-Risk-Prediction/models/cdr_model.pickle", "wb") as f:
        pickle.dump(model, f)
        
def test_model(test_data, test_label):
    
    with open("C:/Users/hp/Credit-Default-Risk-Prediction/models/cdr_model.pickle", "rb") as f:
        loaded_model = pickle.load(f)

    y_pred = loaded_model.predict(test_data)
  
    accuracy_score = metrics.accuracy_score(test_label, y_pred)
    precision_score = metrics.precision_score(test_label, y_pred)
    recall_score = metrics.recall_score(test_label, y_pred)
    f1_score = metrics.f1_score(test_label, y_pred)
    
    return accuracy_score, precision_score, recall_score, f1_score
    
def runner(n_estimators, learning_rate):
    print(n_estimators, learning_rate)
    file_path = "C:/Users/hp/Credit-Default-Risk-Prediction/dataset/credit_risk_dataset.csv"

    train_data, test_data, train_label, test_label = load_dataset(file_path)
    
    train_model(train_data, train_label, n_estimators, learning_rate)
    accuracy_score, precision_score, recall_score, f1_score = test_model(test_data, test_label)
    
    print(accuracy_score, precision_score, recall_score, f1_score)
    return accuracy_score, precision_score, recall_score, f1_score 
    
if __name__== "__main__":
    runner()
    
    
    