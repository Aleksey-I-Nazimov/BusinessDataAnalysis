import pandas as pd
import numpy as np
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score, roc_auc_score, accuracy_score, f1_score

from flask import Flask, jsonify, request
import logging
import os
import signal
import threading
import json


def evaluate_results(y_test, y_predict):
    xtr='Classification results: '
    f1 = f1_score(y_test, y_predict)
    xtr=xtr+"f1= %.2f%% " % (f1 * 100.0)
    roc = roc_auc_score(y_test, y_predict)
    xtr=xtr+"roc= %.2f%% " % (roc * 100.0)
    rec = recall_score(y_test, y_predict, average='binary')
    xtr=xtr+"recall= %.2f%% " % (rec * 100.0)
    prc = precision_score(y_test, y_predict, average='binary')
    xtr=xtr+"precision= %.2f%% " % (prc * 100.0)
    print (xtr)
    return xtr


class TreeClassifier:
    def __init__(self):
        self.model = xgb.XGBClassifier()
        
    def fit (self):
        data_frame = pd.read_csv("wine.data", header=None)
        target = 0;
        features = [1,2,3,4,5,6,7,8,9,10,11,12,13]
        
        # Let's use the 1 and the 3 class. The 2nd class is deleted:---------------------
        filtered_data = data_frame[data_frame[target]!=1]
        filtered_data.loc[filtered_data[target] == 2, target] = 0
        filtered_data.loc[filtered_data[target] == 3, target] = 1

        # Selecting features and splitting dataset:--------------------------------------
        x_data = filtered_data[features]
        y_data = filtered_data[target]
        x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.25, random_state=7)
        
        # Dumping the X_TRAIN for testing:-----------------------------------------------
        x_train.to_json("x_train.json")
        
        # Fitting and testing:-----------------------------------------------------------
        self.model.fit(x_train,y_train)
        y_predict = self.transform(x_frame=x_test)
        status=evaluate_results(y_test, y_predict)
        
        return status
    
    def transform (self,x_frame):
        y_predict = self.model.predict(x_frame)
        return y_predict
        

        
logging.basicConfig(filename='task-9-server.log', encoding='utf-8', level=logging.DEBUG)
web_server=Flask("task-9")
classifier = TreeClassifier()


@web_server.route("/",methods=["GET"])
def wakeUp():
    result = classifier.fit()
    logging.info("Fitting on the wine dataset: result: {}".format(result))
    return result;

@web_server.route("/off",methods=["GET"])
def off():
    sig = getattr(signal, "SIGKILL", signal.SIGTERM)
    os.kill(os.getpid(), sig)

@web_server.route('/predict',methods=["POST"])
def predict():
    
    # Getting the request body:----------------------------------------------------------
    rq_string = request.get_data(as_text=True);
    limit =5000;
    if len(rq_string)>limit :
        logging.debug("  --> Request data {}".format(rq_string[0:limit]))
    else :
        logging.debug("  --> Request data {}".format(rq_string))
        
    x_frame = pd.read_json(rq_string,orient='records')
    
    # Predictions:-----------------------------------------------------------------------
    logging.info("  --> The input\n{}".format(x_frame.head(10)))
    predictions = classifier.transform(x_frame=x_frame)
   
    data = {"success": True}
    data["y"] = predictions.tolist()
    
    logging.info("  --> Response {}".format(data));
    return jsonify(data)

def flask_runnable():
    web_server.run(debug=True, port=10000, use_reloader=False)
    

flask_runnable()
    
#threading.Thread(target=flask_runnable, daemon=True).start()

