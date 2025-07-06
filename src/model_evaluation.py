
import os
import pandas as pd
import numpy as np
import pickle
import json
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

# Logging configuration
log_dir='logs'
os.makedirs(log_dir,exist_ok=True)


#logging configuration
logger=logging.getLogger('data_ingestion')
logger.setLevel('DEBUG')

console_handler=logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path=os.path.join(log_dir,'data_ingestion.log')
file_handler=logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)    

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_model(file_path):
    try:
        with open(file_path,'rb') as f:
            model= pickle.load(f)
        logger.debug(f"Model loaded successfully from {file_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading model from {file_path}: {e}")
        raise
    
def load_data(file_path):
    try:
        df=pd.read_csv(file_path)
        logger.debug(f"Data loaded successfully from {file_path}")
        return df
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}")
        raise
    
def evaluate_model(clf,x_test,y_test):
    try:
        y_pred=clf.predict(x_test)
        y_pred_proba=clf.predict_proba(x_test)[:,1]
        accuracy=accuracy_score(y_test,y_pred)
        precision=precision_score(y_test,y_pred)
        recall=recall_score(y_test,y_pred)
        roc_auc=roc_auc_score(y_test,y_pred_proba)
        metrics={
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'roc_auc': roc_auc
        }
        logger.debug("Model evaluation completed successfully")
        logger.debug(f"Evaluation metrics: {metrics}")
        return metrics
    except Exception as e:
        logger.error(f"Error during model evaluation: {e}")
        raise
    
def save_metrics(metrics,file_path):
    try:
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(metrics, f)
        logger.debug(f"Metrics saved successfully to {file_path}")
    except Exception as e:
        logger.error(f"Error saving metrics to {file_path}: {e}")
        raise
def main():
    try:
        clf=load_model('./models/model.pkl')
        test_data=load_data('./data/processed/test_tfidf.csv')
        x_test=test_data.iloc[:,:-1].values
        y_test=test_data.iloc[:,-1].values
        metrics=evaluate_model(clf,x_test,y_test)
        save_metrics(metrics,'reports/metrics.json')
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        raise
if __name__ == "__main__":
    main()
    
    