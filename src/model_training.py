
import os
import pandas as pd
import numpy as np
import pickle
import logging
from sklearn.ensemble import RandomForestClassifier

#logging module

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

def load_data(file_path):
    try:
        df=pd.read_csv(file_path)
        logger.debug(f"Data loaded successfully from {file_path}")
        
        return df
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}")
        raise
#use type hinting for better code clarity
def train_model(x_train,y_train,params):
    try:
        if x_train.shape[0]!= y_train.shape[0]:
            raise ValueError("Number of samples in features and target do not match")
        clf= RandomForestClassifier(n_estimators=params['n_estimators'],random_state=params['random_state'])
        clf.fit(x_train,y_train)
        logger.debug("Model training completed successfully")
        return clf
    except Exception as e:
        logger.error(f"Error during model training: {e}")
        raise
    
    
def save_model(model,file_path):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump(model, f)
        logger.debug(f"Model saved successfully to {file_path}")
    except Exception as e:
        logger.error(f"Error saving model to {file_path}: {e}")
        raise
    
def main():
    try:
        params={'n_estimators':25,'random_state':42 }
        train_data=load_data('./data/processed/train_tfidf.csv  ')
        x_train=train_data.drop(columns=['label'])
        y_train=train_data['label']
        clf=train_model(x_train,y_train,params)
        save_model(clf,'./models/random_forest_model.pkl')
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        raise
if __name__ == "__main__":
    main()
    