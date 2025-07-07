
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import logging

# Logging configuration
log_dir='logs'
os.makedirs(log_dir,exist_ok=True)


#logging configuration
logger=logging.getLogger('data_preprocessing')
logger.setLevel('DEBUG')

console_handler=logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path=os.path.join(log_dir,'data_preprocessing.log')
file_handler=logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)    

logger.addHandler(console_handler)
logger.addHandler(file_handler)

#now load the preprocessed data
def load_data(file_path):
    try:
        df=pd.read_csv(file_path)
        df.fillna('',inplace=True)
        logger.debug(f"Data loaded successfully from {file_path}")
        return df
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}")
        raise

def apply_tfidf(train_data,test_data,max_features=50):
    try:
        vectorizer = TfidfVectorizer(max_features=max_features)
        x_train=train_data['text'].values
        y_train=train_data['target'].values
        x_test=test_data['text'].values
        y_test=test_data['target'].values
        logger.debug("Starting TF-IDF transformation")
        x_train_bow= vectorizer.fit_transform(x_train)
        x_test_bow = vectorizer.transform(x_test)
        train_df=pd.DataFrame(x_train_bow.toarray())
        train_df['label']=y_train
        test_df=pd.DataFrame(x_test_bow.toarray())
        test_df['label']=y_test
    
        
        
        logger.debug("TF-IDF transformation completed successfully")
        
        return train_df, test_df
    
    except Exception as e:
        logger.error(f"Error during TF-IDF transformation: {e}")
        raise
    
def save_data(df,file_path):
    try:
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        df.to_csv(file_path, index=False)
        logger.debug(f"Data saved successfully to {file_path}")
    except Exception as e:
        logger.error(f"Error saving data to {file_path}: {e}")
        raise
    
def main():
    try:
        max_features=50
        train_data=load_data('./data/interim/train_processed.csv')
        test_data=load_data('./data/interim/test_processed.csv')
        train_df,test_df=apply_tfidf(train_data,test_data,max_features)
        save_data(train_df,os.path.join('./data','processed','train_tfidf.csv'  ))
        save_data(test_df,os.path.join('./data','processed','test_tfidf.csv'  ))
        logger.debug('TF-IDF feature engineering completed successfully')
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        raise
    
if __name__ == "__main__":
    main()
    
    
