
import pandas as pd
import os # to make directories
from sklearn.model_selection import train_test_split
import logging

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
def preprocess_data(df):
    try:
        df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)
        df.rename(columns={'v1':'target', 'v2':'text'}, inplace=True)
        logger.debug('Data preprocessing completed successfully')
        return df
    except Exception as e:
        logger.error(f"Error during data preprocessing: {e}")
        raise
    
def save_data(train_data,test_data,data_path):
    try:
        raw_data_path = os.path.join(data_path, 'raw')
        os.makedirs(raw_data_path, exist_ok=True)
        train_file = os.path.join(raw_data_path, 'train.csv')
        test_file = os.path.join(raw_data_path, 'test.csv')
        
        train_data.to_csv(train_file, index=False)
        test_data.to_csv(test_file, index=False)
        
        logger.debug(f"Data saved successfully to {data_path}")
    except Exception as e:
        logger.error(f"Error saving data: {e}")
        raise
def main():
    try:
        test_size= 0.2
        data_path='D:\MLOPS\DVC-AWS -S3-END-TO-END\experiments\spam.csv'
        df=load_data(data_path)
        final_df=preprocess_data(df)
        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42)
        save_data(train_data, test_data, './data')
        #

    except Exception as e:
        logger.error(f"An error occurred in the main function: {e}")
        raise
if __name__ == "__main__":
    main()
    logger.info("Data ingestion completed successfully.")
    