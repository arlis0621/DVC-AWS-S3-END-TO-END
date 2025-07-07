
import os
import logging
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from nltk.stem.porter import PorterStemmer

from nltk.corpus import stopwords
import string 
import nltk
nltk.download('stopwords')
nltk.download('punkt')


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


def transform_text(text):
    ps=PorterStemmer()
    text=text.lower()
    text=nltk.word_tokenize(text)
    text=[word for word in text if word .isalnum()]
    text=[word for word in text if word not in stopwords.words('english') and word not in string.punctuation]
    text=[ps.stem(word) for word in text]
    #now the below command joins  the text list into a string
    return ' '.join(text)

def preprocess_df(df,text_col,target_col):
    try:
        logger.debug("Starting preprocessing of DataFrame")
        encode=LabelEncoder()
        
        df[target_col]=encode.fit_transform(df[target_col])
        df=df.drop_duplicates(keep="first")
        logger.debug("Label encoding and duplicate removal completed")
        df.loc[:,text_col]=df[text_col].apply(transform_text)
        #way of applying the function to a column
        logger.debug("Text transformation completed")
        return df
    
        
    except Exception as e:
        logger.error(f"Error during DataFrame preprocessing: {e}")
        raise
def main():
    try:
        text_col='text'
        target_col='target'
        #fetch the raw data
        train_data=pd.read_csv('./data/raw/test.csv')
        test_data=pd.read_csv('./data/raw/test.csv')
        
        logger.debug('Data loaded successfully')
        train_processed_data=preprocess_df(train_data,text_col,target_col)
        test_processed_data=preprocess_df(test_data,text_col,target_col)
        data_path=os.path.join('./data','interim')
        os.makedirs(data_path, exist_ok=True)
        train_processed_data.to_csv(os.path.join(data_path,'train_processed.csv'),index=False)
        test_processed_data.to_csv(os.path.join(data_path,'test_processed.csv'),index=False)
        
        logger.debug('Data preprocessing completed successfully')
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        raise
if __name__ == "__main__":
    main()
        
    