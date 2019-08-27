# import libraries
import sys
import os.path
import pickle
import pandas as pd
import numpy as np
from sqlalchemy import create_engine,MetaData

import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

import re

import sklearn
from sklearn.pipeline import Pipeline,FeatureUnion
import xgboost
from sklearn.multioutput import MultiOutputClassifier
import joblib

from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import classification_report,precision_score,recall_score,roc_auc_score,balanced_accuracy_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.decomposition import LatentDirichletAllocation,TruncatedSVD

def load_data(database_filepath):
    
    # load data from database
    engine = create_engine(database_filepath)
    df = pd.read_sql_table('CleanMessages',con=engine)
    X = df['message'].values
    Y = df[[x for x in df.columns.values if x not in ['id','genre','message','original']]].values
    category_names = df[[x for x in df.columns.values if x not in ['id','genre','message','original']]].columns.values
    Y.drop(columns='child_alone',inplace=True)

    return X,Y,category_names


def tokenize(text):
    
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize text
    tokens = word_tokenize(text)
    
    # lemmatize andremove stop words
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return tokens


def build_model(model_path):
    
    xg_params = {
        'max_depth':5,
        'colsample_bytree':0.8
    }
    
    if ~os.path.isfile(model_path):
        pipeline = Pipeline([
        ('countvec',CountVectorizer(tokenizer=tokenize)),
        ('tfidf',TfidfTransformer()),
        #     ('lsa',TruncatedSVD(random_state=42,n_components=100)),
        ('clf',MultiOutputClassifier(xgboost.XGBClassifier(random_state=42,n_jobs=10,**xg_params),n_jobs=10))
        ])
    
    else:
        pipeline = pickle.load(open(model_path,'rb'))

    return pipeline



def evaluate_model(model, X_test, Y_test, category_names):
    
    preds_test = model.predict(X_test)

    for i,col in enumerate(category_names):
        print(f'Predictions for {col}')
        print(classification_report(Y_test[col],preds_test[...,i]))

    print(f'Precision score: {precision_score(Y_test,preds_test,average="macro")}')
    print(f'Recall: {recall_score(Y_test,preds_test,average="macro")}')
    print(f'ROC: {roc_auc_score(Y_test,preds_test,average="macro")}')


def save_model(model, model_filepath):
    
    # condition to check if file already exists
    if ~os.path.isfile(model_filepath):
        joblib.dump(model, model_filepath, compress = 1)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model(model_filepath)
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()