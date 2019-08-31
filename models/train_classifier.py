# import libraries
import sys
import os.path
import pickle
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, MetaData

import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

import re

import sklearn
from sklearn.pipeline import Pipeline, FeatureUnion
import xgboost
from sklearn.multioutput import MultiOutputClassifier
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, precision_score,\
                            recall_score, roc_auc_score,\
                            balanced_accuracy_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


def load_data(database_filepath):
    """load data from sql database

    Arguments:
    database_filepath -- path for sql database file with processed messages data
    """
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('CleanMessages', con=engine)
    X = df['message']
    Y = df[[x for x in df.columns.values if x not in ['id', 'genre',
                                                      'message', 'original']]]
    Y.drop(columns='child_alone', inplace=True)
    category_names = Y.columns.values
    return X, Y, category_names


def tokenize(text):
    """Function to tokenize text - remove stopwords and lemmatize.

    Arguments:
    text -- string to be tokenized before modeling
    """

    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()

    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # tokenize text
    tokens = word_tokenize(text)

    # lemmatize andremove stop words
    tokens = [lemmatizer.lemmatize(word) for word in tokens
              if word not in stop_words]

    return tokens


def build_model(model_path):
    """Function to build a pipeline for classifying messages using SVC.

    Arguments:
    model_path -- path to pkl file for saved model to check if it already exists
    """

    if ~os.path.isfile(model_path):

        # SVC parameters to run grid search over
        params = {
            'clf__estimator__kernel': ['linear', 'rbf'],
            'clf__estimator__C': [0.1, 1, 5]
        }

        pipeline = Pipeline([
                             ('countvec', CountVectorizer(tokenizer=tokenize)),
                             ('tfidf', TfidfTransformer()),
                             ('lsa', TruncatedSVD(random_state=42,
                                                  n_components=100)),
                             ('clf',
                              MultiOutputClassifier(sklearn.svm.SVC(random_state=42,
                                                                    class_weight='balanced',
                                                                    gamma='scale')))
                           ])
        model = GridSearchCV(pipeline, params, cv=5, scoring='f1_samples')

    else:
        model = pickle.load(open(model_path, 'rb'))

    return model


def evaluate_model(model, X_test, Y_test, category_names):
    """Evaluate performance of trained model on test data.

    Arguments:
    model -- trained model whose performance is to be tested
    X_test -- feature matrix for test data
    Y_test -- labels matrix for test data
    category_names -- multi label category names
    """

    preds_test = model.predict(X_test)

    for i, col in enumerate(category_names):
        print(f'Predictions for {col}')
        print(classification_report(Y_test[col], preds_test[..., i]))

    print(f'Precision score: {precision_score(Y_test,preds_test, average="macro")}')
    print(f'Recall: {recall_score(Y_test, preds_test, average="macro")}')
    print(f'ROC: {roc_auc_score(Y_test, preds_test, average="macro")}')


def save_model(model, model_filepath):
    """Save Model to specified path.

    Arguments:
    model -- trained model/pipeline to be saved
    model_filepath -- path to save the model to
    """

    # condition to check if file already exists
    if ~os.path.isfile(model_filepath):
        joblib.dump(model, model_filepath, compress=1)


def main():
    # Main execution flow

    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model(model_filepath)

        print('Training model...')

        if ~os.path.isfile(model_filepath):
            model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()