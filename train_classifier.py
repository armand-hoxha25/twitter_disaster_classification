import sys
from sqlalchemy import create_engine
import numpy as np
import pandas as pd
import os
import re

from sklearn.metrics import accuracy_score, make_scorer, f1_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
#from skmultilearn.model_selection import iterative_train_test_split 
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
import pickle
def load_data(database_filepath):
    """
    Load the data from the .db file

    Keyword arguments:
    database_filepath (str) -- the path to the .db file  
    """

    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql('cleaned_data',engine)
    X = df['message'].values
    Y = df.iloc[:,-34:]
    category_names = Y.columns
    return X, Y, category_names

def tokenize(text):
    """
    Tokenize a sentence:
    1 - transform to lower case
    2 - remove non-alphanumeric entities
    3 - stem the data using the PorterStemmer
    4 - remove stop words 

    Keyword arguments:
    text (str) -- the sentence/word to be tokenized
    """

    text = text.lower()
    text= re.sub(r"^[ a-z0-9 ]","",text)
    text= text.split()
    stemmer = PorterStemmer()
    text = [stemmer.stem(word) for word in text]
    for w in text:
        if w in stopwords.words('english'):
            text.remove(w)
    return text


def build_model():
    """
    Build an Scikit-Learn Pipeline object
    Pipeline:
    Count Vectorizer --> TF-IDF Transformation --> MultioutputClassifier(RandomForestClassifier)
    
    """

    
    pipeline = Pipeline(steps=[
    ('count_vector', CountVectorizer()),
    ('tfidf',TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators=100), n_jobs = -1))
    ])
    return pipeline

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Print evaluation metrics.
    The first output is an overall accuracy score across all classes.
    It is then followed by classification_report of each class, which includes the f1_score, accuracy, and recall.

    Keyword arguments:
    model (sklearn.pipeline.Pipeline) -- the pipeline object to be evaluated, must have a classifier at its last step
    X_test, Y_test (numpy arrays) -- the testing parts from the train_test_split of the data
    category_names (list) -- the names of each class
    """

    print("overall Accuracy is : {}".format(model.score(X_test,Y_test)))
    for n in range(len(category_names)):
        print(category_names[n])
        print(classification_report(Y_test.iloc[n+1,:].values,model.predict(X_test)[n,:]))


def save_model(model, model_filepath):
    """
    Save the model in a pickle file to be used later

    Keyword arguments:
    real -- the real part (default 0.0)
    imag -- the imaginary part (default 0.0)
    """

    filename = model_filepath
    pickle.dump(model, open(filename, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
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