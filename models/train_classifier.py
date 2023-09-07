import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn import set_config
import pickle

def load_data(database_filepath):
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('table', con=engine).astype(str).dropna()
    X = df['message']
    y = df.drop('message', axis=1)
    return X, y, y.columns

def tokenize(text):
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    words = word_tokenize(text)
    for tok in words:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens

def build_model():
    pipeline = Pipeline([
        ('text_pipeline', Pipeline([
                ('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer())
        ])),
        ('clf', RandomForestClassifier())
    ])
    parameters = {
        'clf__max_depth': [800, 1000],
        'clf__n_estimators': [500, 800]
    }
    model = GridSearchCV(pipeline, param_grid=parameters)
    return model

def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)
    i = 0
    while i < len(Y_test.columns):
        print(f'Column:{Y_test.columns[i]}\n {classification_report(np.array(Y_test)[i], y_pred[i])}\n')
        i += 1

def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))

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