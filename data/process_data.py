# import libraries
import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import pickle
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


def load_data(messages_filepath, categories_filepath):
    # Read csv files from given paths
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    # Merge dfs
    df = messages.merge(categories, on='id')
    # Set feature values and names
    categories = categories['categories'].str.split(';', expand=True)
    row = categories.loc[0]
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames
    for column in categories:
        categories[column] = categories[column].apply(lambda x: int(x[-1:]))
    # Drop unneccesary id and category
    df = df.drop(['categories', 'id'], axis=1)
    # Concat dfs
    df = pd.concat([df, categories], axis=1).drop_duplicates().dropna()

    return df

def save_data(df, database_filename):
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('table', con=engine, index=False, if_exists='replace')

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_name = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)
        
        print('Saving data...\n    DATABASE: {}'.format(database_name))
        save_data(df, database_name)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()