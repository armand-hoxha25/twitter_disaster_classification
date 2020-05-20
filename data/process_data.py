import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Load the messages and categories from the respective .csv files, and merge them.

    Keyword arguments:
    messages_filepath (str) -- path to the messages .csv file
    categories_filepath (str) -- path to the categories .csv file
    
    Returns:
    df (pandas.DataFrame) -- the result dataframe after concatenating the data along the ID column
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages,categories, on='id')

    return df
def category_values(cat_list):
    """
    obtain the values for each category

    Keyword arguments:
    cat_list (str) -- value of the categories column
    
    Returns:
    values (numpy.array) -- a numpy array with the values for that row
    """    
    all_cats = cat_list.split(';')
    
    values=[x.split('-')[1] for x in all_cats]
    return np.array(pd.to_numeric(values))

def category_names(cat_list):
    """
    obtain the values for each category

    Keyword arguments:
    cat_list (str) -- value of the categories column
    
    Returns:
    values (list) -- a numpy array with the names of the categories
    """      
    all_cats = cat_list.split(';')
    
    values=[x.split('-')[0] for x in all_cats]
    return np.array(values)    

def clean_data(df):
    """
    clean the dataframe by:
    1. expanding the categories to  0, 1 encodings
    2. dropping unnecessary columns
    3. dropping duplicates
    4. replacing values that cannot be interpreted
    5. droping NaN values

    Keyword arguments:
    df (pandas.DataFrame) -- pandas dataframe with messages and their categories
    
    Returns:
    df (pandas.DataFrame) -- a cleaned up pandas dataframe
    """  
    categories = df['categories']
    categories_expanded = categories.apply(category_values)
    categories_expanded = categories_expanded.apply(pd.Series)
    
    cat_names = category_names(categories.values[0])
    categories_expanded.columns = cat_names
    df = pd.concat([df, categories_expanded], axis =1)
    df.drop(columns = ['categories','related','child_alone'])
    df = df.drop_duplicates(subset='message')
    df.replace(to_replace=2, value=1,inplace=True)
    df = df.dropna(subset=['request'])
    print(df.shape)
    return df

def save_data(df, database_filename):
    """
    saves a dataframe to sqlite database

    Keyword arguments:
    df (pandas.DataFrame) -- the cleaned up dataframe
    database_filename (str) -- databasefilename
    """  
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('cleaned_data', engine, index=False, if_exists='replace')  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
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