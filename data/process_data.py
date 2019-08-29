import sys
import pandas as pd
import sqlalchemy
import numpy as np

def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.drop_duplicates().merge(categories.drop_duplicates(),how='outer')
    return df

def clean_data(df):
    categories = df['categories'].str.split(pat=';',expand=True)
    row = categories.loc[0,:].copy()
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames
    for column in category_colnames:
    # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1]
        
        # set values to 0 or 1 (anything above 1 is 1)
        categories[column] = categories[column].apply(lambda x: 1 if int(x)!=0 else 0)
        
        # convert column from string to numeric
        categories[column] = categories[column].astype(np.int8)
    
    # drop the original categories column from `df`
    df.drop(columns='categories',inplace=True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories],axis=1)

    # drop duplicates
    df.drop_duplicates(inplace=True)

    return df



def save_data(df, database_filename):
    engine = sqlalchemy.create_engine('sqlite:///'+database_filename)
    df.to_sql('CleanMessages', engine, index=False, if_exists='replace')  


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