# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 09:53:41 2019

@author: Jimny
"""

# Import needed packages
import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Define main function here
def main():
    # Open and read in the cleaned data from 2013 to 2016
    with open('Cleaned Data/2013_2016_cleaned.csv', 'rb') as input_all:
        df_all = pd.read_csv(input_all , sep=',',  header = 0, encoding = 'utf-8')

    # Select the columns we needed
    select_columns = ['ev_month', 'ev_weekday', 'ev_state', 
                      'ev_highest_injury', 'damage', 'far_part',
                      'acft_category', 'type_fly', 'CICTTEvent', 'CICTTPhase']
    df_select = df_all.loc[ : , select_columns]
    
    # Change column names for convenience
    df_select.columns = ['Month', 'Weekday', 'State', 
                         'Injury_Level', 'Damage', 'Part',
                         'Aircraft', 'Flight_Type', 'Event', 'Phase']
    
    # Print NA values counts for checking tnad then dorp rows with NA values
    print('Check number of NA values from selected columns:\n',
          df_select.isnull().sum())
    
    df_select.dropna(axis=0, inplace = True)
    df_select.reset_index(drop = True, inplace = True)

    # Separate predictors and responses
    df_X = df_select.drop(['Injury_Level'], axis = 1)
    df_y = df_select.loc[: ,  'Injury_Level' ]
    
    # Using One-hot encoder to transform the dataset
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(df_X)
    df_X = pd.DataFrame(enc.transform(df_X).toarray(), columns = enc.get_feature_names(list(df_X.columns)))
    
    # Generate principle components
    pca = PCA(n_components=2)
    X_PCA = pd.DataFrame(pca.fit_transform(df_X))
    
    # Open a csv file to write the result
    with open('PCA_results.csv','w', newline = '') as out_data:
        X_PCA.to_csv(out_data, sep=',',index = False)
        
main()