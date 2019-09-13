#!/usr/bin/env python
# coding: utf-8


import pandas as pd
data= pd.read_csv('data.csv')
from sklearn.utils import shuffle
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from category_encoders import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split 
data=data.drop(columns=['Unnamed: 0','Photo','Flag','Club Logo','Contract Valid Until'])

def clean_release_clause(row):
    try:
        data= row.loc[row['Release Clause'].isnull(),'Release Clause'] = row['Value']
        return data
    except:
        return('error')
def convert_value(row):
    if row:
        row=row.replace('€','')
        row=row.replace('M','000000')
        row=row.replace('K','000')
        row=row.replace('.','')
        return int(row)
    
def transform(row):
    clean_data= pd.DataFrame()
    
    clean_data['name']=row['Name']
    clean_data['age']=row['Age']
    clean_data['value']=row['Value'].apply(convert_value)
    clean_data['wage']=row['Wage'].apply(convert_value)
    clean_data['release_clause']=clean_release_clause(row)
    clean_data['release_clause']=clean_data['release_clause'].apply(convert_value)
    
    clean_data[['nationality','club',
                'skill moves','position', 'Crossing',
       'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys', 'Dribbling',
       'Curve', 'FKAccuracy', 'LongPassing', 'BallControl', 'Acceleration',
       'SprintSpeed', 'Agility', 'Reactions', 'Balance', 'ShotPower',
       'Jumping', 'Stamina', 'Strength', 'LongShots', 'Aggression',
       'Interceptions', 'Positioning', 'Vision', 'Penalties', 'Composure',
       'Marking', 'StandingTackle', 'SlidingTackle', 'GKDiving', 'GKHandling',
       'GKKicking', 'GKPositioning', 'GKReflexes']]=row[['Nationality','Club',
                                         'Skill Moves','Position','Crossing',
       'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys', 'Dribbling',
       'Curve', 'FKAccuracy', 'LongPassing', 'BallControl', 'Acceleration',
       'SprintSpeed', 'Agility', 'Reactions', 'Balance', 'ShotPower',
       'Jumping', 'Stamina', 'Strength', 'LongShots', 'Aggression',
       'Interceptions', 'Positioning', 'Vision', 'Penalties', 'Composure',
       'Marking', 'StandingTackle', 'SlidingTackle', 'GKDiving', 'GKHandling',
       'GKKicking', 'GKPositioning', 'GKReflexes']]
    clean_data['label']=row['Overall']
    
    return shuffle(clean_data)

def fill_missing(row):
    columns={'constant_replacement':['release_clause'],
             
            'mean_replacememt':['age','Crossing',
       'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys', 'Dribbling',
       'Curve', 'FKAccuracy', 'LongPassing', 'BallControl', 'Acceleration',
       'SprintSpeed', 'Agility', 'Reactions', 'Balance', 'ShotPower',
       'Jumping', 'Stamina', 'Strength', 'LongShots', 'Aggression',
       'Interceptions', 'Positioning', 'Vision', 'Penalties', 'Composure',
       'Marking', 'StandingTackle', 'SlidingTackle', 'GKDiving', 'GKHandling',
       'GKKicking', 'GKPositioning', 'GKReflexes'],
             'no_replacement':['name','value','wage','club','nationality','skill moves','position','label']}
    
    constant_impute_df= row[columns['constant_replacement']]
    constant_impute=SimpleImputer(missing_values=pd.np.nan,fill_value=0)
    const_imputing= pd.DataFrame(constant_impute.fit_transform(constant_impute_df))
    const_imputing.columns= constant_impute_df.columns
    
    mean_impute_df=row[columns['mean_replacememt']]
    mean_impute=SimpleImputer(missing_values=pd.np.nan,strategy='mean')
    mean_imputing=pd.DataFrame(mean_impute.fit_transform(mean_impute_df))
    mean_imputing.columns= mean_impute_df.columns
    
    no_replacement_df= row[columns['no_replacement']]
    return pd.concat([no_replacement_df,mean_imputing,const_imputing],axis=1)

def encode(row):
    columns=['nationality','club','position']
   
    other_columns=list(set(row.columns)-set(['label']))
    enc=TargetEncoder(cols=columns, min_samples_leaf=20,smoothing=1.0).fit(row[other_columns],row['label'])
    
    encoded_train=enc.transform(row[other_columns],row['label'])
    
    encoded_train['label']=row['label']
    
    return encoded_train

def normalize(row):
    min_max= MinMaxScaler()
    training_columns=list(set(row.columns)-set(['label','name']))
    normalized_df=pd.DataFrame(min_max.fit_transform(row[training_columns],row['label']))
    normalized_df.columns=training_columns
    normalized_label=pd.concat([normalized_df,row[['label','name']]],axis=1)
    return shuffle(normalized_label)

def training(row):
    final_df=pd.DataFrame()
    final_df[['label','name']]=row[['label','name']]
    
    linear=LinearRegression()
    x_features= list(set(row.columns)-set(['label','name']))
    X_train, X_test, y_train, y_test = train_test_split(row[x_features], row['label'], test_size=0.2, random_state=42)
    linear.fit(X_train,y_train)
    pred=linear.predict(X_test)
    final_df['prediction']=pd.DataFrame([int(w) for w in pred])
    
    return final_df
    
def preprocessing(row):
    imputed_df=pd.DataFrame()
    
    imputed_df= fill_missing(transform(row))
    
    encoded_df=encode(imputed_df)
    
    normalized_df=normalize(encoded_df).dropna()
    
    train=training(normalized_df)
    
    
    
    return train

