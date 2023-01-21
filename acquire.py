import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error


#Removes warnings and imporves asthenics
import warnings
warnings.filterwarnings("ignore")

from env import get_connection

"""
This is how you get rid of the Unnamed: 0 column:

#read_csv(filename, index_col=0)
#to_csv(filename, index=False)
"""


def wrangle_zillow():
    """
    This function reads the zillow data from Codeup db into a df.
    Changes the names to be more readable.
    Drops null values.
    """
    filename = "zillow_2017.csv"
    
    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else:
        
        # read the SQL query into a dataframe
        query = """
        SELECT taxvaluedollarcnt, bedroomcnt, bathroomcnt,
        calculatedfinishedsquarefeet, transactiondate
        FROM properties_2017
        LEFT JOIN predictions_2017 USING (parcelid)
        WHERE propertylandusetypeid LIKE 261 AND transactiondate like '2017%%';
        """
        df = pd.read_sql(query, get_connection('zillow'))
        
        # Remove NAs. No significant change to data. tax_values upper outliers were affected the most.
        df = df.dropna()
        df.rename(columns = {'bedroomcnt': 'bedrooms', 'bathroomcnt': 'bathrooms',
                             'calculatedfinishedsquarefeet': 'sqft', 'taxvaluedollarcnt':'tax_value'}, inplace=True)
        cols_outliers = ['bedrooms', 'bathrooms', 'sqft', 'tax_value']
        for col in cols_outliers:
            df = df[df[col] <= df[col].quantile(q=0.99)]
        
        # Write that dataframe to disk for later. Called "caching" the data for later.
        df.to_csv(filename, index=False)
        
        # Return the dataframe to the calling code
        return df

def train_validate(df, stratify_col = None, random_seed=1969):
    """
    This function takes in a DataFrame and column name for the stratify argument (defualt is None).
    It will split the data into three parts for training, testing and validating.
    Note: DO NOT specify the target(stratify) column for continuous data. It will error.
    """
    #This is logic to set the stratify argument:
    stratify_arg = ''
    if stratify_col != None:
        stratify_arg = df[stratify_col]
    else:
        stratify_arg = None
    
    #This splits the DataFrame into 'train' and 'test':
    train, test = train_test_split(df, train_size=.7, stratify=stratify_arg, random_state = random_seed)
    
    #The length of the stratify column changed and needs to be adjusted:
    if stratify_col != None:
        stratify_arg = train[stratify_col]
        
    #This splits the larger 'train' DataFrame into a smaller 'train' and 'validate' DataFrames:
    train, validate = train_test_split(train, test_size=.2, stratify=stratify_arg, random_state = random_seed)
    return train, validate, test


def train_val_test(train, val, test, target_col):
    """
    Seperates out the target variable and creates a series with only the target variable to test accuracy.
    """
    #Seperating out the target variable
    X_train = train.drop(columns=[target_col])
    y_train = train[target_col]

    X_val = val.drop(columns = [target_col])
    y_val = val[target_col]

    X_test = test.drop(columns = [target_col])
    y_test = test[target_col]
    return X_train, y_train, X_val, y_val, X_test, y_test

def find_baseline(y_train):
    """
    This function shows a comparison in baselines for mean and median.
    Output is the RMSE error when using both mean and median.
    """    
    # Train set
    bl_df = pd.DataFrame({'actual':y_train,
                          'mean_bl':y_train.mean(),
                          'median_bl':y_train.median()}
                        )
    
    rmse_train_mean = mean_squared_error(bl_df['actual'],
                                         bl_df['mean_bl'],
                                         squared=False)
    
    rmse_train_median = mean_squared_error(bl_df['actual'],
                                           bl_df['median_bl'],
                                           squared=False)


    #Print the findings and difference between each:
    print(f'RMSE Mean training baseline: {round(rmse_train_mean,0):,.0f}')
    print("*****************************************")
    print(f'RMSE Median training baseline: {round(rmse_train_median,0):,.0f}')
    
    return min(rmse_train_mean, rmse_train_median)

def check_p_val(p_val, h0, ha, alpha=0.05):

    if p_val < alpha:
        print(f'We have evidence to reject the null hypothesis. {ha}')
    else:
        print(f'We do not have evidence to reject the null hypothesis. {h0}')

        
        
def scale_zillow(train, val, test, scaler_model = 1, cont_columns = ['sqft']):
    """
    This takes in the train, validate and test DataFrames, scales the cont_columns using the
    selected scaler and returns the DataFrames.
    *** Inputs ***
    train: DataFrame
    validate: DataFrame
    test: DataFrame
    scaler_model (1 = MinMaxScaler, 2 = StandardScaler, else = RobustScaler)
    - default = MinMaxScaler
    cont_columns: List of columns to scale in DataFrames
    *** Outputs ***
    train: DataFrame with cont_columns scaled.
    val: DataFrame with cont_columns scaled.
    test: DataFrame with cont_columns scaled.
    """
    #Create the scaler
    if scaler_model == 1:
        scaler = MinMaxScaler()
    elif scaler_model == 2:
        scaler = StandardScaler()
    else:
        scaler = RobustScaler()
    
    #Make a copy
    train_scaled = train.copy()
    val_scaled = val.copy()
    test_scaled = test.copy()

    
    #Fit the scaler
    scaler = scaler.fit(train[cont_columns])
    
    #Build the new DataFrames
    train_scaled[cont_columns] = pd.DataFrame(scaler.transform(train[cont_columns]),
                                                  columns=train[cont_columns].columns.values).set_index([train.index.values])

    val_scaled[cont_columns] = pd.DataFrame(scaler.transform(val[cont_columns]),
                                                  columns=val[cont_columns].columns.values).set_index([val.index.values])

    test_scaled[cont_columns] = pd.DataFrame(scaler.transform(test[cont_columns]),
                                                 columns=test[cont_columns].columns.values).set_index([test.index.values])
    #Sending them back
    return train_scaled, val_scaled, test_scaled

def explore_relationships(feature_list, train, target_col, visuals = False):
    """
    This function takes in a list of features, grabs the .describe() metrics associated with the target column.
    *** Inputs ***
    feature_list: List of DataFrame column names to iterate through and compare to target column.
    train: Panda's DataFrame to explore.
    target_col: String. Title of target column.
    *** Output ***
    DataFrame with metrics to explore
    """
    metrics = []
    for feature in feature_list:
        num_items = train[feature].unique()
        num_items.sort()
        for item in num_items:
            temp_df = train[train[feature] == item][target_col].describe()
            temp_metrics = {
                'comparison' : f'{item}_{feature}',
                'count' : round(temp_df[0],0),
                'mean' : round(temp_df[1],0),
                'std' : round(temp_df[2],0),
                'min' : round(temp_df[3],0),
                '25%' : round(temp_df[4],0),
                '50%' : round(temp_df[5],0),
                '75%' : round(temp_df[6],0),
                'max' : round(temp_df[7],0)}
            metrics.append(temp_metrics)

    feature_per_item = pd.DataFrame(metrics)
    if visuals == True:
        sns.lineplot(data=feature_per_item, x='comparison', y='25%', legend='brief').set(title=f'{target_col} to {feature} comparison')
        sns.lineplot(data=feature_per_item, x='comparison', y='mean', markers=True)
        sns.lineplot(data=feature_per_item, x='comparison', y='50%')
        sns.lineplot(data=feature_per_item, x='comparison', y='75%')
        plt.ylabel(f'{target_col}')
        plt.xlabel(f'{item}_{feature}')
        
    return feature_per_item