#Python packages
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

#sklearn modeules
#Sklearn
from sklearn.preprocessing import (MinMaxScaler, RobustScaler,
                                   StandardScaler, QuantileTransformer)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.metrics import mean_squared_error


#Removes warnings and imporves asthenics
import warnings
warnings.filterwarnings("ignore")

#Connection information to acquire data
from env import get_connection

    ##################### Aquire and Prepare Functions #####################

#Sets variables to automatically pass into functions
random_seed = 1969
alpha = 0.05
target_col = 'tax_value'

def wrangle_zillow():
    """
    This function reads the zillow data from Codeup db into a df.
    Changes the names to be more readable.
    Drops null values.
    """
    filename = "zillow_2017.csv"
    
    if os.path.isfile(filename):
        return pd.read_csv(filename, parse_dates=['transactiondate'])
    else:
        
        # read the SQL query into a dataframe
        #First query
        query = """
        SELECT taxvaluedollarcnt, bedroomcnt, bathroomcnt,
        calculatedfinishedsquarefeet, transactiondate
        FROM properties_2017
        LEFT JOIN predictions_2017 USING (parcelid)
        WHERE propertylandusetypeid LIKE 261 AND
        transactiondate like '2017%%';
        """
        
        #Final query
        query2 = """
        SELECT taxvaluedollarcnt, bedroomcnt,
        bathroomcnt, calculatedfinishedsquarefeet,
        transactiondate, hashottuborspa, decktypeid,
        garagecarcnt, poolcnt, fips, latitude, longitude
        FROM properties_2017
        LEFT JOIN predictions_2017 USING (parcelid)
        WHERE propertylandusetypeid LIKE 261 AND
        transactiondate like '2017%%';
        """
        df = pd.read_sql(query2, get_connection('zillow'))
        
        # Remove NAs. No significant change to data.
        # tax_values upper outliers were affected the most.
        df.rename(columns = {'bedroomcnt': 'bedrooms',
                             'bathroomcnt': 'bathrooms',
                             'calculatedfinishedsquarefeet': 'sqft',
                             'taxvaluedollarcnt':'tax_value', 
                             'hashottuborspa' : 'hottub_spa', 
                             'decktypeid': 'deck', 
                             'poolcnt': 'pool',
                             'fips':'County'}, 
                  inplace=True)
        df.County = df.County.map({6037.0:'Los Angeles', 6059.0:'Orange', 6111.0:'Ventura'})
        df['latitude'] = df['latitude'] / 10000000
        df['longitude'] = df['longitude'] / 100000000
        
        # Changes to panda's datetime format
        df['transactiondate'] = pd.to_datetime(df['transactiondate'])
        
        # Creating bins for continuous data
        sqft_bins = [0, 200, 400, 600, 800, 1000, 1200, 1400,
                     1600, 1800, 2000, 2200, 2400, 2600, 2800,
                     3000, 3200, 3400, 3600, 3800, 4000, 4200,
                     4400, 4600, 4800, 5000]        
        bin_labels = [200, 400, 600, 800, 1000, 1200, 1400, 1600,
                      1800, 2000, 2200, 2400, 2600, 2800, 3000,
                      3200, 3400, 3600, 3800, 4000, 4200, 4400,
                      4600, 4800, 5000]        
        df['sqft_bins'] = pd.cut(df.sqft, bins = sqft_bins,
                                 labels = bin_labels)        
        value_bins = [0, 400000, 800000, 1200000, 1600000, 30000000]        
        value_bin_labels = ['$400k', '$800k', '$1.2m', '$1.5m', '$1.5m+']
        df['value_bins'] = pd.cut(df.tax_value, bins = value_bins,
                                  labels = value_bin_labels)

        # Changes column data to a more useful format
        df['hottub_spa'] = df['hottub_spa'].notna().astype('int')
        df['deck'] = df['deck'].notna().astype('int')
        df['pool'] = df['pool'].notna().astype('int')
        df['has_garages'] = df['garagecarcnt'].notna().astype('int')
        df['garagecarcnt'].fillna(0, inplace=True)
        df['num_of_features'] = df[['pool','deck','hottub_spa', 'has_garages']].sum(axis=1)

        #Removing rows with null values
        df = df.dropna()

        # Removes outliers in the primary columns
        cols_outliers = ['bedrooms', 'bathrooms', 'sqft', 'tax_value']
        for col in cols_outliers:
            df = df[df[col] <= df[col].quantile(q=0.99)]
        
        # Write that dataframe to disk for later. Called "caching" the data for later.
        df.to_csv(filename, index=False)
        
        # Return the dataframe to the calling code
        return df
    
    
    
    
    
    
    ##################### Explore Functions ##########################

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

def check_p_val(p_val, h0, ha, alpha=0.05):
    """
    Checks if p value is significant or not and prints the associated string
    """
    
    #Pretty self explanitory.
    if p_val < alpha:
        print(f'We have evidence to reject the null hypothesis.')
        print(f'{ha}')
    else:
        print(f'We do not have evidence to reject the null hypothesis.')
        print(f'{h0}')

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
        sns.lineplot(data=feature_per_item, x='comparison', y='25%',
                             legend='brief').set(title=f'{target_col} to {feature} comparison',
                                                 xlabel =f'{feature}', ylabel = f'{target_col}')
        sns.lineplot(data=feature_per_item, x='comparison', y='mean', markers=True)
        sns.lineplot(data=feature_per_item, x='comparison', y='50%')
        sns.lineplot(data=feature_per_item, x='comparison', y='75%')
        plt.ylabel(f'{target_col}')
        plt.xlabel(f'{item}_{feature}')
        
    return feature_per_item

def q1_stats_test(train):
    """
    Does a stats test for the first question.
    """

    h0 = "Square feet and tax value are independent."
    ha = "Square feet and tax value have a relationship."
    
    s, p_val = stats.spearmanr(train['sqft'], train['tax_value'])
    
    check_p_val(p_val, h0, ha, alpha=0.05)
    
def q2_stats_test(train):
    """
    Does a stats test for the second question.
    """

    h0 = "Bedrooms and tax value are independent."
    ha = "Bedrooms feet and tax value have a relationship."

    s, p_val = stats.spearmanr(train['bedrooms'], train['tax_value'])
    
    check_p_val(p_val, h0, ha, alpha=0.05)
    
def q3_stats_test(train):
    """
    Does a stats test for the third question.
    """    
    
    h0 = "Number of bathrooms and home value are independent."
    ha = "Number of bathrooms and home value have a relationship."

    s, p_val = stats.spearmanr(train['bathrooms'], train['tax_value'])
    
    check_p_val(p_val, h0, ha, alpha=0.05)
    
def q4_stats_test(train):
    """
    Does a stats test for the third question.
    """

    h0 = "Number of features and home value are independent."
    ha = "Number of features and home value have a relationship."
    
    s, p_val = stats.spearmanr(train['num_of_features'], train['tax_value'])
    
    check_p_val(p_val, h0, ha, alpha=0.05)

##################### Functions for visuals ########################

def explore_sqft(train):
    #This graphic only works if you run the notebook cells in order
    sns.set_style('whitegrid', rc={'figure.facecolor':'gainsboro'})
    sns.relplot(data=train, x='sqft', y='tax_value', marker=".",
                height=6, aspect=1).set(title='Home value versus Square feet',
                                         xlabel="Square Feet",
                                         ylabel="Home Value")

    sns.kdeplot(data=train, x='sqft', y='tax_value', c='tomato')
    plt.xticks([0, 1000, 2000, 3000, 4000, 5000], ['0k', '1k', '2k', '3k', '4k', '5k'])
    plt.yticks([0, 400000, 800000, 1200000, 1600000, 2000000, 2400000],['$0','$400k','$800k','$1.2m', '$1.6m', '$2m', '$2.4m'])
    plt.xlim(left=0, right=4500)
    plt.ylim(bottom=0)
    plt.show()

def explore_bedrooms(train):
    """
    Visuals to show relationship between the number
    of bedrooms and a house's value.
    """
    metrics = []
    num_items = train['bedrooms'].unique()
    num_items.sort()
    for item in num_items:
        temp_df = train[train['bedrooms'] == item]['tax_value'].describe()
        temp_metrics = {
            'comparison' : f'{item}',
            'count' : round(temp_df[0],0),
            'mean' : round(temp_df[1],0),
            'std' : round(temp_df[2],0),
            'min' : round(temp_df[3],0),
            '25%' : round(temp_df[4],0),
            '50%' : round(temp_df[5],0),
            '75%' : round(temp_df[6],0),
            'max' : round(temp_df[7],0)}
        metrics.append(temp_metrics)

    bedrooms_df = pd.DataFrame(metrics)
    
    fig, ax = plt.subplots(facecolor='gainsboro', edgecolor='dimgray')
    sns.set_style('whitegrid', rc={'figure.facecolor':'gainsboro'})
    sns.lineplot(ax=ax, data=bedrooms_df, x='comparison', y='75%', label = '75%').set(title="How the number of bedrooms affect a house's value")
    sns.lineplot(ax=ax, data=bedrooms_df, x='comparison', y='mean', label = 'mean')
    sns.lineplot(ax=ax, data=bedrooms_df, x='comparison', y='50%' , label = '50%')
    sns.lineplot(ax=ax, data=bedrooms_df, x='comparison', y='25%', label = '25%')
    plt.xlim(left=0, right =3)
    plt.ylabel('Home Value')
    plt.xlabel('Bedrooms')
    plt.xticks([0, 1, 2, 3, 4, 5, 6], ['0', '1', '2', '3', '4', '5', '6'])
    plt.yticks([200000, 400000, 600000, 800000],['$200k','$400k','$600k', '$800k'])
    plt.legend(title="Home Value Percentile", framealpha=1, facecolor="whitesmoke", edgecolor='dimgray')
    plt.show()
    
def explore_bathrooms(train):
    """
    Visuals to show relationship between the number of bathrooms and a house's value.
    """
    metrics = []
    num_items = train['bathrooms'].unique()
    num_items.sort()
    for item in num_items:
        temp_df = train[train['bathrooms'] == item]['tax_value'].describe()
        temp_metrics = {
            'comparison' : f'{item}',
            'count' : round(temp_df[0],0),
            'mean' : round(temp_df[1],0),
            'std' : round(temp_df[2],0),
            'min' : round(temp_df[3],0),
            '25%' : round(temp_df[4],0),
            '50%' : round(temp_df[5],0),
            '75%' : round(temp_df[6],0),
            'max' : round(temp_df[7],0)}
        metrics.append(temp_metrics)

    bathrooms_df = pd.DataFrame(metrics)
    
    fig, ax = plt.subplots(facecolor='gainsboro', edgecolor='dimgray')
    sns.set_style('whitegrid', rc={'figure.facecolor':'gainsboro'})
    sns.lineplot(ax=ax, data=bathrooms_df, x='comparison', y='75%', label = '75%').set(title="How the number of bathrooms affect a house's value")
    sns.lineplot(ax=ax, data=bathrooms_df, x='comparison', y='mean', label = 'mean')
    sns.lineplot(ax=ax, data=bathrooms_df, x='comparison', y='50%' , label = '50%')
    sns.lineplot(ax=ax, data=bathrooms_df, x='comparison', y='25%', label = '25%')
    plt.xlim(left=0, right =3)
    plt.ylabel('Home Value')
    plt.xlabel('Bathrooms')
    plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
    plt.yticks([200000, 400000, 600000, 800000, 1000000, 1200000, ],['$200k','$400k','$600k','$800k','$1m','$1.2m'])
    plt.legend(title="Home Value Percentile", framealpha=1, facecolor="whitesmoke", edgecolor='dimgray')
    plt.show()
    
def explore_num_features(train):
    """
    This function takes in the train dataframe and shows the relationship
    between the number of features a house has and it's value.
    """
    
    #Defines the variables to explore
    feature_list = ['num_of_features']
    target_col = 'tax_value'
    
    #Uses the function above to get data to plot
    feature_per_item = explore_relationships(feature_list, train, target_col)
    
    #Plotting code
    fig, ax = plt.subplots(facecolor='gainsboro', edgecolor='dimgray')
    sns.set_style('whitegrid', rc={'figure.facecolor':'gainsboro'})
    sns.lineplot(ax=ax, data=feature_per_item, x='comparison', y='75%',
                 label = '75%').set(title="How the number of features affect a house's value")
    sns.lineplot(ax=ax, data=feature_per_item, x='comparison', y='mean',
                 label = 'mean')
    sns.lineplot(ax=ax, data=feature_per_item, x='comparison', y='50%' ,
                 label = '50%')
    sns.lineplot(ax=ax, data=feature_per_item, x='comparison', y='25%',
                 label = '25%')
    plt.xlim(left=0, right =3)
    plt.ylabel('Home Value')
    plt.xlabel('Number of Features')
    plt.xticks([0, 1, 2, 3], ['0', '1', '2', '3'])
    plt.yticks([400000,800000,1200000],['$400k','$800k','$1.2m'])
    plt.legend(title="Home Value Percentile", framealpha=1,
               facecolor="whitesmoke", edgecolor='dimgray')
    plt.show()

    
def explore_counties(train):
    sns.set_style('whitegrid', rc={'figure.facecolor':'gainsboro'})
    sns.relplot(data= train, x=train.latitude, y=train.longitude, hue=train.County, marker='.', palette='muted')
    plt.ylabel('Longitude', fontsize=12)
    plt.xlabel('Latitude', fontsize=12)
    plt.title("Counties in California")
    plt.show()
    
def explore_value_loc(train):
    train.rename(columns = {'value_bins':'House values'}, inplace=True)
    
    sns.set_style('whitegrid', rc={'figure.facecolor':'gainsboro'})
    fig = sns.relplot(data = train, x='latitude', y='longitude', height=6.5, hue='House values', marker='.', palette='muted')
    plt.ylabel('Longitude', fontsize=12)
    plt.xlabel('Latitude', fontsize=12)
    plt.title("House locaitons")
    plt.show()
    
    
##################### Modeling Functions ##########################

def train_val_test(train, val, test, target_col):
    """
    Seperates out the target variable and creates
    a series with only the target variable to test accuracy.
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
    # Grabs two baselines caclucated from the Train set mean and median.
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

    baseline = min(rmse_train_mean, rmse_train_median)
    #Print the findings and difference between each:
    print(f'RMSE baseline: {round(baseline,0):,.0f}')
    print("**********************************")
    
    #Returns the baseline with the lowest error
    return baseline

def scale_zillow(train, val, test, scaler_model = 1, cont_columns = ['sqft']):
    """
    This takes in the train, validate and test DataFrames,
    scales the cont_columns using the
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

def find_model_scores(df):
    """
    This function takes in the target DataFrame, runs the data against four
    machine learning models and outputs some visuals.
    """
    #Creates a copy so the original data is not affected
    ml_df = df.copy()

    #Drops columns not used in modeling
    ml_df = df.drop(columns=['transactiondate', 'sqft_bins',
                             'value_bins', 'County'])
    #Creates dummy columns
    ml_df = pd.get_dummies(columns=['bedrooms', 'bathrooms',
                                    'num_of_features', 'garagecarcnt'],
                           data=ml_df)
    #Splits data into train, validate and test datasets
    train, val, test = train_validate(ml_df)
    
    #Scales continuous data#Scaling the data
    train, val, test = scale_zillow(train, val, test, scaler_model = 3,
                                    cont_columns = ['sqft'])

    #Seperate target column from feature columns
    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test(train, val, test, target_col)
    
    #Eastablishes the standard to beat
    baseline = find_baseline(y_train)
    
    #List for gathering metrics
    rmse_scores = []

    
    """ *** Builds and fits Linear Regression Model (OLS) *** """
    
    
    lm = LinearRegression(normalize=True)
    lm.fit(X_train, y_train)
    
    #Train data
    lm_preds = pd.DataFrame({'actual':y_train})
    lm_preds['pred_lm'] = lm.predict(X_train)
    
    #Validate data
    lm_val_preds = pd.DataFrame({'actual':y_val})
    lm_val_preds['lm_val_preds'] = lm.predict(X_val)
    
    #Finds score on Train and Validate data
    rmse_train = mean_squared_error(lm_preds['actual'],
                                    lm_preds['pred_lm'],
                                    squared=False) 
    rmse_val = mean_squared_error(lm_val_preds['actual'],
                                  lm_val_preds['lm_val_preds'],
                                  squared=False) 

    #Adds score to metrics list for later comparison
    rmse_scores.append({'Model':'OLS Linear',
                    'RMSE on Train': round(rmse_train,0),
                    'RMSE on Validate': round(rmse_val,0)})
    
    
    """ *** Builds and fits Lasso Lars Model *** """
   
    
    lars = LassoLars(alpha=.25)
    lars.fit(X_train, y_train)
    
    #Train data
    ll_preds = pd.DataFrame({'actual':y_train})
    ll_preds['pred_ll'] = lars.predict(X_train)
    
    #Validate data
    ll_val_preds = pd.DataFrame({'actual':y_val})
    ll_val_preds['ll_val_preds'] = lars.predict(X_val)
    
    #Finds score on Train and Validate data
    rmse_train = mean_squared_error(ll_preds['actual'],
                                    ll_preds['pred_ll'],
                                    squared=False)
    rmse_val = mean_squared_error(ll_val_preds['actual'],
                                  ll_val_preds['ll_val_preds'],
                                  squared=False)
    
    #Adds score to metrics list for later comparison
    rmse_scores.append({'Model':'Lasso Lars',
                    'RMSE on Train': round(rmse_train,0),
                    'RMSE on Validate': round(rmse_val,0)})
    
    
    """ *** Builds and fits Tweedie Regressor (GLM) Model *** """
    
    glm = TweedieRegressor(power=1, alpha=1)    
    glm.fit(X_train, y_train)

    #Train data
    glm_preds = pd.DataFrame({'actual':y_train})
    glm_preds['pred_glm'] = glm.predict(X_train)
    
    #Validate data
    glm_val_preds = pd.DataFrame({'actual':y_val})
    glm_val_preds['glm_val_preds'] = glm.predict(X_val)
    
    #Finds score on Train and Validate data
    rmse_train = mean_squared_error(glm_preds['actual'],
                                    glm_preds['pred_glm'],
                                    squared=False) 
    rmse_val = mean_squared_error(glm_val_preds['actual'],
                                  glm_val_preds['glm_val_preds'],
                                  squared=False)
    
    #Adds score to metrics list for later comparison
    rmse_scores.append({'Model':'Tweedie',
                        'RMSE on Train': round(rmse_train,0),
                        'RMSE on Validate': round(rmse_val,0)})
    
    
    """ *** Builds and fits Polynomial regression Model *** """

    
    #Polynomial Regression part:
    # make the polynomial features to get a new set of features
    pf = PolynomialFeatures(degree=1)

    # fit and transform X_train_scaled
    X_train_degree2 = pf.fit_transform(X_train)

    # transform X_validate_scaled & X_test_scaled
    X_validate_degree2 = pf.transform(X_val)
    X_test_degree2 = pf.transform(X_test)

    #Polynomial Regression being fed into Linear Regression:
    lm2 = LinearRegression(normalize=True)
    lm2.fit(X_train_degree2, y_train)

    #Train data
    lm2_preds = pd.DataFrame({'actual':y_train})
    lm2_preds['pred_lm2'] = lm2.predict(X_train_degree2)

    #Validate data
    lm2_val_preds = pd.DataFrame({'actual':y_val})
    lm2_val_preds['lm2_val_preds'] = lm2.predict(X_validate_degree2)

    #Finds score on Train and Validate data
    rmse_train = mean_squared_error(lm2_preds['actual'],
                                    lm2_preds['pred_lm2'],
                                    squared=False) 
    rmse_val = mean_squared_error(lm2_val_preds['actual'],
                                  lm2_val_preds['lm2_val_preds'],
                                  squared=False)

    #Adds score to metrics list for later comparison
    rmse_scores.append({'Model':'Polynomial',
                        'RMSE on Train': round(rmse_train,0),
                        'RMSE on Validate': round(rmse_val,0)})
    
    """ *** Later comparison section to display results *** """
    
    #Builds and displays results DataFrame
    rmse_scores = pd.DataFrame(rmse_scores)
    rmse_scores['Difference'] = round(rmse_scores['RMSE on Train'] - rmse_scores['RMSE on Validate'],2)    
    
    #Results were too close so had to look at the numbers
    print(rmse_scores)
    
    #Building variables for plotting
    rmse_min = min([rmse_scores['RMSE on Train'].min(),
                    rmse_scores['RMSE on Validate'].min(), baseline])
    rmse_max = max([rmse_scores['RMSE on Train'].max(),
                    rmse_scores['RMSE on Validate'].max(), baseline])

    lower_limit = rmse_min * 0.8
    upper_limit = rmse_max * 1.05


    x = np.arange(len(rmse_scores))  # the label locations
    width = 0.25  # the width of the bars

    fig, ax = plt.subplots(facecolor="gainsboro")
    rects1 = ax.bar(x - width/2, rmse_scores['RMSE on Train'],
                    width, label='Training data', color='#4e5e33',
                    edgecolor='dimgray') #Codeup dark green
    rects2 = ax.bar(x + width/2, rmse_scores['RMSE on Validate'],
                    width, label='Validation data', color='#8bc34b',
                    edgecolor='dimgray') #Codeup light green

    # Need to have baseline input:
    plt.axhline(baseline, label="Baseline Error", c='tomato', linestyle=':')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.axhspan(0, baseline, facecolor='palegreen', alpha=0.2)
    ax.axhspan(baseline, upper_limit, facecolor='red', alpha=0.3)
    ax.set_ylabel('RMS Error')
    ax.set_xlabel('Machine Learning Models')
    ax.set_title('Model Error Scores')
    ax.set_xticks(x, rmse_scores['Model'])

    plt.ylim(bottom=lower_limit, top = upper_limit)

    ax.legend(loc='upper right', framealpha=.9, facecolor="whitesmoke",
              edgecolor='darkolivegreen')

    #ax.bar_label(rects1, padding=4)
    #ax.bar_label(rects2, padding=4)
    fig.tight_layout()
    #plt.savefig('best_model_all_features.png')
    plt.show()


def final_test(df):
    """
    This function takes in the target DataFrame, runs the data against the
    machine learning model selected for the final test and outputs some visuals.
    """
    
    #List to capture scores
    final_rmse_scores = []
    
    #Creates a copy so the original data is not affected
    ml_df = df.copy()

    #Drops columns not used in modeling
    ml_df = df.drop(columns=['transactiondate', 'sqft_bins',
                             'value_bins', 'County'])
    #Creates dummy columns
    ml_df = pd.get_dummies(columns=['bedrooms', 'bathrooms',
                                    'num_of_features', 'garagecarcnt'],
                           data=ml_df)
    #Splits data into train, validate and test datasets
    train, val, test = train_validate(ml_df)
    
    #Scales continuous data#Scaling the data
    train, val, test = scale_zillow(train, val, test, scaler_model = 3,
                                    cont_columns = ['sqft'])

    #Seperate target column from feature columns
    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test(train, val,
                                                                    test, target_col)
    
    #Eastablishes the standard to beat
    baseline = find_baseline(y_train)
    
    """ *** Builds and fits Lasso Lars Model *** """  
    
    lars = LassoLars(alpha=.25)
    lars.fit(X_train, y_train)
    
    #Train data
    ll_preds = pd.DataFrame({'actual':y_train})
    ll_preds['pred_ll'] = lars.predict(X_train)
    
    #Validate data
    ll_val_preds = pd.DataFrame({'actual':y_val})
    ll_val_preds['ll_val_preds'] = lars.predict(X_val)
    
    #Test data
    ll_test_preds = pd.DataFrame({'actual':y_test})
    ll_test_preds['ll_test_preds'] = lars.predict(X_test)
    
    #Finds score on Train and Validate data
    rmse_train = mean_squared_error(ll_preds['actual'],
                                    ll_preds['pred_ll'],
                                    squared=False)
    rmse_val = mean_squared_error(ll_val_preds['actual'],
                                  ll_val_preds['ll_val_preds'],
                                  squared=False)
    rmse_test = mean_squared_error(ll_test_preds['actual'],
                              ll_test_preds['ll_test_preds'],
                              squared=False)
    
    #Adds score to metrics list for comparison
    final_rmse_scores.append({'Model':'Lasso Lars',
                              'RMSE on Train': round(rmse_train,0), 
                              'RMSE on Validate': round(rmse_val,0), 
                              'RMSE on Test': round(rmse_test,0)})
    # Turn scores into a DataFrame
    final_rmse_scores = pd.DataFrame(data = final_rmse_scores)
    print(final_rmse_scores)
    
    #Create visuals to show the results
    fig, ax = plt.subplots(facecolor="gainsboro")

    plt.figure(figsize=(4,4))
    ax.set_title('Lasso Lars results')
    ax.axhspan(0, baseline, facecolor='palegreen', alpha=0.2)
    ax.axhspan(baseline, ymax=450000, facecolor='red', alpha=0.3)

    #x_pos = [0.5, 1, 1.5]
    width = 0.25

    bar1 = ax.bar(0.5, height=final_rmse_scores['RMSE on Train'],width =width, color=('#4e5e33'), label='Train', edgecolor='dimgray')
    bar2 = ax.bar(1, height= final_rmse_scores['RMSE on Validate'], width =width, color=('#8bc34b'), label='Validate', edgecolor='dimgray')
    bar3 = ax.bar(1.5, height=final_rmse_scores['RMSE on Test'], width =width, color=('tomato'), label='Test', edgecolor='dimgray')

    # Need to have baseline input:
    ax.axhline(baseline, label="Baseline", c='tomato', linestyle=':')
    ax.set_xticks([0.5, 1.0, 1.5], ['Training', 'Validation', 'Test']) 
    ax.set_ylim(bottom=200000, top=400000)
    #Zoom into the important area
    #plt.ylim(bottom=200000, top=400000)
    ax.legend(loc='upper right', framealpha=.9, facecolor="whitesmoke", edgecolor='darkolivegreen')