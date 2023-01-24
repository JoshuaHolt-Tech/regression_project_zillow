# Project Description:

Zillow would like to improve their machine learning model to make predictions on the 2017 data set. They have
tasked you with finding features to improve their model.

# Goals:
	•	Explore features to find the key drivers of property value for single family properties.
	•	Construct a ML regression model that predicts a home's value.
	•	Deliver a report that a non-data scientist can read through and understand what steps were taken, why and what was the outcome?
	•	Develop recommendations to improve predictions.

# Initial Thoughts:
	⁃	My initial hypothesis is that square footage will be a primary driver. 
	⁃	Bedrooms and bathrooms could also be drivers.

# Project Planning (lay out your process through the data science pipeline)
	1.	Aquire data from the Codeup SQL database.
	2.	Prepare data
		⁃ Create calculated columns from existing data:
            ⁃ number of features
            - home value bins
            - square foot bins
	3.	Explore data in search of drivers.
		⁃	Answer the following initial questions:
                - Is there a relationship between a house's square footage and it's value?
                - Does the number of bedrooms affect the value of a house?
	4.	Develop a Model to predict home value
		⁃	Use drivers identified in explore to build predictive models of different types.
		⁃	Evaluate models on train and validate data.
		⁃	Select the best model based on lowest RMSE.
		⁃	Evaluate the best model on test data.
	5.	Draw Conclusions.


# Data Dictionary:


| Feature | Description |
|:--------|:-----------|
|bathroomcnt| Number of bathrooms in home including fractional bathrooms|
|bedroomcnt| Number of bedrooms in home|
|decktypeid| Type of deck (if any) present on parcel|
|calculatedfinishedsquarefeet|Calculated total finished living area of the home|
|fips| Federal Information Processing Standard code - see https://en.wikipedia.org/wiki/FIPS_county_code for more details|
|garagecarcnt|Total number of garages on the lot including an attached garage|
|hashottuborspa| Does the home have a hot tub or spa|
|latitude'	Latitude of the middle of the parcel multiplied by 10e6|
|longitude'	Longitude of the middle of the parcel multiplied by 10e6|
|poolcnt| Number of pools on the lot (if any)|
|taxvaluedollarcnt|	The total tax assessed value of the parcel|


# Steps to Reproduce:
1. Copy this repo.
2. Get credentials from Codeup to query their zillow dataset.
3. Ensure the acquire.py, env.py zillow_data_dictionary.xlsx and final_zillow_functions.py are in the same folder as the final notebook.
3. Run the final notebook.

# Takeaways:

- The number of bedrooms, bathrooms, features and square footage all have a positive linear relationship with home value.
- No sales dates for the end of 2017. This would be a slower part of the season but other things could account for this such as wildfires or other events.
- Using the Pandas' [dot]describe() function helped with initial exploration and on multiple levels of home value compared to a single feature.

# Conclusions:

- OLS Linear and Polynomial performed well on the training and validate data.
- Lasso Lars performed well on the train, validation and test data sets.

# Recommendations

- Zillow makes 6 of their 8 billion dollars annual from property sales.
- 6% commission. Their incentive is to sell homes fast but not necessarily for the best price.

- Create 'Zillow Pages™' of locally available resources to improve your home.
- Targeted advertising to conduct home improvements during the fall and winter.

**"If you are planning to sell, improve your homes now."**

- Target areas with higher property values. They are less affected by high interest rates due to taking loans on assets versus from mortgage companies.
- Aligns the incentives of Zillow and home owners:
    - sell quickly
    - best price

# Next Steps
- Explore a better location based model. Putting latitude and longitude in reduced the RMSE by about 10,000 for all models.
- Create a way to automatically map fips to state and counties. This would make the database more robust.
- Dig in deeper on time series analysis for why no sales were listed in Sept, Oct, Nov or Dec. Establish a good start date for advertising campaign.

