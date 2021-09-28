
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import env


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

sql = """
SELECT prop.*,
       pred.logerror,
       pred.transactiondate,
       air.airconditioningdesc,
       arch.architecturalstyledesc,
       build.buildingclassdesc,
       heat.heatingorsystemdesc,
       landuse.propertylandusedesc,
       story.storydesc,
       construct.typeconstructiondesc
FROM   properties_2017 prop
       INNER JOIN (SELECT parcelid,
                   Max(transactiondate) transactiondate
                   FROM   predictions_2017
                   GROUP  BY parcelid) pred
               USING (parcelid)
               			JOIN predictions_2017 as pred USING (parcelid, transactiondate)
       LEFT JOIN airconditioningtype air USING (airconditioningtypeid)
       LEFT JOIN architecturalstyletype arch USING (architecturalstyletypeid)
       LEFT JOIN buildingclasstype build USING (buildingclasstypeid)
       LEFT JOIN heatingorsystemtype heat USING (heatingorsystemtypeid)
       LEFT JOIN propertylandusetype landuse USING (propertylandusetypeid)
       LEFT JOIN storytype story USING (storytypeid)
       LEFT JOIN typeconstructiontype construct USING (typeconstructiontypeid)
WHERE  prop.latitude IS NOT NULL
       AND prop.longitude IS NOT NULL
"""

def get_db_url(database):
    from env import host, user, password
    url = f'mysql+pymysql://{user}:{password}@{host}/{database}'
    return url

# acquire zillow data using the query
def get_zillow(sql):
    url = get_db_url('zillow')
    zillow_df = pd.read_sql(sql, url, index_col='id')
    return zillow_df


def handle_missing_values(df, prop_required_column = .5, prop_required_row = .70):
	#function that will drop rows or columns based on the percent of values that are missing:\
	#handle_missing_values(df, prop_required_column, prop_required_row
    threshold = int(round(prop_required_column*len(df.index),0))
    df = df.dropna(axis=1, thresh=threshold)
    threshold = int(round(prop_required_row*len(df.columns),0))
    df.dropna(axis=0, thresh=threshold, inplace=True)
    return df

def remove_columns(df, cols_to_remove):
	#remove columns not needed
    df = df.drop(columns=cols_to_remove)
    return df

def final_df(df):
    # drop unnecessary columns
    dropcols = ['taxamount', 'fips', 
                'yearbuilt', 'county',
                'lotsizesquarefeet', 'regionidcity', 'regionidzip', 
                'structuretaxvaluedollarcnt', 'landtaxvaluedollarcnt', 
                         ]

    df = remove_columns(df, dropcols)
    return df

def wrangle_zillow():
    if os.path.isfile('zillow_cached.csv') == False:
        df = get_zillow(sql)
        df.to_csv('zillow_cached.csv',index = False)
    else:
        df = pd.read_csv('zillow_cached.csv')

    # Restrict df to only properties that meet single use criteria
    single_use = [261, 262, 263, 264, 266, 268, 273, 276, 279]
    df = df[df.propertylandusetypeid.isin(single_use)]

    # Restrict df to only those properties with at least 1 bath & bed and 350 sqft area
    df = df[(df.bedroomcnt > 0) & (df.bathroomcnt > 0) & ((df.unitcnt<=1)|df.unitcnt.isnull())\
            & (df.calculatedfinishedsquarefeet>350)]

    # Handle missing values i.e. drop columns and rows based on a threshold
    df = handle_missing_values(df)

    # Add column for counties
    df['county'] = df['fips'].apply(
        lambda x: 'Los Angeles' if x == 6037\
        else 'Orange' if x == 6059\
        else 'Ventura')

    # drop unnecessary columns
    dropcols = ['parcelid',
         'calculatedbathnbr',
         'finishedsquarefeet12',
         'fullbathcnt',
         'heatingorsystemtypeid',
         'propertycountylandusecode',
         'propertylandusetypeid',
         'propertyzoningdesc',
         'censustractandblock',
         'propertylandusedesc', 'transactiondate', 'heatingorsystemdesc']

    df = remove_columns(df, dropcols)

    # replace nulls in unitcnt with 1
    df.unitcnt.fillna(1, inplace = True)

    # assume that since this is Southern CA, null means 'None' for heating system
    # df.heatingorsystemdesc.fillna('None', inplace = True)

    # replace nulls with median values for select columns
    df.lotsizesquarefeet.fillna(7313, inplace = True)
    df.buildingqualitytypeid.fillna(6.0, inplace = True)

    # Columns to look for outliers
    df = df[df.taxvaluedollarcnt < 5_000_000]
    df = df[df.calculatedfinishedsquarefeet < 8000]

    # Just to be sure we caught all nulls, drop them here
    df = df.dropna()

    return df

def min_max_scaler(train, validate, test):
    '''
    Uses the train & test datasets created by the split_my_data function
    Returns 3 items: mm_scaler, train_scaled_mm, test_scaled_mm
    This is a linear transformation. Values will lie between 0 and 1
    '''
    num_vars = list(train.select_dtypes('number').columns)
    scaler = MinMaxScaler()
    train[num_vars] = scaler.fit_transform(train[num_vars])
    valid[num_vars] = scaler.transform(valid[num_vars])
    test[num_vars] = scaler.transform(test[num_vars])
    return scaler, train, validate, test

def train_validate_test_split(df):
    train_and_validate, test = train_test_split(df, train_size=0.8, random_state=123)
    train, validate = train_test_split(train_and_validate, train_size=0.75, random_state=123)
    return train, validate, test

def create_features(df):
    df['age'] = 2017 - df.yearbuilt
    df['age_bin'] = pd.cut(df.age, 
                           bins = [0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140],
                           labels = [0, .066, .133, .20, .266, .333, .40, .466, .533, 
                                     .60, .666, .733, .8, .866, .933])

    # create taxrate variable
    df['taxrate'] = df.taxamount/df.taxvaluedollarcnt*100

    # create acres variable
    df['acres'] = df.lotsizesquarefeet/43560

    # bin acres
    df['acres_bin'] = pd.cut(df.acres, bins = [0, .10, .15, .25, .5, 1, 5, 10, 20, 50, 200], 
                       labels = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9])

    # square feet bin
    df['sqft_bin'] = pd.cut(df.calculatedfinishedsquarefeet, 
                            bins = [0, 800, 1000, 1250, 1500, 2000, 2500, 3000, 4000, 7000, 12000],
                            labels = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9]
                       )

    # dollar per square foot-structure
    df['structure_dollar_per_sqft'] = df.structuretaxvaluedollarcnt/df.calculatedfinishedsquarefeet


    df['structure_dollar_sqft_bin'] = pd.cut(df.structure_dollar_per_sqft, 
                                             bins = [0, 25, 50, 75, 100, 150, 200, 300, 500, 1000, 1500],
                                             labels = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9]
                                            )


    # dollar per square foot-land
    df['land_dollar_per_sqft'] = df.landtaxvaluedollarcnt/df.lotsizesquarefeet

    df['lot_dollar_sqft_bin'] = pd.cut(df.land_dollar_per_sqft, bins = [0, 1, 5, 20, 50, 100, 250, 500, 1000, 1500, 2000],
                                       labels = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9]
                                      )


    # update datatypes of binned values to be float
    df = df.astype({'sqft_bin': 'float64', 'acres_bin': 'float64', 'age_bin': 'float64',
                    'structure_dollar_sqft_bin': 'float64', 'lot_dollar_sqft_bin': 'float64'})


    # ratio of bathrooms to bedrooms
    df['bath_bed_ratio'] = df.bathroomcnt/df.bedroomcnt

    # 12447 is the ID for city of LA. 
    # I confirmed through sampling and plotting, as well as looking up a few addresses.
    df['cola'] = df['regionidcity'].apply(lambda x: 1 if x == 12447.0 else 0)

    return df

def split(df, target_var):
    '''
    This function takes in the dataframe and target variable name as arguments and then
    splits the dataframe into train (56%), validate (24%), & test (20%)
    It will return a list containing the following dataframes: train (for exploration), 
    X_train, X_validate, X_test, y_train, y_validate, y_test
    '''
    # split df into train_validate (80%) and test (20%)
    train_validate, test = train_test_split(df, test_size=.20, random_state=13)
    # split train_validate into train(70% of 80% = 56%) and validate (30% of 80% = 24%)
    train, validate = train_test_split(train_validate, test_size=.3, random_state=13)

    # create X_train by dropping the target variable 
    X_train = train.drop(columns=[target_var])
    # create y_train by keeping only the target variable.
    y_train = train[[target_var]]

    # create X_validate by dropping the target variable 
    X_validate = validate.drop(columns=[target_var])
    # create y_validate by keeping only the target variable.
    y_validate = validate[[target_var]]

    # create X_test by dropping the target variable 
    X_test = test.drop(columns=[target_var])
    # create y_test by keeping only the target variable.
    y_test = test[[target_var]]

    partitions = [train, X_train, X_validate, X_test, y_train, y_validate, y_test]
    return partitions

def scaled_data(X_train, X_validate, X_test, y_train, y_validate, y_test):

    # Make the scaler
    scaler = MinMaxScaler()

    # Fit the scaler
    scaler.fit(X_train)

    # Use the scaler
    X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns)
    X_validate_scaled = pd.DataFrame(scaler.transform(X_validate), columns=X_validate.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    
    # Make y_values separate dataframes
    y_train = pd.DataFrame(y_train)
    y_validate = pd.DataFrame(y_validate)
    y_test = pd.DataFrame(y_test)
    #Unscaled data for later
    X_unscaled= pd.DataFrame(scaler.inverse_transform(X_test), columns=X_test.columns)
    return X_train_scaled, X_validate_scaled, X_test_scaled, y_train, y_validate, y_test, X_unscaled


def correlation_exploration(train, x_string, y_string):
    '''
    This function takes in a df, a string for an x-axis variable in the df, 
    and a string for a y-axis variable in the df and displays a scatter plot, the r-
    squared value, and the p-value. It explores the correlation between input the x 
    and y variables.
    '''
    r, p = stats.pearsonr(train[x_string], train[y_string])
    df.plot.scatter(x_string, y_string)
    plt.title(f"{x_string}'s Relationship with {y_string}")
    print(f'The p-value is: {p}. There is {round(p,3)}% chance that we see these results by chance.')
    print(f'r = {round(r, 2)}')
    plt.show()