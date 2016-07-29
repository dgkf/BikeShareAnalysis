import os, sys, math, re, copy                        # foundational
import sqlite3                                        # data import
import pandas as pd, numpy as np, datetime            # datahandling & analysis
import dill                                           # import/export

from matplotlib import pyplot as plt, colors          # preliminary plotting

import sklearn as sk                                  # scikit-learn and accessories
from sklearn.linear_model import ElasticNet           #
from sklearn.neighbors import KNeighborsRegressor     #
from sklearn.cross_validation import cross_val_score  #
from sklearn.pipeline import Pipeline                 #
from sklearn.preprocessing import StandardScaler      #

dill.settings['recurse'] = True

# Query to access data from station_usage table in babs_archive.db database
_BABS_QUERY = 'SELECT * FROM station_usage'
_BABS_CSV_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                              os.path.pardir, 'datafiles', 'babs_archives')
_ELEVATION_CSV_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)),
             os.path.pardir,
             'datafiles', 'generated_files', 'sf_topo.csv')

# TOD csv's directory
_TOD_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)),
             os.path.pardir, 'datafiles', 'tod_archives')

# Path to database file
_BABS_DATABASE_PATH = os.path.join(
                        os.path.dirname(os.path.realpath(__file__)),
                        os.path.pardir,
                        'datafiles', 'sqlite_database', 'babs_archives.db')

# Circumference of the earth, used for converting lat-long degrees to kilometers
_EARTH_CIRCUMFERENCE = 40075.16 # kilometer

class ColumnSelectTransformer(sk.base.BaseEstimator, sk.base.TransformerMixin):
    def __init__(self, column_name_list=None):
        self.column_name_list = column_name_list

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        try:
            return X[self.column_name_list]
        except:
            print('Column Select Transformer: Column not found in input X.')
            return X[[column for column in self.column_name_list if column in X.columns.values]]


class featuresFromLatLongTransformer(sk.base.BaseEstimator, sk.base.TransformerMixin):
    race_features = ['Percent White','Percent Black','Percent American & Alaskan Native','Percent Asian','Percent Pacific Islander','Percent Other Race','Percent Two-or-more Races','Percent Hispanic & Latino']

    def __init__(self, model_params_dict=None, features_list=None):
        self.features_from_lat_long = {k: {'model':fitModel(**v), 'unit':v['unit']} for k, v in model_params_dict.items()}
        self.features_list = features_list

    def __getitem__(self, i):
        return self.features_from_lat_long[i]

    def setFeatures(self, features_list=None):
        self.features_list = features_list

    def fit(self, X, y):
        return self

    def transform(self, X):
        return self.transformByLatLong(X['Latitude'], X['Longitude'])

    def transformByLatLong(self, lat, long, model_name=None):
        # load default features list if none is specified (if no default was set, all features are used)
        if not model_name: model_name = self.features_list

        # make sure the lat and long are list in case a single prediction is made
        try: lat[0]
        except: lat = [lat]
        try: long[0]
        except: long = [long]

        # add in all races if any one race is passed
        if not not model_name and any([r in model_name for r in self.race_features]):
            model_name = ([model_name] if type(model_name) is not list else model_name) + [r for r in self.race_features if r not in model_name]

        # make predictions
        predictions_df = pd.DataFrame({k:v['model'].predict(pd.DataFrame({'Latitude': lat, 'Longitude': long})) \
                   for k,v in self.features_from_lat_long.items() if (not model_name or k in model_name)})

        # normalize all races to total to 100% if a race prediction is made
        if not model_name or any([r in model_name for r in self.race_features]):
            predictions_df[self.race_features] = predictions_df[self.race_features].divide(np.sum(predictions_df[self.race_features].values, axis=1), axis=0)

        # return predictions dataframe
        return predictions_df


def cleanTODColumnNames(df):
    df.columns = map(lambda x: re.sub(r"\([0-9]+\)", "", x).strip(), df.columns)
    return df


def fitModel(model, dataset, train_col, **kargs):
    print('Fitting model for location-based "%s" predictions.' % train_col)
    print('  Cross validation scores: %s' % ['%0.4f' % cvs for cvs in cross_val_score(model, dataset, dataset[train_col], cv=5)])
    return model.fit(dataset, dataset[train_col])


def retrainModel():
    ### Load Data ###
    # Load Station Data
    conn = sqlite3.connect(_BABS_DATABASE_PATH)
    station_data = pd.read_sql(_BABS_QUERY, conn)
    station_data['Usage'] = 0.5*(station_data['Start Count'] + station_data['End Count'])

    # Elevation
    elevation_df      = pd.read_csv(_ELEVATION_CSV_PATH, skipinitialspace=True)

    # Population by Age
    tod_pop_by_age_df = pd.read_csv(os.path.join(_TOD_DIR, 'TOD_Population_by_Age.csv'), skipinitialspace=True, thousands=",")
    tod_pop_by_age_df = cleanTODColumnNames(tod_pop_by_age_df).dropna(axis=0)
    tod_pop_by_age_df[['Ages 0 - 17', 'Ages 18 - 24', 'Ages 25 - 39', 'Ages 40 - 59', 'Ages 60 +']] = tod_pop_by_age_df[['Percent Ages 0 - 17', 'Percent Ages 18 - 24', 'Percent Ages 25 - 39', 'Percent Ages 40 - 59', 'Percent Ages 60 +']].mul(tod_pop_by_age_df['Total Population'].values, axis='index') / 100

    # Race
    tod_race_df = pd.read_csv(os.path.join(_TOD_DIR, 'TOD_Race.csv'), skipinitialspace=True, thousands=",")
    tod_race_df = cleanTODColumnNames(tod_race_df).dropna(axis=0)
    tod_race_df[["Percent White", "Percent Black or African American", "Percent American Indian and Alaska Native", "Percent Asian", "Percent Native Hawaiian and Other Pacific Islander", "Percent Some Other Race", "Percent Two or More Races", "Percent Hispanic or Latino"]] = tod_race_df[['White alone', 'Black or African American alone', 'American Indian and Alaska Native alone', 'Asian alone', 'Native Hawaiian and Other Pacific Islander alone', 'Some Other Race alone', 'Two or More Races', 'Hispanic or Latino']].divide(tod_race_df['Total'].values, axis='index')

    # Employment
    tod_employment_df = pd.read_csv(os.path.join(_TOD_DIR, 'TOD_Employment.csv'), skipinitialspace=True, thousands=",")
    tod_employment_df = cleanTODColumnNames(tod_employment_df).dropna(axis=0)

    # Household & Transportation Affordability index
    tod_house_and_trans_df = pd.read_csv(os.path.join(_TOD_DIR, 'TOD_Household_and_Transportation_Affordability_Index.csv'), skipinitialspace=True, thousands=",")
    tod_house_and_trans_df = cleanTODColumnNames(tod_house_and_trans_df).dropna(axis=0)

    # Household Tenure
    tod_tenure_df = pd.read_csv(os.path.join(_TOD_DIR, 'TOD_Tenure.csv'), skipinitialspace=True, thousands=",")
    tod_tenure_df = cleanTODColumnNames(tod_tenure_df).dropna(axis=0)

    # Household Income
    tod_household_income_df = pd.read_csv(os.path.join(_TOD_DIR, 'TOD_Household_Income.csv'), skipinitialspace=True, thousands=",")
    tod_household_income_df = cleanTODColumnNames(tod_household_income_df).dropna(axis=0)

    # Population Density
    tod_density_df = pd.read_csv(os.path.join(_TOD_DIR, 'TOD_Density.csv'), skipinitialspace=True, thousands=",")
    tod_density_df = cleanTODColumnNames(tod_density_df).dropna(axis=0)

    # Vehicle Ownership
    tod_vehicle_ownership_df = pd.read_csv(os.path.join(_TOD_DIR, 'TOD_Vehicle_Ownership.csv'), skipinitialspace=True, thousands=",")
    tod_vehicle_ownership_df = cleanTODColumnNames(tod_vehicle_ownership_df).dropna(axis=0)

    ### Build Models For Lat Long Interpolation Transform of Features ###
    # base location based association model
    base_KNN = Pipeline([('column_select', ColumnSelectTransformer(column_name_list=['Latitude', 'Longitude'])),
                         ('knn_reg',       KNeighborsRegressor(radius=0.1)) ])

    # Model info for fitting interpolation models for transit features
    features_from_lat_long_model_params = \
         {'Elevation': {'model':copy.deepcopy(base_KNN), 'dataset': elevation_df, 'train_col': 'Elevation', 'unit': 'km'},
          'Percent Ages 0 to 17': {'model':copy.deepcopy(base_KNN), 'dataset': tod_pop_by_age_df, 'train_col': 'Percent Ages 0 - 17', 'unit': '%'},
          'Percent Ages 18 to 24': {'model':copy.deepcopy(base_KNN), 'dataset': tod_pop_by_age_df, 'train_col': 'Percent Ages 18 - 24', 'unit': '%'},
          'Percent Ages 25 to 39': {'model':copy.deepcopy(base_KNN), 'dataset': tod_pop_by_age_df, 'train_col': 'Percent Ages 25 - 39', 'unit': '%'},
          'Percent Ages 40 to 59': {'model':copy.deepcopy(base_KNN), 'dataset': tod_pop_by_age_df, 'train_col': 'Percent Ages 40 - 59', 'unit': '%'},
          'Percent Ages 60+': {'model':copy.deepcopy(base_KNN), 'dataset': tod_pop_by_age_df, 'train_col': 'Percent Ages 60 +', 'unit': '%'},
          'Employment': {'model':copy.deepcopy(base_KNN), 'dataset': tod_employment_df, 'train_col': '2009 Jobs per Acre', 'unit': 'Jobs/Acre'},
          'Percent of Income for Housing': {'model':copy.deepcopy(base_KNN), 'dataset': tod_house_and_trans_df, 'train_col': 'Regional Typical Household Housing Costs % Income', 'unit': '%'},
          'Percent of Income for Transportation': {'model':copy.deepcopy(base_KNN), 'dataset': tod_house_and_trans_df, 'train_col': 'Regional Typical Household Transportation Costs % Income', 'unit': '%'},
          'Population Density': {'model':copy.deepcopy(base_KNN), 'dataset': tod_density_df, 'train_col': 'Population Density  (Population per Acre)', 'unit': 'Population/Acre'},
          'Vehicle Ownership': {'model':copy.deepcopy(base_KNN), 'dataset': tod_vehicle_ownership_df, 'train_col': 'Average number of vehicles available per occupied housing unit', 'unit': 'Vehicles/Household'},
          'Percent Owned Housing': {'model':copy.deepcopy(base_KNN), 'dataset': tod_tenure_df, 'train_col': 'Percent owner occupied', 'unit': '%'},
          'Percent Rented Housing': {'model':copy.deepcopy(base_KNN), 'dataset': tod_tenure_df, 'train_col': 'Percent renter occupied', 'unit': '%'},
          'Median Household Income': {'model':copy.deepcopy(base_KNN), 'dataset': tod_household_income_df, 'train_col': 'Median household income', 'unit': '$'},
          'Percent Households with Income <$25k': {'model':copy.deepcopy(base_KNN), 'dataset': tod_household_income_df, 'train_col': 'Percent household income less than $25,000', 'unit': '%'},
          'Percent Households with Income between $25k and $50k': {'model':copy.deepcopy(base_KNN), 'dataset': tod_household_income_df, 'train_col': 'Percent household income between $25,000 and $49,999', 'unit': '%'},
          'Percent Households with Income between $50k and $75k': {'model':copy.deepcopy(base_KNN), 'dataset': tod_household_income_df, 'train_col': 'Percent household income between $50,000 and $74,999', 'unit': '%'},
          'Percent Households with Income >$75k': {'model':copy.deepcopy(base_KNN), 'dataset': tod_household_income_df, 'train_col': 'Percent household income of $75,000 or more', 'unit': '%'},
          'Percent White': {'model':copy.deepcopy(base_KNN), 'dataset': tod_race_df, 'train_col': 'Percent White', 'unit': '%'},
          'Percent Black': {'model':copy.deepcopy(base_KNN), 'dataset': tod_race_df, 'train_col': 'Percent Black or African American', 'unit': '%'},
          'Percent American & Alaskan Native': {'model':copy.deepcopy(base_KNN), 'dataset': tod_race_df, 'train_col': 'Percent American Indian and Alaska Native', 'unit': '%'},
          'Percent Asian': {'model':copy.deepcopy(base_KNN), 'dataset': tod_race_df, 'train_col': 'Percent Asian', 'unit': '%'},
          'Percent Pacific Islander': {'model':copy.deepcopy(base_KNN), 'dataset': tod_race_df, 'train_col': 'Percent Native Hawaiian and Other Pacific Islander', 'unit': '%'},
          'Percent Other Race': {'model':copy.deepcopy(base_KNN), 'dataset': tod_race_df, 'train_col': 'Percent Some Other Race', 'unit': '%'},
          'Percent Two-or-more Races': {'model':copy.deepcopy(base_KNN), 'dataset': tod_race_df, 'train_col': 'Percent Two or More Races', 'unit': '%'},
          'Percent Hispanic & Latino': {'model':copy.deepcopy(base_KNN), 'dataset': tod_race_df, 'train_col': 'Percent Hispanic or Latino', 'unit': '%'}}

    # Fit transformer for getting features from latitude, longitude
    features_from_lat_long_tf = featuresFromLatLongTransformer(features_from_lat_long_model_params)

    ### Build Model for Predicting Bike Share Usage ###
    # Linear Model for Predicting Location Success
    babs_usage_model = Pipeline([('column_select',          ColumnSelectTransformer(column_name_list=['Latitude', 'Longitude'])),
                                   ('features_from_lat_long', features_from_lat_long_tf),
                                   ('feature_scaling',        StandardScaler()),
                                   ('lin_reg',                ElasticNet()) ]) #LinearSVR(C=0.1, loss='squared_epsilon_insensitive')) ])

    # Fit model against usage
    babs_usage_model.fit(station_data, station_data['Usage'])
    print("\n\nFitting Parameters:\n%s" %  '\n'.join([str(k)+': '+str(v) for (k,v) in \
          zip(babs_usage_model.named_steps['features_from_lat_long'].features_from_lat_long.keys(), babs_usage_model.named_steps['lin_reg'].coef_)]))

    print("\n\nWriting model to file ... ")
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.path.pardir, 'models', 'babs_usage_model.dill'), 'w') as f:
        dill.dump(babs_usage_model, f)


def remapStationLocations():
    # Load Station Data
    conn = sqlite3.connect(_BABS_DATABASE_PATH)
    station_data = pd.read_sql(_BABS_QUERY, conn)
    station_data['Usage'] = 0.5*(station_data['Start Count'] + station_data['End Count'])

    station_data[['Latitude', 'Longitude', 'Landmark']].to_json(os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        os.path.pardir, 'static', 'json', 'station_locations.json'),
        orient="records")


def remapStationUsage():
    _LATEST_DATA = datetime.datetime(2015,8,31)
    trip_dir = os.path.join(_BABS_CSV_PATH, 'trip_data')
    read_csv_params = {"usecols": ['Start Date', 'Start Terminal', 'End Terminal', 'Subscriber Type']}

    # Trip Data
    trip_data = pd.concat([pd.read_csv(fp, **read_csv_params) for fp in [os.path.join(trip_dir,f) for f in os.listdir(trip_dir)]])

    # Load Station Data
    conn = sqlite3.connect(_BABS_DATABASE_PATH)
    station_data = pd.read_sql(_BABS_QUERY, conn, parse_dates=['Installation'])
    station_data['LatLng'] = [{'Lat': station_data['Latitude'][i], 'Lng': station_data['Longitude'][i]} for i in xrange(len(station_data))]

    # Aggregate trips and reindex with latitude and longitude of each terminal instead of station ID
    trip_data = trip_data.groupby(['Start Terminal', 'End Terminal']).agg({'Start Date': 'count'})
    trip_data.columns = ['TripCount']

    # Flatten dataset so that it can easily be indexed in D3
    trip_data.reset_index(inplace=True)

    # Merge with station information about installation
    trip_data = pd.merge(trip_data, station_data[['Station ID', 'Installation', 'LatLng', 'Landmark']], left_on='Start Terminal', right_on='Station ID')
    trip_data.rename(columns={'Installation': 'StartInstallation', 'LatLng': 'StartLatLng'}, inplace=True)
    trip_data = trip_data[trip_data['Landmark'] == "San Francisco"]
    trip_data.drop(['Station ID', 'Landmark'], 1, inplace=True)
    trip_data = pd.merge(trip_data, station_data[['Station ID', 'Installation', 'LatLng', 'Landmark']], left_on='End Terminal', right_on='Station ID')
    trip_data.rename(columns={'Installation': 'EndInstallation', 'LatLng': 'EndLatLng'}, inplace=True)
    trip_data = trip_data[trip_data['Landmark'] == "San Francisco"]
    trip_data.drop(['Station ID', 'Landmark'], 1, inplace=True)

    # Calculate average daily trips
    trip_data['AvgDailyTrips'] = trip_data['TripCount'] / np.minimum((_LATEST_DATA - trip_data['StartInstallation']) / np.timedelta64(1, 'D'), (_LATEST_DATA - trip_data['EndInstallation']) / np.timedelta64(1, 'D'))

    # Write out aggregated data to a file
    trip_data[['StartLatLng', 'EndLatLng', 'AvgDailyTrips']] \
        .to_json(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                              os.path.pardir, 'static', 'json', 'trip_agg.json'),
                 orient='records')


def main(argv):
    remapStationUsage()
    sys.exit()

    # Load Station Data
    conn = sqlite3.connect(_BABS_DATABASE_PATH)
    station_data = pd.read_sql(_BABS_QUERY, conn)
    station_data['Usage'] = 0.5*(station_data['Start Count'] + station_data['End Count'])

    print("\n\nLoading model to file ... ")
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.path.pardir, 'models', 'babs_usage_model.dill'), 'r') as f:
            babs_usage_model = dill.load(f)

    # Interpolate all features from feature models
    lats, longs = np.meshgrid(np.linspace(37.7,37.82,50), np.linspace(-122.53,-122.35,50))
    transformed_features = babs_usage_model.named_steps['features_from_lat_long'].transform(pd.DataFrame({'Latitude': lats.reshape(1,-1).squeeze(), 'Longitude': longs.reshape(1,-1).squeeze()}))

    prediction_features = pd.DataFrame({'Latitude': lats.reshape(1,-1).squeeze(), 'Longitude': longs.reshape(1,-1).squeeze()})
    usage_predictions = babs_usage_model.predict(prediction_features)
    usage_predictions[transformed_features['Elevation']<0] = np.nan
    usage_predictions = np.lib.scimath.logn(100, usage_predictions - np.nanmin(usage_predictions) + 1)
    usage_predictions[np.where(np.isnan(usage_predictions))] = 0

    plt.contourf(longs, lats, usage_predictions.reshape(50,50),
                 norm=colors.Normalize(np.mean(usage_predictions)-(2*np.std(usage_predictions)), np.mean(usage_predictions)+(2*np.std(usage_predictions)), clip=True),
                 levels=np.linspace(0.,max(usage_predictions),300))
    plt.contour(longs, lats, (transformed_features['Elevation']).reshape(50,50), linewidth=0.2, colors='white')
    plt.scatter(station_data[station_data['Landmark']=='San Francisco']['Longitude'], station_data[station_data['Landmark']=='San Francisco']['Latitude'], s=2, )
    plt.show()

if __name__ == "__main__":
    main(sys.argv[1:])
