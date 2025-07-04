

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,r2_score


seed = 42


df = pd.read_csv('uber data/uber-raw-data-apr14.csv')
print(df)

print(df.info())
df['Date/Time'] = pd.to_datetime(df['Date/Time'])

df['Hour'] = df['Date/Time'].dt.hour
df['Day'] = df['Date/Time'].dt.day
df['Month'] = df['Date/Time'].dt.month
df['DayOfWeek'] = df['Date/Time'].dt.dayofweek

print(df['Hour'].info())





#EDA

plt.figure(figsize = (10,6))
sns.countplot(x='Hour', data=df)
plt.title('trips per hour')
plt.xlabel('Hour of the day')
plt.ylabel('Number of trips')
plt.show()

plt.figure(figsize = (10,6))
sns.countplot(x = 'DayOfWeek',data = df)
plt.title('trips per day of the week')
plt.xlabel('Day of the week')
plt.ylabel('Number of trips')
plt.show()

#df = pd.get_dummies(df,columns=['Base'],drop_first=True)

#aggregate data by time

trip_data = df.groupby(['Hour','Day','DayOfWeek','Month']).agg({
    'Lat':'mean',
    'Lon':'mean',
    'Date/Time':'count'
}).reset_index()
trip_data.rename(columns={'Date/Time':'Trips'},inplace=True)

# Group by hour and date to get trip counts
trip_data = df.groupby(['Hour', 'Day', 'DayOfWeek', 'Month'])[['Lat', 'Lon']].count().reset_index()
trip_data.rename(columns={'Lat': 'Trips'}, inplace=True)  # Using Lat count as trip count

# Features and target
X = trip_data[['Hour', 'Day', 'DayOfWeek', 'Month']]
y = trip_data['Trips']



X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)

#model building
rfr = RandomForestRegressor(random_state=42)
rfr.fit(X_train,y_train)

#prediction

y_prediction = rfr.predict(X_test)

#evaluating our model

print("mean square error :",mean_squared_error(y_test,y_prediction))
print("r^2 score :",r2_score(y_test,y_prediction))

plt.figure(figsize=(10,6))
plt.scatter(y_test,y_prediction)
plt.xlabel('Actual Trips')
plt.ylabel('Predicted trips')
plt.title("actual trips vs predicted trips")


plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')


plt.show()

importances = rfr.feature_importances_
feat_name = X.columns
plt.figure(figsize=(10,6))
sns.barplot(x = importances,y = feat_name)
plt.title('feature Importance')
plt.xlabel('importance')
plt.ylabel('features')
plt.show()

import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np

import os
import xgboost as xgb
from sklearn.model_selection import KFold
from xgboost import plot_importance,plot_tree
from sklearn.model_selection import train_test_split

from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV,TimeSeriesSplit

def PlotDecomposition(result):
  plt.figure(figsize=(10,6))
  plt.subplot(4,1,1)
  plt.plot(result.observed,label = 'Observed',lw = 1)
  plt.legend(loc='upper left')
  plt.subplot(4,1,2)
  plt.plot(result.trend,label='Trend',lw=1)
  plt.legend(loc='upper left')
  plt.subplot(4, 1, 3)
  plt.plot(result.seasonal, label='Seasonality',lw=1)
  plt.legend(loc='upper left')
  plt.subplot(4, 1, 4)
  plt.plot(result.resid, label='Residuals',lw=1)
  plt.legend(loc='upper left')
  plt.show()

def CalulateError(pred,sales):
  percentual_error = []
  for A_i,B_i in zip(pred,sales):
    percentual_error= abs((A_i-B_i)/B_i)
    percentual_error.append(percentual_error)
  return sum(percentual_error)/len(percentual_error)


import matplotlib.pyplot as plt

def PlotPrediction(plots, title):
    plt.figure(figsize=(18, 8))
    
    for plot in plots:
        plt.plot(plot[0], plot[1], label=plot[2], linestyle=plot[3], color=plot[4], lw=1)
    
    plt.xlabel('Date')
    plt.ylabel('Trips')
    plt.legend(loc='upper left')
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

window_size = 24

def create_lagged_features(data, window_size):
    X, y = [], []
    # Should be range(len(data) - window_size), not len(data, -window_size)
    for i in range(len(data) - window_size):
        X.append(data[i : i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)



# Set path
data_directory = 'C:/Users/ajaym/OneDrive/Desktop/internship/uber trip/uber data'
files = []

# Check for directory
if not os.path.exists(data_directory):
    raise FileNotFoundError(f"Directory not found at {data_directory}")

# Find CSV files
for dirname, _, filenames in os.walk(data_directory):
    for filename in filenames:
        if filename.endswith('.csv'):
            files.append(os.path.join(dirname, filename))

if not files:
    raise FileNotFoundError(f"No CSV files found in {data_directory}")

# Read CSV files
dataFrames = [pd.read_csv(file, encoding='latin-1') for file in files]
uber2014 = pd.concat(dataFrames, ignore_index=True)

# Parse and clean datetime
uber2014['Date/Time'] = pd.to_datetime(uber2014['Date/Time'], format='%m/%d/%Y %H:%M:%S', errors='coerce')
uber2014 = uber2014.dropna(subset=['Date/Time'])

# Sort and reindex
uber2014 = uber2014.sort_values(by='Date/Time')
uber2014 = uber2014.rename(columns={'Date/Time': 'Date'})
uber2014.set_index('Date', inplace=True)

# Preview data
print(uber2014.head())


# Set path to your Uber data
data_directory = 'C:/Users/ajaym/OneDrive/Desktop/internship/uber trip/uber data'
files = []

# Collect CSV files
for dirname, _, filenames in os.walk(data_directory):
    for filename in filenames:
        if filename.endswith('.csv'):
            files.append(os.path.join(dirname, filename))

if not files:
    raise FileNotFoundError(f"No CSV files found in {data_directory}")

# Load and combine CSVs
dataFrames = [pd.read_csv(file, encoding='latin-1') for file in files]
uber2014 = pd.concat(dataFrames, ignore_index=True)

# Parse datetime column
uber2014['Date/Time'] = pd.to_datetime(uber2014['Date/Time'], format='%m/%d/%Y %H:%M:%S', errors='coerce')
uber2014 = uber2014.dropna(subset=['Date/Time'])

uber2014.set_index('Date/Time', inplace=True)

# (Optional) Rename the index
uber2014.index.name = 'Date'

# ✅ Resample hourly
hourly_counts = uber2014['Base'].resample('H').count()

# Reset index for visualization
hourly_counts = hourly_counts.reset_index()
hourly_counts.columns = ['Date', 'Count']

# Show results
print(hourly_counts.head())

print(uber2014.index.min())
print(uber2014.index.max())




import os
import pandas as pd
import matplotlib.pyplot as plt # Ensure matplotlib is imported here as well


# Set path to your Uber data
data_directory = 'C:/Users/ajaym/OneDrive/Desktop/internship/uber trip/uber data'
files = []

# Collect CSV files
for dirname, _, filenames in os.walk(data_directory):
    for filename in filenames:
        if filename.endswith('.csv'):
            files.append(os.path.join(dirname, filename))

if not files:
    raise FileNotFoundError(f"No CSV files found in {data_directory}")

# Load and combine CSVs
dataFrames = [pd.read_csv(file, encoding='latin-1') for file in files]
# Use a temporary name for the combined raw data to avoid overwriting uber2014 prematurely
raw_uber2014 = pd.concat(dataFrames, ignore_index=True)

raw_uber2014['Date/Time'] = pd.to_datetime(raw_uber2014['Date/Time'], format='%m/%d/%Y %H:%M:%S', errors='coerce')
raw_uber2014 = raw_uber2014.dropna(subset=['Date/Time'])

raw_uber2014.set_index('Date/Time', inplace=True)

# (Optional) Rename the index
raw_uber2014.index.name = 'Date'


hourly_counts = raw_uber2014['Base'].resample('H').count()

# Convert the Series to a DataFrame and set the DatetimeIndex
uber2014 = hourly_counts.to_frame(name='Count') # Convert Series to DataFrame
uber2014.index.name = 'Date' # Ensure the index name is 'Date' if you want it


print(uber2014.head())


plt.figure(figsize=(20,8))

plt.plot(uber2014.index, uber2014['Count'], linewidth=1, color='blue') # Plotting Date on x-axis
plt.xticks(rotation = 30,ha = 'right')
plt.xlabel('Date')
plt.ylabel('Number of Trips')
plt.title('Hourly Uber Trip Counts (April-September 2014)')
plt.show()

result = seasonal_decompose(uber2014['Count'],model = 'add', period = 24*1)

PlotDecomposition(result)

cutoff_date = '2014-09-15 00:00:00'

plt.figure(figsize=(20,8))
# Use uber2014.index for plotting the trend line to ensure date axis
plt.plot(uber2014.index, result.trend, linewidth = 1,color = 'red')
plt.axvline(x = pd.Timestamp(cutoff_date),color = 'black',linestyle = '--',linewidth = 1)

plt.xticks(rotation = 30, ha = 'right')
plt.show()

# Split the data using the DatetimeIndex
uber2014_train = uber2014.loc[:cutoff_date]
uber2014_test = uber2014.loc[cutoff_date:]

plt.figure(figsize=[15, 5])
# Now uber2014_train.index and uber2014_test.index are DatetimeIndex objects
plt.plot(uber2014_train.index, uber2014_train['Count'], label='TRAINING SET', linestyle='-', lw=1)
plt.plot(uber2014_test.index, uber2014_test['Count'], label='TESTSET', linestyle='-', lw=1)
plt.title('Train / Test Sets')
plt.xlabel('Date')
plt.ylabel('Count')
plt.legend()
plt.show()

from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

import numpy as np


X_train, y_train = create_lagged_features(uber2014_train['Count'].values, window_size)

test_data = np.concatenate([
    uber2014_train['Count'].values[-window_size:],
    uber2014_test['Count'].values
])

X_test, y_test = create_lagged_features(test_data, window_size)



model = LinearRegression()
model.fit(X_train,y_train)
y_predd = model.predict(X_test)


mse = mean_squared_error(y_test,y_predd)
print(f'Mean Squared Error: {mse:.2f}')

plt.figure(figsize=(15, 5))
plt.plot(y_test, label='Actual', linestyle='-', lw=1)
plt.plot(y_predd, label='Predicted', linestyle='-', lw=1)
plt.title('Actual vs Predicted')
plt.xlabel('Time')
plt.ylabel('Count')
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.show()

seed = 12345

print(uber2014_test['Count'].describe())

#XGBOOST for improving accuracy of the model



tscv = TimeSeriesSplit(n_splits = 5)
xgb_param_grid = {
    'n_estimators':[100,200,300],
    'max_depth':[3,6,9],
    'learning_rate':[0.01,0.1,0.3],
    'subsample':[0.6,0.8,1.0]
}

xgb_model = xgb.XGBRegressor(random_state = seed)

xgb_grid_search = GridSearchCV(estimator = xgb_model,param_grid = xgb_param_grid,
                               cv = tscv,scoring = 'neg_mean_squared_error',verbose = 1,n_jobs = -1)

xgb_grid_search.fit(X_train,y_train)

print("Best XGBoost parameters: ",xgb_grid_search.best_params_)
print("Best XGBoost score: ",xgb_grid_search.best_score_)


xgb_predictions = xgb_grid_search.best_estimator_.predict(X_test)

PlotPrediction([
    (uber2014_test.index,uber2014_test['Count'],'Test','-','blue'),
    (uber2014_test.index,xgb_predictions,'XGBoost Predicted','-','red')
],'Uber 2014 Trips : XGBoost Predictions vs Test')

xgb_mape = mean_absolute_percentage_error(uber2014_test['Count'],xgb_predictions)
print(f'XGBoost MAPE: {xgb_mape:.2f}')



gbr_param_grid = {
    'n_estimators':[100,200,300],
    'learning_rate':[0.01,0.1],
    'max_depth': [3,4,5],
    'min_samples_split':[2,5,10],
    'min_samples_leaf':[1,2,4],
    'max_features':['sqrt','log2']
}

gbr_model = GradientBoostingRegressor(random_state = seed)

gbr_grid_search = GridSearchCV(estimator = gbr_model,param_grid = gbr_param_grid,cv=tscv,n_jobs=-1,verbose=1,scoring='neg_mean_squared_error')

gbr_grid_search.fit(X_train,y_train)

print("best random forest regressor: ",gbr_grid_search.best_params_)
print("best random forest regressor score: ",gbr_grid_search.best_score_)

gbr_predictions = gbr_grid_search.best_estimator_.predict(X_test)


import matplotlib.pyplot as plt

# Plotting actual vs predicted on the same graph
plt.figure(figsize=(12, 6))
plt.plot(uber2014_test.index, uber2014_test['Count'], label='Test', linestyle='-', color='blue')
plt.plot(uber2014_test.index, gbr_predictions, label='Gradient Boosting Predicted', linestyle='-', color='red')

plt.title('Uber2014 trips: Gradient Boosting predictions vs Test')
plt.xlabel('Date')
plt.ylabel('Trip Count')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


xgb_mape = mean_absolute_percentage_error(uber2014_test['Count'], xgb_predictions)
print(f'XGBoost MAPE:\t\t{xgb_mape:.2%}')


#RANDOM FOREST MODEL
rf_param_grid = {
   
  'n_estimators': [100, 200, 300],
  'max_depth': [10, 20, 30],
  'min_samples_split': [2, 5, 10],
  'min_samples_leaf': [1, 2, 4],
  'max_features': [None, 'sqrt', 'log2']
}
rf_model = RandomForestRegressor(random_state=seed)


rf_grid_search = GridSearchCV(estimator=rf_model, param_grid=rf_param_grid, cv=tscv,
n_jobs=-1, scoring='neg_mean_absolute_percentage_error',verbose = 1)
rf_grid_search.fit(X_train, y_train)

rf_predictions = rf_grid_search.best_estimator_.predict(X_test)




plt.figure(figsize=(12, 6))
plt.plot(uber2014_test.index, uber2014_test['Count'], label='Test', linestyle='-', color='blue')
plt.plot(uber2014_test.index, rf_predictions, label='Random Forest Predictions', linestyle='-', color='red')

plt.title('Uber2014 trips:Random forest predictions vs Test')
plt.xlabel('Date')
plt.ylabel('Trip Count')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


rf_mape = mean_absolute_percentage_error(uber2014_test['Count'], rf_predictions)
print(f'Random Forest Mean Percentage Error:\t{rf_mape:.2%}')


#Gradient bossted regression tree model

gbr_param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1],
    'max_depth': [3, 4, 5],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

gbr_model = GradientBoostingRegressor(random_state=seed)

gbr_grid_search = GridSearchCV(estimator=gbr_model, param_grid=gbr_param_grid,
cv=tscv, n_jobs=-1, scoring='neg_mean_absolute_percentage_error',verbose = 1)
gbr_grid_search.fit(X_train, y_train)

print("Best Random Forest parameters:", gbr_grid_search.best_params_)

gbr_predictions = gbr_grid_search.best_estimator_.predict(X_test)




plt.figure(figsize=(12, 6))
plt.plot(uber2014_test.index, uber2014_test['Count'], label='Test', linestyle='-', color='blue')
plt.plot(uber2014_test.index, gbr_predictions, label='GBRT Predictions', linestyle='-', color='red')

plt.title('Uber2014 trips:GBRT Predictions vs Test')
plt.xlabel('Date')
plt.ylabel('Trip Count')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

gbr_mape = mean_absolute_percentage_error(y_test, gbr_predictions)
print(f'GBTR Percentage Error:\t{gbr_mape:.2%}')


import matplotlib.pyplot as plt

# Create a single figure
plt.figure(figsize=(14, 7))

# Plot actual test data
plt.plot(uber2014_test.index, uber2014_test['Count'], label='Test', linestyle='-', color='gray')

# Plot predictions from all models
plt.plot(uber2014_test.index, xgb_predictions, label='XGBoost Predictions', linestyle='--', color='red')
plt.plot(uber2014_test.index, gbr_predictions, label='GBRT Predictions', linestyle='--', color='orange')
plt.plot(uber2014_test.index, rf_predictions, label='Random Forest Predictions', linestyle='--', color='green')

# Customize the plot
plt.title('Uber 2014 Trips: All Models Predictions vs Test')
plt.xlabel('Date')
plt.ylabel('Trip Count')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


print(f'XGBoost MAPE:\t\t\t{xgb_mape:.2%}')
print(f'Random Forest MAPE:\t\t{rf_mape:.2%}')
print(f'GBTR Percentage Error:\t\t{gbr_mape:.2%}')


from sklearn.base import BaseEstimator, RegressorMixin

class WeightedEnsembleRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, models, weights):
        self.models = models
        self.weights = weights

    def fit(self, X, y):
        for model in self.models:
            model.fit(X, y)
        return self

    def predict(self, X):
        predictions = [model.predict(X) for model in self.models]
        return sum(w * p for w, p in zip(self.weights, predictions))


PlotPrediction([
    (uber2014_test.index, uber2014_test['Count'], 'Test', '-', 'gray'),
    (uber2014_test.index, ensemble_predictions, 'Ensemble Predictions', '--', 'purple')
], 'Uber 2014 Trips: Ensemble Predictions vs Test')


ensemble_mape = mean_absolute_percentage_error(uber2014_test['Count'],
ensemble_predictions)
print(f'Ensemble MAPE:\t{ensemble_mape:.2%}')



print(f'XGBoost MAPE:\t\t{xgb_mape:.2%}')

print(f'Random Forest MAPE:\t{rf_mape:.2%}')
print(f'GBTR MAPE:\t\t{gbr_mape:.2%}')
print(f'Ensemble MAPE:\t\t{ensemble_mape:.2%}')



