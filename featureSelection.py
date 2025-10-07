import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from datetime import datetime
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanAbsoluteError
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.layers import LSTM, LSTMCell, RNN, Dense
from keras.layers import Conv1D
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
import shap


# Load the data
df = pd.read_csv('patient_data.csv')

# Replace space strings with np.nan
df = df.replace('                                  ', np.nan)
df = df.replace('       ', np.nan)

##//////////////////////// FEATURE ENGINEERING /////////////////////////////////

print(df.head(10))

print(df.isnull().sum())  
print("DIVIDE")
print(df.nunique()) 
datatypes = df.dtypes
datatypes.transpose()
statistics = df.describe(include='all')
# Printing the statistics
print(statistics)
# Exporting to a CSV file
statistics.to_csv('feature_statistics.csv')
datatypes.to_csv('dataTypes.csv')
# Convert the 'Time' column to datetime objects
try:
    df['Time'] = pd.to_datetime(df['Time'], format='%Y-%m-%d %H:%M:%S')
except ValueError:
    try:
        df['Time'] = pd.to_datetime(df['Time'], format='%m/%d/%Y %H:%M')
    except ValueError:
        df['Time'] = pd.to_datetime(df['Time'], format='%H:%M')

# Now you can use the timestamp method
df['timestamp_s'] = df['Time'].map(datetime.timestamp)

# Calculate day_sin and day_cos
day = 24 * 60 * 60
df['day_sin'] = np.sin(df['timestamp_s'] * (2 * np.pi / day))
df['day_cos'] = np.cos(df['timestamp_s'] * (2 * np.pi / day))

## Age of diagonsis

df['Age_of_diagnosis'] = df["GA (w)"] + df["DOL"]/7

## Ventilation Effiency 

df["Ventilation_effiency"] = df["Vte"] /(df["PIP"] - df["DOL"])

## Respiration Health
df["compliance"] = pd.to_numeric(df["compliance"], errors='coerce')
df["resistance"] = pd.to_numeric(df["resistance"], errors='coerce')

df["Respiration_index"] = df["compliance"] / (df["resistance"])

## Oxgygen to CO2 ratio

df["Oxgen_etCO2_ratio"] = df["FiO2"] / (df["etCO2"])

## Is low birth weight

df["is_Low_birth_Rate"] = (df["BW (g)"] < 2500).astype(int)


## Ventilation Duration


df["Ventilation_period"] = df.groupby('Study No').cumcount() * 60

##Average volume per breath

df["Vte"] = pd.to_numeric(df["Vte"], errors='coerce')
df["rate"] = pd.to_numeric(df["rate"], errors='coerce')
df["trigger rate"] = pd.to_numeric(df["rate"], errors='coerce')
df['Vte_rate_ratio'] = df['Vte'] / df['rate']


## create a binary feature that indicates
## whether a patient with a ventilation event also has BPD
df['Ventilation_BPD'] = ((df['Ventilation mode'].notna()) & (df['Diagnosis'].str.contains('BPD'))).astype(int)

# Assuming df is your DataFrame
df['risk_factor'] = 0

# Add 10 if BPD is present
df.loc[df['BPD'] == 'present', 'risk_factor'] += 10

# Add 5 if GA (w) is less than 37
df.loc[df['BW (g)'] < 1000, 'risk_factor'] += 10
df.loc[(df['BW (g)'] >= 1000) & (df['BW (g)'] < 1500), 'risk_factor'] += 7
df.loc[(df['BW (g)'] >= 1500) & (df['BW (g)'] < 2000), 'risk_factor'] += 5
df.loc[df['BW (g)'] >= 2500, 'risk_factor'] += 0

# Add 5 if BW (g) is less than 2500
df.loc[df['GA (w)'] < 37, 'risk_factor'] += 5

# Add 1 if Sex is male
df.loc[df['Sex'] == '1', 'risk_factor'] += 1


# Sample 50 rows and create a scatter plot
df.sample(50).plot.scatter('day_sin', 'day_cos').set_aspect('equal')

##//////////////////////// DISPLAYING DATA //////////////////////////////////////////////

cat_cols = df[["Study No", "Diagnosis", "Ventilation mode"]]

num_cols = df.drop(columns=["Study No", "Diagnosis", "Ventilation mode", "Hospital No"])
print(num_cols.head())

##for feature in num_cols:
##    plt.figure(figsize=(10, 6))
##    plt.plot(df.index, df[feature])
##    plt.title(f"Graph of {feature} over index (frequency)")
##    plt.xlabel('Index (Frequency)')
##    plt.ylabel(feature)
##    plt.show()
    
##for col in cat_cols:
##    plt.figure(figsize=[15,7])
##    sns.countplot(df,x=df[col]).set(title= col+' Value Distribution')
##    plt.show()

##for col in num_cols:
##    plt.figure(figsize=[15,7])
##    sns.displot(df[col],kde=True).set(title= col+' Histogram')
##    plt.axvline(df[col].mean(),color='r', label='Mean')
##    plt.axvline(df[col].median(),color='y', linestyle='--',label='Median')
##    plt.legend()
##    plt.show()
    
## /////////////// Handling Missing Values//////////////////////////////////////////


for col in df.columns:
    if col in ['BPD', 'DOL', 'PIP', 'PEEP', 'MAP' ,'Vte', 'Itime','rate',
               'trigger rate', 'compliance', 'resistance', 'FiO2', 'etCO2']:
        # Group by study number (patient ID) and replace missing values with the median
        df[col] = df.groupby('Study No')[col].transform(lambda x: x.fillna(x.median()))

# Save the updated dataset (replace 'imputed_data.csv' with your desired output file path)

# Create an instance of the OneHotEncoder
encoder = OneHotEncoder(handle_unknown='ignore')

# Fit the encoder and transform the data
encoder_df = pd.DataFrame(encoder.fit_transform(df[['Ventilation mode', 'Diagnosis']]).toarray())

# Get the feature names from the encoder
feature_names = encoder.get_feature_names_out(input_features=['Ventilation mode', 'Diagnosis'])

# Convert the array of feature names to a list
feature_names_list = feature_names.tolist()

# Assign the new column names to the dataframe
encoder_df.columns = feature_names_list


# View the updated DataFrame
df = df.join(encoder_df)
print(df)

# Replace infinities with NaN
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Drop rows with missing values
df.dropna(inplace=True)

# Drop unnecessary columns
df = df.drop(columns=["Hospital No", "BW z-score", "Time", 'Ventilation mode', 'Diagnosis', 'Study No'], axis=1)

### Save the processed data to a csv file
##df.to_csv('patient_data2.csv', index=False)



print(df.isnull().sum())



#///////////////SPLIT AND SCALE DATA //////////////////////////////////////////////


def add_feature_noising(df, noise_level=0.02):
    noisy_df = df.copy()
    # Apply noising only to numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        # Calculate noise scale based on the column's standard deviation
        scale = df[col].std() * noise_level
        # Generate noise
        noise = np.random.normal(loc=0, scale=scale, size=df[col].shape)
        # Add noise to the column
        noisy_df[col] += noise
    return noisy_df

# Apply feature noising
noisy_df = add_feature_noising(df, noise_level=0.02)

# Optional: Create a combined dataset of original and noised data
df = pd.concat([df, noisy_df], ignore_index=True)

# Save the augmented dataset
df.to_csv('augmented_patient_data.csv', index=False)

print(f"Original size: {len(df)}, Augmented size: {len(df)}")


##///////////////////// SPLIT AND SCALE DATA ////////////////////////////////////////////////

# Split 70:20:10


n = len(df)
train_df = df[0:int(n*0.7)]
val_df = df[int(n*0.7):int(n*0.9)]
test_df = df[int(n*0.9):]

##scale 
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(train_df)
train_df.loc[:, train_df.columns] = scaler.transform(train_df[train_df.columns])
val_df.loc[:, val_df.columns] = scaler.transform(val_df[val_df.columns])
test_df.loc[:, test_df.columns] = scaler.transform(test_df[test_df.columns])


train_df.to_csv('train2.csv')
val_df.to_csv('val2.csv')
test_df.to_csv('test2.csv')

##/////////////////////////FEATURE SELECTION ///////////////////////////////////////////////



from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

# Assuming 'label' is the column you want to predict
X_train = train_df.drop('etCO2', axis=1)
y_train = train_df['etCO2']




##  UNIVARIATE SELECTION 
# Apply SelectKBest class to extract top 10 best features
bestfeatures = SelectKBest(score_func=f_regression, k=48)
fit = bestfeatures.fit(X_train, y_train)

dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X_train.columns)
 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  
print(featureScores.nlargest(10,'Score'))  
top_features_select = featureScores.sort_values(by='Score', ascending=False).reset_index(drop=True).head(10)


# Plot the top 5 features
plt.figure(figsize=(10, 6))
plt.barh(top_features_select['Specs'], top_features_select['Score'])
plt.xlabel('Score')
plt.ylabel('Features')
plt.title('Top 10 Features based on SelectKBest Scores')
plt.gca().invert_yaxis()  # Invert y-axis to have the highest score at the top
plt.show()

#  Feature Importance 
model = RandomForestRegressor()
model.fit(X_train, y_train)
print(model.feature_importances_)

# Get feature importances
importances = model.feature_importances_

# Convert feature importances into a pandas DataFrame
importances_df = pd.DataFrame({'feature': X_train.columns, 'importance': importances})

# Sort DataFrame by importance
importances_df = importances_df.sort_values('importance', ascending=False)

# Specify the number of features to show
n_features = 6  # Change this to the number of features you want to show

# Select the top n features
top_features_df = importances_df.head(n_features)

# Plot feature importances for the top n features
plt.figure(figsize=(10, 6))
plt.barh(top_features_df['feature'], top_features_df['importance'])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Top {} Feature Importance'.format(n_features))
plt.gca().invert_yaxis()  # Invert y-axis to have features with highest importance at the top
plt.show()


#/////////////////////SPECIFYING  FEATURE /////////////////////

# Top Features for Feature Importance
# Get the top n features

top_features = top_features_df['feature'].tolist()
# If 'etCO2' is one of the features you want to include, add it to the list
if 'etCO2' not in top_features:
   top_features.append('etCO2')

## Select these features from your datasets
train_df = train_df[top_features]
val_df = val_df[top_features]
test_df = test_df[top_features]

# Save the new datasets
train_df.to_csv('train2.csv')
val_df.to_csv('val2.csv')
test_df.to_csv('test2.csv')


# Top features from the selectKBest features

### Get the top 5 features
##top_features_select = featureScores.nlargest(10,'Score')
####Get the names of the top features
##top_features_names = top_features_select['Specs'].tolist()
##
### Print the list of top features names
##print(top_features_names)
##
##if 'etCO2' not in top_features_select:
##    top_features_names.append('etCO2')
##
### Select these features from your datasets
##train_df = train_df[top_features_names]
##val_df = val_df[top_features_names]
##test_df = test_df[top_features_names]
##
##
### Save the new datasets
##train_df.to_csv('train2.csv')
##val_df.to_csv('val2.csv')
##test_df.to_csv('test2.csv')



##//////////////////////////////////////////////////////////////////////////

class DataWindow():

    def __init__(self, input_width, label_width, shift,train_df=train_df, val_df=val_df, test_df=test_df,label_columns=None):
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in enumerate(label_columns)}
            self.column_indices = {name: i for i, name in enumerate(train_df.columns)}
            self.input_width = input_width
            self.label_width = label_width
            self.shift = shift
            self.total_window_size = input_width + shift
            self.input_slice = slice(0, input_width)
            self.input_indices = np.arange(self.total_window_size)[self.input_slice]
            self.label_start = self.total_window_size - self.label_width
            self.labels_slice = slice(self.label_start, None)
            self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def split_to_inputs_labels(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack([labels[:,:,self.column_indices[name]] for name in self.label_columns], axis=-1)
            inputs.set_shape([None, self.input_width, None])
            labels.set_shape([None, self.label_width, None])
            return inputs, labels

    def plot(self, model=None, plot_col='etCO2', max_subplots=1):
        inputs, labels = self.sample_batch
        plt.figure(figsize=(12, 8))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))
        for n in range(max_n):
            plt.subplot(1, 1, n+1)
            plt.ylabel(f'{plot_col} [scaled]')
            plt.xlabel('Time (Hours)')
##            plt.title(f"Window {n+1}", fontsize=10, loc='left')
            plt.plot(self.input_indices, inputs[n, :, plot_col_index],
             label='Inputs', marker='.', zorder=-10)

            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col,None)
            else:
                label_col_index = plot_col_index

            if label_col_index is None:
                continue

            plt.scatter(self.label_indices, labels[n, :, label_col_index],
                        edgecolors='k', marker='s',
                        label='Labels',c='blue',s=64)
            if model is not None:
                predictions = model(inputs)
                plt.scatter(self.label_indices, predictions[n, :, label_col_index],marker='X',
                            edgecolors='k', label='Predictions',c='red', s=64)

            if n == 0:
                plt.legend()
                plt.xlabel('Time (Hours)')
                

    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(data=data,targets=None,
                                                                  sequence_length=self.total_window_size,sequence_stride=1,
                                                                  shuffle=True,batch_size=32)
        ds = ds.map(self.split_to_inputs_labels)
        return ds


    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)

    @property
    def sample_batch(self):
        result = getattr(self, '_sample_batch', None)
        if result is None:
            result = next(iter(self.train))
            self._sample_batch = result
            return result


##/////////////////////////Linear model///////////////////////////////////////////////
from tensorflow.keras.metrics import MeanAbsolutePercentageError
SizeOfWindow = 24 
def compile_and_fit(model, window, patience=3, max_epochs=50):
    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=patience,
                                   mode='min')
    model.compile(loss=MeanSquaredError(),
                  optimizer=Adam(),
                  metrics=[MeanAbsolutePercentageError()])
    history = model.fit(window.train,
                       epochs=max_epochs,
                        validation_data=window.val,
                        callbacks=[early_stopping])
    return history

## linear model single-step

val_performance = {}
performance = {}

wide_window_linear = DataWindow(input_width=24, label_width=24, shift=1, label_columns=['etCO2'])
linear = Sequential([
    Dense(units=1)
])
history = compile_and_fit(linear, wide_window_linear)
val_performance['Linear'] = linear.evaluate(wide_window_linear.val)
performance['Linear'] = linear.evaluate(wide_window_linear.test, verbose=0)
wide_window_linear.plot(linear)
plt.xlabel('Time (h)')
plt.suptitle("Time Series With Linear Regression When The Window Size is 1")
plt.show()

## linear model multi-step

ms_val_performance = {}
ms_performance = {}
wide_window_linear_multi = DataWindow(input_width=24, label_width=24, shift=SizeOfWindow, label_columns=['etCO2'])
ms_linear = Sequential([
Dense(1, kernel_initializer=tf.initializers.zeros)
])

history = compile_and_fit(ms_linear, wide_window_linear_multi)
ms_val_performance['Linear'] = ms_linear.evaluate(wide_window_linear_multi.val)
ms_performance['Linear'] = ms_linear.evaluate(wide_window_linear_multi.test, verbose=0)
wide_window_linear_multi.plot(linear)
plt.xlabel('Time (h)')
plt.suptitle(f"Time Series With Linear Regression When The Window Size is {SizeOfWindow}")
plt.show()



##/////////////////// GRADIENT TREE BOOSTING  WINDOW = 1 //////////////////////////////

# Assuming df is your DataFrame and it's already sorted by the index



from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit


X_train = train_df.drop('etCO2', axis=1)
y_train = train_df['etCO2']
X_val = val_df.drop('etCO2', axis=1)
y_val = val_df['etCO2']
X_test = test_df.drop('etCO2', axis=1)
y_test = test_df['etCO2']

# Combine your training and validation data into one
df_train_val = pd.concat([X_train, X_val])
# Combine your training and validation dataframes into one
y_train_val = pd.concat([y_train, y_val])

##Number of folds 
n_splits = 3
window_size = 1
tss = TimeSeriesSplit(n_splits=n_splits, test_size=window_size)

preds = []
scores = []

for train_idx, val_idx in tss.split(df_train_val):
    X_train_fold = df_train_val.iloc[train_idx]
    y_train_fold = y_train_val.iloc[train_idx]
    X_val_fold = df_train_val.iloc[val_idx]
    y_val_fold = y_train_val.iloc[val_idx]

    reg = xgb.XGBRegressor(base_score=0.5, booster='gbtree',    
                           n_estimators=1000,
                           early_stopping_rounds=50,
                           objective='reg:squarederror',
                           max_depth=3,
                           learning_rate=0.01)
    reg.fit(X_train_fold, y_train_fold,
            eval_set=[(X_train_fold, y_train_fold), (X_val_fold, y_val_fold)],
            verbose=100)

    y_pred = reg.predict(X_val_fold)
    preds.append(y_pred)
    score = np.sqrt(mean_squared_error(y_val_fold, y_pred))
    scores.append(score)

print(f'Score across folds {np.mean(scores):0.4f}')
print(f'Fold scores:{scores}')

##Plot the graph
def plot_boost(model, X_val, y_val, num_subplots, window_size=24, size=1):
    # Get predictions
    predictions = model.predict(X_val)
    
    plt.figure(figsize=(12, 8))
    
    for i in range(num_subplots):
        # Calculate start and end indices for the window
        start = i * window_size
        end = (i+1) * window_size

        plt.subplot(num_subplots, 1, i+1)
        plt.ylabel('etCO2 [scaled]')
        plt.plot(np.arange(len(y_val[start:end])), y_val[start:end], label='Actual', marker='s', zorder=-10, color='blue', linestyle='-')

        plt.scatter(np.arange(len(y_val[start:end])), y_val[start:end], edgecolors='k', label='Labels', c='blue', s=64, marker='s')
        plt.scatter(np.arange(len(predictions[start:end])), predictions[start:end], marker='X',
                    edgecolors='k', label='Predictions', c='red', s=64)
##        plt.title(f"Window {i+1}", fontsize=10, loc='left')
        plt.legend()
        plt.xlabel('Time (Hours)')
        plt.xlim(0, 24)  
        plt.suptitle(f"Time Series With Gradient Boost When The Window Size is {size}")

    plt.tight_layout()
    plt.show()


## Calculate the evaluation matrics

plot_boost(reg, X_val, y_val,1, window_size=24, size=1)
# Calculate the evaluation metrics
y_pred_val = reg.predict(X_val)
val_mse = mean_absolute_percentage_error(y_val, y_pred_val)
val_mae = mean_absolute_percentage_error(y_val, y_pred_val)

y_pred_test = reg.predict(X_test)
test_mse = mean_absolute_percentage_error(y_test, y_pred_test)
test_mae = mean_absolute_percentage_error(y_test, y_pred_test)

val_performance['XGBoost'] = ['XGBoost Val', val_mae]
performance['XGBoost'] = ['XGBoost Test', test_mae]

print(f'Validation performance: {val_performance}')
print(f'Test performance: {performance}')

##///////////////////////// GRADIENT TREE BOOSTING  WINDOW = 24  /////////////////////////////////////////////////////////////////


# Assuming df is your DataFrame and it's already sorted by the index



from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import GridSearchCV


X_train = train_df.drop('etCO2', axis=1)
y_train = train_df['etCO2']
X_val = val_df.drop('etCO2', axis=1)
y_val = val_df['etCO2']
X_test = test_df.drop('etCO2', axis=1)
y_test = test_df['etCO2']


df_train_val = pd.concat([X_train, X_val])
y_train_val = pd.concat([y_train, y_val])

n_splits = 3
window_size = SizeOfWindow
tss = TimeSeriesSplit(n_splits=n_splits, test_size=window_size)

preds = []
scores = []

for train_idx, val_idx in tss.split(df_train_val):
    X_train_fold = df_train_val.iloc[train_idx]
    y_train_fold = y_train_val.iloc[train_idx]
    X_val_fold = df_train_val.iloc[val_idx]
    y_val_fold = y_train_val.iloc[val_idx]

    param_grid = {
    'max_depth': [7, 9],  
    'learning_rate': [0.01, 0.2],  
    'n_estimators': [100, 500],  
    'colsample_bytree': [0.3, 0.7], 
    }

    # Initialize the GridSearchCV object for the XGBRegressor
    grid_clf = GridSearchCV(estimator=xgb.XGBRegressor(objective='reg:squarederror'), 
                        param_grid=param_grid, 
                        scoring='neg_mean_squared_error', 
                        cv=5, 
                        n_jobs=-1)
    # Fit the grid search to the combined training and validation data
    grid_clf.fit(df_train_val, y_train_val)
    # Print the best parameters and the best score
    print(f"Best parameters found: {grid_clf.best_params_}")
    print(f"Best score found: {-grid_clf.best_score_}")

    # Use the best parameters to train a new model

    reg = xgb.XGBRegressor(**grid_clf.best_params_)

##    reg = xgb.XGBRegressor(base_score=0.5, booster='gbtree',    
##                           n_estimators=1000,
##                           early_stopping_rounds=50,
##                           objective='reg:squarederror',
##                           max_depth=3,
##                           learning_rate=0.01)
    reg.fit(X_train_fold, y_train_fold,
            eval_set=[(X_train_fold, y_train_fold), (X_val_fold, y_val_fold)],
            verbose=100)

    y_pred = reg.predict(X_val_fold)
    preds.append(y_pred)
    score = np.sqrt(mean_squared_error(y_val_fold, y_pred))
    scores.append(score)

print(f'Score across folds {np.mean(scores):0.4f}')
print(f'Fold scores:{scores}')


## Calculate the evaluation matrics

plot_boost(reg, X_val, y_val,1, window_size=24, size=SizeOfWindow)
y_pred_val = reg.predict(X_val)
val_mse = mean_squared_error(y_val, y_pred_val)
val_mae = mean_absolute_percentage_error(y_val, y_pred_val)*100

y_pred_test = reg.predict(X_test)
test_mse = mean_squared_error(y_test, y_pred_test)
test_mae = mean_absolute_percentage_error(y_test, y_pred_test)*100


### Add the Xgboost results to the dictionaries

ms_val_performance['XGBoost'] = ['XGBoost Val', val_mae]
ms_performance['XGBoost'] = ['XGBoost Test', test_mae]


# Create an explainer for the model using training data
explainer = shap.Explainer(reg.predict, X_train)
# Compute SHAP values for all examples in the test set
shap_values_test = explainer(X_test)
# Visualize the first prediction's explanation for the test set
shap.plots.waterfall(shap_values_test[0])
shap.summary_plot(shap_values_test, X_test, plot_type="bar")
# For a single feature, say 'Feature1'



##import lime
##from lime import lime_tabular
##
### Define the feature columns and the target column
##features = test_df.columns
##target = 'etCO2'
##
### Create a LIME explainer object
##explainer = lime_tabular.LimeTabularExplainer(X_train.to_numpy(), 
##                                              feature_names=features, 
##                                              class_names=[target], 
##                                              verbose=True, 
##                                              mode='regression')
##
### Choose a random instance for explanation
##i = np.random.randint(0, X_test.shape[0])
### Get the explanation for the instance
##
### Create a LIME explainer object for regression models with understandable names
##explainer = lime_tabular.LimeTabularExplainer(X_train.to_numpy(),
##                                              feature_names=features.tolist(), 
##                                              class_names=['Predicted etCO2 level'], 
##                                              verbose=True, 
##                                              mode='regression')
##
### Choose a specific instance for explanation
##idx_to_explain = 42  # For instance, replace 42 with the index of the specific instance you want to explain
### Get the explanation for that instance
##explanation = explainer.explain_instance(X_test.iloc[idx_to_explain].to_numpy(), 
##                                         reg.predict, 
##                                         num_features=5, 
##                                         num_samples=500) 
##
####exp = explainer.explain_instance(X_test.iloc[i], reg.predict, num_features=5)
##### Predict the value using your model
####predicted_value = reg.predict(X_test.iloc[[i]])
### Save the explanation to an HTML file
##explanation.save_to_file('/Users/gabriellafawaz/Documents/Project/Lime/gradient_boosting.html')




##////////////////////////DEEP NEURAL NETWORKS /////////////////////////////////////////
#### Deep neural network: first hidden layer with 64 neurons 
##
##wide_window_dnn = DataWindow(input_width=24, label_width=24, shift=1, label_columns=['etCO2'])
##dense = Sequential([
##    Dense(units=64, activation='relu'),
##    Dense(units=64, activation='relu'),
##    Dense(units=1)
##])
##history = compile_and_fit(dense, wide_window_dnn)
##val_performance['DNN'] = dense.evaluate(wide_window_dnn.val)
##performance['DNN'] = dense.evaluate(wide_window_dnn.test, verbose=0)
##wide_window_dnn.plot(dense)
##plt.xlabel('Time (h)')
##plt.suptitle("Time Series With DNN When The Window Size is 1")
##plt.show()
##
##
##mae_val = [v[1] for v in val_performance.values()]
##mae_test = [v[1] for v in performance.values()]
##x = np.arange(len(performance))
##
####fig, ax = plt.subplots(figsize=(12, 8))
##fig, ax = plt.subplots()
##ax.bar(x - 0.15, mae_val, width=0.25, color='black', edgecolor='black',  label='Validation')
##ax.bar(x + 0.15, mae_test, width=0.25, color='white', edgecolor='black',  hatch='/', label='Test')
##ax.set_ylabel('Mean absolute error')
##ax.set_xlabel('Models')
##
##for index, value in enumerate(mae_val):
##    plt.text(x=index - 0.15, y=value+0.0025, s=str(round(value, 3)), ha='center')
##for index, value in enumerate(mae_test):
##    plt.text(x=index + 0.15, y=value+0.0025, s=str(round(value, 3)), ha='center')
##
##
####plt.ylim(0, 0.1)
##plt.xticks(ticks=x, labels=performance.keys())
##plt.legend(loc='best')
##plt.suptitle("Histogram of MAE of Models when Window Size is 1")
##plt.show()



## Deep neural network: multipstep 
ms_wide_window_dnn = DataWindow(input_width=24, label_width=24, shift=SizeOfWindow, label_columns=['etCO2'])
ms_dense = Sequential([
Dense(64, activation='relu'),
Dense(64, activation='relu'),
Dense(1, kernel_initializer=tf.initializers.zeros),
])
history = compile_and_fit(ms_dense, ms_wide_window_dnn)
ms_val_performance['Dense'] = ms_dense.evaluate(ms_wide_window_dnn.val)
ms_performance['Dense'] = ms_dense.evaluate(ms_wide_window_dnn.test, verbose=0)
ms_wide_window_dnn.plot(ms_dense)
plt.xlabel('Time (h)')
plt.suptitle(f"Time Series With DNN When The Window Size is {SizeOfWindow}")
plt.show()


ms_mae_val = [v[1] for v in ms_val_performance.values()]
ms_mae_test = [v[1] for v in ms_performance.values()]
x = np.arange(len(ms_performance))
##fig, ax = plt.subplots()
##ax.bar(x - 0.15, ms_mae_val, width=0.25, color='black', edgecolor='black',  label='Validation')
##ax.bar(x + 0.15, ms_mae_test, width=0.25, color='white', edgecolor='black',  hatch='/', label='Test')
##ax.set_ylabel('Mean absolute error')
##ax.set_xlabel('Models')
##for index, value in enumerate(ms_mae_val):
##    plt.text(x=index - 0.15, y=value+0.0025, s=str(round(value, 3)), ha='center')
##for index, value in enumerate(ms_mae_test):
##    plt.text(x=index + 0.15, y=value+0.0025, s=str(round(value, 3)), ha='center')
##plt.ylim(0, 0.4)
##plt.xticks(ticks=x, labels=ms_performance.keys())
##plt.legend(loc='best')
##plt.suptitle(f"Histogram of MAE of Models When The Window Size is {SizeOfWindow}")
##plt.show()

##////////////////////////LSTM //////////////////////////////////////////

####LSTM as a single-step model
##
##wide_window_lstm = DataWindow(input_width=24, label_width=24, shift=1, label_columns=['etCO2'])
##lstm_model = Sequential([
##    LSTM(32, return_sequences=True),
##    Dense(units=1)
##])
##history = compile_and_fit(lstm_model, wide_window_lstm)
####val_performance = {}
####performance = {}
##val_performance['LSTM'] = lstm_model.evaluate(wide_window_lstm.val)
##performance['LSTM'] = lstm_model.evaluate(wide_window_lstm.test, verbose=0)
##wide_window_lstm.plot(lstm_model)
##plt.suptitle("Time Series With LSTM When The Window Size is 1")
##plt.show()
##
##lstm_mae_val = [v[1] for v in val_performance.values()]
##lstm_mae_test = [v[1] for v in performance.values()]
##x = np.arange(len(performance))
###fig, ax = plt.subplots()
##fig, ax = plt.subplots(figsize=(10, 8))
##ax.bar(x - 0.15, lstm_mae_val, width=0.25, color='black', edgecolor='black',  label='Validation')
##ax.bar(x + 0.15, lstm_mae_test, width=0.25, color='white', edgecolor='black',  hatch='/', label='Test')
##ax.set_ylabel('Mean absolute error')
##ax.set_xlabel('Models')
##for index, value in enumerate(lstm_mae_val):
##    plt.text(x=index - 0.15, y=value+0.0025, s=str(round(value, 3)), ha='center')
##for index, value in enumerate(lstm_mae_test):
##    plt.text(x=index + 0.15, y=value+0.0025, s=str(round(value, 3)), ha='center')
##plt.ylim(0, 0.4)
##plt.xticks(ticks=x, labels=performance.keys())
##plt.legend(loc='best')
##plt.suptitle("Histogram of MAE of Models When The Window Size is 1")
##plt.show()



##LSTM as a mult-step model
multi_window_lstm = DataWindow(input_width=24, label_width=24, shift=SizeOfWindow,label_columns=['etCO2'])
ms_lstm_model = Sequential([
    LSTM(32, return_sequences=True),
    Dense(1, kernel_initializer=tf.initializers.zeros), ])

history = compile_and_fit(ms_lstm_model, multi_window_lstm)
##ms_val_performance = {}
##ms_performance = {}
ms_val_performance['LSTM'] = ms_lstm_model.evaluate(multi_window_lstm.val)
ms_performance['LSTM'] = ms_lstm_model.evaluate(multi_window_lstm.test, verbose=0)
multi_window_lstm.plot(ms_lstm_model)
plt.suptitle(f"Time Series With LSTM When The Window Size is {SizeOfWindow}")
plt.show()

ms_mae_val = [v[1] for v in ms_val_performance.values()]
ms_mae_test = [v[1] for v in ms_performance.values()]
x = np.arange(len(ms_performance))
##fig, ax = plt.subplots()
##fig, ax = plt.subplots(figsize=(10, 8))
##ax.bar(x - 0.15, ms_mae_val, width=0.25, color='black', edgecolor='black',  label='Validation')
##ax.bar(x + 0.15, ms_mae_test, width=0.25, color='white', edgecolor='black',  hatch='/', label='Test')
##ax.set_ylabel('Mean absolute error')
##ax.set_xlabel('Models')
##for index, value in enumerate(ms_mae_val):
##    plt.text(x=index - 0.15, y=value+0.0025, s=str(round(value, 3)), ha='center')
##for index, value in enumerate(ms_mae_test):
##    plt.text(x=index + 0.15, y=value+0.0025, s=str(round(value, 3)), ha='center')
##plt.ylim(0, 0.4)
##plt.xticks(ticks=x, labels=ms_performance.keys())
##plt.legend(loc='best')
##plt.suptitle(f"Histogram of MAE of Models When The Window Size is {SizeOfWindow}")
##plt.show()


##///////////////////////////// CONVOLUTION CNN //////////////////////////////////////////

####Convolution window
##KERNEL_WIDTH = 3
##conv_window = DataWindow(input_width=KERNEL_WIDTH, label_width=1, shift=1, label_columns=['etCO2'])
##LABEL_WIDTH = 24
##INPUT_WIDTH = LABEL_WIDTH + KERNEL_WIDTH - 1
##wide_conv_window = DataWindow(input_width=INPUT_WIDTH,label_width=LABEL_WIDTH, shift=1,
##                              label_columns=['etCO2'])
##cnn_model = Sequential([
##    Conv1D(filters=32,
##          kernel_size=(KERNEL_WIDTH,),
##          activation='relu'),
##    Dense(units=32, activation='relu'),
##    Dense(units=1)
##])
##
##history = compile_and_fit(cnn_model, conv_window)
####val_performance = {}
####performance = {}
##val_performance['CNN'] = cnn_model.evaluate(conv_window.val)
##performance['CNN'] = cnn_model.evaluate(conv_window.test, verbose=0)
##wide_conv_window.plot(cnn_model)
##plt.suptitle("Time Series of CNN When The Window Size is 1")
##plt.show()
##
##mae_val = [v[1] for v in val_performance.values()]
##mae_test = [v[1] for v in performance.values()]
##x = np.arange(len(performance))
####fig, ax = plt.subplots()
##fig, ax = plt.subplots(figsize=(10, 8))
##ax.bar(x - 0.15, mae_val, width=0.25, color='black', edgecolor='black',  label='Validation')
##ax.bar(x + 0.15, mae_test, width=0.25, color='white', edgecolor='black',  hatch='/', label='Test')
##ax.set_ylabel('Mean absolute error')
##ax.set_xlabel('Models')
##
##for index, value in enumerate(mae_val):
##    plt.text(x=index - 0.15, y=value+0.0025, s=str(round(value, 3)), ha='center')
##for index, value in enumerate(mae_test):
##    plt.text(x=index + 0.15, y=value+0.0025, s=str(round(value, 3)), ha='center')
##
##
##plt.ylim(0, 1.0)
##plt.xticks(ticks=x, labels=performance.keys())
##plt.legend(loc='best')
##plt.suptitle("Histogram of MAE  of Models When The Window Size is 1")
##plt.show()


## convolution  multiple window 

KERNEL_WIDTH = 3
LABEL_WIDTH = 24
INPUT_WIDTH = LABEL_WIDTH + KERNEL_WIDTH - 1
multi_window = DataWindow(input_width=INPUT_WIDTH, label_width=LABEL_WIDTH, shift=SizeOfWindow, label_columns=['etCO2'])
ms_cnn_model = Sequential([
    Conv1D(32, activation='relu',kernel_size=(KERNEL_WIDTH)),
    Dense(units=32, activation='relu'),
    Dense(1, kernel_initializer=tf.initializers.zeros),
])

history = compile_and_fit(ms_cnn_model, multi_window)
##ms_val_performance = {}
##ms_performance = {}
ms_val_performance['CNN'] = ms_cnn_model.evaluate(multi_window.val)
ms_performance['CNN'] = ms_cnn_model.evaluate(multi_window.test, verbose=0)
multi_window .plot(ms_cnn_model)
plt.suptitle(f"Time Series of CNN When The Window Size is {SizeOfWindow}")
plt.show()

ms_mae_val = [v[1] for v in ms_val_performance.values()]
ms_mae_test = [v[1] for v in ms_performance.values()]
x = np.arange(len(ms_performance))
#fig, ax = plt.subplots()
##fig, ax = plt.subplots(figsize=(10, 8))
##ax.bar(x - 0.15, ms_mae_val, width=0.25, color='black', edgecolor='black',  label='Validation')
##ax.bar(x + 0.15, ms_mae_test, width=0.25, color='white', edgecolor='black',  hatch='/', label='Test')
##ax.set_ylabel('Mean absolute error')
##ax.set_xlabel('Models')
##for index, value in enumerate(ms_mae_val):
##    plt.text(x=index - 0.15, y=value+0.0025, s=str(round(value, 3)), ha='center')
##for index, value in enumerate(ms_mae_test):
##    plt.text(x=index + 0.15, y=value+0.0025, s=str(round(value, 3)), ha='center')
##plt.ylim(0, 0.4)
##plt.xticks(ticks=x, labels=ms_performance.keys())
##plt.legend(loc='best')
##plt.suptitle(f"Histogram of MAE of Models When The Window Size is {SizeOfWindow}")
##plt.show()


##/////////////////////// AUTOREGRESSION LSTM /////////////////////////////



## Autoregression LSTM
multi_window = DataWindow(input_width=24, label_width=24, shift=SizeOfWindow,label_columns=['etCO2'])
class AutoRegressive(tf.keras.Model):
    def __init__(self, units, out_steps):
        super().__init__()
        self.out_steps = out_steps
        self.units = units
        self.lstm_cell = LSTMCell(units)
        self.lstm_rnn = RNN(self.lstm_cell, return_state=True)
        self.dense = Dense(train_df.shape[1])
    def warmup(self, inputs):
        x, *state = self.lstm_rnn(inputs)
        prediction = self.dense(x)
        return prediction, state
    def call(self, inputs, training=None):
        predictions = []
        prediction, state = self.warmup(inputs)
        predictions.append(prediction)
        for n in range(1, self.out_steps):
            x = prediction
            x, state = self.lstm_cell(x, states=state, training=training)
            prediction = self.dense(x)
            predictions.append(prediction)
        predictions = tf.stack(predictions)
        predictions = tf.transpose(predictions, [1, 0, 2])
        return predictions[:, :, :1]  


AR_LSTM = AutoRegressive(units=64, out_steps=24)
history = compile_and_fit(AR_LSTM, multi_window)
##ms_val_performance = {}
##ms_performance = {}
ms_val_performance['AR - LSTM'] = AR_LSTM.evaluate(multi_window.val)
ms_performance['AR - LSTM'] = AR_LSTM.evaluate(multi_window.test,  verbose=0)
multi_window.plot(AR_LSTM)
plt.suptitle(f"Time Series of AR - LSTM When The Window Size is {SizeOfWindow}")
plt.show()

ms_mae_val = [v[1] for v in ms_val_performance.values()]
ms_mae_test = [v[1] for v in ms_performance.values()]
x = np.arange(len(ms_performance))
##fig, ax = plt.subplots()
fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(x - 0.15, ms_mae_val, width=0.25, color='black', edgecolor='black',  label='Validation')
ax.bar(x + 0.15, ms_mae_test, width=0.25, color='white', edgecolor='black',  hatch='/', label='Test')
ax.set_ylabel('Mean absolute percentage error')
ax.set_xlabel('Models')
for index, value in enumerate(ms_mae_val):
    plt.text(x=index - 0.15, y=value+0.0025, s=str(round(value, 3)), ha='center')
for index, value in enumerate(ms_mae_test):
    plt.text(x=index + 0.15, y=value+0.0025, s=str(round(value, 3)), ha='center')
plt.ylim(0, 100)
plt.xticks(ticks=x, labels=ms_performance.keys())
plt.legend(loc='best')
plt.suptitle(f"Histogram of MAPE of All the Models When The Window Size is {SizeOfWindow}")
plt.tight_layout()
plt.show()

print(ms_val_performance)
print(ms_performance)
