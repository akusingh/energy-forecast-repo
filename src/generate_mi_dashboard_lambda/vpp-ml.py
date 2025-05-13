import boto3
import pandas as pd
from io import StringIO
import os
from datetime import datetime, timezone

def lambda_handler(event, context):
    s3 = boto3.client('s3')
    input_bucket = os.environ.get('S3_BUCKET_NAME')
    raw_key = event['key']
    transform_key = 'transformed_charge_data.csv'

    try:
        response_raw = s3.get_object(Bucket=input_bucket, Key=raw_key)
		response_trans = s3.get_object(Bucket=input_bucket, Key=transform_key)
        csv_content_raw = response_raw['Body'].read().decode('utf-8')
		csv_content_transform = response_trans['Body'].read().decode('utf-8')
        df_raw = pd.read_csv(StringIO(csv_content_raw))
		df_transform = pd.read_csv(StringIO(csv_content_transform))
		
	except Exception as e:
        print(f"Error processing S3 object: {e}")
        return {
            'statusCode': 500,
            'body': f'Error processing S3 object: {e}'
        }

# Focusing on CHARGE_WATT_HOUR as it was mentioned as problematic
if 'CHARGE_WATT_HOUR' in df_transform.columns:
    print(df_transform['CHARGE_WATT_HOUR'].describe())
else:
    print("'CHARGE_WATT_HOUR' column not found in the DataFrame.")
	
	
print(hourly_consumption.head())
print(hourly_consumption.index)

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(hourly_consumption['total_hourly_consumption'], label='Total Hourly Consumption')
plt.title('Total Hourly Energy Consumption')
plt.xlabel('Timestamp')
plt.ylabel('Watt-hour')
plt.legend()
plt.grid(True)
plt.show()

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

# Autocorrelation Function (ACF)
plot_acf(hourly_consumption['total_hourly_consumption'], lags=50)
plt.title('Autocorrelation Function (ACF)')
plt.xlabel('Lag (in hours)')
plt.ylabel('Correlation')
plt.show()

# Partial Autocorrelation Function (PACF)
plot_pacf(hourly_consumption['total_hourly_consumption'], lags=50, method='ywm')
plt.title('Partial Autocorrelation Function (PACF)')
plt.xlabel('Lag (in hours)')
plt.ylabel('Partial Correlation')
plt.show()

forecast_horizon = 168  # 7 days * 24 hours
train_data = hourly_consumption[:-forecast_horizon]
test_data = hourly_consumption[-forecast_horizon:]

print("Training data shape:", train_data.shape)
print("Testing data shape:", test_data.shape)

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(train_data['total_hourly_consumption'], label='Training Data')
plt.plot(test_data['total_hourly_consumption'], label='Testing Data')
plt.title('Training and Testing Data Split')
plt.xlabel('Timestamp')
plt.ylabel('Watt-hour')
plt.legend()
plt.grid(True)
plt.show()

def create_lagged_features(df, lags):
    for lag in lags:
        df[f'lag_{lag}'] = df['total_hourly_consumption'].shift(lag)
    return df

lags = [1, 2, 3, 24, 168]

# Create lagged features for the training data and drop NaN rows
train_data_lagged = create_lagged_features(train_data.copy(), lags).dropna()
X_train = train_data_lagged.drop('total_hourly_consumption', axis=1)
y_train = train_data_lagged['total_hourly_consumption']

# Create lagged features for the test data
test_data_lagged_with_history = create_lagged_features(test_data_with_history.copy(), lags)

# The first 'max(lags)' rows of the test set will have NaN due to the shift.
# We need to drop these to align the features with the forecast period.
test_data_lagged = test_data_lagged_with_history.dropna()
X_test = test_data_lagged.drop('total_hourly_consumption', axis=1)
y_test = test_data_lagged['total_hourly_consumption']

print("Training features shape:", X_train.shape)
print("Training target shape:", y_train.shape)
print("Testing features shape:", X_test.shape)
print("Testing target shape:", y_test.shape)

print("\nTraining features head:")
print(X_train.head())
print("\nTesting features head:")
print(X_test.head())
print("\nTesting target head:")
print(y_test.head())

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# --- Train and Evaluate Linear Regression ---
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)
predictions_lr = model_lr.predict(X_test)

mse_lr = mean_squared_error(y_test, predictions_lr)
rmse_lr = np.sqrt(mse_lr)
mae_lr = mean_absolute_error(y_test, predictions_lr)

print("\nLinear Regression Results:")
print(f'Mean Squared Error (MSE): {mse_lr:.2f}')
print(f'Root Mean Squared Error (RMSE): {rmse_lr:.2f}')
print(f'Mean Absolute Error (MAE): {mae_lr:.2f}')

plt.figure(figsize=(12, 6))
plt.plot(y_test.index, y_test.values, label='Actual Consumption')
plt.plot(y_test.index, predictions_lr, label='Linear Regression Predictions')
plt.title('Actual vs. Predicted Hourly Energy Consumption (Linear Regression)')
plt.xlabel('Timestamp')
plt.ylabel('Watt-hour')
plt.legend()
plt.grid(True)
plt.show()

lr_predictions_df = pd.DataFrame({'timestamp': y_test.index, 'predicted_consumption': predictions_lr})
lr_predictions_df.set_index('timestamp', inplace=True)
print("\nLinear Regression Predictions:")
print(lr_predictions_df.head())

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# --- Train and Evaluate Random Forest ---
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
rf_mse = mean_squared_error(y_test, rf_predictions)
rf_rmse = np.sqrt(rf_mse)
rf_mae = mean_absolute_error(y_test, rf_predictions)

print("\nRandom Forest Regression Results:")
print(f'Mean Squared Error (MSE): {rf_mse:.2f}')
print(f'Root Mean Squared Error (RMSE): {rf_rmse:.2f}')
print(f'Mean Absolute Error (MAE): {rf_mae:.2f}')

plt.figure(figsize=(12, 6))
plt.plot(y_test.index, y_test.values, label='Actual Consumption')
plt.plot(y_test.index, rf_predictions, label='Random Forest Predictions')
plt.title('Actual vs. Predicted Hourly Energy Consumption (Random Forest)')
plt.xlabel('Timestamp')
plt.ylabel('Watt-hour')
plt.legend()
plt.grid(True)
plt.show()

rf_predictions_df = pd.DataFrame({'timestamp': y_test.index, 'predicted_consumption': rf_predictions})
rf_predictions_df.set_index('timestamp', inplace=True)
print("\nRandom Forest Predictions:")
print(rf_predictions_df.head())


import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# --- Train and Evaluate XGBoost ---
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
xgb_model.fit(X_train, y_train)
xgb_predictions = xgb_model.predict(X_test)
xgb_mse = mean_squared_error(y_test, xgb_predictions)
xgb_rmse = np.sqrt(xgb_mse)
xgb_mae = mean_absolute_error(y_test, xgb_predictions)

print("\nGradient Boosting (XGBoost) Regression Results:")
print(f'Mean Squared Error (MSE): {xgb_mse:.2f}')
print(f'Root Mean Squared Error (RMSE): {xgb_rmse:.2f}')
print(f'Mean Absolute Error (MAE): {xgb_mae:.2f}')

plt.figure(figsize=(12, 6))
plt.plot(y_test.index, y_test.values, label='Actual Consumption')
plt.plot(y_test.index, xgb_predictions, label='XGBoost Predictions')
plt.title('Actual vs. Predicted Hourly Energy Consumption (XGBoost)')
plt.xlabel('Timestamp')
plt.ylabel('Watt-hour')
plt.legend()
plt.grid(True)
plt.show()

xgb_predictions_df = pd.DataFrame({'timestamp': y_test.index, 'predicted_consumption': xgb_predictions})
xgb_predictions_df.set_index('timestamp', inplace=True)
print("\nXGBoost Predictions:")
print(xgb_predictions_df.head())

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Create a DataFrame with the results
results_df = pd.DataFrame({
    'Model': ['Linear Regression', 'Random Forest', 'XGBoost'],
    'RMSE': [rmse_lr, rf_rmse, xgb_rmse],
    'MAE': [mae_lr, rf_mae, xgb_mae]
})

# Set the 'Model' column as the index for easier plotting
results_df.set_index('Model', inplace=True)

# Plotting RMSE
plt.figure(figsize=(10, 6))
results_df['RMSE'].sort_values(ascending=True).plot(kind='bar', color='skyblue')
plt.title('Comparison of RMSE for Different Models')
plt.ylabel('Root Mean Squared Error (RMSE)')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--')
plt.tight_layout()
plt.show()

# Plotting MAE
plt.figure(figsize=(10, 6))
results_df['MAE'].sort_values(ascending=True).plot(kind='bar', color='lightcoral')
plt.title('Comparison of MAE for Different Models')
plt.ylabel('Mean Absolute Error (MAE)')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--')
plt.tight_layout()
plt.show()

print("\nError Statistics:")
print(results_df)

import matplotlib.pyplot as plt
import pandas as pd

# --- Plot Actual vs. Predicted (XGBoost - Zoomed In) ---
plt.figure(figsize=(15, 6))
plt.plot(y_test.index, y_test.values, label='Actual Consumption', color='blue')
plt.plot(y_test.index, xgb_predictions, label='XGBoost Predictions', color='red', alpha=0.7)
plt.title('Actual vs. Predicted Hourly Energy Consumption (XGBoost - Last 72 Hours)')
plt.xlabel('Timestamp')
plt.ylabel('Watt-hour')
plt.legend()
plt.grid(True)
plt.xlim(y_test.index[-72], y_test.index[-1])  # Zoom in on the last 72 hours
plt.tight_layout()
plt.show()

# --- Error Statistics Summary ---
print("\nError Statistics:")
results_df = pd.DataFrame({
    'Model': ['Linear Regression', 'Random Forest', 'XGBoost'],
    'MSE': [mse_lr, rf_mse, xgb_mse],
    'RMSE': [rmse_lr, rf_rmse, xgb_rmse],
    'MAE': [mae_lr, rf_mae, xgb_mae]
})
print(results_df)

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

# ACF of the original hourly consumption
plot_acf(hourly_consumption['total_hourly_consumption'], lags=50)
plt.title('ACF of Original Hourly Consumption')
plt.xlabel('Lag (in hours)')
plt.ylabel('Correlation')
plt.show()

# PACF of the original hourly consumption
plot_pacf(hourly_consumption['total_hourly_consumption'], lags=50, method='ywm')
plt.title('PACF of Original Hourly Consumption')
plt.xlabel('Lag (in hours)')
plt.ylabel('Partial Correlation')
plt.show()

from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Fit the SARIMA model
order = (1, 0, 1)             # (p, d, q)
seasonal_order = (1, 0, 1, 24) # (P, D, Q, s)
sarima_model = SARIMAX(train_data['total_hourly_consumption'], order=order, seasonal_order=seasonal_order)
sarima_fit = sarima_model.fit(disp=False)

# Make predictions on the test data
sarima_predictions = sarima_fit.predict(start=len(train_data), end=len(hourly_consumption)-1)

# Evaluate the model
sarima_mse = mean_squared_error(test_data['total_hourly_consumption'], sarima_predictions)
sarima_rmse = np.sqrt(sarima_mse)
sarima_mae = mean_absolute_error(test_data['total_hourly_consumption'], sarima_predictions)

print("\nSARIMA(1, 0, 1)(1, 0, 1)24 Results:")
print(f'Mean Squared Error (MSE): {sarima_mse:.2f}')
print(f'Root Mean Squared Error (RMSE): {sarima_rmse:.2f}')
print(f'Mean Absolute Error (MAE): {sarima_mae:.2f}')

# Plot actual vs. predicted
plt.figure(figsize=(12, 6))
plt.plot(test_data.index, test_data['total_hourly_consumption'], label='Actual Consumption')
plt.plot(test_data.index, sarima_predictions, label='SARIMA Predictions', color='red')
plt.title('Actual vs. Predicted Hourly Energy Consumption (SARIMA(1, 0, 1)(1, 0, 1)24)')
plt.xlabel('Timestamp')
plt.ylabel('Watt-hour')
plt.legend()
plt.grid(True)
plt.show()

sarima_predictions_df = pd.DataFrame({'timestamp': test_data.index, 'predicted_consumption': sarima_predictions})
sarima_predictions_df.set_index('timestamp', inplace=True)
print("\nSARIMA Predictions:")
print(sarima_predictions_df.head())

from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import pandas as pd

# Fit the SARIMA model
order = (1, 0, 1)             # (p, d, q)
seasonal_order = (1, 0, 1, 24) # (P, D, Q, s)
sarima_model = SARIMAX(train_data['total_hourly_consumption'], order=order, seasonal_order=seasonal_order)
sarima_fit = sarima_model.fit(disp=False)

# Make predictions on the test data
sarima_predictions = sarima_fit.predict(start=len(train_data), end=len(hourly_consumption)-1)

# Evaluate the model
sarima_mse = mean_squared_error(test_data['total_hourly_consumption'], sarima_predictions)
sarima_rmse = np.sqrt(sarima_mse)
sarima_mae = mean_absolute_error(test_data['total_hourly_consumption'], sarima_predictions)

print("\nSARIMA(1, 0, 1)(1, 0, 1)24 Results:")
print(f'Mean Squared Error (MSE): {sarima_mse:.2f}')
print(f'Root Mean Squared Error (RMSE): {sarima_rmse:.2f}')
print(f'Mean Absolute Error (MAE): {sarima_mae:.2f}')

import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd

# Define the parameter grid to search
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 4, 5],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}

# Initialize the XGBoost Regressor
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

# Perform GridSearchCV
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=3, verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Get the best parameters and the best score
best_params = grid_search.best_params_
best_score = np.sqrt(-grid_search.best_score_)  # Convert negative MSE to RMSE

print("\nBest Hyperparameters:", best_params)
print("Best RMSE:", best_score)

# Evaluate the model with the best hyperparameters on the test set
best_xgb_model = xgb.XGBRegressor(objective='reg:squarederror', **best_params, random_state=42)
best_xgb_model.fit(X_train, y_train)
best_xgb_predictions = best_xgb_model.predict(X_test)

best_xgb_mse = mean_squared_error(y_test, best_xgb_predictions)
best_xgb_rmse = np.sqrt(best_xgb_mse)
best_xgb_mae = mean_absolute_error(y_test, best_xgb_predictions)

print("\nBest XGBoost Model Results on Test Set:")
print(f'Mean Squared Error (MSE): {best_xgb_mse:.2f}')
print(f'Root Mean Squared Error (RMSE): {best_xgb_rmse:.2f}')
print(f'Mean Absolute Error (MAE): {best_xgb_mae:.2f}')

# Plot actual vs. predicted with the best XGBoost model
plt.figure(figsize=(12, 6))
plt.plot(y_test.index, y_test.values, label='Actual Consumption')
plt.plot(y_test.index, best_xgb_predictions, label='Best XGBoost Predictions', color='green')
plt.title('Actual vs. Predicted Hourly Energy Consumption (Best XGBoost Model)')
plt.xlabel('Timestamp')
plt.ylabel('Watt-hour')
plt.legend()
plt.grid(True)
plt.show()

best_xgb_predictions_df = pd.DataFrame({'timestamp': y_test.index, 'predicted_consumption': best_xgb_predictions})
best_xgb_predictions_df.set_index('timestamp', inplace=True)
print("\nBest XGBoost Predictions:")
print(best_xgb_predictions_df.head())

import pandas as pd
import numpy as np

def create_lstm_data(data, seq_length):
    """
    Creates sequences of data for LSTM input.

    Args:
        data (pd.Series): The time series data.
        seq_length (int): The length of the input sequences.

    Returns:
        tuple: A tuple containing:
            - X (np.array): Input sequences of shape (number_of_sequences, seq_length, 1).
            - y (np.array): Corresponding output values of shape (number_of_sequences,).
            - data_lstm (pd.DataFrame):  A DataFrame containing the original data and shifted values.
    """
    data_lstm = pd.DataFrame(data.copy())  # Create a copy
    for i in range(seq_length):
        data_lstm[f't-{i+1}'] = data_lstm['total_hourly_consumption'].shift(i + 1)

    data_lstm.dropna(inplace=True)

    # Convert to numpy array
    data_np = data_lstm['total_hourly_consumption'].values  # Only the target variable

    X = []
    y = []
    for i in range(len(data_np) - seq_length):
        X.append(data_np[i:i + seq_length])
        y.append(data_np[i + seq_length])

    X = np.array(X)
    y = np.array(y)

    # Reshape X to be (number_of_sequences, seq_length, 1)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    return X, y, data_lstm

# --- Create LSTM Data ---
seq_length = 24  # Length of the input sequence

X_lstm, y_lstm, data_lstm = create_lstm_data(hourly_consumption, seq_length)

# --- Split Data for LSTM ---
split_point_lstm = -(forecast_horizon)  # Split based on the forecast horizon
X_train_lstm = X_lstm[:split_point_lstm]
y_train_lstm = y_lstm[:split_point_lstm]
X_test_lstm = X_lstm[split_point_lstm:]
y_test_lstm = y_lstm[split_point_lstm:]

print("LSTM Training data shape X:", X_train_lstm.shape)
print("LSTM Training data shape y:", y_train_lstm.shape)
print("LSTM Testing data shape X:", X_test_lstm.shape)
print("LSTM Testing data shape y:", y_test_lstm.shape)

print("\nLSTM Training data X[0]:")
print(X_train_lstm[0])
print("\nLSTM Training data y[0]:")
print(y_train_lstm[0])
print("\nLSTM Testing data X[0]:")
print(X_test_lstm[0])
print("\nLSTM Testing data y[0]:")
print(y_test_lstm[0])

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Define the LSTM model
lstm_model = Sequential()
lstm_model.add(LSTM(units=50, activation='relu', input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])))
lstm_model.add(Dense(units=1))

# Compile the model
lstm_model.compile(optimizer='adam', loss='mse')

# Print the model summary
lstm_model.summary()

# Train the LSTM model
epochs = 50
batch_size = 32
history = lstm_model.fit(X_train_lstm, y_train_lstm, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=1)

# Evaluate the model on the test data
lstm_predictions = lstm_model.predict(X_test_lstm)

# Evaluate the predictions
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt

lstm_mse = mean_squared_error(y_test_lstm, lstm_predictions)
lstm_rmse = np.sqrt(lstm_mse)
lstm_mae = mean_absolute_error(y_test_lstm, lstm_predictions)

print("\nLSTM Model Results:")
print(f'Mean Squared Error (MSE): {lstm_mse:.2f}')
print(f'Root Mean Squared Error (RMSE): {lstm_rmse:.2f}')
print(f'Mean Absolute Error (MAE): {lstm_mae:.2f}')

# Plot actual vs. predicted
plt.figure(figsize=(12, 6))
plt.plot(hourly_consumption.index[-len(y_test_lstm):], y_test_lstm, label='Actual Consumption')
plt.plot(hourly_consumption.index[-len(y_test_lstm):], lstm_predictions, label='LSTM Predictions', color='purple')
plt.title('Actual vs. Predicted Hourly Energy Consumption (LSTM)')
plt.xlabel('Timestamp')
plt.ylabel('Watt-hour')
plt.legend()
plt.grid(True)
plt.show()

lstm_predictions_df = pd.DataFrame({'timestamp': hourly_consumption.index[-len(y_test_lstm):], 'predicted_consumption': lstm_predictions.flatten()})
lstm_predictions_df.set_index('timestamp', inplace=True)
print("\nLSTM Predictions:")
print(lstm_predictions_df.head())
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Ensure we have the prediction dataframes for all models
# lr_predictions_df, rf_predictions_df, best_xgb_predictions_df,
# sarima_predictions_df, lstm_predictions_df

plt.figure(figsize=(18, 10))

# Plot actual consumption
plt.plot(y_test.index, y_test.values, label='Actual Consumption', color='blue', linewidth=2)

# Plot Linear Regression predictions
plt.plot(lr_predictions_df.index, lr_predictions_df['predicted_consumption'], label='Linear Regression', color='orange', linestyle='--')

# Plot Random Forest predictions
plt.plot(rf_predictions_df.index, rf_predictions_df['predicted_consumption'], label='Random Forest', color='green', linestyle='--')

# Plot Tuned XGBoost predictions
plt.plot(best_xgb_predictions_df.index, best_xgb_predictions_df['predicted_consumption'], label='Tuned XGBoost', color='red', linestyle='--')

# Plot SARIMA predictions
plt.plot(sarima_predictions_df.index, sarima_predictions_df['predicted_consumption'], label='SARIMA', color='purple', linestyle='--')

# Plot LSTM predictions
plt.plot(lstm_predictions_df.index, lstm_predictions_df['predicted_consumption'], label='LSTM', color='brown', linestyle='--')

plt.title('Comparison of Hourly Energy Consumption Predictions Across Models', fontsize=16)
plt.xlabel('Timestamp', fontsize=12)
plt.ylabel('Watt-hour', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True)
plt.tight_layout()
plt.show()

import pandas as pd
import numpy as np

# --- Prepare data for future predictions (using the last known values) ---
last_known_consumption = hourly_consumption['total_hourly_consumption'].iloc[-max(lags):].values.flatten()

future_predictions_lr = []
current_history_lr = list(last_known_consumption)

future_predictions_rf = []
current_history_rf = list(last_known_consumption)

future_predictions_xgb = []
current_history_xgb = list(last_known_consumption)

n_future_hours = 168
lags_used = [1, 2, 3, 24, 168]

for i in range(n_future_hours):
    # Linear Regression
    features_lr = np.array([current_history_lr[-lag] if len(current_history_lr) >= lag else 0 for lag in lags_used]).reshape(1, -1)
    next_pred_lr = model_lr.predict(features_lr)[0]
    future_predictions_lr.append(next_pred_lr)
    current_history_lr.append(next_pred_lr)

    # Random Forest
    features_rf = np.array([current_history_rf[-lag] if len(current_history_rf) >= lag else 0 for lag in lags_used]).reshape(1, -1)
    next_pred_rf = rf_model.predict(features_rf)[0]
    future_predictions_rf.append(next_pred_rf)
    current_history_rf.append(next_pred_rf)

    # Tuned XGBoost
    features_xgb = np.array([current_history_xgb[-lag] if len(current_history_xgb) >= lag else 0 for lag in lags_used]).reshape(1, -1)
    next_pred_xgb = best_xgb_model.predict(features_xgb)[0]
    future_predictions_xgb.append(next_pred_xgb)
    current_history_xgb.append(next_pred_xgb)

# Create future timestamps
last_timestamp = hourly_consumption.index[-1]
future_timestamps = pd.date_range(start=last_timestamp + pd.Timedelta(hours=1), periods=n_future_hours, freq='H')

# Create DataFrames for future predictions
future_lr_df = pd.DataFrame({'timestamp': future_timestamps, 'predicted_consumption': future_predictions_lr}).set_index('timestamp')
future_rf_df = pd.DataFrame({'timestamp': future_timestamps, 'predicted_consumption': future_predictions_rf}).set_index('timestamp')
future_xgb_df = pd.DataFrame({'timestamp': future_timestamps, 'predicted_consumption': future_predictions_xgb}).set_index('timestamp')

print("\nLinear Regression Predictions for Next Week:")
print(future_lr_df.head())
print("\nRandom Forest Predictions for Next Week:")
print(future_rf_df.head())
print("\nTuned XGBoost Predictions for Next Week:")
print(future_xgb_df.head())

# Forecast the next 168 hours using the fitted SARIMA model
sarima_future_predictions = sarima_fit.get_forecast(steps=n_future_hours).predicted_mean
sarima_future_timestamps = pd.date_range(start=last_timestamp + pd.Timedelta(hours=1), periods=n_future_hours, freq='H')
future_sarima_df = pd.DataFrame({'timestamp': sarima_future_timestamps, 'predicted_consumption': sarima_future_predictions}).set_index('timestamp')

print("\nSARIMA Predictions for Next Week:")
print(future_sarima_df.head())

# Prepare the last sequence from the training data
last_sequence = X_test_lstm[-1].reshape(1, seq_length, 1)
future_predictions_lstm = []

for _ in range(n_future_hours):
    next_prediction = lstm_model.predict(last_sequence)
    future_predictions_lstm.append(next_prediction[0, 0])
    # Update the sequence by shifting and appending the new prediction
    last_sequence = np.roll(last_sequence, shift=-1, axis=1)
    last_sequence[0, -1, 0] = next_prediction[0, 0]

future_lstm_timestamps = pd.date_range(start=last_timestamp + pd.Timedelta(hours=1), periods=n_future_hours, freq='H')
future_lstm_df = pd.DataFrame({'timestamp': future_lstm_timestamps, 'predicted_consumption': future_predictions_lstm}).set_index('timestamp')

print("\nLSTM Predictions for Next Week:")
print(future_lstm_df.head())

import matplotlib.pyplot as plt

plt.figure(figsize=(18, 10))

# Plot predictions for each model
plt.plot(future_lr_df.index, future_lr_df['predicted_consumption'], label='Linear Regression (Next Week)', color='orange', linestyle='-')
plt.plot(future_rf_df.index, future_rf_df['predicted_consumption'], label='Random Forest (Next Week)', color='green', linestyle='-')
plt.plot(future_xgb_df.index, future_xgb_df['predicted_consumption'], label='Tuned XGBoost (Next Week)', color='red', linestyle='-')
plt.plot(future_sarima_df.index, future_sarima_df['predicted_consumption'], label='SARIMA (Next Week)', color='purple', linestyle='-')
plt.plot(future_lstm_df.index, future_lstm_df['predicted_consumption'], label='LSTM (Next Week)', color='brown', linestyle='-')

plt.title('Hourly Energy Consumption Predictions for the Next Week', fontsize=16)
plt.xlabel('Timestamp', fontsize=12)
plt.ylabel('Watt-hour', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True)
plt.tight_layout()
plt.show()
