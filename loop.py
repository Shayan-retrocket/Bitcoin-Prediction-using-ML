import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tabulate import tabulate
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, SimpleRNN, Conv1D, MaxPooling1D, Bidirectional, Dense, BatchNormalization, Dropout, Layer
from scipy.stats import pearsonr
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.feature_selection import mutual_info_regression, SelectKBest
from tensorflow.keras.regularizers import l2
import tensorflow as tf


btc_data = pd.read_csv('BTC-USD.csv', index_col='Date', parse_dates=True)

btc_data['Open_scaled'] = MinMaxScaler().fit_transform(btc_data[['Open']])
btc_data['Volume_scaled'] = MinMaxScaler().fit_transform(btc_data[['Volume']])
btc_data['High_scaled'] = MinMaxScaler().fit_transform(btc_data[['High']])
btc_data['Low_scaled'] = MinMaxScaler().fit_transform(btc_data[['Low']])

date_series = btc_data.index
btc_data['Date_sin'] = np.sin(2 * np.pi * date_series.dayofyear / 365)
btc_data['Date_cos'] = np.cos(2 * np.pi * date_series.dayofyear / 365)

features = ['Close', 'Date_sin', 'Open_scaled' , 'Volume_scaled',  'Date_cos' , 'High_scaled' ,'Low_scaled' ]
data_scaled = MinMaxScaler().fit_transform(btc_data[features])
btc_data['Close_scaled'] = MinMaxScaler().fit_transform(btc_data[['Close']])

def create_sequences_multi(data, seq_length):
    sequences, targets = [], []
    for i in range(len(data) - seq_length):
        seq = data[i : i + seq_length]
        target = data[i + seq_length, 0]  
        sequences.append(seq)
        targets.append(target)
    return np.array(sequences), np.array(targets)



seq_length = 10
epochs = 250
batch_size = 16
num_runs = 10
results = []

class CentralizedWeightLayer(Layer):
    def __init__(self, **kwargs):
        super(CentralizedWeightLayer, self).__init__(**kwargs)
        self.w = None  

    def build(self, input_shape):
        super(CentralizedWeightLayer, self).build(input_shape)
        if self.w is None:
            self.w = self.add_weight(
                shape=(int(input_shape[-1]),),
                initializer='ones',
                name='centralized_weight',
                trainable=True
            )

    def call(self, x):
        return self.w * x

    def compute_output_shape(self, input_shape):
        return input_shape


best_models = {'LSTM': None, 'GRU': None, 'RNN': None}
best_rmse = {'LSTM': float('inf'), 'GRU': float('inf'), 'RNN': float('inf')}


for run in range(num_runs):

    targets_multi = MinMaxScaler().fit_transform(btc_data[['Close']])

    mutual_info_values = mutual_info_regression(data_scaled.reshape(-1, len(features)), targets_multi)

    k = 5  
    selector = SelectKBest(mutual_info_regression, k=k)
    selected_features = selector.fit_transform(data_scaled.reshape(-1, len(features)), targets_multi)
    sequences_multi, targets_multi = create_sequences_multi(selected_features, seq_length)

    selected_features = selected_features.reshape(-1, k)  

  
    sequences_multi, targets_multi = create_sequences_multi(selected_features, seq_length)
    split_multi = int(0.8 * len(sequences_multi))


    
    X_train_multi, y_train_multi = sequences_multi[:split_multi], targets_multi[:split_multi].ravel()
    X_test_multi, y_test_multi = sequences_multi[split_multi:], targets_multi[split_multi:].ravel()
    model_multi_lstm = Sequential()
    model_multi_lstm.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(seq_length, k)))
    model_multi_lstm.add(MaxPooling1D(pool_size=2))
    model_multi_lstm.add(Bidirectional(LSTM(100,input_shape=(seq_length, k), return_sequences=True)))
    model_multi_lstm.add(CentralizedWeightLayer()) 
    model_multi_lstm.add(BatchNormalization())
    model_multi_lstm.add(Dropout(0.1))  
    model_multi_lstm.add(Bidirectional(LSTM(100, return_sequences=True)))
    model_multi_lstm.add(CentralizedWeightLayer()) 
    model_multi_lstm.add(BatchNormalization())
    model_multi_lstm.add(Dropout(0.1))
    model_multi_lstm.add(Bidirectional(LSTM(100)))
    model_multi_lstm.add(CentralizedWeightLayer()) 
    model_multi_lstm.add(BatchNormalization())
    model_multi_lstm.add(Dropout(0.1))
    model_multi_lstm.add(Dense(1,kernel_regularizer=l2(0.01)))
    optimizer_lstm = tf.keras.optimizers.legacy.Adam(learning_rate=0.001)
    model_multi_lstm.compile(optimizer=optimizer_lstm, loss='mean_squared_error')
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)


    model_multi_gru = Sequential()
    model_multi_gru.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(seq_length, k)))
    model_multi_gru.add(Bidirectional(GRU(100, input_shape=(seq_length, k), return_sequences=True)))
    model_multi_gru.add(BatchNormalization())
    model_multi_gru.add(CentralizedWeightLayer()) 
    model_multi_gru.add(Dropout(0.1))    
    model_multi_gru.add(Bidirectional(GRU(100, return_sequences=True )))
    model_multi_gru.add(BatchNormalization())  
    model_multi_gru.add(CentralizedWeightLayer()) 
    model_multi_gru.add(Dropout(0.1))    
    model_multi_gru.add(Bidirectional(GRU(100)))
    model_multi_gru.add(BatchNormalization())
    model_multi_gru.add(CentralizedWeightLayer()) 
    model_multi_gru.add(Dropout(0.1))    
    model_multi_gru.add(Dense(1,kernel_regularizer=l2(0.01)))
    optimizer_gru = tf.keras.optimizers.legacy.Adam(learning_rate=0.001)
    model_multi_gru.compile(optimizer=optimizer_gru, loss='mean_squared_error')
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)


    model_multi_rnn = Sequential()
    model_multi_rnn.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(seq_length, k)))
    model_multi_rnn.add(SimpleRNN(100, input_shape=(seq_length, k), return_sequences=True) )
    model_multi_rnn.add(BatchNormalization())
    model_multi_gru.add(CentralizedWeightLayer()) 
    model_multi_rnn.add(SimpleRNN(100, return_sequences=True))
    model_multi_rnn.add(BatchNormalization())
    model_multi_gru.add(CentralizedWeightLayer()) 
    model_multi_rnn.add(SimpleRNN(100))
    model_multi_rnn.add(Dense(1))
    optimizer_rnn = tf.keras.optimizers.legacy.Adam(learning_rate=0.001)  
    model_multi_rnn.compile(optimizer=optimizer_rnn, loss='mean_squared_error')

    history_multi_lstm = model_multi_lstm.fit(X_train_multi, y_train_multi, epochs=epochs, batch_size=batch_size, validation_data=(X_test_multi, y_test_multi), verbose=2 , callbacks=[ early_stopping ])
    history_multi_gru = model_multi_gru.fit(X_train_multi, y_train_multi, epochs=epochs, batch_size=batch_size, validation_data=(X_test_multi, y_test_multi), verbose=2 , callbacks=[early_stopping ])
    history_multi_rnn = model_multi_rnn.fit(X_train_multi, y_train_multi, epochs=epochs, batch_size=batch_size, validation_data=(X_test_multi, y_test_multi), verbose=2 )


    y_pred_multi_lstm = model_multi_lstm.predict(X_test_multi)
    y_pred_inv_multi_lstm = MinMaxScaler().fit(btc_data[['Close']]).inverse_transform(y_pred_multi_lstm)
    y_test_inv_multi_lstm = MinMaxScaler().fit(btc_data[['Close']]).inverse_transform(y_test_multi.reshape(-1, 1))

    y_pred_multi_gru = model_multi_gru.predict(X_test_multi)
    y_pred_inv_multi_gru = MinMaxScaler().fit(btc_data[['Close']]).inverse_transform(y_pred_multi_gru)
    y_test_inv_multi_gru = MinMaxScaler().fit(btc_data[['Close']]).inverse_transform(y_test_multi.reshape(-1, 1))

    y_pred_multi_rnn = model_multi_rnn.predict(X_test_multi)
    y_pred_inv_multi_rnn = MinMaxScaler().fit(btc_data[['Close']]).inverse_transform(y_pred_multi_rnn)
    y_test_inv_multi_rnn = MinMaxScaler().fit(btc_data[['Close']]).inverse_transform(y_test_multi.reshape(-1, 1))

    rmse_lstm = np.sqrt(mean_squared_error(y_test_inv_multi_lstm, y_pred_inv_multi_lstm))
    mae_lstm = mean_absolute_error(y_test_inv_multi_lstm, y_pred_inv_multi_lstm)
    mape_lstm = np.mean(np.abs((y_test_inv_multi_lstm - y_pred_inv_multi_lstm) / y_test_inv_multi_lstm)) * 100
    bias_lstm = np.mean(y_pred_inv_multi_lstm - y_test_inv_multi_lstm)

    rmse_gru = np.sqrt(mean_squared_error(y_test_inv_multi_gru, y_pred_inv_multi_gru))
    mae_gru = mean_absolute_error(y_test_inv_multi_gru, y_pred_inv_multi_gru)
    mape_gru = np.mean(np.abs((y_test_inv_multi_gru - y_pred_inv_multi_gru) / y_test_inv_multi_gru)) * 100
    bias_gru = np.mean(y_pred_inv_multi_gru - y_test_inv_multi_gru)

    rmse_rnn = np.sqrt(mean_squared_error(y_test_inv_multi_rnn, y_pred_inv_multi_rnn))
    mae_rnn = mean_absolute_error(y_test_inv_multi_rnn, y_pred_inv_multi_rnn)
    mape_rnn = np.mean(np.abs((y_test_inv_multi_rnn - y_pred_inv_multi_rnn) / y_test_inv_multi_rnn)) * 100
    bias_rnn = np.mean(y_pred_inv_multi_rnn - y_test_inv_multi_rnn)

    corr_lstm, _ = pearsonr(y_test_inv_multi_lstm.flatten(), y_pred_inv_multi_lstm.flatten())
    corr_gru, _ = pearsonr(y_test_inv_multi_gru.flatten(), y_pred_inv_multi_gru.flatten())
    corr_rnn, _ = pearsonr(y_test_inv_multi_rnn.flatten(), y_pred_inv_multi_rnn.flatten())

    std_dev_lstm = np.std(y_test_inv_multi_lstm - y_pred_inv_multi_lstm)
    std_dev_gru = np.std(y_test_inv_multi_gru - y_pred_inv_multi_gru)
    std_dev_rnn = np.std(y_test_inv_multi_rnn - y_pred_inv_multi_rnn)

    r2_lstm = r2_score(y_test_inv_multi_lstm, y_pred_inv_multi_lstm)
    r2_gru = r2_score(y_test_inv_multi_gru, y_pred_inv_multi_gru)
    r2_rnn = r2_score(y_test_inv_multi_rnn, y_pred_inv_multi_rnn)
   

    # Update the best model for each method
    if rmse_lstm < best_rmse['LSTM']:
        best_rmse['LSTM'] = rmse_lstm
        best_models['LSTM'] = model_multi_lstm

    if rmse_gru < best_rmse['GRU']:
        best_rmse['GRU'] = rmse_gru
        best_models['GRU'] = model_multi_gru

    if rmse_rnn < best_rmse['RNN']:
        best_rmse['RNN'] = rmse_rnn
        best_models['RNN'] = model_multi_rnn
   
    results.append({
        'Method': 'LSTM',
        'Run': run + 1,
        'RMSE': rmse_lstm,
        'MAE': mae_lstm,
        'MAPE': mape_lstm,
        'Bias': bias_lstm,
        'Correlation': corr_lstm,
        'R2': r2_lstm,
        'Std Deviation': std_dev_lstm
    })
     
    results.append({
        'Method': 'GRU',
        'Run': run + 1,
        'RMSE': rmse_gru,
        'MAE': mae_gru,
        'MAPE': mape_gru,
        'Bias': bias_gru,
        'Correlation': corr_gru,
        'R2': r2_gru,
        'Std Deviation': std_dev_gru
    })
     
    results.append({
        'Method': 'RNN',
        'Run': run + 1,
        'RMSE': rmse_rnn,
        'MAE': mae_rnn,
        'MAPE': mape_rnn,
        'Bias': bias_rnn,
        'Correlation': corr_rnn,
        'R2': r2_rnn,
        'Std Deviation': std_dev_rnn
    })


results_df = pd.DataFrame(results)

# Calculate average results
average_results = results_df.groupby('Method').mean().reset_index()
average_results['Run'] = 'Average'

# Combine individual results with average results
all_results = pd.concat([results_df, average_results], ignore_index=True)

# Display a table summarizing the results for all runs
formatted_results = []
for _, result in all_results.iterrows():
    formatted_result = {key: round(value, 4) if isinstance(value, (int, float)) else value for key, value in result.items()}
    formatted_results.append(formatted_result)

table = tabulate(formatted_results, headers='keys', tablefmt='pretty')
print(table)

# Plot actual vs. predicted prices for the best model in each method


plt.figure(figsize=(15, 8))
y_test_inv = MinMaxScaler().fit(btc_data[['Close']]).inverse_transform(y_test_multi.reshape(-1, 1))
plt.plot(date_series[split_multi + seq_length:], y_test_inv, label=f'Actual Price ', linewidth=2)

for method, best_model in best_models.items():
    X_test = np.reshape(X_test_multi, (X_test_multi.shape[0], seq_length, k))
    y_pred = best_model.predict(X_test)

    y_pred_inv = MinMaxScaler().fit(btc_data[['Close']]).inverse_transform(y_pred)
    y_test_inv = MinMaxScaler().fit(btc_data[['Close']]).inverse_transform(y_test_multi.reshape(-1, 1))

    plt.plot(date_series[split_multi + seq_length:], y_pred_inv, label=f'Predicted Price - {method}', linewidth=2)

plt.title('Actual vs. Predicted Prices for Best Models')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

