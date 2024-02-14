import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tabulate import tabulate
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, SimpleRNN, Conv1D, MaxPooling1D, Bidirectional, Dense, BatchNormalization, Dropout, Layer
from scipy.stats import pearsonr
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.feature_selection import mutual_info_regression, SelectKBest
from tensorflow.keras.regularizers import l2
import matplotlib.dates as mdates

btc_data = pd.read_csv('BTC-USD.csv', index_col='Date', parse_dates=True)

date_series = btc_data.index
btc_data['Date_sin'] = np.sin(2 * np.pi * date_series.dayofyear / 365)
btc_data['Date_cos'] = np.cos(2 * np.pi * date_series.dayofyear / 365)

btc_data.drop(["Adj Close", "Volume"], inplace=True, axis=1)
k = 5

data_scaled = MinMaxScaler().fit_transform(btc_data)

selector = SelectKBest(mutual_info_regression, k=k)

X_new = selector.fit_transform(data_scaled[:, :k], data_scaled[:, 3])

selected_features = X_new.reshape(-1, k)

seq_length = 30  # Adjusted to predict the next 30 days
epochs = 150
batch_size = 16

def create_sequences_multi(data, seq_length):
    sequences, targets = [], []
    for i in range(len(data) - seq_length):
        seq = data[i: i + seq_length]
        target = data[i + seq_length, 0]
        sequences.append(seq)
        targets.append(target)
    return np.array(sequences), np.array(targets)

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


sequences_multi, targets_multi = create_sequences_multi(selected_features, seq_length)
split_multi = int(0.8 * len(sequences_multi))
X_train_multi, y_train_multi = sequences_multi[:split_multi], targets_multi[:split_multi]
X_test_multi, y_test_multi = sequences_multi[split_multi:], targets_multi[split_multi:]

model_multi_lstm = Sequential()
model_multi_lstm.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(seq_length, k)))
model_multi_lstm.add(MaxPooling1D(pool_size=2))
model_multi_lstm.add(Bidirectional(LSTM(50, input_shape=(seq_length, k), return_sequences=True)))
model_multi_lstm.add(CentralizedWeightLayer())
model_multi_lstm.add(BatchNormalization())
model_multi_lstm.add(Dropout(0.1))
model_multi_lstm.add(Bidirectional(LSTM(50, return_sequences=True)))
model_multi_lstm.add(CentralizedWeightLayer())
model_multi_lstm.add(BatchNormalization())
model_multi_lstm.add(Dropout(0.1))
model_multi_lstm.add(Bidirectional(LSTM(50)))
model_multi_lstm.add(CentralizedWeightLayer())
model_multi_lstm.add(BatchNormalization())
model_multi_lstm.add(Dropout(0.1))
model_multi_lstm.add(Dense(1, kernel_regularizer=l2(0.01)))
optimizer_lstm = Adam(learning_rate=0.001)
model_multi_lstm.compile(optimizer=optimizer_lstm, loss='mean_squared_error')
early_stopping_lstm = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

model_multi_gru = Sequential()
model_multi_gru.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(seq_length, k)))
model_multi_gru.add(Bidirectional(GRU(50, input_shape=(seq_length,k), return_sequences=True)))
model_multi_gru.add(BatchNormalization())
model_multi_gru.add(CentralizedWeightLayer()) 
model_multi_gru.add(Dropout(0.1))    
model_multi_gru.add(Bidirectional(GRU(50, return_sequences=True)))
model_multi_gru.add(BatchNormalization())  
model_multi_gru.add(CentralizedWeightLayer()) 
model_multi_gru.add(Dropout(0.1))    
model_multi_gru.add(Bidirectional(GRU(50)))
model_multi_gru.add(BatchNormalization())
model_multi_gru.add(CentralizedWeightLayer()) 
model_multi_gru.add(Dropout(0.1))    
model_multi_gru.add(Dense(1,kernel_regularizer=l2(0.01)))
optimizer_gru = Adam(learning_rate=0.001)
model_multi_gru.compile(optimizer=optimizer_gru, loss='mean_squared_error')
early_stopping_gru = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)


model_multi_rnn = Sequential()
model_multi_rnn.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(seq_length, k)))
model_multi_rnn.add(SimpleRNN(50, input_shape=(seq_length,k), return_sequences=True) )
model_multi_rnn.add(BatchNormalization())
model_multi_rnn.add(CentralizedWeightLayer()) 
model_multi_rnn.add(SimpleRNN(50, return_sequences=True))
model_multi_rnn.add(BatchNormalization())
model_multi_rnn.add(CentralizedWeightLayer()) 
model_multi_rnn.add(SimpleRNN(50))
model_multi_rnn.add(Dense(1,kernel_regularizer=l2(0.01)))
optimizer_rnn = Adam(learning_rate=0.001)  
model_multi_rnn.compile(optimizer=optimizer_rnn, loss='mean_squared_error')

history_multi_lstm = model_multi_lstm.fit(X_train_multi, y_train_multi, epochs=epochs, batch_size=batch_size, validation_data=(X_test_multi, y_test_multi), verbose=2, callbacks=[early_stopping_lstm])
history_multi_gru = model_multi_gru.fit(X_train_multi, y_train_multi, epochs=epochs, batch_size=batch_size, validation_data=(X_test_multi, y_test_multi), verbose=2, callbacks=[early_stopping_gru])
history_multi_rnn = model_multi_rnn.fit(X_train_multi, y_train_multi, epochs=epochs, batch_size=batch_size, validation_data=(X_test_multi, y_test_multi), verbose=2)

# Predicting the next 30 days
future_dates = pd.date_range(btc_data.index[-1] + pd.Timedelta(days=1), periods=30, freq='D')
future_features = selected_features[-seq_length:]
future_sequences = np.array([future_features[i:i+seq_length] for i in range(len(future_features)-seq_length+1)])

future_pred_multi_lstm = model_multi_lstm.predict(future_sequences)
future_pred_inv_multi_lstm = MinMaxScaler().fit(btc_data[['Close']]).inverse_transform(future_pred_multi_lstm.reshape(-1, 1))

future_pred_multi_gru = model_multi_gru.predict(future_sequences)
future_pred_inv_multi_gru = MinMaxScaler().fit(btc_data[['Close']]).inverse_transform(future_pred_multi_gru.reshape(-1, 1))

future_pred_multi_rnn = model_multi_rnn.predict(future_sequences)
future_pred_inv_multi_rnn = MinMaxScaler().fit(btc_data[['Close']]).inverse_transform(future_pred_multi_rnn.reshape(-1, 1))

# Evaluation metrics for the last 30 days
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

table = [
    ['LSTM', round(rmse_lstm, 4), round(mae_lstm, 4), round(mape_lstm, 4), round(bias_lstm, 4), round(corr_lstm, 4), round(std_dev_lstm, 4), round(r2_lstm, 4)],
    ['GRU', round(rmse_gru, 4), round(mae_gru, 4), round(mape_gru, 4), round(bias_gru, 4), round(corr_gru, 4), round(std_dev_gru, 4), round(r2_gru, 4)],
    ['SimpleRNN', round(rmse_rnn, 4), round(mae_rnn, 4), round(mape_rnn, 4), round(bias_rnn, 4), round(corr_rnn, 4), round(std_dev_rnn, 4), round(r2_rnn, 4)]
]
print(tabulate(table, headers=['Method', 'RMSE', 'MAE', 'MAPE', 'Bias', 'Correlation', 'Std Deviation' , 'R2'], tablefmt='pretty'))



# Assuming future_pred_inv_multi_lstm, future_pred_inv_multi_gru, and future_pred_inv_multi_rnn are 1D arrays
last_date = future_dates[-1]
last_index = len(future_pred_inv_multi_lstm) - 1

price_lstm = future_pred_inv_multi_lstm[last_index][0]
price_gru = future_pred_inv_multi_gru[last_index][0]
price_rnn = future_pred_inv_multi_rnn[last_index][0]

print(f"Date: {last_date}, LSTM Prediction: {price_lstm}, GRU Prediction: {price_gru}, SimpleRNN Prediction: {price_rnn}")


plt.subplot(3, 1, 1)
plt.plot(btc_data.index[-len(y_test_inv_multi_lstm):], y_test_inv_multi_lstm, label='Actual Prices', color='black', linestyle='-', linewidth=1)
plt.plot(btc_data.index[-len(y_pred_inv_multi_lstm):], y_pred_inv_multi_lstm, label='Predicted Prices (LSTM)', color='black', linestyle='-', linewidth=2)
plt.title('Actual and Predicted Prices (LSTM)')
plt.xlabel('Date')
plt.ylabel('Bitcoin Price')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(btc_data.index[-len(y_test_inv_multi_gru):], y_test_inv_multi_gru, label='Actual Prices', color='black', linestyle='-', linewidth=1)
plt.plot(btc_data.index[-len(y_pred_inv_multi_gru):], y_pred_inv_multi_gru, label='Predicted Prices (GRU)', color='black', linestyle='-', linewidth=2)
plt.title('Actual and Predicted Prices (GRU)')
plt.xlabel('Date')
plt.ylabel('Bitcoin Price')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(btc_data.index[-len(y_test_inv_multi_rnn):], y_test_inv_multi_rnn, label='Actual Prices', color='black', linestyle='-', linewidth=1)
plt.plot(btc_data.index[-len(y_pred_inv_multi_rnn):], y_pred_inv_multi_rnn, label='Predicted Prices (SimpleRNN)', color='black', linestyle='-', linewidth=2)
plt.title('Actual and Predicted Prices (SimpleRNN)')
plt.xlabel('Date')
plt.ylabel('Bitcoin Price')
plt.legend()

plt.show()


