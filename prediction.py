import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tabulate import tabulate
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, SimpleRNN , Conv1D, MaxPooling1D, Bidirectional, Dense, BatchNormalization, Dropout
from scipy.stats import pearsonr
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping , LearningRateScheduler




btc_data = pd.read_csv('BTC-USD.csv', index_col='Date', parse_dates=True)

btc_data['Open_scaled'] = MinMaxScaler().fit_transform(btc_data[['Open']])
btc_data['Volume_scaled'] = MinMaxScaler().fit_transform(btc_data[['Volume']])

date_series = btc_data.index
btc_data['Date_sin'] = np.sin(2 * np.pi * date_series.dayofyear / 365)
btc_data['Date_cos'] = np.cos(2 * np.pi * date_series.dayofyear / 365)

features = ['Close', 'Open_scaled', 'Volume_scaled', 'Date_sin', 'Date_cos']
data_scaled = MinMaxScaler().fit_transform(btc_data[features])

def create_sequences_multi(data, seq_length):
    sequences, targets = [], []
    for i in range(len(data) - seq_length):
        seq = data[i : i + seq_length]
        target = data[i + seq_length, 0]  
        sequences.append(seq)
        targets.append(target)
    return np.array(sequences), np.array(targets)

def lr_schedule(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * 0.9 
    

seq_length = 10  
epochs = 150
batch_size = 64

sequences_multi, targets_multi = create_sequences_multi(data_scaled, seq_length)

split_multi = int(0.8 * len(sequences_multi))
X_train_multi, y_train_multi = sequences_multi[:split_multi], targets_multi[:split_multi]
X_test_multi, y_test_multi = sequences_multi[split_multi:], targets_multi[split_multi:]

X_train_multi = np.reshape(X_train_multi, (X_train_multi.shape[0], seq_length, len(features)))
X_test_multi = np.reshape(X_test_multi, (X_test_multi.shape[0], seq_length, len(features)))

model_multi_lstm = Sequential()

model_multi_lstm.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(seq_length, len(features))))
model_multi_lstm.add(MaxPooling1D(pool_size=2))

model_multi_lstm.add(Bidirectional(LSTM(50, return_sequences=True)))
model_multi_lstm.add(BatchNormalization())
model_multi_lstm.add(Dropout(0.2)) 
model_multi_lstm.add(Bidirectional(LSTM(50, return_sequences=True)))
model_multi_lstm.add(BatchNormalization())
model_multi_lstm.add(Dropout(0.2))  
model_multi_lstm.add(Bidirectional(LSTM(50)))
model_multi_lstm.add(BatchNormalization())
model_multi_lstm.add(Dropout(0.2)) 
model_multi_lstm.add(Dense(1))
optimizer_lstm = Adam(learning_rate=0.01)
model_multi_lstm.compile(optimizer=optimizer_lstm, loss='mean_squared_error')
lr_scheduler_lstm = LearningRateScheduler(lr_schedule)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)






model_multi_gru = Sequential()
model_multi_gru.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(seq_length, len(features))))
model_multi_gru.add(Bidirectional(GRU(50, input_shape=(seq_length, len(features)), return_sequences=True)))
model_multi_lstm.add(BatchNormalization())  
model_multi_gru.add(Bidirectional(GRU(50, return_sequences=True)))
model_multi_lstm.add(BatchNormalization())  
model_multi_gru.add(Bidirectional(GRU(50)))
model_multi_lstm.add(BatchNormalization())  
model_multi_gru.add(Dense(1))
optimizer_gru = Adam(learning_rate=0.01)  
model_multi_gru.compile(optimizer=optimizer_gru, loss='mean_squared_error')





model_multi_rnn = Sequential()
model_multi_rnn.add(SimpleRNN(50, input_shape=(seq_length, len(features)), return_sequences=True))
model_multi_rnn.add(SimpleRNN(50, return_sequences=True))
model_multi_rnn.add(SimpleRNN(50))
model_multi_rnn.add(Dense(1))
optimizer_rnn = Adam(learning_rate=0.01)  
model_multi_rnn.compile(optimizer=optimizer_rnn, loss='mean_squared_error')



history_multi_lstm = model_multi_lstm.fit(X_train_multi, y_train_multi, epochs=epochs, batch_size=batch_size, validation_data=(X_test_multi, y_test_multi), verbose=2)


history_multi_gru = model_multi_gru.fit(X_train_multi, y_train_multi, epochs=epochs, batch_size=batch_size, validation_data=(X_test_multi, y_test_multi), verbose=2)

history_multi_rnn = model_multi_rnn.fit(X_train_multi, y_train_multi, epochs=epochs, batch_size=batch_size, validation_data=(X_test_multi, y_test_multi), verbose=2)

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





table = [
    ['LSTM', rmse_lstm, mae_lstm, mape_lstm, bias_lstm, corr_lstm, std_dev_lstm],
    ['GRU', rmse_gru, mae_gru, mape_gru, bias_gru, corr_gru, std_dev_gru],
    ['SimpleRNN', rmse_rnn, mae_rnn, mape_rnn, bias_rnn, corr_rnn, std_dev_rnn]
]


print(tabulate(table, headers=['Method', 'RMSE', 'MAE', 'MAPE', 'Bias', 'Correlation', 'Std Deviation'], tablefmt='pretty'))

plt.subplot(3, 1, 1)
plt.plot(btc_data.index[-len(y_test_inv_multi_lstm):], y_test_inv_multi_lstm, label='Actual Prices', color='blue')
plt.plot(btc_data.index[-len(y_pred_inv_multi_lstm):], y_pred_inv_multi_lstm, label='Predicted Prices (LSTM)', color='orange')
plt.title('Actual and Predicted Prices (LSTM)')
plt.xlabel('Date')
plt.ylabel('Bitcoin Price')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(btc_data.index[-len(y_test_inv_multi_gru):], y_test_inv_multi_gru, label='Actual Prices', color='blue')
plt.plot(btc_data.index[-len(y_pred_inv_multi_gru):], y_pred_inv_multi_gru, label='Predicted Prices (GRU)', color='green')
plt.title('Actual and Predicted Prices (GRU)')
plt.xlabel('Date')
plt.ylabel('Bitcoin Price')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(btc_data.index[-len(y_test_inv_multi_rnn):], y_test_inv_multi_rnn, label='Actual Prices', color='blue')
plt.plot(btc_data.index[-len(y_pred_inv_multi_rnn):], y_pred_inv_multi_rnn, label='Predicted Prices (SimpleRNN)', color='red')
plt.title('Actual and Predicted Prices (SimpleRNN)')
plt.xlabel('Date')
plt.ylabel('Bitcoin Price')
plt.legend()

plt.tight_layout()
plt.show()