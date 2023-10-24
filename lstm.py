import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

df = pd.read_csv('BTC-USD.csv')

df.fillna(method='ffill', inplace=True)

data = df[['Close']].values.astype(float)

scaler = MinMaxScaler()
data = scaler.fit_transform(data)

train_size = int(len(data) * 0.67)
train, test = data[0:train_size], data[train_size:len(data)]

def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data)-seq_length):
        sequences.append(data[i:i+seq_length])
    return np.array(sequences)

seq_length = 10
X_train = create_sequences(train, seq_length)
y_train = train[seq_length:]
X_test = create_sequences(test, seq_length)
y_test = test[seq_length:]

model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(seq_length, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X_train, y_train, epochs=10, batch_size=32)

train_preds = model.predict(X_train)
test_preds = model.predict(X_test)

train_preds = scaler.inverse_transform(train_preds)
test_preds = scaler.inverse_transform(test_preds)
y_train = scaler.inverse_transform(y_train)
y_test = scaler.inverse_transform(y_test)

train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))

print(f'Training RMSE: {train_rmse}')
print(f'Testing RMSE: {test_rmse}')

plt.figure(figsize=(12, 6))
plt.plot(y_test, label='Actual Prices')
plt.plot(test_preds, label='Predicted Prices')
plt.title('Bitcoin Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
