import pandas as pd
import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import matplotlib.pyplot as plt

# Yatırım fonu sembolü
fund_symbol = "AAPL"
data = yf.download(fund_symbol, start="2020-01-01", end="2024-06-25")
data = data.dropna()
data = data[['Close']]

# Veriyi ölçeklendirme
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data)

# Eğitim ve test verisi oluşturma
training_data_len = int(np.ceil(len(scaled_data) * 0.8))
train_data = scaled_data[0:int(training_data_len), :]

# X_train ve y_train veri setlerini oluşturma
X_train = []
y_train = []

for i in range(60, len(train_data)):
    X_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])

# Dizi ve tensörlere dönüştürme
X_train, y_train = np.array(X_train), np.array(y_train)

# Veriyi yeniden şekillendirme (LSTM girişi için 3-boyutlu veri gereklidir)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# LSTM modelini oluşturma
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=25))
model.add(Dense(units=1))

# Modeli derleme
model.compile(optimizer='adam', loss='mean_squared_error')

# Modeli eğitme
model.fit(X_train, y_train, batch_size=1, epochs=1)

# Test verisi oluşturma
test_data = scaled_data[training_data_len - 60:, :]

# X_test ve y_test veri setlerini oluşturma
X_test = []
y_test = data.iloc[training_data_len:, :].values  # DataFrame'den numpy dizisine dönüştürme

for i in range(60, len(test_data)):
    X_test.append(test_data[i-60:i, 0])

# Veriyi yeniden şekillendirme
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Tahminler yapma
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)

# RMSE hesaplama
rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
print(f"RMSE: {rmse}")

# Tahminleri görselleştirme
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions

# plt.figure(figsize=(16,8))
# plt.title('Model')
# plt.xlabel('Tarih')
# plt.ylabel('Kapanış Fiyatı USD')
# plt.plot(train['Close'])
# plt.plot(valid[['Close', 'Predictions']])
# plt.legend(['Eğitim', 'Gerçek', 'Tahminler'], loc='lower right')
# plt.show()

# Generate future dates for prediction (next 30 days)
future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=30, freq='D')

# Scale the test data including the future dates
future_test_data = np.append(test_data, scaled_data[-60:, :])

# Reshape future_test_data to maintain its 3-dimensional structure
future_test_data = np.reshape(future_test_data, (-1, 1))

X_future = []

for i in range(60, len(future_test_data)):
    X_future.append(future_test_data[i-60:i, 0])

X_future = np.array(X_future)
X_future = np.reshape(X_future, (X_future.shape[0], X_future.shape[1], 1))

# Make predictions for the future dates
future_predictions = model.predict(X_future)
future_predictions = scaler.inverse_transform(future_predictions)

# Slice future predictions to match future_dates length
future_predictions = future_predictions[:30]

future_predictions_updated = []


data_once = yf.download(fund_symbol, start="2024-06-25", end="2024-06-26")
value = data_once.iloc[0, 0]  # Assuming it's the first row and first column

first = future_predictions[0]
print(value)
print(first[0])
result = abs(value-first[0])
for item in future_predictions:
    future_predictions_updated.append(item+result) 

# Flatten future_predictions if needed
# future_predictions = future_predictions.flatten()

# Visualize predictions including future dates
plt.figure(figsize=(24,8))
plt.title('Model')
plt.xlabel('Date')
plt.ylabel('Closing Price (USD)')
# plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.plot(future_dates, future_predictions_updated, linestyle='dashed', color='b')
plt.legend(['Training', 'Actual', 'Predictions', 'Future Predictions'], loc='lower right')
plt.show()



