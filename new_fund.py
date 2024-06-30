import pandas as pd
import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, LSTM # type: ignore
import matplotlib.pyplot as plt

# Yatırım fonu sembolü
fund_symbol = "TSLA"
data = yf.download(fund_symbol, start="2015-01-01", end="2024-06-25")
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
model.fit(X_train, y_train, batch_size=1, epochs=10)  # Eğitim dönemini 10'a çıkardık

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

# Gelecek 1 yılı tahmin etme
future_predictions = []
last_60_days = scaled_data[-60:]

for i in range(365):  # 1 yıl = 365 gün
    X_future = last_60_days[-60:]
    X_future = np.reshape(X_future, (1, X_future.shape[0], 1))
    future_pred = model.predict(X_future)
    future_predictions.append(future_pred[0, 0])
    last_60_days = np.append(last_60_days, future_pred, axis=0)
    last_60_days = last_60_days[1:]

# Tahminleri orijinal ölçeğe geri dönüştürme
future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# Gelecek yılın tarihlerini oluşturma
last_date = data.index[-1]
future_dates = pd.date_range(last_date, periods=30, freq='D')

# Gelecek tahminlerini görselleştirme
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date')
plt.ylabel('Close Price USD ($)')
# plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.plot(future_dates, future_predictions, color='orange', label='Future Predictions')
plt.legend(['Train', 'Val', 'Predictions', 'Future Predictions'], loc='lower right')
plt.show()
