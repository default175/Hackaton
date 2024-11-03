# Импортируем необходимые библиотеки
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Загрузка данных
data = pd.read_csv('C:/Users/Demo/Desktop/bus_bts.csv')  # Замените на путь к вашим данным

# Предварительная обработка данных
data = data.dropna()  # Удаляем пропуски

# Преобразуем дату в формат datetime
data['geton_date'] = pd.to_datetime(data['geton_date'])

# Используем только нужные столбцы для анализа
data = data[['geton_date', 'user_count']]

# Устанавливаем дату как индекс
data.set_index('geton_date', inplace=True)

# Нормализация данных
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data[['user_count']])

# Разделение на тренировочный и тестовый наборы
train_size = int(len(data_scaled) * 0.8)
train, test = data_scaled[0:train_size], data_scaled[train_size:]

# Создание последовательностей для LSTM
def create_sequences(data, seq_length):
    x, y = [], []
    for i in range(len(data) - seq_length):
        x.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(x), np.array(y)

seq_length = 10  # Длина последовательности
X_train, y_train = create_sequences(train, seq_length)
X_test, y_test = create_sequences(test, seq_length)

# Изменение формы для LSTM
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Базовая модель линейной регрессии
model_lr = LinearRegression()
X_train_lr = np.arange(len(y_train)).reshape(-1, 1)
X_test_lr = np.arange(len(y_test)).reshape(-1, 1)
model_lr.fit(X_train_lr, y_train)
pred_lr = model_lr.predict(X_test_lr)

# Построение LSTM модели
model_lstm = Sequential()
model_lstm.add(LSTM(50, activation='relu', input_shape=(X_train.shape[1], 1)))
model_lstm.add(Dense(1))
model_lstm.compile(optimizer='adam', loss='mse')

# Обучение модели
model_lstm.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Прогнозирование на тестовых данных
pred_lstm = model_lstm.predict(X_test)

# Оценка модели
# Переводим данные обратно к исходной шкале
pred_lstm = scaler.inverse_transform(pred_lstm)
y_test = scaler.inverse_transform(y_test)

mae = mean_absolute_error(y_test, pred_lstm)
rmse = np.sqrt(mean_squared_error(y_test, pred_lstm))

print(f"MAE (LSTM): {mae}")
print(f"RMSE (LSTM): {rmse}")

# Оценка линейной регрессии
pred_lr = scaler.inverse_transform(pred_lr.reshape(-1, 1))
mae_lr = mean_absolute_error(y_test, pred_lr)
rmse_lr = np.sqrt(mean_squared_error(y_test, pred_lr))

print(f"MAE (Linear Regression): {mae_lr}")
print(f"RMSE (Linear Regression): {rmse_lr}")

# Построение графиков
plt.figure(figsize=(14, 7))
plt.plot(y_test, label='Real')
plt.plot(pred_lstm, label='LSTM Prediction')
plt.title('LSTM Prediction vs Real Data')
plt.legend()
plt.show()

plt.figure(figsize=(14, 7))
plt.plot(y_test, label='Real')
plt.plot(pred_lr, label='Linear Regression Prediction')
plt.title('Linear Regression Prediction vs Real Data')
plt.legend()
plt.show()
