import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder


data = pd.read_csv('pricing.csv')


X = data.drop('quantity', axis=1)
y = data['quantity']

scaler = StandardScaler()
X = scaler.fit_transform(X)


encoder = OneHotEncoder()
X = encoder.fit_transform(X).toarray()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

import tensorflow as tf


model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(64, activation='sigmoid', input_shape=(X_train.shape[1],)))
model.add(tf.keras.layers.Dense(32, activation='sigmoid'))
model.add(tf.keras.layers.Dense(16, activation='sigmoid'))
model.add(tf.keras.layers.Dense(1))


model.compile(optimizer='adam', loss='mean_squared_error')


batch_size = 32

for i in range(0, len(X_train), batch_size):
    X_batch = X_train[i:i+batch_size]
    y_batch = y_train[i:i+batch_size]
    model.fit(X_batch, y_batch, epochs=1, verbose=0)



import matplotlib.pyplot as plt


mse_history = []
for i in range(0, len(X_train), batch_size):
    X_batch = X_train[i:i+batch_size]
    y_batch = y_train[i:i+batch_size]
    mse = model.evaluate(X_batch, y_batch, verbose=0)
    mse_history.append(mse)

mse_ma = np.convolve(mse_history, np.ones(100)/100, mode='valid')


plt.plot(range(len(mse_ma)), mse_ma)
plt.xlabel('Number of instances learned')
