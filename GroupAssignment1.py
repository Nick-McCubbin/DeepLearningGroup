import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("C:/Users/weste/Desktop/MSBA/Spring 2023/BZAN 554 - Deep Learning/pricing.csv")
len(data)
data.head()
data.isna().sum() #nas present in last row of data
data = data[:-1] #remove na's from dataset
data.isna().sum() #na's removed

len(data)
data.head()

X1 = data['sku']
X2 = data['price']
X3 = data['order']
X4 = data['duration']
X5 = data['category']
X = np.array(np.column_stack((X1,X2,X3,X4,X5)))
Y = data['quantity']


#specify architecture
inputs = tf.keras.layers.Input(shape=(X.shape[1],), name = 'input')
hidden1 = tf.keras.layers.Dense(units = 5, activation = 'sigmoid', name = 'hidden1')(inputs)
hidden2 = tf.keras.layers.Dense(units = 5, activation = 'sigmoid', name = 'hidden2')(hidden1)
hidden3 = tf.keras.layers.Dense(units = 5, activation = 'sigmoid', name = 'hidden3')(hidden2)
output = tf.keras.layers.Dense(units = 1, activation = 'linear', name = 'output')(hidden3)

model = tf.keras.Model(inputs = inputs, outputs = output)

# model = tf.keras.models.Sequential([tf.keras.layers.Input(shape=(X.shape[1],), name = 'input'), tf.keras.layers.Dense(units = 5, activation = 'sigmoid', name = 'hidden1'), tf.keras.layers.Dense(units = 5, activation = 'sigmoid', name = 'hidden2'), tf.keras.layers.Dense(units = 5, activation = 'sigmoid', name = 'hidden3'), tf.keras.layers.Dense(units = 1, activation = 'linear', name = 'output')])

model.compile(loss = 'mse', optimizer = tf.keras.optimizers.SGD(learning_rate = 0.001))

history = model.fit(x = X, y = Y, batch_size = 5000, epochs = 10)

# for i in data.iterrows():
#     X1 = data['sku']
#     X2 = data['price']
#     X3 = data['order']
#     X4 = data['duration']
#     X5 = data['category']
#     X = np.array(np.column_stack((X1,X2,X3,X4,X5)))
#     Y = data['quantity']
#     model = tf.keras.Model(inputs = inputs, outputs = output)
#     model.compile(loss = 'mse', optimizer = tf.keras.optimizers.SGD(learning_rate = 0.001))
#     history = model.fit(x= X, y = Y, batch_size = 100, epochs = 2)


plt.plot(history.history['loss'])
plt.show()

