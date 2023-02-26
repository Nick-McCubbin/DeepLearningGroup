
# import tensorflow as tf
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# data = pd.read_csv('pricing.csv')
# len(data)
# data.head()
# data.isna().sum() #nas present in last row of data
# data = data[:-1] #remove na's from dataset
# data.isna().sum() #na's removed
# len(data)
# data.head()

# #Indexing
# # X1 = data[0]
# # X2 = data[1]
# # X3 = data[2]
# # X4 = data[3]
# # X5 = data[4]
# # X = np.array(np.column_stack((X1,X2,X3,X4,X5)))
# # Y = data[5]

# X1 = data['sku']
# X2 = data['price']
# X3 = data['order']
# X4 = data['duration']
# X5 = data['category']
# X = np.array(np.column_stack((X1,X2,X3,X4,X5)))
# Y = data['quantity']

# ## Specifing Architecture
# inputs = tf.keras.layers.Input(shape=(X.shape[1],), name = 'input')
# hidden1 = tf.keras.layers.Dense(units = 5, activation = 'sigmoid', name = 'hidden1')(inputs)
# hidden2 = tf.keras.layers.Dense(units = 5, activation = 'sigmoid', name = 'hidden2')(hidden1)
# hidden3 = tf.keras.layers.Dense(units = 5, activation = 'sigmoid', name = 'hidden3')(hidden2)
# output = tf.keras.layers.Dense(units = 1, activation = 'linear', name = 'output')(hidden3)
# model = tf.keras.Model(inputs = inputs, outputs = output)
# model.compile(loss = 'mse', optimizer = tf.keras.optimizers.SGD(learning_rate = 0.001))
# history = model.fit(x = X, y = Y, batch_size = 1, epochs = 10)


# ## Plotting the Loss Curve
# plt.plot(history.history['loss'])
# plt.title("Learning Curve")
# plt.xlabel("Iterations")
# plt.ylabel("Loss")
# plt.show()


## Attempt 2

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('pricing.csv')
len(data)
data.head()
data.isna().sum() #nas present in last row of data
data = data[:-1] #remove na's from dataset
data.isna().sum() #na's removed
len(data)
data.head()

X = data[['sku', 'price', 'order', 'duration', 'category']]
Y = data['quantity']

## Specifing Architecture
inputs = tf.keras.layers.Input(shape=(X.shape[1],), name = 'input')
hidden1 = tf.keras.layers.Dense(units = 5, activation = 'sigmoid', name = 'hidden1')(inputs)
hidden2 = tf.keras.layers.Dense(units = 5, activation = 'sigmoid', name = 'hidden2')(hidden1)
hidden3 = tf.keras.layers.Dense(units = 5, activation = 'sigmoid', name = 'hidden3')(hidden2)
output = tf.keras.layers.Dense(units = 1, activation = 'linear', name = 'output')(hidden3)
model = tf.keras.Model(inputs = inputs, outputs = output)
model.compile(loss = 'mse', optimizer = tf.keras.optimizers.SGD(learning_rate = 0.001))

## Incremental learning loop
history = {'loss': []}
for row in X.itertuples():
    x_i = np.array(row[1:]).reshape(1, -1)
    y_i = np.array(Y[row.Index]).reshape(1, -1)
    loss = model.train_on_batch(x_i, y_i)
    history['loss'].append(loss)

## Plotting the Loss Curve
plt.plot(history['loss'])
plt.title("Learning Curve")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.show()


