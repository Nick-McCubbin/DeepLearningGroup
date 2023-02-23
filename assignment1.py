import numpy as np
import pandas as pd
import tensorflow as tf

data = pd.read_csv("pricing.csv")
#input_shape = (5,)

# Define the model
inputs = tf.keras.layers.Input(shape=(5,), name='input')
hidden1 = tf.keras.layers.Dense(units=5, activation="sigmoid", name = 'hidden1')(inputs)
hidden2 = tf.keras.layers.Dense(units=5, activation="sigmoid", name= 'hidden2')(hidden1)
hidden3 = tf.keras.layers.Dense(units=5, activation="sigmoid", name= 'hidden3')(hidden2)
output = tf.keras.layers.Dense(units=1, activation = "linear", name= 'output')(hidden3)

model = tf.keras.Model(inputs = inputs, outputs = output)

model.compile(loss = 'mse', optimizer = tf.keras.optimizers.SGD(learning_rate = 0.001))

#Iterate over all rows in the dataframe to incrementally train the model
for i, row in data.iterrows():
    sku = row['sku']
    price = row['price']
    order = row['order']
    duration = row['duration']
    category = row['category']
    quantity = row['quantity']
    X = [np.column_stack([[sku, price, order, duration, category]])]
    y = [quantity]
    model.train_on_batch(X, y)
