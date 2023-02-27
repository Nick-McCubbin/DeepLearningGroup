import numpy as np
import pandas as pd
import tensorflow as tf
import datetime
import time
import psutil
import csv

data = pd.read_csv("pricing.csv")
data = data[:-1]
data = data.sample(n = 100)

dprice = data['price'].max() - data['price'].min()
dorder = data['order'].max() - data['order'].min()
dduration = data['duration'].max() - data['duration'].min()
dquantity = data['quantity'].max() - data['quantity'].min()

#normalize the variables
diffs = {'price': dprice, 'order': dorder, 'duration': dduration, 'quantity': dquantity}
for col, diffs in diffs.items():
    data[col] = data[col]/diffs
data.head()

# v4 = pd.get_dummies(data, columns=['category'])
# data['category'] = v4.to_numpy() 

#model architecture
inputs = tf.keras.layers.Input(shape=(3,), name='input')
hidden1 = tf.keras.layers.Dense(units=5, activation="sigmoid", name='hidden1')(inputs)
hidden2 = tf.keras.layers.Dense(units=3, activation="sigmoid", name='hidden2')(hidden1)
hidden3 = tf.keras.layers.Dense(units=2, activation="sigmoid", name='hidden3')(hidden2)
output = tf.keras.layers.Dense(units=1, activation="linear", name='output')(hidden3)
model = tf.keras.Model(inputs=inputs, outputs=output)

# Compile the model
model.compile(loss='mse', optimizer=tf.keras.optimizers.SGD(learning_rate=0.01))

time_start = datetime.datetime.now()
ram_usage = [] #initialize ram usage
with open('loss.csv', mode='w') as loss_file:
    loss_writer = csv.writer(loss_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    loss_writer.writerow(['loss'])
    #read in rows 1 at a time
    for i, row in data.iterrows():
        X = np.array(row[['price', 'order', 'duration']]).reshape(1,3) 
        y = np.array(row['quantity']).reshape(1, 1)
        history = model.fit(x = X, y = y, batch_size = 1, epochs = 1)
        loss_value = history.history['loss'][0]
        loss_writer.writerow([loss_value])
        process = psutil.Process()
        ram_usage.append(process.memory_info().rss)
print(datetime.datetime.now() - time_start)
