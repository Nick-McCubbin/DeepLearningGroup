import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import datetime
import time

data = pd.read_csv("pricing.csv")
data = data[:-1]
data = data.sample(n = 5000)

dprice = data['price'].max() - data['price'].min()
dorder = data['order'].max() - data['order'].min()
dduration = data['duration'].max() - data['duration'].min()
dquantity = data['quantity'].max() - data['quantity'].min()

#dummies for category
# data = pd.get_dummies(data, columns=['category'])
# data['category'] = v4.to_numpy() 

#normalize the variables
diffs = {'price': dprice, 'order': dorder, 'duration': dduration, 'quantity': dquantity}
for col, diffs in diffs.items():
    data[col] = data[col]/diffs
data.head()

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
loss_history = []
#read in rows 1 at a time
for i, row in data.iterrows():
    X = np.array(row[['price', 'order', 'duration']]).reshape(1,3)
    y = np.array(row['quantity']).reshape(1, 1)
    history = model.fit(x = X, y = y, batch_size = 1, epochs = 2)
    loss_history.append(history.history['loss'])
print(datetime.datetime.now() - time_start)

plt.plot(loss_history)
plt.show()

test = pd.read_csv("pricing_test.csv")
test.columns =['sku', 'price', 'quantity', 'order', 'duration', 'category']

diffs = {'price': dprice, 'order': dorder, 'duration': dduration, 'quantity': dquantity}
for col, diffs in diffs.items():
    test[col] = test[col]/diffs
test.head()

X_test = test[['price', 'order', 'duration']].values
y_test = test['quantity'].values
yhat = model.predict(x=X_test)
yhat
model.evaluate(X,y)
model.summary()
correlation = np.corrcoef(yhat.flatten(), y_test)[0, 1]
correlation
plt.plot(yhat)
plt.show()
