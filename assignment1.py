import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import datetime
import time
import psutil

data = pd.read_csv("pricing.csv")
data = data[:-1]
# data = data.sample(n = 1000)

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
model.compile(loss='mse', optimizer=tf.keras.optimizers.SGD(learning_rate=0.001))

time_start = datetime.datetime.now()
loss_history = [] #initialize loss
ram_usage = [] #initialize ram usage
#read in rows 1 at a time
for i, row in data.iterrows():
    X = np.array(row[['price', 'order', 'duration']]).reshape(1,3) 
    y = np.array(row['quantity']).reshape(1, 1)
    history = model.fit(x = X, y = y, batch_size = 1, epochs = 1)
    loss_history.append(history.history['loss'])
    process = psutil.Process()
    ram_usage.append(process.memory_info().rss)
print(datetime.datetime.now() - time_start)

plt.plot(loss_history)
plt.xlabel('Number of instances learned')
plt.ylabel('Loss')
plt.show()

plt.plot(ram_usage)
plt.xlabel('Number of instances learned')
plt.ylabel('RAM usage (bytes)')
plt.show()

test = pd.read_csv("pricing_test.csv")
test.columns =['sku', 'price', 'quantity', 'order', 'duration', 'category']

diffs = {'price': dprice, 'order': dorder, 'duration': dduration, 'quantity': dquantity}
for col, diffs in diffs.items():
    test[col] = test[col]/diffs
test.head()
# v4_test = pd.get_dummies(test, columns=['category'])
# test['category'] = v4_test.to_numpy() 

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

##################################################################################

#variable importance plot
weights = model.get_layer('hidden1').get_weights()[0]

importance = np.abs(weights).sum(axis=1)

importance /= importance.sum()

# create a bar plot of variable importance
plt.bar(['price', 'order', 'duration'], importance)
plt.xlabel('Variable')
plt.ylabel('Importance')
plt.title('Variable Importance Plot')
plt.show()

##################################################################################

#partial dependence plots

#price
price_range = np.linspace(test['price'].min(), test['price'].max(), num=50)
order_median = np.median(test['order'])
duration_median = np.median(test['duration'])

partial_dependence = np.zeros_like(price_range)

for i, price in enumerate(price_range):
    X_test_partial = np.array([[price, order_median, duration_median]])
    partial_dependence[i] = model.predict(X_test_partial)

plt.plot(price_range, partial_dependence)
plt.xlabel('Price')
plt.ylabel('Partial dependence')
plt.show()

#order
order_range = np.linspace(test['order'].min(), test['order'].max(), num=50)
price_median = np.median(test['price'])
duration_median = np.median(test['duration'])

partial_dependence2 = np.zeros_like(order_range)

for i, order in enumerate(order_range):
    X_test_partial = np.array([[price_median, order, duration_median]])
    partial_dependence2[i] = model.predict(X_test_partial)

plt.plot(order_range, partial_dependence2)
plt.xlabel('Order')
plt.ylabel('Partial dependence')
plt.show()

#duration
duration_range = np.linspace(test['duration'].min(), test['duration'].max(), num=50)
price_median = np.median(test['price'])
order_median = np.median(test['order'])

partial_dependence3 = np.zeros_like(duration_range)

for i, duration in enumerate(duration_range):
    X_test_partial = np.array([[price_median, order_median, duration]])
    partial_dependence3[i] = model.predict(X_test_partial)

plt.plot(duration_range, partial_dependence3)
plt.xlabel('Duration')
plt.ylabel('Partial dependence')
plt.show()
