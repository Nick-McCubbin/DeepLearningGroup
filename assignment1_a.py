import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import datetime
import time
import psutil
import sys 
import csv

data_generator = pd.read_csv("pricing_dummies.csv", chunksize = 1) 
diff_v1 = 550.0850988 #range of price
diff_v2 =  317426220.0 #range of order
diff_v3 = 6357.4524 #range of duration
diff_y = 4164.0 #range of quantity

# data = pd.read_csv("small_pricing_extra.csv")
#list of unique categories
#unique_category = ['category_0.0', 'category_1.0', 'category_2.0', 'category_3.0', 'category_4.0', 'category_5.0', 'category_6.0', 'category_7.0', 'category_8.0', 'category_9.0', 'category_10.0', 'category_11.0', 'category_12.0', 'category_13.0', 'category_14.0', 'category_15.0', 'category_16.0', 'category_17.0', 'category_18.0', 'category_19.0', 'category_20.0', 'category_21.0', 'category_22.0', 'category_23.0', 'category_24.0', 'category_25.0', 'category_26.0', 'category_27.0', 'category_28.0', 'category_29.0', 'category_30.0', 'category_31.0', 'category_32.0']

inputs = tf.keras.layers.Input(shape=(36,), name='input')
hidden1 = tf.keras.layers.Dense(units=36, activation="sigmoid", name= 'hidden1')(inputs)
hidden2 = tf.keras.layers.Dense(units=36, activation="sigmoid", name= 'hidden2')(hidden1)
hidden3 = tf.keras.layers.Dense(units=36, activation="sigmoid", name= 'hidden3')(hidden2)
output = tf.keras.layers.Dense(units=1, activation = "linear", name= 'output')(hidden3)
model = tf.keras.Model(inputs=inputs, outputs = output)

model.compile(loss='mse', optimizer=tf.keras.optimizers.SGD(learning_rate=0.001))

counter = 0
start = datetime.datetime.now()
with open('loss.csv', mode='w') as loss_file:
    loss_writer = csv.writer(loss_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    loss_writer.writerow(['loss', 'ram_usage'])
    for data in data_generator:
        if counter == 10000:
            break
       #break
        if (any(data.isna().sum()) > 0):
            continue
        data['price'] = data['price']/diff_v1
        data['order'] = data['order']/diff_v2
        data['duration'] = data['duration'] / diff_v3
        data['quantity'] = data['quantity'] / diff_y
        X_a = data.drop('sku', axis = 1)
        X_b = X_a.drop('quantity', axis = 1)
        X_c = X_b.drop(X_b.columns[0], axis = 1)
        X = X_c.to_numpy()
        y = data['quantity'].to_numpy()
        # X = np.array(np.column_stack((X1,X2,X3, X4)))
        # y_n = data["quantity"]/diff_y
        # y = y_n.to_numpy() 
        # history = model.fit(x=X,y=y,batch_size = 1, epochs = 1)
        # loss = model.train_on_batch(X, y)
        loss_writer.writerow([model.train_on_batch(X, y), psutil.Process().memory_info().rss])
        counter += 1
        # loss_writer.writerow([loss])
print(datetime.datetime.now() - start)

#read in loss and ram usage
loss_ram = pd.read_csv("loss.csv")

#loss plot
plt.plot(loss_ram['loss'])
plt.title('Loss')
plt.xlabel('Number of instances learned')
plt.ylabel('Loss')
plt.show()

#ram usage plot
plt.plot(loss_ram['ram_usage'])
plt.xlabel('Number of instances learned')
plt.ylabel('RAM usage (bytes)')
plt.show()

#reaad in testing data
test = pd.read_csv("pricing_test.csv")
test.columns =['sku', 'price', 'quantity', 'order', 'duration', 'category']
test = pd.get_dummies(test, columns = ['category'])
test.insert(5, 'category_0', 0)
test.insert(9, 'category_4', 0)
test.insert(18, 'category_13', 0)
test.insert(21, 'category_16', 0)
test.insert(25, 'category_20', 0)
test.insert(28, 'category_23', 0)
test.insert(30, 'category_25', 0)

#normalize testing data
test['price'] = test['price']/diff_v1
test['order'] = test['order']/diff_v2
test['duration'] = test['duration'] / diff_v3
test['quantity'] = test['quantity'] / diff_y

#create the testing set
X_test = test[['price', 'order', 'duration', 'category_0', 'category_1', 'category_2', 'category_3', 'category_4', 'category_5', 'category_6', 'category_7', 'category_8', 'category_9', 'category_10', 'category_11', 'category_12', 'category_13', 'category_14', 'category_15', 'category_16', 'category_17', 'category_18', 'category_19', 'category_20', 'category_21', 'category_22', 'category_23', 'category_24', 'category_25', 'category_26', 'category_27', 'category_28', 'category_29', 'category_30', 'category_31', 'category_32']].values
y_test = test['quantity'].values

#predict on testing set
yhat = model.predict(x=X_test)
yhat
model.evaluate(X,y)
model.summary()
correlation = np.corrcoef(yhat.flatten(), y_test)[0, 1]
correlation
# plt.plot(yhat)
# plt.show()

#variable importance plot
weights = model.get_layer('hidden1').get_weights()[0]

importance = np.abs(weights).sum(axis=1)

importance /= importance.sum()

# create a bar plot of variable importance
plt.bar(['price', 'order', 'duration', 'category_0', 'category_1', 'category_2', 'category_3', 'category_4', 'category_5', 'category_6', 'category_7', 'category_8', 'category_9', 'category_10', 'category_11', 'category_12', 'category_13', 'category_14', 'category_15', 'category_16', 'category_17', 'category_18', 'category_19', 'category_20', 'category_21', 'category_22', 'category_23', 'category_24', 'category_25', 'category_26', 'category_27', 'category_28', 'category_29', 'category_30', 'category_31', 'category_32'], importance)
plt.xlabel('Variable')
plt.ylabel('Importance')
plt.title('Variable Importance Plot')
plt.show()

##################################################################################

#partial dependence plots - THESE NEED TO BE FIXED

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
