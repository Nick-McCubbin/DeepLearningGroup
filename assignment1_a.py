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
# data_generator = data_generator_big.sample(n = 10000)
diff_v1 = 550.0850988 #range of price
diff_v2 =  317426220.0 #range of order
diff_v3 = 6357.4524 #range of duration
diff_y = 4164.0 #range of quantity

# data = pd.read_csv("small_pricing_extra.csv")
#list of unique categories
unique_category = ['category_0.0', 'category_1.0', 'category_2.0', 'category_3.0', 'category_4.0', 'category_5.0', 'category_6.0', 'category_7.0', 'category_8.0', 'category_9.0', 'category_10.0', 'category_11.0', 'category_12.0', 'category_13.0', 'category_14.0', 'category_15.0', 'category_16.0', 'category_17.0', 'category_18.0', 'category_19.0', 'category_20.0', 'category_21.0', 'category_22.0', 'category_23.0', 'category_24.0', 'category_25.0', 'category_26.0', 'category_27.0', 'category_28.0', 'category_29.0', 'category_30.0', 'category_31.0', 'category_32.0']

inputs = tf.keras.layers.Input(shape=(37,), name='input')
hidden1 = tf.keras.layers.Dense(units=36, activation="sigmoid", name= 'hidden1')(inputs)
hidden2 = tf.keras.layers.Dense(units=36, activation="sigmoid", name= 'hidden2')(hidden1)
hidden3 = tf.keras.layers.Dense(units=36, activation="sigmoid", name= 'hidden3')(hidden2)
output = tf.keras.layers.Dense(units=1, activation = "linear", name= 'output')(hidden3)
model = tf.keras.Model(inputs=inputs, outputs = output)

model.compile(loss='mse', optimizer=tf.keras.optimizers.SGD(learning_rate=0.001))

start = datetime.datetime.now()
with open('loss.csv', mode='w') as loss_file:
    loss_writer = csv.writer(loss_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    loss_writer.writerow(['loss'])
    for data in data_generator:
        # break
        if (any(data.isna().sum()) > 0):
            continue
        # v1 = data["price"]/diff_v1
        # X1 = v1.to_numpy() 
        # v2 = data["order"]/diff_v2 
        # X2 = v2.to_numpy() 
        # v3 = data["duration"]/diff_v3
        # X3 = v3.to_numpy()
        # data_dummies = pd.get_dummies(data, columns=['category'])
        data['price'] = data['price']/diff_v1
        data['order'] = data['order']/diff_v2
        data['duration'] = data['duration'] / diff_v3
        data['quantity'] = data['quantity'] / diff_y
        X_a = data.drop('sku', axis = 1)
        X_b = X_a.drop('quantity', axis = 1)
        X = X_b.to_numpy()
        y = data['quantity'].to_numpy()
        # X = np.array(np.column_stack((X1,X2,X3, X4)))
        # y_n = data["quantity"]/diff_y
        # y = y_n.to_numpy() 
        # history = model.fit(x=X,y=y,batch_size = 1, epochs = 1)
        loss = model.train_on_batch(X, y)
        loss_writer.writerow([loss])
        # process = psutil.Process()
        # ram_usage.append(process.memory_info().rss)
print(datetime.datetime.now() - start)

loss = pd.read_csv("loss.csv")
plt.plot(loss)
plt.title('Loss')
plt.xlabel('Number of instances learned')
plt.ylabel('Loss')
plt.show()

# plt.plot(ram_usage)
# plt.xlabel('Number of instances learned')
# plt.ylabel('RAM usage (bytes)')
# plt.show()

# sys.getsizeof(history)
