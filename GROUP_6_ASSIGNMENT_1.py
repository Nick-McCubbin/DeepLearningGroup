########################
########################
# GROUP 6 ASSIGNMENT 1 #
# MEMBERS: WESTENA ANDERSON, NICK MCCUBBIN, MICHALEA SHOFNER, CARRIE BETH WORKMAN#
########################
########################


import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import datetime
import time
import psutil
import sys
import csv
from sklearn.metrics import r2_score


#list of unique categories
#unique_category = ['category_0.0', 'category_1.0', 'category_2.0', 'category_3.0', 'category_4.0', 'category_5.0', 
# 'category_6.0', 'category_7.0', 'category_8.0', 'category_9.0', 'category_10.0', 'category_11.0', 'category_12.0', 
# 'category_13.0', 'category_14.0', 'category_15.0', 'category_16.0', 'category_17.0', 'category_18.0', 'category_19.0', 
# 'category_20.0', 'category_21.0', 'category_22.0', 'category_23.0', 'category_24.0', 'category_25.0', 'category_26.0', 
# 'category_27.0', 'category_28.0', 'category_29.0', 'category_30.0', 'category_31.0', 'category_32.0']

# FOR NORMALIZATION OF VARIABLES
diff_v1 = 550.0850988 #range of price
diff_v2 =  317426220.0 #range of order
diff_v3 = 6357.4524 #range of duration
diff_y = 4164.0 #range of quantity


# DATA PREPROCESSING FUNCTION (FOR MINIMIZING STORAGE WITHIN LOOP)
def preprocess(data):
    if(any(data.isna().sum()) > 0):
        return None
    data['price'] /= diff_v1
    data['order'] /= diff_v2
    data['duration'] /= diff_v3
    data['quantity'] /= diff_y
    X = data.drop(['sku', 'quantity'], axis = 1).iloc[:,1:].to_numpy()
    y = data['quantity'].to_numpy()
    return X, y


#SPECIFY ARCHITECTURE
inputs = tf.keras.layers.Input(shape=(36,), name='input')
hidden1 = tf.keras.layers.Dense(units=15, activation="sigmoid", name= 'hidden1')(inputs)
hidden2 = tf.keras.layers.Dense(units=10, activation="sigmoid", name= 'hidden2')(hidden1)
hidden3 = tf.keras.layers.Dense(units=5, activation="sigmoid", name= 'hidden3')(hidden2)
output = tf.keras.layers.Dense(units=1, activation = "linear", name= 'output')(hidden3)

###########################################################
###########################
#  MODELING
###########################
###########################################################

#INITIAL OBJECTS: ONLY RUN WHEN BEGINNING MODEL. COMMENT OUT IF RESUMING MODEL FROM A SAVED MODEL.
data_generator = pd.read_csv("PATH/pricing_dummies.csv", chunksize = 1)
counter = 0
model = tf.keras.Model(inputs=inputs, outputs = output)
model.compile(loss='mse', optimizer=tf.keras.optimizers.SGD(learning_rate=0.0001))

########################################

##NOTE: DELETE LOSS.CSV EACH TIME YOU RESTART A MODEL FROM THE BEGINNING. WE ARE USING APPEND MODE
## TO RESUME WRITING TO LOSS.CSV WHEN WE RESUME THE MODEL.

start = datetime.datetime.now()
with open('PATH/loss1.csv', mode='a', newline = '') as loss_file:
    loss_writer = csv.writer(loss_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for data in data_generator:
        preprocessed_data = preprocess(data)
        if preprocessed_data is not None:
            X,y = preprocessed_data
            loss_writer.writerow([model.train_on_batch(X, y), psutil.Process().memory_info().rss])
            counter += 1
            print(f"Counter: {counter}")
        if counter == 10000000: # THIS VALUE CAN BE CHANGED. WE RAN ON 10,000,000 INSTANCES.
            break
        if (any(data.isna().sum()) > 0):
            continue
        X = data.drop(['sku', 'quantity'], axis = 1).iloc[:,1:].to_numpy()
        y = data['quantity'].to_numpy()
        loss_writer.writerow([model.train_on_batch(X, y), psutil.Process().memory_info().rss])
        print(f"Counter: {counter}")
        counter += 1
        if counter % 100000 == 0: #SAVE MODEL EVERY 100,000 INSTANCES
            model.save('PATH/nnmodel.h5')

print(datetime.datetime.now() - start) #TOTAL TIME TO RUN MODEL - WILL INCLUDE ANY DOWNTIME, HOWEVER.

#RESUMED OBJECTS: ONLY RUN WHEN RESUMING A SAVED MODEL. COMMENT OUT IF BEGINNING A NEW MODEL.
#NEED TO RUN THIS CHUNK, THEN THE LOOP.
row_end = counter
data_generator = pd.read_csv('PATH/pricing_dummies.csv', chunksize = 1, skiprows = range(1, row_end))
model = tf.keras.models.load_model('PATH/nnmodel.h5')
model.compile(loss='mse', optimizer=tf.keras.optimizers.SGD(learning_rate=0.00001))


###########################################################
###########################
#  LOSS AND RAM PLOTS
###########################
###########################################################

#read in loss and ram usage
loss_ram = pd.read_csv("PATH/loss.csv")
loss_ram.columns = ['loss', 'ram_usage']
loss_ram.head()
len(loss_ram)


#LOSS PLT - OVERALL. MOVING AVERAGE IN NEXT CHUNK
plt.plot(loss_ram['loss'], "#8098fc")
plt.title('Loss')
plt.xlabel('Number of instances learned')
plt.ylabel('Loss (MSE)')
plt.show()

##MOVING AVG OF MSE USING EXPONENTIALLY WEIGHTED MOVING AVG
ewma1 = pd.Series(loss_ram['loss']).ewm(alpha = 0.1).mean()
plt.plot(ewma1, '#8098fc')
plt.xlabel('Instances Learned')
plt.ylabel('Exponentially Weighted Moving Average (EWMA) of Loss')
plt.title('Moving Average of Loss Across 10,000,000 Instances Learned')
plt.show()

#MOVING AVG OF MSE USING EXPONENTIALLY WEIGHTED MOVING AVG (FIRST 5K ROWS)
subsetloss = loss_ram[:5000]
ewma = pd.Series(subsetloss['loss']).ewm(alpha=0.1).mean()
plt.plot(ewma, '#8098fc')
plt.xlabel('Instances Learned')
plt.ylabel('Exponentially Weighted Moving Average (EWMA) of Loss')
plt.title('Moving Average of Loss Across First 5,000 Instances Learned')
plt.show()


#RAM USAGE PLOT
#we find the average ram usage across every 100,000 instances in order to also plot the avg ram usage over time
groups = loss_ram['ram_usage'].groupby(loss_ram.index // 100000)
group_avg = groups.mean()

x = np.linspace(0, len(loss_ram), len(group_avg))
y1 = np.interp(x, group_avg.index * 1000000, group_avg)
plt.plot(loss_ram['ram_usage'] / (1024*1024*1024), '#8098fc') #multiplication to convert to GB from KB
plt.plot(x, y1 / (1024*1024*1024), "#ee6f68")
plt.xlabel('Instances Learned')
plt.ylabel('RAM usage (GB)')
plt.yticks(np.arange(0, 4, 0.5))
plt.legend(['RAM usage', 'Average RAM usage (per 100,000)'])
plt.title('RAM Usage Over 10 Million Instances')
plt.show()


###########################################################
###########################
#  PREDICTIONS & MODEL EVALUATIONS
###########################
###########################################################

#read in testing data
test = pd.read_csv("PATH/pricing_test.csv")
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
y_test = test['quantity'].values #this is y_true


#predict on testing set
yhat = model.predict(x=X_test) #y_hat is y_pred
yhat = yhat.flatten()
model.evaluate(X_test,y_test)
model.summary()
correlation = np.corrcoef(yhat, y_test)[0, 1]
correlation


#write y and yhat to a csv
counter2 = 0
with open ('PATH/predictions1mil.csv', mode='w', newline = '') as predictions_file:
    writer = csv.writer(predictions_file)
    writer.writerow(['yhat','y_test'])
    for i in range(len(yhat)):
        print(f"Counter2: {counter2}")
        counter2 += 1
        writer.writerow([yhat[i], y_test[i]])


predictions = pd.read_csv('PATH/predictions.csv')
predictions.head()
len(predictions)

# r2 function and evaluation

def r2_score_fn(y_test, y_hat):

    numerator = 0
    denominator = 0
    ybar = np.mean(y_test)

    for i in range(len(y_test)):
        numerator = (y_test[i] - y_hat[i])**2 ##SSres
        denominator = (y_test[i] - ybar)**2 ##SStot

    r2 = 1 - (np.sum(numerator)/np.sum(denominator))
    return r2

r2_score_fn(y_test, yhat) #predicting from stored vars
r2_score(predictions['y_test'], predictions['yhat']) #can also predict from predictions.csv
r2_score(y_test, yhat) #just checking that our r2 function matches the sklearn r2 function

###########################################

#VARIABLE IMPORTANCE PLOT

performance_before = np.corrcoef(y_test,yhat)[0,1]
performance_before

importance = list()
for ind in range(1,37):
    X_test_cp = np.copy(X_test)
    variable = np.random.permutation(np.copy(X_test_cp[:,ind]))
    X_test_cp[:,ind] = variable
    yhat2 = model.predict(X_test_cp)
    performance_after = np.corrcoef(y_test,yhat2.flatten())[0,1]
    importance.append(performance_before - performance_after)

importance

plt.bar(range(len(importance)), importance)
plt.xticks(range(len(importance)), ['price', 'order', 'duration', 'category_0', 'category_1', 'category_2', 'category_3', 'category_4', 'category_5', 'category_6', 'category_7', 'category_8', 'category_9', 'category_10', 'category_11', 'category_12', 'category_13', 'category_14', 'category_15', 'category_16', 'category_17', 'category_18', 'category_19', 'category_20', 'category_21', 'category_22', 'category_23', 'category_24', 'category_25', 'category_26', 'category_27', 'category_28', 'category_29', 'category_30', 'category_31', 'category_32'], rotation = 90)
plt.xlabel('Variable')
plt.ylabel('Importance')
plt.title('Variable Importance Plot')
plt.show()

###########################################

#PARTIAL DEPENDENCE PLOTS

#price
v = np.linspace(test['price'].min(), test['price'].max(), 15)

means = []
for i in v:
    X_test_cp = np.copy(X_test)
    X_test_cp[:,1] = i
    yhat = model.predict(X_test_cp)
    means.append(np.mean(yhat))

plt.plot(v, means)
plt.xlabel('Price')
plt.ylabel('Average Predicted Quantity')
plt.show()

#order
v2 = np.linspace(test['order'].min(), test['order'].max(), 15)

means2 = []
for i in v2:
    X_test_cp = np.copy(X_test)
    X_test_cp[:,2] = i
    yhat = model.predict(X_test_cp)
    means2.append(np.mean(yhat))

plt.plot(v2, means2)
plt.xlabel('Order')
plt.ylabel('Average Predicted Quantity')
plt.show()

#duration
v3 = np.linspace(test['duration'].min(), test['duration'].max(), 15)

means3 = []
for i in v3:
    X_test_cp = np.copy(X_test)
    X_test_cp[:,3] = i
    yhat = model.predict(X_test_cp)
    means3.append(np.mean(yhat))

plt.plot(v, means3)
plt.xlabel('Duration')
plt.ylabel('Average Predicted Quantity')
plt.show()
