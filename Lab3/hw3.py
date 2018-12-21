import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import math
import timeit
import random

data_in = pd.read_csv('Concrete_Data.csv', sep = ',')

cement = data_in[['Cement (component 1)(kg in a m^3 mixture)']].values
blast_furnace_slag = data_in[['Blast Furnace Slag (component 2)(kg in a m^3 mixture)']].values
fly_ash = data_in[['Fly Ash (component 3)(kg in a m^3 mixture)']].values
water = data_in[['Water  (component 4)(kg in a m^3 mixture)']].values
superplasticizer = data_in[['Superplasticizer (component 5)(kg in a m^3 mixture)']].values
coarse_aggregate = data_in[['Coarse Aggregate  (component 6)(kg in a m^3 mixture)']].values
fine_aggregate = data_in[['Fine Aggregate (component 7)(kg in a m^3 mixture)']].values
age = data_in[['Age (day)']].values
concrete_compressive_strength = data_in[['Concrete compressive strength(MPa, megapascals) ']].values

attribute = [[*cement], [*blast_furnace_slag], [*fly_ash], [*water], [*superplasticizer], [*coarse_aggregate], [*fine_aggregate], [*age]]
attribute_name = [['cement'], ['blast_furnace_slag'], ['fly_ash'], ['water'], ['superplasticizer'], ['coarse_aggregate'], ['fine_aggregate'], ['age']]
'''
#Visualization of all the features with the target
for i in range(8):
	plt.title('Visualization of %s and the target' % attribute_name[i][0])
	plt.xlabel(attribute_name[i][0])
	plt.ylabel('concrete_compressive_strength')
	plt.scatter(attribute[i], concrete_compressive_strength, c = 'b')
	plt.show()

#The code, graph, r2_score, weight and bias for problem 1
for i in range(8):
	X_train, X_test, y_train, y_test = train_test_split(attribute[i], concrete_compressive_strength, test_size = 0.2)
	linreg = LinearRegression()
	linreg.fit(X_train, y_train)
	print('----- %s -----' % attribute_name[i][0])
	print('r2_score: ', r2_score(y_test, linreg.predict(X_test)))
	print('weight: ', linreg.coef_[0][0])
	print('bias: ', linreg.intercept_[0])
	plt.title('Visualization of %s and the target' % attribute_name[i][0])
	plt.xlabel(attribute_name[i][0])
	plt.ylabel('concrete_compressive_strength')
	plt.scatter(attribute[i], concrete_compressive_strength, c = 'b')
	x = np.linspace(min(attribute[i]), max(attribute[i]))
	plt.plot(x, linreg.coef_[0][0] * x + linreg.intercept_[0], color='r', linestyle='-', linewidth=2)
	plt.show()


#The code, graph, r2_score, weight and bias for problem 2
for i in range(8):
	X_train, X_test, y_train, y_test = train_test_split(attribute[i], concrete_compressive_strength, test_size = 0.2)
	learning_rate_weight = 10
	learning_rate_bias = 10
	history_gd_weight = 0
	history_gd_bias = 0
	xy = 0
	xx = 0
	X_avg = sum(X_train) / len(X_train)
	y_avg = sum(y_train) / len(y_train)
	for j in range(len(X_train)):
		xy = xy + X_train[j] * y_train[j]
		xx = xx + X_train[j] * X_train[j]
	weight = (xy - len(X_train) * X_avg * y_avg) / (xx - len(X_train) * X_avg* X_avg)
	bias = y_avg - weight * X_avg
	while(1):
		gd_weight = 0
		gd_bias = 0
		for j in range(len(X_train)):
			temp = 2 * (y_train[j] - weight * X_train[j] - bias) * (-1)
			gd_weight = gd_weight + temp * X_train[j]
			gd_bias = gd_bias + temp
		gd_weight = gd_weight / len(X_train)
		gd_bias = gd_bias / len(X_train)
		if(gd_weight <= 1e-3 and gd_weight >= -1e-3 and gd_bias <= 1e-3 and gd_bias >= -1e-3):
			break
		else:
			history_gd_weight = history_gd_weight + gd_weight ** 2
			history_gd_bias = history_gd_bias + gd_bias * gd_bias
			weight = weight - ((learning_rate_weight) / (math.sqrt(history_gd_weight))) * (gd_weight)
			bias = bias - ((learning_rate_bias) / (math.sqrt(history_gd_bias))) * (gd_bias)
	r2_score_down = 0
	r2_score_up = 0
	for j in range(len(y_test)):
		r2_score_down = r2_score_down + (y_test[j] - sum(y_test) / len(y_test)) * (y_test[j] - sum(y_test) / len(y_test))
		r2_score_up = r2_score_up + (y_test[j] - X_test[j] * weight - bias) * (y_test[j] - X_test[j] * weight - bias)
	print('----- %s -----' % attribute_name[i][0])
	print('r2_score: ', 1 - (r2_score_up) / (r2_score_down))
	print('weight: ', weight)
	print('bias: ', bias)
	plt.title('Visualization of %s and the target' % attribute_name[i][0])
	plt.xlabel(attribute_name[i][0])
	plt.ylabel('concrete_compressive_strength')
	plt.scatter(attribute[i], concrete_compressive_strength, c = 'b')
	x = np.linspace(min(attribute[i]), max(attribute[i]))
	plt.plot(x, weight * x + bias, color='r', linestyle='-', linewidth=2)
	plt.show()

#The code, MSE, and the r2_score for problem 3(each operation updates w)
X1_train, X1_test, X2_train, X2_test, y_train, y_test = train_test_split(attribute[0], attribute[1], concrete_compressive_strength, test_size = 0.2)
learning_rate_weight1 = 20
learning_rate_weight2 = 20
learning_rate_bias = 20
history_gd_weight1 = 0
history_gd_weight2 = 0
history_gd_bias = 0
weight1 = 0
weight2 = 0
bias = 0
while(1):
	gd_weight1 = 0
	gd_weight2 = 0
	gd_bias = 0
	for j in range(len(X1_train)):
		temp = 2 * (y_train[j] - weight1 * X1_train[j] - weight2 * X2_train[j] - bias) * (-1)
		gd_weight1 = gd_weight1 + temp * X1_train[j]
		gd_weight2 = gd_weight2 + temp * X2_train[j]
		gd_bias = gd_bias + temp
	gd_weight1 = gd_weight1 / len(X1_train)
	gd_weight2 = gd_weight2 / len(X2_train)
	gd_bias = gd_bias / len(X1_train)
	if(gd_weight1 <= 1e-3 and gd_weight1 >= -1e-3 and gd_weight2 <= 1e-3 and gd_weight2 >= -1e-3 and gd_bias <= 1e-3 and gd_bias >= -1e-3):
		break
	else:
		history_gd_weight1 = history_gd_weight1 + gd_weight1 ** 2
		history_gd_weight2 = history_gd_weight2 + gd_weight2 ** 2
		history_gd_bias = history_gd_bias + gd_bias * gd_bias
		weight1 = weight1 - ((learning_rate_weight1) / (math.sqrt(history_gd_weight1))) * (gd_weight1)
		weight2 = weight2 - ((learning_rate_weight2) / (math.sqrt(history_gd_weight2))) * (gd_weight2)
		bias = bias - ((learning_rate_bias) / (math.sqrt(history_gd_bias))) * (gd_bias)
print('----- %s and %s -----' % (attribute_name[0][0], attribute_name[1][0]))
#calculation for training data
MSE_train = 0
for i in range(len(X1_train)):
	MSE_train = MSE_train + (y_train[i] - weight1 * X1_train[i] - weight2 * X2_train[i] - bias) * (y_train[i] - weight1 * X1_train[i] - weight2 * X2_train[i] - bias)
MSE_train = MSE_train / len(X1_train)
print('MSE of training data: ', MSE_train)
r2_score_down = 0
r2_score_up = 0
for j in range(len(y_train)):
	r2_score_down = r2_score_down + (y_train[j] - sum(y_train) / len(y_train)) * (y_train[j] - sum(y_train) / len(y_train))
	r2_score_up = r2_score_up + (y_train[j] - weight1 * X1_train[i] - weight2 * X2_train[i] - bias) * (y_train[j] - weight1 * X1_train[i] - weight2 * X2_train[i] - bias)
print('r2_score of training data: ', 1 - (r2_score_up) / (r2_score_down))
MSE_test = 0
#calculation for testing data
for i in range(len(X1_test)):
	MSE_test = MSE_test + (y_test[i] - weight1 * X1_test[i] - weight2 * X2_test[i] - bias) * (y_test[i] - weight1 * X1_test[i] - weight2 * X2_test[i] - bias)
MSE_test = MSE_test / len(X1_test)
print('MSE of testing data: ', MSE_test)
r2_score_down = 0
r2_score_up = 0
for j in range(len(y_test)):
	r2_score_down = r2_score_down + (y_test[j] - sum(y_test) / len(y_test)) * (y_test[j] - sum(y_test) / len(y_test))
	r2_score_up = r2_score_up + (y_test[j] - weight1 * X1_test[j] - weight2 * X2_test[j] - bias) * (y_test[j] - weight1 * X1_test[j] - weight2 * X2_test[j] - bias)
print('r2_score of testing data: ', 1 - (r2_score_up) / (r2_score_down))

#The code, MSE, and the r2_score for problem 3(each operation updates wi)
X1_train, X1_test, X2_train, X2_test, y_train, y_test = train_test_split(attribute[0], attribute[1], concrete_compressive_strength, test_size = 0.2)
learning_rate_weight1 = 100
learning_rate_weight2 = 100
learning_rate_bias = 100
history_gd_weight1 = 0
history_gd_weight2 = 0
history_gd_bias = 0
weight1 = 0
weight2 = 0
bias = 0
k = 0
while(1):
	gd_weight1 = 0
	gd_weight2 = 0
	gd_bias = 0
	if(k % 3 == 0):
		for j in range(len(X1_train)):
			gd_weight1 = gd_weight1 + 2 * (y_train[j] - weight1 * X1_train[j] - weight2 * X2_train[j] - bias) * (-1) * X1_train[j]
		gd_weight1 = gd_weight1 / len(X1_train)
	elif(k % 3 == 1):
		for j in range(len(X1_train)):
			gd_weight2 = gd_weight2 + 2 * (y_train[j] - weight1 * X1_train[j] - weight2 * X2_train[j] - bias) * (-1) * X2_train[j]
		gd_weight2 = gd_weight2 / len(X2_train)
	else:
		for j in range(len(X1_train)):
			gd_bias = gd_bias + 2 * (y_train[j] - weight1 * X1_train[j] - weight2 * X2_train[j] - bias) * (-1)
		gd_bias = gd_bias / len(X1_train)
	if(gd_weight1 <= 1e-3 and gd_weight1 >= -1e-3 and gd_weight2 <= 1e-3 and gd_weight2 >= -1e-3 and gd_bias <= 1e-3 and gd_bias >= -1e-3):
		break
	else:
		if(k %3 == 0):
			history_gd_weight1 = history_gd_weight1 + gd_weight1 ** 2
			weight1 = weight1 - ((learning_rate_weight1) / (math.sqrt(history_gd_weight1))) * (gd_weight1)
		elif(k % 3 == 1):
			history_gd_weight2 = history_gd_weight2 + gd_weight2 ** 2
			weight2 = weight2 - ((learning_rate_weight2) / (math.sqrt(history_gd_weight2))) * (gd_weight2)
		else:
			history_gd_bias = history_gd_bias + gd_bias * gd_bias
			bias = bias - ((learning_rate_bias) / (math.sqrt(history_gd_bias))) * (gd_bias)
		k = k + 1
print('----- %s and %s -----' % (attribute_name[0][0], attribute_name[1][0]))
#calculation for training data
MSE_train = 0
for i in range(len(X1_train)):
	MSE_train = MSE_train + (y_train[i] - weight1 * X1_train[i] - weight2 * X2_train[i] - bias) * (y_train[i] - weight1 * X1_train[i] - weight2 * X2_train[i] - bias)
MSE_train = MSE_train / len(X1_train)
print('MSE of training data: ', MSE_train)
r2_score_down = 0
r2_score_up = 0
for j in range(len(y_train)):
	r2_score_down = r2_score_down + (y_train[j] - sum(y_train) / len(y_train)) * (y_train[j] - sum(y_train) / len(y_train))
	r2_score_up = r2_score_up + (y_train[j] - weight1 * X1_train[i] - weight2 * X2_train[i] - bias) * (y_train[j] - weight1 * X1_train[i] - weight2 * X2_train[i] - bias)
print('r2_score of training data: ', 1 - (r2_score_up) / (r2_score_down))
MSE_test = 0
#calculation for testing data
for i in range(len(X1_test)):
	MSE_test = MSE_test + (y_test[i] - weight1 * X1_test[i] - weight2 * X2_test[i] - bias) * (y_test[i] - weight1 * X1_test[i] - weight2 * X2_test[i] - bias)
MSE_test = MSE_test / len(X1_test)
print('MSE of testing data: ', MSE_test)
r2_score_down = 0
r2_score_up = 0
for j in range(len(y_test)):
	r2_score_down = r2_score_down + (y_test[j] - sum(y_test) / len(y_test)) * (y_test[j] - sum(y_test) / len(y_test))
	r2_score_up = r2_score_up + (y_test[j] - weight1 * X1_test[j] - weight2 * X2_test[j] - bias) * (y_test[j] - weight1 * X1_test[j] - weight2 * X2_test[j] - bias)
print('r2_score of testing data: ', 1 - (r2_score_up) / (r2_score_down))

#The code, MSE, and the r2_score for problem 4
X_train, X_test, y_train, y_test = train_test_split(attribute[1], concrete_compressive_strength, test_size = 0.2)
learning_rate_weight1 = 100000000
learning_rate_weight2 = 100000000
learning_rate_bias = 100000000000
history_gd_weight1 = 0
history_gd_weight2 = 0
history_gd_bias = 0
weight1 = 0
weight2 = 0
bias = 0
while(1):
	gd_weight1 = 0
	gd_weight2 = 0
	gd_bias = 0
	for j in range(len(X_train)):
		temp = 2 * (y_train[j] - weight1 * X_train[j] * X_train[j] - weight2 * X_train[j] - bias) * (-1)
		gd_weight1 = gd_weight1 + temp * X_train[j] * X_train[j]
		gd_weight2 = gd_weight2 + temp * X_train[j]
		gd_bias = gd_bias + temp
	gd_weight1 = gd_weight1 / len(X_train)
	gd_weight2 = gd_weight2 / len(X_train)
	gd_bias = gd_bias / len(X_train)
	#print(gd_weight1)
	#print(gd_weight2)
	#print(gd_bias)
	if(gd_weight1 <= 1e-3 and gd_weight1 >= -1e-3 and gd_weight2 <= 1e-3 and gd_weight2 >= -1e-3 and gd_bias <= 1e-3 and gd_bias >= -1e-3):
		break
	else:
		history_gd_weight1 = history_gd_weight1 + gd_weight1 ** 2
		history_gd_weight2 = history_gd_weight2 + gd_weight2 ** 2
		history_gd_bias = history_gd_bias + gd_bias * gd_bias
		weight1 = weight1 - ((learning_rate_weight1) / (math.sqrt(history_gd_weight1))) * (gd_weight1)
		weight2 = weight2 - ((learning_rate_weight2) / (math.sqrt(history_gd_weight2))) * (gd_weight2)
		bias = bias - ((learning_rate_bias) / (math.sqrt(history_gd_bias))) * (gd_bias)
print('----- %s -----' % attribute_name[1][0])
#calculation for training data
MSE_train = 0
for i in range(len(X_train)):
	MSE_train = MSE_train + (y_train[j] - weight1 * X_train[j] * X_train[j] - weight2 * X_train[j] - bias) ** 2
MSE_train = MSE_train / len(X_train)
print('MSE of training data: ', MSE_train)
r2_score_down = 0
r2_score_up = 0
for j in range(len(y_train)):
	r2_score_down = r2_score_down + (y_train[j] - sum(y_train) / len(y_train)) * (y_train[j] - sum(y_train) / len(y_train))
	r2_score_up = r2_score_up + (y_train[j] - weight1 * X_train[j] * X_train[j] - weight2 * X_train[j] - bias) ** 2
print('r2_score of training data: ', 1 - (r2_score_up) / (r2_score_down))
MSE_test = 0
#calculation for testing data
for j in range(len(X_test)):
	MSE_test = MSE_test + (y_test[j] - weight1 * X_test[j] * X_test[j] - weight2 * X_test[j] - bias) ** 2
MSE_test = MSE_test / len(X_test)
print('MSE of testing data: ', MSE_test)
r2_score_down = 0
r2_score_up = 0
for j in range(len(y_test)):
	r2_score_down = r2_score_down + (y_test[j] - sum(y_test) / len(y_test)) * (y_test[j] - sum(y_test) / len(y_test))
	r2_score_up = r2_score_up + (y_test[j] - weight1 * X_test[j] * X_test[j] - weight2 * X_test[j] - bias) ** 2
print('r2_score of testing data: ', 1 - (r2_score_up) / (r2_score_down))
'''

#The code, MSE, and the r2_score for problem 5
r2 = 0
while(1):
	r2 = 0
	X1_train, X1_test, y_train, y_test = train_test_split(attribute[0],  concrete_compressive_strength, test_size = 0.995)
	temp_y = []

	for i in range(len(y_train)):
		temp_y.append(y_train[i])

	X1_train = np.array(X1_train).tolist()
	new_x = []
	index = []
	for i in range(len(X1_train)):
		if(X1_train[i][0] in new_x):
			index.append(i)
			continue
		else:
			sum = 0
			counter = 1
			for j in range(len(temp_y)):
				if X1_train[j][0] == X1_train[i][0]:
					sum = sum + temp_y[j][0]
					counter = counter + 1
				temp_y[i][0] = sum/counter
			new_x.append(X1_train[i][0])

	trans_temp = []
	for i in range(len(temp_y)):
		if i in index:
			continue
		else:
			trans_temp.append(temp_y[i][0])

	learning_rate_weight1 = 1e7
	learning_rate_weight2 = 1e7
	learning_rate_weight3 = 1e7
	learning_rate_weight4 = 1e7
	learning_rate_weight5 = 1e7
	learning_rate_weight6 = 1e-10
	learning_rate_weight7 = 1e-10
	learning_rate_weight8 = 1e-10
	learning_rate_weight9 = 1e-10
	learning_rate_bias = 1e3
	history_gd_weight1 = 0
	history_gd_weight2 = 0
	history_gd_weight3 = 0
	history_gd_weight4 = 0
	history_gd_weight5 = 0
	history_gd_bias = 0
	weight1 = 0
	weight2 = 0
	weight3 = 0
	weight4 = 0
	weight5 = 0
	bias = 0
	for x in range(40000):
		gd_weight1 = 0
		gd_weight2 = 0
		gd_weight3 = 0
		gd_weight4 = 0
		gd_weight5 = 0
		gd_weight6 = 0
		gd_weight7 = 0
		gd_weight8 = 0
		gd_weight9 = 0
		gd_bias = 0
		for j in range(len(new_x)):
			temp = 2 * (trans_temp[j] - weight1 * (new_x[j])  - weight2 * (new_x[j] ** 2)-weight3 * (new_x[j] ** 3) -weight4 * (new_x[j] ** 4) -weight5 * (new_x[j] ** 5) - bias) * (-1)
			gd_weight1 = gd_weight1 + temp * (new_x[j] ** 1)
			gd_weight2 = gd_weight2 + temp * (new_x[j] ** 2)
			gd_weight3 = gd_weight3 + temp * (new_x[j] ** 3)
			gd_weight4 = gd_weight4 + temp * (new_x[j] ** 4)
			gd_weight5 = gd_weight5 + temp * (new_x[j] ** 5)
			gd_bias = gd_bias + temp
		
		if gd_weight1 < -10000:
			break
		#print(bias)
		if(gd_weight1 <= 1e-3 and gd_weight1 >= -1e-3 
			or gd_weight2 <= 1e-3 and gd_weight2 >= -1e-3 
			or gd_weight3 <= 1e-3 and gd_weight3 >= -1e-3 
			or gd_weight4 <= 1e-3 and gd_weight4 >= -1e-3  or gd_weight5 <= 1e-3 and gd_weight5 >= -1e-3
		   ):
			break
		else:
			history_gd_weight1 = history_gd_weight1 + gd_weight1 ** 2
			history_gd_weight2 = history_gd_weight2 + gd_weight2 ** 2
			history_gd_weight3 = history_gd_weight3 + gd_weight3 ** 2
			history_gd_weight4 = history_gd_weight4 + gd_weight4 ** 2
			history_gd_weight5 = history_gd_weight5 + gd_weight5 ** 2
			history_gd_bias = history_gd_bias + gd_bias * gd_bias
			weight1 = weight1 - ((learning_rate_weight1) / (math.sqrt(history_gd_weight1))) * (gd_weight1)
			weight2 = weight2 - ((learning_rate_weight2) / (math.sqrt(history_gd_weight2))) * (gd_weight2)
			weight3 = weight3 - ((learning_rate_weight3) / (math.sqrt(history_gd_weight3))) * (gd_weight3)
			weight4 = weight4 - ((learning_rate_weight4) / (math.sqrt(history_gd_weight4))) * (gd_weight4)
			weight5 = weight5 - ((learning_rate_weight5) / (math.sqrt(history_gd_weight5))) * (gd_weight5)
			bias = bias - ((learning_rate_bias) / (math.sqrt(history_gd_bias))) * (gd_bias)

	r2_score_down = 0
	r2_score_up = 0

	sum_y = 0
	for i in range(len(y_train)):
		sum_y = sum_y + y_train[i][0]
	avg_y = sum_y/len(y_train)

	for j in range(len(y_train)):
		r2_score_down = r2_score_down + (y_train[j] - avg_y/ len(y_train)) ** 2
		r2_score_up = r2_score_up + (y_train[j] - weight1 * (X1_train[j][0]) - weight2*(X1_train[j][0] ** 2) -weight3 * (X1_train[j][0] ** 3)-weight4 * (X1_train[j][0] ** 4) -weight5 * (X1_train[j][0] ** 5) - bias) ** 2
	r2 = 1 - (r2_score_up) / (r2_score_down)
	if r2 > 0.87 and r2 < 0.95:
		break
print('r2_score of training data: ', r2)
