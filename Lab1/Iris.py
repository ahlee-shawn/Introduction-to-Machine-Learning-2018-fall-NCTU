import csv
import statistics
from statistics import mean
import copy
import random
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def name_to_number(name):
	if name == "Iris-setosa":
		return 0
	if name == "Iris-versicolor":
		return 1
	if name == "Iris-virginica":
		return 2

def print_confusion_matrix(Confusion_Matrix):
	print(np.reshape(Confusion_Matrix, (3, 3)))
	print("Precision of setosa: ", (Confusion_Matrix[0][0] / (Confusion_Matrix[0][0] + Confusion_Matrix[1][0] + Confusion_Matrix[2][0])))
	print("Precision of versicolor: ", (Confusion_Matrix[1][1] / (Confusion_Matrix[0][1] + Confusion_Matrix[1][1] + Confusion_Matrix[2][1])))
	print("Precision of virginica: ", (Confusion_Matrix[2][2] / (Confusion_Matrix[0][2] + Confusion_Matrix[1][2] + Confusion_Matrix[2][2])))
	print("Recall of setosa with: ", (Confusion_Matrix[0][0] / (Confusion_Matrix[0][0] + Confusion_Matrix[0][1] + Confusion_Matrix[0][2])))
	print("Recall of versicolor: ", (Confusion_Matrix[1][1] / (Confusion_Matrix[1][0] + Confusion_Matrix[1][1] + Confusion_Matrix[1][2])))
	print("Recall of virginica): ", (Confusion_Matrix[2][2] / (Confusion_Matrix[2][0] + Confusion_Matrix[2][1] + Confusion_Matrix[2][2])))
	print("Accuracy: ", ((Confusion_Matrix[0][0] + Confusion_Matrix[1][1] + Confusion_Matrix[2][2]) / (Confusion_Matrix[0][0] + Confusion_Matrix[0][1] + Confusion_Matrix[0][2] + Confusion_Matrix[1][0] + Confusion_Matrix[1][1] + Confusion_Matrix[1][2] + Confusion_Matrix[2][0] + Confusion_Matrix[2][1] + Confusion_Matrix[2][2])))

#initialization
sepal_length_data = []
sepal_width_data = []
petal_length_data = []
petal_width_data = []
data = []
target = []
target_name = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]

#open file
with open('iris.csv', newline = '\n') as csvfile:
	reader = csv.reader(csvfile, delimiter = ',')
	a = 0
	for i in reader:
		if a == 1:
			data.append(i[0:4])
			sepal_length_data.append(i[0])
			sepal_width_data.append(i[1])
			petal_length_data.append(i[2])
			petal_width_data.append(i[3])
			target.append(i[4])
		a = 1

#preprocess data
for i in range(len(data)):
	data[i][0] = float(data[i][0])
	data[i][1] = float(data[i][1])
	data[i][2] = float(data[i][2])
	data[i][3] = float(data[i][3])
	sepal_length_data[i] = float(sepal_length_data[i])
	sepal_width_data[i] = float(sepal_width_data[i])
	petal_length_data[i] = float(petal_length_data[i])
	petal_width_data[i] = float(petal_width_data[i])

#mean and standard deviation
print("avg_sepal_length: ", mean(sepal_length_data))
print("avg_sepal_width: ", mean(sepal_width_data))
print("avg_petal_length: ", mean(petal_length_data))
print("avg_petal_width: ", mean(petal_width_data))
print("stdev_sepal_length: ", statistics.stdev(sepal_length_data))
print("stdev_sepal_width: ", statistics.stdev(sepal_width_data))
print("stdev_petal_length: ", statistics.stdev(petal_length_data))
print("stdev_petal_width: ", statistics.stdev(petal_width_data))

#plot
fig = plt.figure()
ax = plt.subplot()
plt.xlabel('sepal_length')
plt.ylabel('sepal_width')
ax.scatter(sepal_length_data[0:50],sepal_width_data[0:50],s=20,label=target_name[0])  
ax.scatter(sepal_length_data[50:100],sepal_width_data[50:100],c='green',s=20,label=target_name[1])
ax.scatter(sepal_length_data[100:150],sepal_width_data[100:150],c='red',s=20,label=target_name[2])
plt.legend(loc = 'best') 

fig = plt.figure()
ax = plt.subplot()
plt.xlabel('petal_length')
plt.ylabel('petal_width')
ax.scatter(petal_length_data[0:50],petal_width_data[0:50],s=20,label=target_name[0])  
ax.scatter(petal_length_data[50:100],petal_width_data[50:100],c='green',s=20,label=target_name[1])
ax.scatter(petal_length_data[100:150],petal_width_data[100:150],c='red',s=20,label=target_name[2])
plt.legend(loc = 'best') 

fig = plt.figure()
ax = plt.subplot()
plt.title('Average value of Iris data')
means = [mean(sepal_length_data),mean(sepal_width_data), mean(petal_length_data),mean(petal_width_data)]
labels = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'] 
ax.bar(range(len(means)), means, tick_label=labels) 

fig = plt.figure()
ax = plt.subplot()
plt.title('Standard deviation of Iris data')
stds = [statistics.stdev(sepal_length_data),statistics.stdev(sepal_width_data), statistics.stdev(petal_length_data),statistics.stdev(petal_width_data)]
labels = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'] 
ax.bar(range(len(stds)), stds, tick_label=labels) 
plt.show()

#split data and target for training and testing
data_train, data_test, target_train, target_test = train_test_split(data, target, test_size = 0.3)

#single decision tree with resubstitution
clf = tree.DecisionTreeClassifier()
clf = clf.fit(data, target)
result = clf.predict(data)
Confusion_Matrix = confusion_matrix(target, result, labels = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"])
print("Confusion Matrix with resubstitution (single decision tree): ")
print_confusion_matrix(Confusion_Matrix)

#single decision tree with k-fold validation
Confusion_Matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
num_folds = 3
subset_size = int(len(data_train) / num_folds)
for i in range(len(data_test)):
	query_data = data_test[i]
	query_target = target_test[i]
	vote = [0, 0, 0]
	for j in range(num_folds):
	    data_train_temp = data_train[ : j * subset_size] + data_train[(j + 1) * subset_size : ]
	    target_train_temp = target_train[ : j * subset_size] + target_train[(j + 1) * subset_size : ]
	    clf = tree.DecisionTreeClassifier()
	    clf = clf.fit(data_train_temp, target_train_temp)
	    result = clf.predict(np.reshape(query_data, (1, -1)))
	    vote[name_to_number(result)] += 1
	if target_name[vote.index(max(vote))] ==  "Iris-setosa":
		if query_target == "Iris-setosa":
			Confusion_Matrix[0][0] += 1
		elif query_target == "Iris-versicolor":
			Confusion_Matrix[1][0] += 1
		elif query_target == "Iris-virginica":
			Confusion_Matrix[2][0] += 1
	elif target_name[vote.index(max(vote))] ==  "Iris-versicolor":
		if query_target == "Iris-setosa":
			Confusion_Matrix[0][1] += 1
		elif query_target == "Iris-versicolor":
			Confusion_Matrix[1][1] += 1
		elif query_target == "Iris-virginica":
			Confusion_Matrix[2][1] += 1
	elif target_name[vote.index(max(vote))] ==  "Iris-virginica":
		if query_target == "Iris-setosa":
			Confusion_Matrix[0][2] += 1
		elif query_target == "Iris-versicolor":
			Confusion_Matrix[1][2] += 1
		elif query_target == "Iris-virginica":
			Confusion_Matrix[2][2] += 1
print("Confusion Matrix with k-fold validation (single decision tree): ")
print_confusion_matrix(Confusion_Matrix)

#random forest model with resubstitution
fault = 0
Confusion_Matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
vote = [[0 for i in range(3)] for j in range(int(len(data)))]
for i in range(100):
	for k in range(len(data)):
		query_data = data[k]
		query_target = target[k]
		rf_data = copy.deepcopy(data)
		rf_query_data = copy.deepcopy(query_data)
		counter = 0
		if random.uniform(0, 1) <= 0.5:
			counter += 1
			rf_query_data.pop(3)
			for j in range(len(data)):
				rf_data[j].pop(3)
		if random.uniform(0, 1) <= 0.5:
			counter += 1
			rf_query_data.pop(2)
			for j in range(len(data)):
				rf_data[j].pop(2)
		if random.uniform(0, 1) <= 0.5:
			counter += 1
			rf_query_data.pop(1)
			for j in range(len(data)):
				rf_data[j].pop(1)
		if random.uniform(0, 1) <= 0.5:
			counter += 1
			rf_query_data.pop(0)
			for j in range(len(data)):
				rf_data[j].pop(0)
		if counter == 4:
			continue
		clf = tree.DecisionTreeClassifier()
		clf = clf.fit(rf_data, target)
		result = clf.predict(np.reshape(rf_query_data, (1, -1)))
		if result == target_name[0]:
			vote[k][0] += 1
		elif result == target_name[1]:
			vote[int(k)][1] += 1
		elif result == target_name[2]:
			vote[int(k)][2] += 1
for k in range(len(data)):
	query_target = target[k]
	if target_name[vote[k].index(max(vote[k]))] ==  "Iris-setosa":
		if query_target == "Iris-setosa":
			Confusion_Matrix[0][0] += 1
		elif query_target == "Iris-versicolor":
			Confusion_Matrix[1][0] += 1
		elif query_target == "Iris-virginica":
			Confusion_Matrix[2][0] += 1
	elif target_name[vote[k].index(max(vote[k]))] ==  "Iris-versicolor":
		if query_target == "Iris-setosa":
			Confusion_Matrix[0][1] += 1
		elif query_target == "Iris-versicolor":
			Confusion_Matrix[1][1] += 1
		elif query_target == "Iris-virginica":
			Confusion_Matrix[2][1] += 1
	elif target_name[vote[k].index(max(vote[k]))] ==  "Iris-virginica":
		if query_target == "Iris-setosa":
			Confusion_Matrix[0][2] += 1
		elif query_target == "Iris-versicolor":
			Confusion_Matrix[1][2] += 1
		elif query_target == "Iris-virginica":
			Confusion_Matrix[2][2] += 1
print("Confusion Matrix with resubstitution (random forest): ")
print_confusion_matrix(Confusion_Matrix)

#random forest model with k-fold validation
num_folds = 3
subset_size = int(len(data_train) / num_folds)
fault = 0
Confusion_Matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
vote = [[0 for i in range(3)] for j in range(int(len(data)))]
for i in range(100):
	for k in range(len(target_test)):
		query_data = data_test[k]
		k_fold_vote = [0, 0, 0]
		rf_data = copy.deepcopy(data_train)
		rf_query_data = copy.deepcopy(query_data)
		counter = 0
		if random.uniform(0, 1) <= 0.5:
			counter += 1
			rf_query_data.pop(3)
			for j in range(len(data_train)):
				rf_data[j].pop(3)
		if random.uniform(0, 1) <= 0.5:
			counter += 1
			rf_query_data.pop(2)
			for j in range(len(data_train)):
				rf_data[j].pop(2)
		if random.uniform(0, 1) <= 0.5:
			counter += 1
			rf_query_data.pop(1)
			for j in range(len(data_train)):
				rf_data[j].pop(1)
		if random.uniform(0, 1) <= 0.5:
			counter += 1
			rf_query_data.pop(0)
			for j in range(len(data_train)):
				rf_data[j].pop(0)
		if counter == 4:
			continue
		for j in range(num_folds):
			data_train_temp = rf_data[ : j * subset_size] + rf_data[(j + 1) * subset_size : ]
			target_train_temp = target_train[ : j * subset_size] + target_train[(j + 1) * subset_size : ]
			clf = tree.DecisionTreeClassifier()
			clf = clf.fit(data_train_temp, target_train_temp)
			result = clf.predict(np.reshape(rf_query_data, (1, -1)))
			k_fold_vote[name_to_number(result)] += 1
		vote[k][k_fold_vote.index(max(k_fold_vote))] += 1
for k in range(len(data_test)):
	query_target = target_test[k]
	if target_name[vote[k].index(max(vote[k]))] ==  "Iris-setosa":
		if query_target == "Iris-setosa":
			Confusion_Matrix[0][0] += 1
		elif query_target == "Iris-versicolor":
			Confusion_Matrix[1][0] += 1
		elif query_target == "Iris-virginica":
			Confusion_Matrix[2][0] += 1
	elif target_name[vote[k].index(max(vote[k]))] ==  "Iris-versicolor":
		if query_target == "Iris-setosa":
			Confusion_Matrix[0][1] += 1
		elif query_target == "Iris-versicolor":
			Confusion_Matrix[1][1] += 1
		elif query_target == "Iris-virginica":
			Confusion_Matrix[2][1] += 1
	elif target_name[vote[k].index(max(vote[k]))] ==  "Iris-virginica":
		if query_target == "Iris-setosa":
			Confusion_Matrix[0][2] += 1
		elif query_target == "Iris-versicolor":
			Confusion_Matrix[1][2] += 1
		elif query_target == "Iris-virginica":
			Confusion_Matrix[2][2] += 1
print("Confusion Matrix with k-fold validation (random forest): ")
print_confusion_matrix(Confusion_Matrix)
