import time
import pandas as pd
import numpy as np
import copy
import random
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

def print_confusion_matrix(Confusion_Matrix):
	print(np.reshape(Confusion_Matrix, (5, 5)))
	print("Precision of level 1: ", (Confusion_Matrix[0][0] / (Confusion_Matrix[0][0] + Confusion_Matrix[1][0] + Confusion_Matrix[2][0] + Confusion_Matrix[3][0] + Confusion_Matrix[4][0])))
	print("Precision of level 2: ", (Confusion_Matrix[1][1] / (Confusion_Matrix[0][1] + Confusion_Matrix[1][1] + Confusion_Matrix[2][1] + Confusion_Matrix[3][1] + Confusion_Matrix[4][1])))
	print("Precision of level 3: ", (Confusion_Matrix[2][2] / (Confusion_Matrix[0][2] + Confusion_Matrix[1][2] + Confusion_Matrix[2][2] + Confusion_Matrix[3][2] + Confusion_Matrix[4][2])))
	print("Precision of level 4: ", (Confusion_Matrix[3][3] / (Confusion_Matrix[0][3] + Confusion_Matrix[1][3] + Confusion_Matrix[2][3] + Confusion_Matrix[3][3] + Confusion_Matrix[4][3])))
	print("Precision of level 5: ", (Confusion_Matrix[4][4] / (Confusion_Matrix[0][4] + Confusion_Matrix[1][4] + Confusion_Matrix[2][4] + Confusion_Matrix[3][4] + Confusion_Matrix[4][4])))
	print("Recall of level 1: ", (Confusion_Matrix[0][0] / (Confusion_Matrix[0][0] + Confusion_Matrix[0][1] + Confusion_Matrix[0][2] + Confusion_Matrix[0][3] + Confusion_Matrix[0][4])))
	print("Recall of level 2: ", (Confusion_Matrix[1][1] / (Confusion_Matrix[1][0] + Confusion_Matrix[1][1] + Confusion_Matrix[1][2] + Confusion_Matrix[1][3] + Confusion_Matrix[1][4])))
	print("Recall of level 3: ", (Confusion_Matrix[2][2] / (Confusion_Matrix[2][0] + Confusion_Matrix[2][1] + Confusion_Matrix[2][2] + Confusion_Matrix[2][3] + Confusion_Matrix[2][4])))
	print("Recall of level 4: ", (Confusion_Matrix[3][3] / (Confusion_Matrix[3][0] + Confusion_Matrix[3][1] + Confusion_Matrix[3][2] + Confusion_Matrix[3][3] + Confusion_Matrix[3][4])))
	print("Recall of level 5: ", (Confusion_Matrix[4][4] / (Confusion_Matrix[4][0] + Confusion_Matrix[4][1] + Confusion_Matrix[4][2] + Confusion_Matrix[4][3] + Confusion_Matrix[4][4])))
	print("Accuracy: ", ((Confusion_Matrix[0][0] + Confusion_Matrix[1][1] + Confusion_Matrix[2][2] + Confusion_Matrix[3][3] + Confusion_Matrix[4][4]) / (Confusion_Matrix[0][0] + Confusion_Matrix[0][1] + Confusion_Matrix[0][2] + Confusion_Matrix[0][3] + Confusion_Matrix[0][4] + Confusion_Matrix[1][0] + Confusion_Matrix[1][1] + Confusion_Matrix[1][2] + Confusion_Matrix[1][3] + Confusion_Matrix[1][4] + Confusion_Matrix[2][0] + Confusion_Matrix[2][1] + Confusion_Matrix[2][2] + Confusion_Matrix[2][3] + Confusion_Matrix[2][4] + Confusion_Matrix[3][0] + Confusion_Matrix[3][1] + Confusion_Matrix[3][2] + Confusion_Matrix[3][3] + Confusion_Matrix[3][4] + Confusion_Matrix[4][0] + Confusion_Matrix[4][1] + Confusion_Matrix[4][2] + Confusion_Matrix[4][3] + Confusion_Matrix[4][4])))

#start timer
start_time = time.time()

#read file
gps = pd.read_csv('googleplaystore.csv',sep = ',')

#delete illegal character
Cate_cnt = gps.Category.value_counts()
for key in Cate_cnt.keys():
	if Cate_cnt[key] <= 1:
		err = key        
err_idx = gps[gps.Category == err].index.values
gps.drop(err_idx,inplace=True)

#delete "+" in 'Installs'
j = 0
for line in gps["Installs"].values:
	gps["Installs"].values[j]="".join(gps["Installs"].values[j].split("+"))
	gps["Installs"].values[j]="".join(gps["Installs"].values[j].split(","))
	gps["Installs"].values[j] = float(gps["Installs"].values[j])
	j += 1
	
j = 0
for line in gps["Price"].values:
	gps["Price"].values[j] = "".join(gps["Price"].values[j].split("$"))
	gps["Price"].values[j] = float(gps["Price"].values[j])
	j += 1
#make 'Installs' < 1000 be 0, >= 1000 and < 10000000 be 1, >= 10000000 be 2	
gps.loc[gps['Installs'] < 1000, 'Installs'] = 0
gps.loc[gps['Installs'] >= 10000000 , 'Installs'] = 4
gps.loc[gps['Installs'] >= 1000000 , 'Installs'] = 3		
gps.loc[gps['Installs'] >= 100000 , 'Installs'] = 2
gps.loc[gps['Installs'] >= 1000 , 'Installs'] = 1

#make 'Type' free be 0,paid be 1
gps.loc[gps['Type'] == 'Free', 'Type'] = 0
gps.loc[gps['Type'] == 'Paid', 'Type'] = 1


#size
j = 0   
for line in gps["Size"].values:
   
    if 'M' in line:
        gps["Size"].values[j] = "".join(gps["Size"].values[j].split("M"))
        gps["Size"].values[j] = float(gps["Size"].values[j])
        gps["Size"].values[j] = (gps["Size"].values[j]) * 1024
        j += 1
        
    elif 'k' in line:
        gps["Size"].values[j] = "".join(gps["Size"].values[j].split("k"))
        gps["Size"].values[j] = float(gps["Size"].values[j])
        j += 1
        
    else:
        gps["Size"].values[j] = 0.0
        j += 1

#print(gps['Size'].value_counts())
j = 0
sum = 0
counts = 0
for line in gps["Size"].values:
	if(gps["Size"].values[j] != 0.0):
		sum += gps["Size"].values[j]
		counts += 1
	j += 1
avg = sum/counts
gps.loc[gps['Size'] == 0.0, 'Size'] = avg

#gps.loc[gps['Size'] < avg, 'Size'] = 0
#gps.loc[gps['Size'] >= 10*avg , 'Size'] = 2
#gps.loc[gps['Size'] >= avg , 'Size'] = 1
	
#make 'Category' be number
gps.loc[gps['Category'] == 'FAMILY', 'Category'] = 0 
gps.loc[gps['Category'] == 'GAME', 'Category'] = 1 
gps.loc[gps['Category'] == 'TOOLS', 'Category'] = 2 
gps.loc[gps['Category'] == 'MEDICAL', 'Category'] = 3 
gps.loc[gps['Category'] == 'BUSINESS', 'Category'] = 4 
gps.loc[gps['Category'] == 'PRODUCTIVITY', 'Category'] = 5 
gps.loc[gps['Category'] == 'PERSONALIZATION', 'Category'] = 6 
gps.loc[gps['Category'] == 'COMMUNICATION', 'Category'] = 7 
gps.loc[gps['Category'] == 'SPORTS', 'Category'] = 8 
gps.loc[gps['Category'] == 'LIFESTYLE', 'Category'] = 9 
gps.loc[gps['Category'] == 'FINANCE', 'Category'] = 10
gps.loc[gps['Category'] == 'HEALTH_AND_FITNESS', 'Category'] = 11
gps.loc[gps['Category'] == 'PHOTOGRAPHY', 'Category'] = 12
gps.loc[gps['Category'] == 'SOCIAL', 'Category'] = 13
gps.loc[gps['Category'] == 'NEWS_AND_MAGAZINES', 'Category'] = 14
gps.loc[gps['Category'] == 'SHOPPING', 'Category'] = 15
gps.loc[gps['Category'] == 'TRAVEL_AND_LOCAL', 'Category'] = 16
gps.loc[gps['Category'] == 'DATING', 'Category'] = 17
gps.loc[gps['Category'] == 'BOOKS_AND_REFERENCE', 'Category'] = 18
gps.loc[gps['Category'] == 'VIDEO_PLAYERS', 'Category'] = 19
gps.loc[gps['Category'] == 'EDUCATION', 'Category'] = 20
gps.loc[gps['Category'] == 'ENTERTAINMENT', 'Category'] = 21
gps.loc[gps['Category'] == 'MAPS_AND_NAVIGATION', 'Category'] = 22
gps.loc[gps['Category'] == 'FOOD_AND_DRINK', 'Category'] = 23
gps.loc[gps['Category'] == 'HOUSE_AND_HOME', 'Category'] = 24
gps.loc[gps['Category'] == 'AUTO_AND_VEHICLES', 'Category'] = 25
gps.loc[gps['Category'] == 'LIBRARIES_AND_DEMO', 'Category'] = 26
gps.loc[gps['Category'] == 'WEATHER', 'Category'] = 27
gps.loc[gps['Category'] == 'ART_AND_DESIGN', 'Category'] = 28
gps.loc[gps['Category'] == 'EVENTS', 'Category'] = 29
gps.loc[gps['Category'] == 'PARENTING', 'Category'] = 30
gps.loc[gps['Category'] == 'COMICS', 'Category'] = 31
gps.loc[gps['Category'] == 'BEAUTY', 'Category'] = 32

#make reviews float
j = 0
for line in gps["Reviews"].values:
	gps["Reviews"].values[j] = float(gps["Reviews"].values[j])
	j += 1

#make 'Content Rating' be numbers
gps.loc[gps['Content Rating'] == 'Everyone', 'Content Rating'] = 0 
gps.loc[gps['Content Rating'] == 'Teen', 'Content Rating'] = 1 
gps.loc[gps['Content Rating'] == 'Mature 17+', 'Content Rating'] = 2 
gps.loc[gps['Content Rating'] == 'Everyone 10+', 'Content Rating'] = 3 
gps.loc[gps['Content Rating'] == 'Adults only 18+', 'Content Rating'] = 4 
gps.loc[gps['Content Rating'] == 'Unrated', 'Content Rating'] = 5

#rating
gps = gps.fillna({'Category':0.0})
gps = gps.fillna({'Rating':0.0})
gps = gps.fillna({'Reviews':0.0})
gps = gps.fillna({'Type':0.0})
gps = gps.fillna({'Price':0.0})
gps = gps.fillna({'Content Rating':0.0})
gps = gps.fillna({'Installs':0.0})
gps = gps.fillna({'Size':0.0})

#delete useless features
gps = gps.drop(['Android Ver'],axis = 1)
gps = gps.drop(['App'],axis = 1)
gps = gps.drop(['Last Updated'],axis = 1)
gps = gps.drop(['Current Ver'],axis = 1)
gps = gps.drop(['Genres'],axis = 1)

temp = gps
category_data_temp = gps[['Category']].values
rating_data_temp = gps[['Rating']].values
reviews_data_temp = gps[['Reviews']].values
type_data_temp = gps[['Type']].values
price_data_temp = gps[['Price']].values
content_rating_data_temp = gps[['Content Rating']].values
target_temp = gps[['Installs']].values
size_data_temp = gps[['Size']].values

category_data = category_data_temp.tolist()
rating_data = rating_data_temp.tolist()
reviews_data = reviews_data_temp.tolist()
type_data = type_data_temp.tolist()
price_data = price_data_temp.tolist()
content_rating_data = content_rating_data_temp.tolist()
size_data = size_data_temp.tolist()
target = target_temp.tolist()
temp = temp.drop(['Installs'],axis = 1)
temp = temp.values
data = temp.tolist()



#split data and target for training and testing
x_train, x_test, y_train, y_test = train_test_split(
	gps[['Category', 'Rating', 'Reviews', 'Size', 'Type', 'Price', 'Content Rating']], gps[['Installs']], test_size = 0.3,random_state = 0)
#data_train, data_test, target_train, target_test = train_test_split(data, target, test_size = 0.3)
data_train = x_train.values
data_train = data_train.tolist()
data_test = x_test.values
data_test = data_test.tolist()
target_train = y_train.values
target_train = target_train.tolist()
target_test = y_test.values
target_test = target_test.tolist()


###################
####PLOT FIGURE####
###################
#plot rating fig
fig = plt.figure()
ax = plt.subplot()
plt.title('Average value and Standard deviation of Rating')
ax.errorbar(1,gps['Rating'].mean(),yerr=gps['Rating'].std(), fmt='o',capsize=5,elinewidth=2)
plt.xlim([0.5,1.5])
ax.set_xticks([])
plt.xlabel("mean= "+format(gps['Rating'].mean(),'.2f')+"      std= "+format(gps['Rating'].std(),'.2f'))
plt.legend(['Errorbar of Rating'],loc='best')
plt.show()

fig = plt.figure()
ax = plt.subplot()
plt.title('Rating scatter')
ax.scatter(range(len(gps['Rating'])),gps['Rating'],s=5)
plt.show()

#plot review fig
fig = plt.figure()
ax = plt.subplot()
plt.title('Average value and Standard deviation of Reviews')
ax.errorbar(1,gps['Reviews'].mean(),yerr=gps['Reviews'].std(), fmt='o',capsize=5,elinewidth=2)
plt.xlim([0.5,1.5])
ax.set_xticks([])
plt.xlabel("mean= "+format(gps['Reviews'].mean(),'.2f')+"      std= "+format(gps['Reviews'].std(),'.2f'))
plt.legend(['Errorbar of Reviews'],loc='best')
plt.show()

fig = plt.figure()
ax = plt.subplot()
plt.title('Reviews scatter')
ax.scatter(range(len(gps['Reviews'])),gps['Reviews'],s=5)
plt.show()

#plot Size fig
fig = plt.figure()
ax = plt.subplot()
plt.title('Average value and Standard deviation of Size')
ax.errorbar(1,gps['Size'].mean(),yerr=gps['Size'].std(), fmt='o',capsize=5,elinewidth=2)
plt.xlim([0.5,1.5])
ax.set_xticks([])
plt.xlabel("mean= "+format(gps['Size'].mean(),'.2f')+"(KB)      std= "+format(gps['Size'].std(),'.2f'))
plt.legend(['Errorbar of Size'],loc='best')
plt.show()

fig = plt.figure()
ax = plt.subplot()
plt.title('Size scatter')
ax.scatter(range(len(gps['Size'])),gps['Size'],s=5)
plt.show()

#plot Installs fig
fig = plt.figure()
ax = plt.subplot()
plt.title('Average value and Standard deviation of Installs distributed')
ax.errorbar(1,gps['Installs'].astype(float).mean(), yerr=gps['Installs'].astype(float).std(), fmt='o', capsize=5, elinewidth=2)
plt.ylim([0,4])
ax.set_xticks([])
plt.xlabel("mean= "+format(gps['Installs'].astype(float).mean(),'.2f')+"      std= "+format(gps['Installs'].astype(float).std(),'.2f'))
plt.legend(['Errorbar of Installs distributed'],loc='best')
plt.show()

fig = plt.figure()
ax = plt.subplot()
plt.title('Bar of Installs distributed')
ax.bar(['<1000','>1000','>100000','>1000000','>10000000'],gps['Installs'].value_counts().sort_index().values)
plt.show()

#plot price fig
fig = plt.figure()
ax = plt.subplot()
plt.title('Average value and Standard deviation of Price')
ax.errorbar(1,gps['Price'].mean(),yerr=gps['Price'].std(), fmt='o',capsize=5,elinewidth=2)
plt.xlim([0.5,1.5])
ax.set_xticks([])
plt.xlabel("mean= $"+format(gps['Price'].mean(),'.2f')+"      std= "+format(gps['Price'].std(),'.2f'))
plt.legend(['Errorbar of Price'],loc='best')
plt.show()

fig = plt.figure()
ax = plt.subplot()
plt.title('Price scatter')
ax.scatter(range(len(gps['Price'])),gps['Price'],s=5)
plt.show()

#####PLOT END#####


#single decision tree with resubstitution
clf = tree.DecisionTreeClassifier()
clf = clf.fit(data, target)
result = clf.predict(data)
Confusion_Matrix = confusion_matrix(target, result, labels = [0, 1, 2, 3, 4])
print("Confusion Matrix with resubstitution (single decision tree): ")
print_confusion_matrix(Confusion_Matrix)
print("--- %s seconds ---" % (time.time() - start_time))

#single decision tree with k-fold validation
Confusion_Matrix = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
num_folds = 2
subset_size = int(len(data_train) / num_folds)
for i in range(len(data_test)):
	query_data = data_test[i]
	query_target = target_test[i]
	vote = [0, 0, 0, 0, 0]
	for j in range(num_folds):
	    data_train_temp = data_train[ : j * subset_size] + data_train[(j + 1) * subset_size : ]
	    target_train_temp = target_train[ : j * subset_size] + target_train[(j + 1) * subset_size : ]
	    clf = tree.DecisionTreeClassifier()
	    clf = clf.fit(data_train_temp, target_train_temp)
	    result = clf.predict(np.reshape(query_data, (1, -1)))
	    vote[result[0]] += 1
	Confusion_Matrix[query_target[0]][vote.index(max(vote))] += 1
print("Confusion Matrix with k-fold validation (single decision tree): ")
print_confusion_matrix(Confusion_Matrix)
print("--- %s seconds ---" % (time.time() - start_time))

#random forest model with resubstitution
Confusion_Matrix = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
vote = [[0 for i in range(5)] for j in range(int(len(data)))]
for i in range(3):
	for k in range(len(data)):
		temp1 = gps
		temp1 = temp1.drop(['Installs'],axis = 1)
		temp1 = temp1.values
		query_data = temp1[k]
		data1 = temp1.tolist()
		query_data = query_data.tolist()

		#rf_data = copy.deepcopy(data)
		#rf_query_data = copy.deepcopy(query_data)
		counter = 0
		select = []
		for j in range(0,7):
			if random.uniform(0,1) <= 0.5:
				select.append(1)
				query_data.pop(6-j)
				counter += 1
			else:
				select.append(0)
		if counter == 7:
			continue
		for j in range((len(data))):
			if(select[0]):
				data1[j].pop(6)
			if(select[1]):
				data1[j].pop(5)
			if(select[2]):
				data1[j].pop(4)
			if(select[3]):
				data1[j].pop(3)
			if(select[4]):
				data1[j].pop(2)
			if(select[5]):
				data1[j].pop(1)
			if(select[6]):
				data1[j].pop(0)
		clf = tree.DecisionTreeClassifier()
		clf = clf.fit(data1, target)
		result = clf.predict(np.reshape(query_data, (1, -1)))
		if result[0] == 0:
			vote[k][0] += 1
		elif result[0] == 1:
			vote[k][1] += 1
		elif result[0] == 2:
			vote[k][2] += 1
		elif result[0] == 3:
			vote[k][3] += 1
		elif result[0] == 4:
			vote[k][4] += 1
for k in range(len(data)):
	query_target = target[k]
	Confusion_Matrix[query_target[0]][vote[k].index(max(vote[k]))] += 1
print("Confusion Matrix with resubstitution (random forest): ")
print_confusion_matrix(Confusion_Matrix)
print("--- %s seconds ---" % (time.time() - start_time))


#random forest model with k-fold validation
num_folds = 2
subset_size = int(len(data_train) / num_folds)
Confusion_Matrix = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
vote = [[0 for i in range(5)] for j in range(int(len(data_test)))]
for i in range(3):
	for k in range(len(target_test)):
		rf_data = x_train.values
		rf_data = rf_data.tolist()
		query_data = x_test.values
		query_data = query_data[k]
		query_data = query_data.tolist()
		
		k_fold_vote = [0, 0, 0, 0, 0]
		#rf_data = copy.deepcopy(data_train)
		#rf_query_data = copy.deepcopy(query_data)
		counter = 0
		select = []
		for j in range(0,7):
			if random.uniform(0,1) <= 0.5:
				select.append(1)
				query_data.pop(6-j)
				counter += 1
			else:
				select.append(0)
		if counter == 7:
			continue
		for j in range((len(data_train))):
			if(select[0]):
				rf_data[j].pop(6)
			if(select[1]):
				rf_data[j].pop(5)
			if(select[2]):
				rf_data[j].pop(4)
			if(select[3]):
				rf_data[j].pop(3)
			if(select[4]):
				rf_data[j].pop(2)
			if(select[5]):
				rf_data[j].pop(1)
			if(select[6]):
				rf_data[j].pop(0)
		
		for j in range(num_folds):
			data_train_temp = rf_data[ : j * subset_size] + rf_data[(j + 1) * subset_size : ]
			target_train_temp = target_train[ : j * subset_size] + target_train[(j + 1) * subset_size : ]
			clf = tree.DecisionTreeClassifier()
			clf = clf.fit(data_train_temp, target_train_temp)
			result = clf.predict(np.reshape(query_data, (1, -1)))
			k_fold_vote[result[0]] += 1
		vote[k][k_fold_vote.index(max(k_fold_vote))] += 1
for k in range(len(data_test)):
	query_target = target_test[k]
	Confusion_Matrix[query_target[0]][vote[k].index(max(vote[k]))] += 1
print("Confusion Matrix with k-fold validation (random forest): ")
print_confusion_matrix(Confusion_Matrix)
print("--- %s seconds ---" % (time.time() - start_time))
