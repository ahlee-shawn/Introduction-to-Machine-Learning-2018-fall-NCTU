import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data_in = pd.read_csv('data_noah.csv', sep = ',')

#the fundamental datasets
x_temp = data_in[['x']].values
y_temp = data_in[['y']].values
target_temp = data_in[['pitch_type']].values

#two more attributes to analyze
spin_temp = data_in[['spin']].values
speed_temp = data_in[['speed']].values

data_x = list(map(float, x_temp))
data_y = list(map(float, y_temp))
data_spin = list(map(float, spin_temp))
data_speed = list(map(float, speed_temp))
data_target = target_temp.tolist()

data_cluster = np.zeros(max(len(data_x), len(data_y))) # i for data_cluster[i] belongs to i(th) cluster (i = 0, 1, 2)
cluster_name = [0, 0, 0]

#number of cluster
k = 3

#create k random centroids
C_x= np.random.randint(low = min(data_x) + 5, high = max(data_x) - 5, size = k)
C_y= np.random.randint(low = min(data_y) + 5, high = max(data_y) - 5, size = k)

#distance function
def distance(x1, y1, x2, y2):
	return(x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)

#K means algorithm
distance_to_C = [0.0, 0.0, 0.0]
while(1):
	change = 0
	#assign each node to the nearest centroid
	for i in range(0, min(len(data_x), len(data_y))):
		for j in range(0, 3):
			distance_to_C[j] = distance(data_x[i], data_y[i], C_x[j], C_y[j])
		if data_cluster[i] != distance_to_C.index(min(distance_to_C)):
			data_cluster[i] = distance_to_C.index(min(distance_to_C))
			change = 1
	if change == 0:
		break
	#find new cluster centroid
	new_C0_x = 0.0
	new_C0_y = 0.0
	count_C0 = 0.0
	new_C1_x = 0.0
	new_C1_y = 0.0
	count_C1 = 0.0
	new_C2_x = 0.0
	new_C2_y = 0.0
	count_C2 = 0.0
	for i in range(0, min(len(data_x), len(data_y))):
		if(data_cluster[i] == 0):
			count_C0 += 1.0
			new_C0_x += data_x[i]
			new_C0_y += data_y[i]
		elif(data_cluster[i] == 1):
			count_C1 += 1.0
			new_C1_x += data_x[i]
			new_C1_y += data_y[i]
		elif(data_cluster[i] == 2):
			count_C2 += 1.0
			new_C2_x += data_x[i]
			new_C2_y += data_y[i]
	if(count_C0 != 0):
		C_x[0] = new_C0_x / count_C0
		C_y[0] = new_C0_y / count_C0
	if(count_C1 != 0):
		C_x[1] = new_C1_x / count_C1
		C_y[1] = new_C1_y / count_C1
	if(count_C2 != 0):
		C_x[2] = new_C2_x / count_C2
		C_y[2] = new_C2_y / count_C2

#divide data into k labels ,calculate cost function and calculate accuarcy by finding the data closest to the centroids to represent each cluster
cluster0_x = []
cluster0_y = []
cluster1_x = []
cluster1_y = []
cluster2_x = []
cluster2_y = []
cost = 0.0
cluster0 = [0, 0, 0]
cluster1 = [0, 0, 0]
cluster2 = [0, 0, 0]
for i in range(0, min(len(data_x), len(data_y))):
	if(data_cluster[i] == 0):
		cluster0_x = np.append(cluster0_x, data_x[i])
		cluster0_y = np.append(cluster0_y, data_y[i])
		cost += distance(data_x[i], data_y[i], C_x[0], C_y[0])
		if(data_target[i][0] == 'FF'):
			cluster0[0] += 1
		elif(data_target[i][0] == 'CH'):
			cluster0[1] += 1
		elif(data_target[i][0] == 'CU'):
			cluster0[2] += 1
	elif(data_cluster[i] == 1):
		cluster1_x = np.append(cluster1_x, data_x[i])
		cluster1_y = np.append(cluster1_y, data_y[i])
		cost += distance(data_x[i], data_y[i], C_x[1], C_y[1])
		if(data_target[i][0] == 'FF'):
			cluster1[0] += 1
		elif(data_target[i][0] == 'CH'):
			cluster1[1] += 1
		elif(data_target[i][0] == 'CU'):
			cluster1[2] += 1
	elif(data_cluster[i] == 2):
		cluster2_x = np.append(cluster2_x, data_x[i])
		cluster2_y = np.append(cluster2_y, data_y[i])
		cost += distance(data_x[i], data_y[i], C_x[2], C_y[2])
		if(data_target[i][0] == 'FF'):
			cluster2[0] += 1
		elif(data_target[i][0] == 'CH'):
			cluster2[1] += 1
		elif(data_target[i][0] == 'CU'):
			cluster2[2] += 1
print("The cost function of x and y is: " + str(cost))
if(cluster0[2] > cluster1[2] and cluster0[2] > cluster2[2]):
	cluster_name[0] = 2
if(cluster1[2] > cluster0[2] and cluster1[2] > cluster2[2]):
	cluster_name[1] = 2
if(cluster2[2] > cluster1[2] and cluster2[2] > cluster0[2]):
	cluster_name[2] = 2
if(cluster0[1] > cluster1[1] and cluster0[1] > cluster2[1]):
	cluster_name[0] = 1
if(cluster1[1] > cluster0[1] and cluster1[1] > cluster2[1]):
	cluster_name[1] = 1
if(cluster2[1] > cluster1[1] and cluster2[1] > cluster0[1]):
	cluster_name[2] = 1
if(cluster0[0] > cluster1[0] and cluster0[0] > cluster2[0]):
	cluster_name[0] = 0
if(cluster1[0] > cluster0[0] and cluster1[0] > cluster2[0]):
	cluster_name[1] = 0
if(cluster2[0] > cluster1[0] and cluster2[0] > cluster0[0]):
	cluster_name[2] = 0
correct = 0.0
for i in range(0, len(data_target)):
	if((data_target[i][0] == 'FF' and data_cluster[i] == 0) or (data_target[i][0] == 'CH' and data_cluster[i] == 1) or (data_target[i][0] == 'CU' and data_cluster[i] == 2)):
		correct += 1.0;
print("The accuarcy of x and y is: " + str(correct / len(data_target)))

#draw the figure
plt.title('K means with xy')
plt.xlabel('X (horizontal movement)')
plt.ylabel('Y (vertical movement)')
plt.scatter(cluster0_x, cluster0_y, c = 'y')
plt.scatter(cluster1_x, cluster1_y, c = 'g')
plt.scatter(cluster2_x, cluster2_y, c = 'b')
plt.scatter(C_x, C_y, c = 'r', marker = '*')
plt.show()

#initialization
cluster_name = [0, 0, 0]

#two other attrbutes to partition
data_cluster = np.zeros(max(len(data_spin), len(data_speed))) # i for data_cluster[i] belongs to i(th) cluster (i = 0, 1, 2)

#create k random centroids
C_x= np.random.randint(low = min(data_spin) + 50, high = max(data_spin) - 50, size = k)
C_y= np.random.randint(low = min(data_speed) + 20, high = max(data_speed), size = k)

#K means algorithm
distance_to_C = [0.0, 0.0, 0.0]
while(1):
	change = 0
	#assign each node to the nearest centroid
	for i in range(0, min(len(data_spin), len(data_speed))):
		for j in range(0, 3):
			distance_to_C[j] = distance(data_spin[i], data_speed[i], C_x[j], C_y[j])
		if data_cluster[i] != distance_to_C.index(min(distance_to_C)):
			data_cluster[i] = distance_to_C.index(min(distance_to_C))
			change = 1
	if change == 0:
		break
	#find new cluster centroid
	new_C0_x = 0.0
	new_C0_y = 0.0
	count_C0 = 0.0
	new_C1_x = 0.0
	new_C1_y = 0.0
	count_C1 = 0.0
	new_C2_x = 0.0
	new_C2_y = 0.0
	count_C2 = 0.0
	for i in range(0, min(len(data_spin), len(data_speed))):
		if(data_cluster[i] == 0):
			count_C0 += 1.0
			new_C0_x += data_spin[i]
			new_C0_y += data_speed[i]
		elif(data_cluster[i] == 1):
			count_C1 += 1.0
			new_C1_x += data_spin[i]
			new_C1_y += data_speed[i]
		elif(data_cluster[i] == 2):
			count_C2 += 1.0
			new_C2_x += data_spin[i]
			new_C2_y += data_speed[i]
	if(count_C0 != 0):
		C_x[0] = new_C0_x / count_C0
		C_y[0] = new_C0_y / count_C0
	if(count_C1 != 0):
		C_x[1] = new_C1_x / count_C1
		C_y[1] = new_C1_y / count_C1
	if(count_C2 != 0):
		C_x[2] = new_C2_x / count_C2
		C_y[2] = new_C2_y / count_C2

#divide data into k labels ,calculate cost function and calculate accuarcy by finding the data closest to the centroids to represent each cluster
cluster0_x = []
cluster0_y = []
cluster1_x = []
cluster1_y = []
cluster2_x = []
cluster2_y = []
cost = 0.0
cluster0 = [0, 0, 0]
cluster1 = [0, 0, 0]
cluster2 = [0, 0, 0]
for i in range(0, min(len(data_spin), len(data_speed))):
	if(data_cluster[i] == 0):
		cluster0_x = np.append(cluster0_x, data_spin[i])
		cluster0_y = np.append(cluster0_y, data_speed[i])
		cost += distance(data_spin[i], data_speed[i], C_x[0], C_y[0])
		if(data_target[i][0] == 'FF'):
			cluster0[0] += 1
		elif(data_target[i][0] == 'CH'):
			cluster0[1] += 1
		elif(data_target[i][0] == 'CU'):
			cluster0[2] += 1
	elif(data_cluster[i] == 1):
		cluster1_x = np.append(cluster1_x, data_spin[i])
		cluster1_y = np.append(cluster1_y, data_speed[i])
		cost += distance(data_spin[i], data_speed[i], C_x[1], C_y[1])
		if(data_target[i][0] == 'FF'):
			cluster1[0] += 1
		elif(data_target[i][0] == 'CH'):
			cluster1[1] += 1
		elif(data_target[i][0] == 'CU'):
			cluster1[2] += 1
	elif(data_cluster[i] == 2):
		cluster2_x = np.append(cluster2_x, data_spin[i])
		cluster2_y = np.append(cluster2_y, data_speed[i])
		cost += distance(data_spin[i], data_speed[i], C_x[2], C_y[2])
		if(data_target[i][0] == 'FF'):
			cluster2[0] += 1
		elif(data_target[i][0] == 'CH'):
			cluster2[1] += 1
		elif(data_target[i][0] == 'CU'):
			cluster2[2] += 1
print("The cost function of x and y is: " + str(cost))
if(cluster0[2] > cluster1[2] and cluster0[2] > cluster2[2]):
	cluster_name[0] = 2
if(cluster1[2] > cluster0[2] and cluster1[2] > cluster2[2]):
	cluster_name[1] = 2
if(cluster2[2] > cluster1[2] and cluster2[2] > cluster0[2]):
	cluster_name[2] = 2
if(cluster0[1] > cluster1[1] and cluster0[1] > cluster2[1]):
	cluster_name[0] = 1
if(cluster1[1] > cluster0[1] and cluster1[1] > cluster2[1]):
	cluster_name[1] = 1
if(cluster2[1] > cluster1[1] and cluster2[1] > cluster0[1]):
	cluster_name[2] = 1
if(cluster0[0] > cluster1[0] and cluster0[0] > cluster2[0]):
	cluster_name[0] = 0
if(cluster1[0] > cluster0[0] and cluster1[0] > cluster2[0]):
	cluster_name[1] = 0
if(cluster2[0] > cluster1[0] and cluster2[0] > cluster0[0]):
	cluster_name[2] = 0
correct = 0.0
for i in range(0, len(data_target)):
	if((data_target[i][0] == 'FF' and data_cluster[i] == 0) or (data_target[i][0] == 'CH' and data_cluster[i] == 1) or (data_target[i][0] == 'CU' and data_cluster[i] == 2)):
		correct += 1.0;
print("The accuarcy of x and y is: " + str(correct / len(data_target)))

#draw the figure
plt.title('K means with spin speed')
plt.xlabel('X (spin)')
plt.ylabel('Y (speed)')
plt.scatter(cluster0_x, cluster0_y, c = 'y')
plt.scatter(cluster1_x, cluster1_y, c = 'g')
plt.scatter(cluster2_x, cluster2_y, c = 'b')
plt.scatter(C_x, C_y, c = 'r', marker = '*')
plt.show()