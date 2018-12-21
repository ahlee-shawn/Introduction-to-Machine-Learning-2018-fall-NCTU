import numpy as np
import kdtree as kd
import matplotlib.pyplot as plt
import copy

x = []
y = []

with open('points.txt','r') as file:
	line = file.readlines()
	for i in line:
		row = i.split()
		x.append(int(row[0]))
		y.append(int(row[1]))

points = []
for i in range(len(x)):
	points.append([x[i],y[i]])


kd_tree = kd.create(points)

#kd.visualize(kd_tree)

print(len(kd_tree.data))

#print(type(kd_tree))

level_list = list(kd.level_order(kd_tree))


data = []

#for i in range(len(level_list)):
#	data.append(level_list[i].data)

#print(data)

#print(type(empty_tree.left))
def plt_tree(kd_tree,min_x,max_x,min_y,max_y,prv_node,branch, depth = 0):
	current = kd_tree.data
	left_sub = kd_tree.left
	right_sub = kd_tree.right
	if kd_tree.data is None:
		return
	#print(left_sub)
	#print(type(left_sub))
	#print(right_sub)
	#print(type(right_sub))
	#print(kd_tree)
	axis = depth % 2

	#verticle
	if axis == 0:
		if branch is not None and prv_node is not None:
			if branch:
				max_y = prv_node[1]
			else:
				min_y = prv_node[1]
		plt.plot([current[0],current[0]], [min_y,max_y], linestyle='-', color='red')
	#horizontal
	elif axis == 1:
		if branch is not None and prv_node is not None:
			if branch:
				max_x = prv_node[0]
			else:
				min_x = prv_node[0]
		plt.plot([min_x,max_x],[current[1],current[1]], linestyle='-', color='blue')

	plt.plot(current[0], current[1], 'ko')

	if left_sub is not None:
		plt_tree(left_sub,min_x,max_x,min_y,max_y,current,True,depth+1)
	

	if right_sub is not None:
		plt_tree(right_sub,min_x,max_x,min_y,max_y,current,False,depth+1)
	


n = len(x)
max_val = 10
min_val = 0
delta = 0
plt.figure("K-d Tree", figsize=(10., 10.))
plt.axis( [min_val-delta, max_val+delta, min_val-delta, max_val+delta] )
 
#plt.grid(b=True, which='major', color='0.75', linestyle='--')
plt.xticks([i for i in range(min_val-delta, max_val+delta, 1)])
plt.yticks([i for i in range(min_val-delta, max_val+delta, 1)])
 
# draw the tree
plt_tree(kd_tree, min_val-delta, max_val+delta, min_val-delta, max_val+delta,None,None)
 
plt.title('K-D Tree')
plt.show()
plt.close()