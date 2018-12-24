import numpy as np

f = open("label.txt", 'r')
label_list = f.read().splitlines()
f.close()
label = np.asarray(label_list)
label = label.astype(np.int)

for i in range (0, label.size):
	label[i] = (int)(label[i] / 1000)

np.savetxt("label_classification.txt", label, delimiter = '\n', fmt = '%d')