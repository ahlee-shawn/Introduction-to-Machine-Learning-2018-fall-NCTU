import matplotlib.pyplot as plt
import numpy as np

f = open("label.txt", 'r')
label_list = f.read().splitlines()
f.close()
label = np.asarray(label_list)
label = label.astype(np.int)

label = np.sort(label)

x = []
y = []

prev = -1
for i in range(label.size):
	if(label[i] != prev):
		x.insert(len(x), label[i])
		y.insert(len(y), 1)
		prev = label[i]
	else:
		y[len(y) - 1] += 1

plt.title("Dataset Distribution")
plt.plot(x, y)
plt.xlabel("Price")
plt.ylabel("Quantity")
plt.show()