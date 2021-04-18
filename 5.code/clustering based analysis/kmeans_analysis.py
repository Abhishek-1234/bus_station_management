import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import pickle
from math import radians,sin,cos
import os


if (os.path.exists(os.getcwd()+'\\b_data.pickle')):
	pass
else:
	data = pd.read_csv('stops_detail.csv')
	no_of_p = []
	lat = data['stop_lat']
	lon = data['stop_lon']
	no_of_p = data['no_of_passengers_avg']

	tid = np.array((lat,lon,no_of_p), dtype = float)
	tid = tid.T

	with open('b_data','wb') as f:
		pickle.dump(tid,f)

f = open('b_data','rb')
A = pickle.load(f)
f.close()

# print(type(A))
# print(A[:3][:3])

a = A[:,0]
b = A[:,1]
c = A[:,2]

# normalization
A[:,0] = (a - a.min())/(a.max() - a.min())
A[:,1] = (b - b.min())/(b.max() - b.min())
A[:,2] = (c - c.min())/(c.max() - c.min())

# print(A[:4][:4])

#########################################################
#elbow method
Error =[]
for i in range(1, 20):
    kmeans = KMeans(n_clusters = i).fit(A)
    kmeans.fit(A)
    Error.append(kmeans.inertia_)

plt.plot(range(1, 20), Error)
plt.title('Elbow method')
plt.xlabel('No of clusters')
plt.ylabel('Error')
plt.show()
#########################################################

k = int(input('Check with some cluster number: '))
kmeans = KMeans(n_clusters = k, max_iter = 1500)
y_means = kmeans.fit_predict(A)


########################################################
print('No. of iterations: ', kmeans.n_iter_)
print('Sum squared of error: ', kmeans.inertia_)
label = kmeans.labels_
dictionary = {}
for k in label:
	if k not in dictionary.keys():
		dictionary[k] = 1
		continue
	dictionary[k] = dictionary[k] + 1

for k,v in dictionary.items():
	print('cluster ',k,': ', v)

for i in kmeans.cluster_centers_:
	print(i)
#########################################################
plt.scatter(A[:,0],A[:,1], c = y_means, cmap = 'rainbow')
plt.xlabel('Lat')
plt.ylabel('Lon')
plt.show()

plt.scatter(A[:,0],A[:,2], c = y_means, cmap = 'rainbow')
plt.xlabel('Lat')
plt.ylabel('Avg_no_passengers')
plt.show()


plt.scatter(A[:,1],A[:,2], c = y_means, cmap = 'rainbow')
plt.xlabel('Lon')
plt.ylabel('Avg_no_passengers')
plt.show()