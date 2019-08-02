# Finding the dominant color in a picture. The exercise is derived from https://www.dataquest.io/blog/tutorial-colors-image-clustering-python/


import matplotlib, scipy, PIL
import matplotlib.pyplot as plt
from matplotlib import image as img
image = img.imread('./sample.jpg')

print(image.shape)
# the shape is 200 x 200 for R, G and B
red = image[:,:,0]
green = image[:,:,1]
blue = image[:,:,2]

# Visualize the data
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
fig = plt.figure()
ax = fig.gca(projection = '3d')
surf = ax.plot_surface(red,blue,green, cmap = cm.coolwarm)
# based on the 3D surface map, the data is dominated by 2 colors (2 strong arms)


# rehape the data so can be converted to pandas dataframe
import numpy as np
import pandas as pd
# use whiten in scipy to normalize the data
from scipy.cluster.vq import whiten, kmeans
re_red = list(np.reshape(red,(40000,1)))
re_green = list(np.reshape(green,(40000,1)))
re_blue = list(np.reshape(blue,(40000,1)))
whiten_red = list(whiten(re_red))
whiten_green = list(whiten(re_green))
whiten_blue = list(whiten(re_blue))

df = pd.DataFrame({'red': re_red,'blue': re_blue,'green': re_green)
df.shape
# The shape is 4000 x 6. That is correct because it has 6 features and each has 200 x 200 pixels.
                   
# Make a numpy array with 0s and fill the table with normalized data
observations = np.zeros((40000,3))
observations[:,0] = whiten_red
observations[:,1] = whiten_green
observations[:,2] = whiten_blue

cluster_centers, distortion = kmeans(observations, 2)

# the cluster_centers are based on whiten data. To get the unscaled cluster_centers, multiply them with their standard deviations
red_std, green_std, blue_std = df[['red','green','blue']].std()
cluster_centers[:,0] = cluster_centers[:,0]*red_std
cluster_centers[:,1] = cluster_centers[:,1]*green_std
cluster_centers[:,2] = cluster_centers[:,2]*blue_std
# add the cluster centers to the scatterplot to see if they do locate at the center of the plot
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(red, green, blue)
ax.scatter(cluster_centers[:,0], cluster_centers[:,1],cluster_centers[:,2], c = 'r')
plt.show()

# normalize the data to the cluster centers to see what color would be dominant 
cluster_centers = cluster_centers/255
plt.imshow(cluster_centers)
plt.show()
