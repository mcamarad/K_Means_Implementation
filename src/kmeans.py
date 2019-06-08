# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 18:01:07 2019

@author: Marcos Camara Donoso
"""
import numpy as np
from copy import deepcopy
from matplotlib import pyplot as plt


class KMeans(object):
    def __init__(self, n_clusters, max_iter=False):
        self.n_clusters= n_clusters
        self.max_iter = max_iter
        self.centroids = np.array([])
        self.mean = 0
        self.std = 0
        self.n = 0
        self.c = 0
        self.clusters=0
    def fit(self, data):
        # Number of training data
        self.n = data.shape[0]
        # Number of features in the data
        self.c = data.shape[1]
        #init parameters
        self.mean = np.mean(data, axis = 0)
        self.std = np.std(data, axis = 0)
        self.centroids = np.random.randn(self.n_clusters, self.c)*self.std + self.mean
        distances = np.zeros((self.n,self.n_clusters))
        centers_old = np.zeros(self.centroids.shape) # to store old centers
        centers_new = deepcopy(self.centroids) # Store new centers
        
        if self.max_iter == False:
            error = np.linalg.norm(centers_new - centers_old)
            # When, after an update, the estimate of that center stays the same, exit loop
            while error != 0:
                # Measure the distance to every center
                for i in range(self.n_clusters):
                    distances[:,i] = np.linalg.norm(data - self.centroids[i], axis=1)
                # Assign all training data to closest center
                self.clusters = np.argmin(distances, axis = 1)
        
                centers_old = deepcopy(centers_new)
                # Calculate mean for every cluster and update the center
                for i in range(self.n_clusters):
                    centers_new[i] = np.mean(data[self.clusters == i], axis=0)
                error = np.linalg.norm(centers_new - centers_old)
        else:
            counter=0
            while i <= self.max_iter:
                # Measure the distance to every center
                for i in range(self.n_clusters):
                    distances[:,i] = np.linalg.norm(data - self.centroids[i], axis=1)
                # Assign all training data to closest center
                self.clusters = np.argmin(distances, axis = 1)
        
                centers_old = deepcopy(centers_new)
                # Calculate mean for every cluster and update the center
                for i in range(self.n_clusters):
                    centers_new[i] = np.mean(data[self.clusters == i], axis=0)
                counter+=1
        self.centroids = centers_new
        return self.clusters, self.centroids

    def plot(self, data, color_cluster):
        for i in range(self.n):
            plt.scatter(data[i, 0], data[i,1], s=7, color = color_cluster[int(self.clusters[i])])
        plt.scatter(self.centroids[:,0], self.centroids[:,1], marker='*', c='black', s=150)
        
if __name__ == "__main__":
    #Let's generate 3 clusters of simulated data
    center_1 = np.array([1,1])
    center_2 = np.array([5,5])
    center_3 = np.array([8,1])
    
    # Generate random data and center it to the three centers
    data_1 = np.random.randn(100, 2) + center_1
    data_2 = np.random.randn(100,2) + center_2
    data_3 = np.random.randn(100,2) + center_3
    
    data = np.concatenate((data_1, data_2, data_3), axis = 0)
    
    kmeans = KMeans(n_clusters=3)
    labels, centroids = kmeans.fit(data)
    kmeans.plot(data, ["green", "yellow", "red"])