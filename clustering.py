import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn import metrics

def kmeans(data, k):
    num_data = np.shape(data)[0]
    # First, randomly pick k data points for the centers of the initial clusters. Maybe you should use the np.random.choice() function.
    # Afterwards, initialize the clusters.
    cluster_mean = data[np.random.choice(range(num_data), k)]
    mean_dist_mat = np.linalg.norm(np.expand_dims(data, axis = 0) - np.expand_dims(cluster_mean, axis = 1), axis = 2)
    cluster_allocation = np.argmin(mean_dist_mat, axis = 0)

    # Update the clusters until convergence.
    updated = True
    init_num = 0
    while updated:
        updated = False

        # Calculate the cluster means.
        cluster_mean = np.array([np.sum(data[cluster_allocation==i],axis=0)/np.sum(cluster_allocation==i) for i in range(k)])

        # Find out which new cluster each data point belongs to
        mean_dist_mat = np.linalg.norm(np.expand_dims(data, axis = 0) - np.expand_dims(cluster_mean, axis = 1), axis = 2)
        new_cluster_allocation = np.argmin(mean_dist_mat, axis = 0)

        # Update the cluster allocation. If nothing changes, exit the loop and return the converged result.
        if not np.array_equal(cluster_allocation, new_cluster_allocation):
            updated = True
            cluster_allocation = new_cluster_allocation
        else:
            loss = np.sum(np.min(mean_dist_mat, axis = 0))
    return cluster_allocation, loss

# This function should return the cluster labels and the value of the objective function for kernel k-means.
def kernel_kmeans(data, k, s, kernel = 'gaussian'):
    # The variable s refers to the value of sigma in the Gaussian kernel.
    num_data = np.shape(data)[0]
    # First, randomly pick k data points for the centers of the initial clusters. Maybe you should use the np.random.choice() function.
    # Afterwards, initialize the clusters.
    cluster_mean = data[np.random.choice(range(num_data), k)]
    mean_dist_mat = np.linalg.norm(np.expand_dims(data, axis = 0) - np.expand_dims(cluster_mean, axis = 1), axis = 2)
    cluster_allocation = np.argmin(mean_dist_mat, axis = 0)
    # Pre-calculate kernel values and save it in a matrix.
    # You can use sklearn.metrics.pairwise_distances(data) to get the pairwise distance of vectors in the data.
    # Gaussian kernel
    if kernel == 'gaussian':
        pre_cal = np.exp(-np.square(metrics.pairwise_distances(data))/(2*(s**2)))
    else:
        print("no such kernel")

    # Update the clusters until convergence.
    updated = True
    init_num = 0
    while updated:
        updated = False
        
        # Find out which new cluster each data point belongs to
        mean_dist_mat = [1-2*np.sum(pre_cal[:,cluster_allocation==idx], axis = 1)/np.sum(cluster_allocation==idx)
                         +np.sum(pre_cal[cluster_allocation==idx][:, cluster_allocation==idx])/(np.sum(cluster_allocation==idx)**2)
                         for idx in range(k)]
        new_cluster_allocation = np.argmin(mean_dist_mat, axis = 0)

        # Update cluster allocation. If nothing changes, exit the loop and return the converged result.
        if not np.array_equal(cluster_allocation, new_cluster_allocation):
            updated = True
            cluster_allocation = new_cluster_allocation
        else:
            loss = np.sum(np.min(mean_dist_mat, axis = 0))
    return cluster_allocation, loss


from sklearn.datasets import make_circles
X, y = make_circles(n_samples=1000, noise = 0.1, factor = 0.3, random_state = 10)
plt.figure()
plt.scatter(X[y == 0, 0], X[y == 0, 1], color='red') 
plt.scatter(X[y == 1, 0], X[y == 1, 1], color='blue') 
plt.show()

# Run k-means clustering. This cell should print the NMI score of your model.
np.random.seed(1)
loss = np.inf
for _ in range(10):
    kmeans_result,loss_tmp = kmeans(X, k=2)
    if loss > loss_tmp:
        loss = loss_tmp
        best_kmeans_result = kmeans_result
        score = metrics.normalized_mutual_info_score(y, kmeans_result)

# Visualize the result of the k-means clustering.
plt.figure()
plt.scatter(X[kmeans_result == 0,0], X[kmeans_result == 0,1], color = 'red')
plt.scatter(X[kmeans_result == 1,0], X[kmeans_result == 1,1], color = 'blue')
plt.show()
print("NMI score of K-means clustering: ",score)

# Run k-means clustering. This cell should print the NMI score of your model.
# Feel free to experiment with hyperparameter s (Use 0.5 as default)
np.random.seed(1)
loss = np.inf
for _ in range(10):
    kkmeans_result,loss_tmp = kernel_kmeans(X, k=2, s = 0.5)
    if loss > loss_tmp:
        loss = loss_tmp
        best_kkmeans_result = kkmeans_result
        score = metrics.normalized_mutual_info_score(y, kkmeans_result)

# Visualize the result of the kernel k-means clustering.
plt.figure()
plt.scatter(X[best_kkmeans_result == 0,0], X[best_kkmeans_result == 0,1], color = 'red')
plt.scatter(X[best_kkmeans_result == 1,0], X[best_kkmeans_result == 1,1], color = 'blue')
plt.show()
print("NMI score of kernel K-means clustering: ",score)
