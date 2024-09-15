import numpy as np
import elbow as elb
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from matplotlib import cm
from sklearn.decomposition import PCA

iris_data = elb.MyClass()      # Create an instance of MyClass
data = iris_data.get_sql_data() # Get the features and labels
X = data[:, :2]

# Plot the fetched data
plt.scatter(X[:, 0], X[:, 1])
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter('{:.0f}'.format))
plt.grid(True)
plt.xlabel("Total Count")
plt.ylabel("Total Amount")
plt.show()

iris_data.elbow_method()

# input("Press Enter to continue...")

# Initialize clusters
k = 20
clusters = {}
np.random.seed(23)


# Distance function (e.g., Euclidean distance)
def distance(x1, x2):
    return np.linalg.norm(x1 - x2)

# Initialize clusters using K-Means++
def initialize_clusters(X, k):
    clusters = []
    idx = np.random.randint(0, X.shape[0])
    clusters.append({'center': X[idx], 'points': []})

    # Choose remaining k-1 cluster centers using K-Means++ technique
    for _ in range(1, k):
        dist_sq = np.array([min([distance(x, c['center'])**2 for c in clusters]) for x in X])
        prob = dist_sq / dist_sq.sum()
        cumulative_prob = np.cumsum(prob)
        r = np.random.rand()

        for i, p in enumerate(cumulative_prob):
            if r < p:
                clusters.append({'center': X[i], 'points': []})
                break

    return clusters

# Assign clusters (E-step)
def assign_clusters(X, clusters, k):
    for idx in range(X.shape[0]):
        dist = []
        curr_x = X[idx]
        
        # Find the nearest cluster center
        for i in range(k):
            dis = distance(curr_x, clusters[i]['center'])
            dist.append(dis)
        
        curr_cluster = np.argmin(dist)
        clusters[curr_cluster]['points'].append(curr_x)
    
    return clusters

# Update cluster centers (M-step)
def update_clusters(clusters, k):
    for i in range(k):
        points = np.array(clusters[i]['points'])
        if points.shape[0] > 0:
            new_center = points.mean(axis=0)
            clusters[i]['center'] = new_center
            clusters[i]['points'] = []  # Reset points after updating
    return clusters

# Predict cluster labels for data points
def pred_cluster(X, clusters, k):
    pred = []
    for i in range(X.shape[0]):
        dist = []
        for j in range(k):
            dist.append(distance(X[i], clusters[j]['center']))
        pred.append(np.argmin(dist))
    return pred

# Function to run K-Means algorithm
def kmeans(X, k, max_iter=100, tol=1e-4):
    clusters = initialize_clusters(X, k)
    for iteration in range(max_iter):
        # Assign points to the nearest cluster
        clusters = assign_clusters(X, clusters, k)

        # Store old cluster centers for convergence check
        old_centers = np.array([cluster['center'] for cluster in clusters])

        # Update cluster centers based on assigned points
        clusters = update_clusters(clusters, k)
        # Check convergence (if centers don't change much, stop)
        new_centers = np.array([cluster['center'] for cluster in clusters])
        diff = np.linalg.norm(new_centers - old_centers)
        print(iteration)
        print(diff)

        if diff < tol:
            print(f"Converged after {iteration+1} iterations.")
            break
    return clusters


def plot_clusters(X, clusters, k):
    colors = plt.cm.get_cmap('tab10', k)  # Colormap for cluster colors
    
    # Get predicted clusters for each point in X
    pred = pred_cluster(X, clusters, k)
    
    for i in range(k):
        # Get all points assigned to cluster i
        cluster_points = X[np.array(pred) == i]
        
        if cluster_points.shape[0] > 0:
            # Plot each cluster's points
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], s=50, color=colors(i), label=f'Cluster {i+1}')
        
        # Plot the cluster centers
        plt.scatter(clusters[i]['center'][0], clusters[i]['center'][1], s=200, color=colors(i), marker='X', edgecolor='k', label=f'Center {i+1}')

    plt.title('Cluster Visualization')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True)
    plt.show(block=True) 


k = 20  # Number of clusters

# Apply PCA to reduce the dimensions (e.g., to 2 dimensions)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Run K-Means algorithm
clusters = kmeans(X_pca, k)

# Plot the resulting clusters
plot_clusters(X, clusters, k)
