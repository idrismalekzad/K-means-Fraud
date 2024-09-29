import numpy as np
import elbow as elb
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from matplotlib import cm
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest


iris_data = elb.MyClass()      # Create an instance of MyClass
data = iris_data.get_sql_data() # Get the features and labels
X = data[:, :2]
max_values = np.max(X, axis=0)

# This will return a 1D array where each element is the maximum of that dimension
print("Maximum values in each dimension before apply filtering :", max_values)
print(f"Counts X-shape before apply filtering : {X.shape[0]}")

# Apply logarithmic transformation###################################################################################
# X_log_transformed = np.log1p(X)  # Use log1p to avoid log(0) issues
# # Now apply MinMaxScaler to the log-transformed data
# scaler = MinMaxScaler(feature_range=(1, 81603))                                        #Logarithm#
# X_scaled = scaler.fit_transform(X_log_transformed)
#####################################################################################################################
# Custom normalization for each dimension
# X_scaled = X

# # Normalize the first dimension (horizontal) to [1, 100000]
# X_scaled[:, 0] = (X_scaled[:, 0] - X_scaled[:, 0].min()) / (X_scaled[:, 0].max() - X_scaled[:, 0].min()) * 100000 + 1
# # Normalize the second dimension (vertical) to [1, 1000000000]
# X_scaled[:, 1] = (X_scaled[:, 1] - X_scaled[:, 1].min()) / (X_scaled[:, 1].max() - X_scaled[:, 1].min()) * 10000000000 + 1

# Apply Min-Max scaling
# scaler = MinMaxScaler(feature_range=(1, 1000))
# X_scaled = scaler.fit_transform(X)

# Apply Z-score normalization (Standardization)
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

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
k = 10
clusters = {}
np.random.seed(23)

# Fit the Isolation Forest model
iso_forest = IsolationForest(contamination=0.005)  # Set contamination based on expected percentage of outliers
outliers = iso_forest.fit_predict(X)

# Filter data: Keep points labeled as '1' (non-outliers)
X_scaled = X[outliers == 1]
max_values = np.max(X_scaled, axis=0)
# This will return a 1D array where each element is the maximum of that dimension
print("Maximum values in each dimension after apply filtering :", max_values)

plt.scatter(X_scaled[:, 0], X_scaled[:, 1])
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter('{:.0f}'.format))
plt.grid(True)
plt.xlabel("Total Count")
plt.ylabel("Total Amount")
plt.show()


scaler = MinMaxScaler(feature_range=(0, 10000))

X_normalized = scaler.fit_transform(X_scaled)

X_scaled = X_normalized

plt.scatter(X_scaled[:, 0], X_scaled[:, 1])
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter('{:.0f}'.format))
plt.grid(True)
plt.xlabel("Total Count")
plt.ylabel("Total Amount")
plt.show()

print(f"Counts X-shape after apply filterin and normalization with MinMaxScaller: {X_scaled.shape[0]}")

# Distance function (e.g., Euclidean distance)
def distance(x1, x2):
    return np.linalg.norm(x1 - x2)

# Initialize clusters using K-Means++
def initialize_clusters(X_scaled, k):
    clusters = []
    idx = np.random.randint(0, X_scaled.shape[0])
    clusters.append({'center': X_scaled[idx], 'points': []})

    # Choose remaining k-1 cluster centers using K-Means++ technique
    for _ in range(1, k):
        dist_sq = np.array([min([distance(x, c['center'])**2 for c in clusters]) for x in X_scaled])
        prob = dist_sq / dist_sq.sum()
        cumulative_prob = np.cumsum(prob)
        r = np.random.rand()

        for i, p in enumerate(cumulative_prob):
            if r < p:
                clusters.append({'center': X_scaled[i], 'points': []})
                break

    return clusters

# Assign clusters (E-step)
def assign_clusters(X_scaled, clusters, k):
    for idx in range(X_scaled.shape[0]):
        dist = []
        curr_x = X_scaled[idx]
        
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
def pred_cluster(X_scaled, clusters, k):
    pred = []
    for i in range(X_scaled.shape[0]):
        dist = []
        for j in range(k):
            dist.append(distance(X_scaled[i], clusters[j]['center']))
        pred.append(np.argmin(dist))
    return pred

# Function to run K-Means algorithm
def kmeans(X_scaled, k, max_iter=100, tol=1e-4):
    clusters = initialize_clusters(X_scaled, k)
    for iteration in range(max_iter):
        # Assign points to the nearest cluster
        clusters = assign_clusters(X_scaled, clusters, k)

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

    clusters = assign_clusters(X_scaled, clusters, k)
    return clusters


def plot_clusters(X_scaled, clusters, k):
    colors = plt.cm.get_cmap('tab10', k)  # Colormap for cluster colors
    
       # Get predicted clusters for each point in X
    pred = pred_cluster(X_scaled, clusters, k)

    for i in range(k):
        # Get all points assigned to cluster i
        cluster_points = X_scaled[np.array(pred) == i]
        
        if cluster_points.shape[0] > 0:
            # Plot each cluster's points
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], s=50, color=colors(i), label=f'Cluster {i+1}')
        
        # Plot the cluster centers
        plt.scatter(clusters[i]['center'][0], clusters[i]['center'][1], s=200, color=colors(i), marker='X', edgecolor='k', label=f'Center {i+1}')

    plt.title('Cluster Visualization')
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter('{:.0f}'.format))
    plt.xlabel('Feature 1 - Count')
    plt.ylabel('Feature 2 - Amount')
    plt.legend()
    plt.grid(True)
    plt.show(block=True) 

# Calculate thresholds for clusters
def calculate_thresholds(clusters):
    thresholds = {}
    for i in range(k):
        points = np.array(clusters[i]['points'])
        if points.shape[0] > 0:
            distances = [distance(point, clusters[i]['center']) for point in points]
            thresholds[i] = np.max(distances)  # Maximum distance as threshold
    return thresholds


# Classify new data point
def classify_new_point(new_point, clusters, thresholds):
    # Calculate the distance to each cluster center using cluster number as keys
    distances_to_centers = {i: distance(new_point, clusters[i]['center']) for i in range(len(clusters))}
    
    # Find the closest cluster number
    closest_cluster = min(distances_to_centers, key=distances_to_centers.get)
    closest_distance = distances_to_centers[closest_cluster]
    
    # Check if the distance is within the threshold for the cluster
    if closest_distance <= thresholds[closest_cluster]:
        return "Normal", closest_cluster
    else:
        return "Abnormal", closest_cluster


# Function to classify multiple new data points
def classify_new_data_points(new_data_points, clusters, thresholds):
    classifications = []
    for new_point in new_data_points:
        status, cluster = classify_new_point(new_point, clusters, thresholds)
        classifications.append((new_point, status, cluster))
    return classifications

def plot_clusters_with_new_points(X_scaled, clusters, k, new_data_classifications):
    colors = plt.cm.get_cmap('tab10', k)  # Colormap for cluster colors

    # Get predicted clusters for each point in X
    pred = pred_cluster(X_scaled, clusters, k)

    # Plot existing clusters
    for i in range(k):
        # Get all points assigned to cluster i
        cluster_points = X_scaled[np.array(pred) == i]

        if cluster_points.shape[0] > 0:
            # Plot each cluster's points
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], s=50, color=colors(i), label=f'Cluster {i+1}')

        # Plot the cluster centers
        plt.scatter(clusters[i]['center'][0], clusters[i]['center'][1], s=200, color=colors(i), marker='X', edgecolor='k', label=f'Center {i+1}')

    # Plot new data points
    for idx, (point, status, cluster) in enumerate(new_data_classifications):
        if status == "Normal":
            # print(f"Normal {point[0], point[1] }")
            plt.scatter(point[0], point[1], s=50, color='green', marker='*', edgecolor='k')
        else:
            # print(f"Abnormal {point[0], point[1] }")
            plt.scatter(point[0], point[1], s=50, color='red', marker='X', edgecolor='k', label=f'New Point {idx+1} (Abnormal)')

    plt.title('Cluster Visualization with New Points')
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter('{:.0f}'.format))
    plt.xlabel('Feature 1 - Count')
    plt.ylabel('Feature 2 - Amount')
    plt.legend()
    plt.grid(True)
    plt.show(block=True)


# Run K-Means algorithm
clusters = kmeans(X_scaled, k)

# Plot the resulting clusters
# plot_clusters(X_scaled, clusters, k)

# Calculate thresholds for each cluster
thresholds = calculate_thresholds(clusters)

# Example: Classify a new data point
new_data_point = iris_data.get_sql_new_data() # Get the features and labels
X_NEWDATA = new_data_point[:, :2] # Example new data point (scaled)

# Apply logarithmic transformation########################################################################
# X_log_transformed = np.log1p(X_NEWDATA)  # Use log1p to avoid log(0) issues
# # Now apply MinMaxScaler to the log-transformed data
#                                                                 #Logarithm#
# X_scaled_NEWDATA = scaler.fit_transform(X_log_transformed)
##########################################################################################################
# Custom normalization for each dimension
# X_normalized= X_NEWDATA

X_normalized = scaler.transform(X_NEWDATA)

max_values = np.max(X_NEWDATA, axis=0)
# This will return a 1D array where each element is the maximum of that dimension
print("Maximum values for new data Incomming :", max_values)
# # Normalize the first dimension (horizontal) to [1, 100000]
# X_normalized[:, 0] = (X_NEWDATA[:, 0] - X_NEWDATA[:, 0].min()) / (X_NEWDATA[:, 0].max() - X_NEWDATA[:, 0].min()) * 100000 + 1

# # Normalize the second dimension (vertical) to [1, 100000000]
# X_normalized[:, 1] = (X_NEWDATA[:, 1] - X_NEWDATA[:, 1].min()) / (X_NEWDATA[:, 1].max() - X_NEWDATA[:, 1].min()) * 10000000000 + 1

# Classify all new data points
new_data_classifications = classify_new_data_points(X_normalized, clusters, thresholds)


# Print the classifications
for idx, (point, status, cluster) in enumerate(new_data_classifications):
    print(f"New data point {idx + 1}: {point}, classified as {status} (Cluster {cluster + 1})")

plot_clusters_with_new_points(X_scaled, clusters, k, new_data_classifications)