import numpy as np
from sklearn.cluster import KMeans
from skimage import io
import matplotlib.pyplot as plt

def cluster_image(image_path, n_clusters=10):
    # Load the image
    image = io.imread(image_path)

    # Normalize the image
    image_normalized = image.astype(float) / 255

    # Reshape for K-means clustering
    image_reshaped = image_normalized.reshape((-1, 4))

    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10).fit(image_reshaped)

    # Check the cluster centers
    print("Cluster centers (in RGB):", kmeans.cluster_centers_)

    # Assign each pixel to the nearest cluster center
    clustered = kmeans.cluster_centers_[kmeans.labels_]

    # Reshape into the original image shape
    clustered_image = clustered.reshape(image.shape)

    # Convert to uint8 by scaling back to [0, 255]
    clustered_image = (clustered_image * 255).astype(np.uint8)

    # Display the results
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Original image
    axs[0].imshow(image)
    axs[0].set_title('Original Image')
    axs[0].axis('off')

    # Clustered image
    axs[1].imshow(clustered_image)
    axs[1].set_title('Clustered Image')
    axs[1].axis('off')

    plt.show()

    # Display the cluster centers as colors
    plt.figure(figsize=(8, 2))
    plt.imshow([kmeans.cluster_centers_], aspect='auto')
    plt.axis('off')
    plt.title('Cluster Center Colors')
    plt.show()

# Path to the uploaded image file
# Ensure this path is correct
image_path = '/content/picture.png'

# Cluster the image
cluster_image(image_path, n_clusters=5)
