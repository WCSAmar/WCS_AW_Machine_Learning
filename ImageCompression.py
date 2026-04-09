import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Step 1: Read the image
image = plt.imread('cat.png')

# Step 2: Normalize if needed (in case values are 0–255)
if image.dtype == np.uint8:
    image = image / 255.0

# Step 3: Reshape image into 2D array (num_pixels, 3)
h, w, c = image.shape
pixels = image.reshape(-1, 3)

# Step 4: Apply K-Means clustering (k=10)
kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
kmeans.fit(pixels)

# Step 5: Replace each pixel with its cluster center
compressed_pixels = kmeans.cluster_centers_[kmeans.labels_]

# Step 6: Reshape back to original image shape
compressed_image = compressed_pixels.reshape(h, w, 3)

# Step 7: Display the compressed image
plt.imshow(compressed_image)
plt.axis('off')
plt.show()

# Step 8: Save the compressed image
plt.imsave('compressed.png', compressed_image)