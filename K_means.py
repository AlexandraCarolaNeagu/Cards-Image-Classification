import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import random
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import precision_score, classification_report

def calculate_accuracy(true_labels, predicted_labels):
    accuracy = accuracy_score(true_labels, predicted_labels)
    return accuracy

# Function to plot images grouped by clusters
def plot_cluster_images(images, cluster_assignments, titles):
    n_clusters = len(images)
    plt.figure(figsize=(15, 5))

    for cluster in range(n_clusters):
        cluster_images = images[cluster]
        cluster_title = f"Cluster {cluster} ({len(cluster_images)} images)"

        for i, image in enumerate(cluster_images):
            plt.subplot(n_clusters, len(cluster_images), i + 1 + cluster * len(cluster_images))

            # Ensure the index is within bounds
            if i < len(cluster_assignments):
                cluster_indices = np.where(cluster_assignments == cluster)[0]
                if i < len(cluster_indices):
                    image_idx = cluster_indices[i]
                    # Ensure the index is within bounds for titles
                    if image_idx < len(titles):
                        plt.title(f"{titles[image_idx]}")
                    else:
                        plt.title("Title not available")
                else:
                    plt.title("Title not available")
            else:
                plt.title("Title not available")

            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.axis('off')
    plt.show()

# Function to load images from a folder and assign labels
def load_images_from_folder(folder, target_shape=None):
    images = []
    labels = []
    label_mapping = {}  # Dicționar pentru a mapa numele folderelor la valori numerice
    label_index = 0
    for card_folder in os.listdir(folder):
        card_folder_path = os.path.join(folder, card_folder)
        if os.path.isdir(card_folder_path):
            for filename in os.listdir(card_folder_path):
                if filename.lower().endswith(('.jpg', '.jpeg')):
                    img = cv2.imread(os.path.join(card_folder_path, filename))
                    if img is not None:
                        # Resize the image to a consistent shape (e.g., 200x200)
                        if target_shape is not None:
                            img = cv2.resize(img, target_shape)
                        images.append(img)
                        labels.append(label_index)  # Adăugați eticheta numerică, nu numele folderului
                    else:
                        print(f"Warning: Unable to load {filename}")
                else:
                    print(f"Warning: Non-JPEG file found: {filename}")
            label_index += 1  # Increment the label index for the next folder
    return images, labels, label_mapping



# Folder paths
train_data_folder = './train'  # Folder with training data
test_data_folder = './test'    # Folder with test data

# Load images and labels from the training folder and resize them to (200, 200)
train_images, train_labels, label_mapping = load_images_from_folder(train_data_folder, target_shape=(200, 200))

# Load images and labels from the test folder and resize them to (200, 200)
test_images, test_labels, label_mapping = load_images_from_folder(test_data_folder, target_shape=(200, 200))
print('# TEST files:', len(test_images))

# Apply K-Means clustering on training data
n_clusters = 53  # User-defined number of clusters (adjust as needed)
kmeans = KMeans(n_clusters=n_clusters, random_state=0)

# Reshape the images and convert them to grayscale
train_image_data = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).flatten() for image in train_images]
train_image_data = np.array(train_image_data)

# Scale the data
scaler = StandardScaler()
scaled_train_data = scaler.fit_transform(train_image_data)

# Train the K-Means model
kmeans.fit(scaled_train_data)

# Get cluster assignments for training data
train_cluster_assignments = kmeans.labels_

# Randomly select an image from the training set
random_index = random.randint(0, len(train_images) - 1)
random_image = train_images[random_index]
random_image_label = train_labels[random_index]

# Visualize a random training image
plt.figure()
plt.imshow(cv2.cvtColor(random_image, cv2.COLOR_BGR2RGB))
plt.title(f"Random Training Image - Label: {random_image_label}")
plt.axis('off')
plt.show()

# Visualize training images grouped by clusters
plot_cluster_images([train_images], train_cluster_assignments, train_labels)

# Apply K-Means to test images
scaled_test_data = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).flatten() for image in test_images]
scaled_test_data = np.array(scaled_test_data)
scaled_test_data = scaler.transform(scaled_test_data)

# Get cluster assignments for test data
test_cluster_assignments = kmeans.predict(scaled_test_data)

test_predictions = kmeans.predict(scaled_test_data)

# Convert the card labels to numerical labels based on the predicted clusters
label_mapping = {label: idx for idx, label in enumerate(np.unique(train_labels))}
numeric_test_labels = [label_mapping[label] for label in test_labels]

# Calculate and print the accuracy for training data
train_accuracy = calculate_accuracy(train_labels, train_cluster_assignments)
print(f'Training Accuracy: {train_accuracy * 100:.2f}%')

# Calculate and print the accuracy for test data
test_accuracy = calculate_accuracy(test_labels, test_cluster_assignments)
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

# Convert test labels to numeric labels based on the label mapping
numeric_test_labels = [label_mapping[label] for label in test_labels]

# Calculate precision for test data
precision = precision_score(test_labels, test_cluster_assignments, average='weighted')
print(f'Precision: {precision * 100:.2f}%')

# Print classification report for more detailed metrics
report = classification_report(test_labels, test_cluster_assignments)
print("Classification Report:")
print(report)

# Calculate confusion matrix for test data
conf_matrix = confusion_matrix(numeric_test_labels, test_cluster_assignments)

# Visualize confusion matrix
plt.figure(figsize=(12, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_mapping.keys(), yticklabels=label_mapping.keys())
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


# Apply K-Means to test images
for i, test_image in enumerate(test_images):
    test_image_data = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY).flatten()
    test_image_data = np.array(test_image_data).reshape(1, -1)  # Reshape for a single sample
    scaled_test_data = scaler.transform(test_image_data)
    test_prediction = kmeans.predict(scaled_test_data)  # Predict classes

    # Visualize the test image with its predicted cluster
    plt.figure()
    plt.imshow(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))
    plt.title(f"Test Image {i + 1} - Cluster {test_prediction[0]} - Card: {test_labels[i]}")
    plt.axis('off')
    plt.show()


