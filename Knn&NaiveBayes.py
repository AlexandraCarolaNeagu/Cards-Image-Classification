import os
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
from skimage.io import imread
from skimage.transform import resize
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Function to load images and labels from the dataset
def load_dataset(root_folder, target_size=(64, 64)):
    images = []
    labels = []
    class_mapping = {}  # To map class names to numeric labels

    for class_label, class_name in enumerate(os.listdir(root_folder)):
        class_mapping[class_label] = class_name
        class_folder = os.path.join(root_folder, class_name)

        for filename in os.listdir(class_folder):
            image_path = os.path.join(class_folder, filename)
            image = imread(image_path)

            # Check if the image has an alpha channel (4th channel)
            if image.shape[2] == 4:
                # Convert RGBA to RGB by discarding the alpha channel
                image = image[:, :, :3]

            image = resize(image, target_size, anti_aliasing=True)  # Resize all images to the same size

            # Check if the shape is correct before appending
            if image.shape == (target_size[0], target_size[1], 3):
                images.append(image.flatten())  # Flatten the image
                labels.append(class_label)
            else:
                print(f"Ignoring image {image_path} due to incorrect shape: {image.shape}")

    # Check if all images have the same shape
    if any(image.shape != images[0].shape for image in images):
        raise ValueError("All input images must have the same shape")

    return np.array(images), np.array(labels), class_mapping

# Function to save results to a CSV file
def save_results(algorithm, y_true, y_pred, class_mapping=None):
    results_df = pd.DataFrame({'True Label': y_true, 'Predicted Label': y_pred})
    results_folder = './results'
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    results_file = os.path.join(results_folder, f'{algorithm}_results.csv')
    results_df.to_csv(results_file, index=False)
    print(f'Results saved for {algorithm} at {results_file}')

# Function to save test images with predictions
def save_test_images(algorithm, X_test, y_true, y_pred, class_mapping, probabilities=None):
    results_folder = './results'
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    for idx, (true_label, pred_label) in enumerate(zip(y_true, y_pred)):
        image = X_test[idx].reshape(64, 64, -1)  # Reshape flattened image
        plt.imshow(image)
        plt.title(f'True Label: {class_mapping[true_label]}\nPredicted Label: {class_mapping[pred_label]}')
        plt.savefig(os.path.join(results_folder, f'{algorithm}_test_image_{idx}.png'))
        plt.close()

        # Plot probability distribution if available
        if probabilities is not None:
            plt.bar(class_mapping.values(), probabilities[idx], color='red')
            plt.title(f'Probability Distribution - Test Image {idx}')
            plt.xlabel('Class')
            plt.ylabel('Probability')
            plt.savefig(os.path.join(results_folder, f'{algorithm}_probability_distribution_{idx}.png'))
            plt.close()

    print(f'Test images saved for {algorithm} in {results_folder}')

# Load your dataset
train_dataset_folder = "./train"  # Change this to the path of your training dataset folder
X_train, y_train, class_mapping = load_dataset(train_dataset_folder)

# Load your test set
test_dataset_folder = "./test"  # Change this to the path of your test dataset folder
X_test, y_test, class_mapping_test = load_dataset(test_dataset_folder)

# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, class_mapping, algorithm):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(len(class_mapping), len(class_mapping)))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=class_mapping.values(), yticklabels=class_mapping.values())
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix - {algorithm}')
    plt.show()

# k-Nearest Neighbors for Regression
# Initialize k-NN regressor
knn_regressor = KNeighborsRegressor(n_neighbors=3)
# Train the model
knn_regressor.fit(X_train.reshape((X_train.shape[0], -1)), y_train)
# Make predictions on the test set
y_test_pred_knn_reg = knn_regressor.predict(X_test.reshape((X_test.shape[0], -1)))
# Calculate mean squared error for k-NN regression
mse_knn_reg = mean_squared_error(y_test, y_test_pred_knn_reg)
print(f'Mean Squared Error for k-NN Regression: {mse_knn_reg}')
# Save results for k-NN regression
save_results('knn_regression', y_test, y_test_pred_knn_reg, class_mapping_test)

# Naive Bayes for Classification
# Initialize Naive Bayes classifier
nb_classifier = GaussianNB()
# Train the model
nb_classifier.fit(X_train.reshape((X_train.shape[0], -1)), y_train)
# Make predictions on the test set
y_test_pred_nb_class = nb_classifier.predict(X_test.reshape((X_test.shape[0], -1)))
# Calculate accuracy and print classification report for Naive Bayes classification
accuracy_nb_class = accuracy_score(y_test, y_test_pred_nb_class)
print(f'Accuracy for Naive Bayes Classification: {accuracy_nb_class}')
classification_report_nb_class = classification_report(y_test, y_test_pred_nb_class, target_names=class_mapping_test.values())
print(f'Classification Report for Naive Bayes Classification:\n{classification_report_nb_class}')
# Save results and test images for Naive Bayes classification
save_results('naive_bayes_classification', y_test, y_test_pred_nb_class, class_mapping_test)
probabilities_nb_class = nb_classifier.predict_proba(X_test.reshape((X_test.shape[0], -1)))
save_test_images('naive_bayes_classification', X_test, y_test, y_test_pred_nb_class, class_mapping_test, probabilities_nb_class)

# Convert class labels to numeric values using the mapping
y_test_numeric = np.array([class_mapping_test[label] for label in y_test])
y_test_pred_nb_class_numeric = np.array([class_mapping_test[label] for label in y_test_pred_nb_class])

# Plot confusion matrix for Naive Bayes Classification
plot_confusion_matrix(y_test_numeric, y_test_pred_nb_class_numeric, class_mapping_test, 'naive_bayes_classification')
