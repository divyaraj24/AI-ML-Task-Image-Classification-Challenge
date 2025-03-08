import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings as wr
wr.filterwarnings('ignore')

'''Displaying sample images from each label - Task 1'''

df = pd.read_csv("dataset_aiml_task/data.csv")

labels = df.iloc[:, 0].unique()
label_dfs = {label: df[df.iloc[:, 0] == label] for label in labels} # Classify the unique labels into a dictionary
print(df.iloc[:, 0].value_counts())  # Count images per label

image_size = int(np.sqrt(df.shape[1] - 1)) # square 28x28 images are there in the dataset

# create 10 subplots for 2 row and 5 columns for each sample image
fig, axes = plt.subplots(2, 5, figsize=(12, 6))

# convert to 1D flattened array using numpy
axes = axes.ravel()


# loop and show the first image of from each label to show sample images
for i, label in enumerate(sorted(labels)):
    sample_image = label_dfs[label].iloc[0, 1:].values.reshape(image_size, image_size)
    axes[i].imshow(sample_image, cmap="gray")
    axes[i].set_title(f'Label {label}')
    axes[i].axis("off")

plt.tight_layout()
plt.show()

# diplay summary statistics and store in a csv file
print(df.describe())
df.describe().to_csv("summary_statistics.csv")

'''Classification Model using Logistic Regression - Task 2'''

x = df.iloc[:, 1:].values # Pixel Values
y = df.iloc[:,0].values # Labels

# Split dataset into 80% for training and 20% for testing
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=42, stratify=y)

# standardizing pixel data to have a mean of 0 and standard deviation of 1 (helps the model to converge faster)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Check if the model file already exists to check whether to train the model or not
model_file = "logistic_regression_model.pkl"
if os.path.exists(model_file):
    # Load the previously trained model
    model = joblib.load(model_file)
    print("Loaded model from file.")
else:
    # Train the logistic regression model
    model = LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='multinomial')
    model.fit(x_train, y_train)
    # Save the trained model for future use
    joblib.dump(model, model_file)
    print("Model trained and saved.")


# Use the model to predict test set labels
y_pred = model.predict(x_test)

# Predict Accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")
print("Classification Report:\n", classification_report(y_test, y_pred))

sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap="Blues", xticklabels=range(10), yticklabels=range(10))
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# Testing the model on random images from the test set

random_indices = np.random.choice(range(x_test.shape[0]), size=3, replace=False)
image_size = 28

for i in random_indices:
    image = x_test[i].reshape(28,28)
    pred_label = model.predict(x_test[i].reshape(1, -1))[0]
    true_label = y_test[i]

    plt.figure()
    plt.imshow(image,cmap="gray")
    plt.title(f"True Label: {true_label} | Predicted: {pred_label}")
    plt.axis("off")
    plt.show()