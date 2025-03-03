import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings as wr
wr.filterwarnings('ignore')


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