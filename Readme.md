### AI-ML-Task-Image-Classification-Challenge:

#### Level - 1: Exploratory Data Analysis


> Dataset is not present in this repository as it was too large to upload to github.


##### Steps: 

- Step - 1: Creating main.py file and importing the required libraries (pandas, numpy and matplotlib).

![step1](images/step1.png)

- Step - 2: Reading the csv data file using pandas and storing it in a dataframe

![step2](images/step2.png)

- Step - 3: Using the dataframe classify the images based on their labels into a dictionary. Create subplots for displaying the images.

![step3](images/step3.png)

- Step - 4: Display the sample images (first) from each particular label category to understand the data distribution.

![step4](images/step4.png)

- Step - 5: Print the summary statistics of the given datset and store them in a csv file.

![step5](images/step5.png)


##### Plot displaying sample images:

![plot](images/plot1.png)

##### Summary/Statistics of the dataset:

![summary statistics](images/summary_statistics.png)


#### Level - 1: Logistic Regression Classification Model

##### Steps: 

- Step 1: Import required libraries like: 'sklearn' for logistic regression model, accuracy prediction and classification report, 'joblib' to store logistic regression model for future usage and 'os' to check whether the model is present or not.

![step1](images/step1_task2_1.png)
![step1](images/step1_task2_2.png)

- Step 2: We take the pixel values as the input values 'x' and the labels as output values 'y'. Split the dataset into two parts - 80% for training and 20% for testing.
Finally, standardize the pixel data (input values) to have mean 0 and standard deviation 1 which helps the logistic regression model to converge faster.

![step2](images/step2_task2.png)

- Step 3: Check if the model is already present in the present working directory using 'os' library in python. If it is already present use the 'joblib' library to load the pickle file. If it's not present make the logistic regression model using 'sklearn' and train it on the training values. Finally, dump the model into a file using 'joblib' library for future usage.

![step3](images/step3_task2.png)

- Step 4: Use the model now to predict the set labels and get the model accuracy score, the classification report and get the confusion matrix as heatmap using 'seaborn' library and plot it using matplotlib.

![step4](images/step4_task2.png)

##### Model Accuracy Score and Classification Report:

![Accuracy and Classification report](images/model_classification%20and%20accuracy%20report.png)

##### Confusion Matrix Heatmap:

![Confusion Matrix](images/confusion_matrix.png)

##### Classifying random images from the test set:

![classify](images/classify.png)
![classify](images/classify1.png)
![classify](images/classify2.png)
![classify](images/classify3.png)
