# Rice Varieties Classification Report

### Table of Contents
* [Introduction](#introduction)
* [Method](#method)
* [Result](#result)
* [Discussion](#discussion)
* [Conclusion](#conclusion)
* [Statement of Collaboration](#statement-of-collaboration)

## Introduction

## Method
### Preprocess the data
* Load the image from the dataset to check the quality of the images.
* Check the size of the images and unify them to the same size.
* Convert labels to categories, `Arborio （1）-> Arborio`.
* Convert images to matrices.
* Compute feature matrix from image matrices.
* Manually add the categories to the feature matrix, like rice classification.
* Standardize the image matrix for later model building.
* Analyze feature data and remove data that affects model accuracy. Remove missing values, redundant features, unnecessary samples, outliers, and duplicate records in the data to reduce redundancy.

* **Choose Key Features**
  * Due to the wide and disparate range across various features from the above pair plot, we normalized and standardized three key features (`Perimeter`(0.71), `Area`(0.65), and `Convex_Area`(0.65)) within the `preprocess_rice_data()` function, as they show a higher correlation with `class`.
![heatmap2](./data_picture/heatmap2.png)
* **Label Encoding**
  *  We also encoded the categorical rice types (`Basmati`, `Jasmine`, `Arborio`, `Ipsala`, and `Karacadag`) by mapping them to numerical values ranging from 0 to 4.
* **Normalize and Standardize**
  * Minmax Normalization makes all eigenvalues ​​between [0, 1], eliminating the impact of different eigenvalue magnitudes. Standardization converts data into a standard normal distribution with a mean of 0 and a standard deviation of 1, which improves the efficiency of subsequent operations on the data.  
![df_pre](./data_picture/df_pre.png)

#### First Training Model
* **Logistic Regression model**
    We used logistic regression models to classify the types of rice.
* **Feature Pairing and Model Training**
  * We paired different types of rice to train different models. For example, Type 1 vs. Type 2, Type 1 vs. Type 3, and so on.
  * Each pair was used to train a separate instance of the logistic regression model. In the end, we have a total of 10 models, each trained on two of the types of rices. The following are 10 Logistic Regression models and their corresponding **log losses**:  
![logistic_regression](./data_picture/logistic_regression.png)
* **Scoring Mechanism**
  * After training, each model's predictions on the test set were combined into a single matrix.  
![output_matrix](./data_picture/out_matrix.png)  
Each row is the prediction of each model for the test set, and each column is the prediction result of different models for the same sample.
  * The final classification was based on a scoring mechanism. For each test sample, the most frequent prediction, that is **mode**, obtained from all these models was the final output in this scoring approach. Thus, the final classification takes much from multiple models applied to further enable the overall accuracy to increase significantly.
* **Evaluate**
* Train_Accuracy:  
![train_accuracy](./data_picture/train_accuracy.png)
* Test_Accuracy:  
![test_accuracy](./data_picture/test_accuracy.png)
* Classification Report:  
![classification_report](./data_picture/classification_report.png)

### Question
* **Where does your model fit in the fitting graph?**
  * We think that our model is in a good fit. The training accuracy and test accuracy are both high and close. And according to the Classification report, for all rice classifications, the model's precision, recall and f1-score are at high values

* **What are the next models you are thinking of and why?**
  * Since our goal is to distinguish 5 types of rice, which is a multi-classification problem, using **Neural Networks** can meet the needs of having multiple outputs. A neural network can be implemented by using a sigmoid activation function in the output layer, allowing each output node to predict the class independently.

### Conclusion section
* **What is the conclusion of your 1st model?**
  * For problems that require multiple classifications, using a logistic regression model seems to require more steps and details that require attention. Since there are a total of 5 types of rice that we trained this time, we were able to build 10 models to distinguish them separately. However, if the types of rice continue to increase, then continuing to use the logistic regression model to judge the efficiency will decrease, and the statistics and judgment of the results of each model will also become complicated.
* **What can be done to possibly improve it?**
   * First of all, in terms of the use of the model, it seems to be a better decision to use a neural network rather than a logistic regression model, because this is a multi-classification problem, and neural networks have better performance for predicting tasks belonging to multiple categories. Then, for the final statistical method of the test results using the logistic regression model, we adopted the mode method, which may produce multiple different modes and lead to confusion in the test results. Perhaps it would be a better idea to use the sum or maximum probability of each model's probability for the test result.

## Result

## Discussion

In this project, we aimed to classify five different types of rice: Arborio, Basmati, Ipsala, Jasmine, and Karacadag using a variety of machine learning and deep learning methods. The main parts of the project are divided into data preprocessing, feature extraction, model selection and training, hyperparameter tuning, and model evaluation.

### Data Preprocessing

We started with a dataset that consists of 75,000 images of rice grains—15,000 of each type of rice, each being an image of size 250 by 250 pixels containing one rice grain in the middle of the picture, and the rest of the image with a black background. Preprocessing the data involves the extraction of useful features from the images.

### Feature Extraction

Our initial approach was to use OpenCV to extract geometrical features from the rice grain. We selected this approach because of its simplicity and our hypothesis that different rice varieties are likely to have distinct characteristic shapes. We binarized the image and found the contours in the image to compute different geometrical features for each of the contours.

### Standardizing Features

Standardization ensures all features have the same scale, which helps the model train well. We standardized our features using `StandardScaler`.

### Model Selection and Training

Initially, we selected logistic regression models and trained binary classifiers for pairs of the selected classes, calculating the corresponding log loss values for each. Finally, by using these log loss values, a voting classifier was built. This served as our baseline before moving to more complex models. We were pleasantly surprised by how well logistic regression performed on our dataset. It might indicate strong discriminativity among features but also raises concerns about high complexity in the dataset. Typically, if a logistic regression model works well for a task, it suggests that the differences are more subtle and non-linear.



### Deep Learning Model and Hyperparameter Tuning

Given the promising results from logistic regression, we moved on to explore deep learning models. We used convolutional neural networks (CNNs) due to their effectiveness in image classification tasks. Our CNN architecture was designed to capture the intricate patterns and textures in rice grains that logistic regression might miss.

To further improve performance, we used Keras Tuner for hyperparameter search. We hypothesized the presence of complex nonlinear relationships in our data, beyond what simpler models could capture. Indeed, in support of this hypothesis, the deep learning model showed improved performance at the expense of a decrease in interpretability.

### Results and Discussion

Both logistic regression and deep learning models were trained, and both showed high accuracy on the test set. However, this result is not without questions about its reliability. Further details on the results are discussed below:

### Reliability of Results

We achieved about 97% accuracy on the test set; therefore, it remains highly predictive. However, the dataset is generated artificially, and images are on black backgrounds, which may not be the case in reality. So, the model's performance might degrade while working in real applications.

### Limitations and Improvements

The primary limitation of our model is overfitting. We can improve this by increasing the size of the data, as well as using data augmentation techniques and regularization methods. Other approaches to try for further performance improvement may include trying out additional models and complex network architectures.

### Future Work

In the future, transfer learning can be tried using pre-trained convolutional neural networks (CNNs), such as ResNet and VGG, for feature extraction and classification. Further methods of hyperparameter tuning—such as Bayesian optimization—might be considered in an attempt to find better hyperparameter configurations.

Overall, it has been a practical learning process in building a machine and deep learning project from scratch: data preprocessing, feature extraction, model selection and training, and hyperparameter tuning. The results are not perfect; however, this process gave us insight into machine learning and deep learning, enabling us to identify problems and solve them within a project.


## Conclusion

## Statement of Collaboration
#### Jiawei Huang
    