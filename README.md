# CSE151A_Group_Project-Rice_Classification

#### Team member:
* Po-Yu Lai
* chaowen cao
* Xinheng Wang
* Jiawei Huang
* Zhenhan Hu
* Shiwei Yang

## Data Exploration

We performed the following data exploration steps:

1. **Image Data**:
   - Described the number of classes, number of images, size of images, and checked if sizes are uniform.
   - Example code:
     ```python
     import os
     import cv2
     import numpy as np
     import pandas as pd
     import matplotlib.pyplot as plt

     def extract_features(image_path):
         # Read the image
         image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
         # Check if the image is loaded successfully
         if image is None:
             print(f"Failed to load image: {image_path}")
             return []

         # Binary thresholding
         _, thresh = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
         # Find contours
         contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

         # Initialize feature list
         features_list = []

         for contour in contours:
             # Calculate features
             area = cv2.contourArea(contour)
             perimeter = cv2.arcLength(contour, True)
             x, y, w, h = cv2.boundingRect(contour)
             major_axis_length = max(w, h)
             minor_axis_length = min(w, h)
             eccentricity = 0  # Default value for eccentricity
             if len(contour) >= 5:  # fitEllipse needs at least 5 points
                 (x, y), (MA, ma), angle = cv2.fitEllipse(contour)
                 if MA < ma:
                     MA, ma = ma, MA
                 if MA > 0 and ma > 0:
                     eccentricity = np.sqrt(1 - (ma/MA)**2)
             hull = cv2.convexHull(contour)
             convex_area = cv2.contourArea(hull)
             extent = area / (w * h) if w * h > 0 else 0

             # Append features to list
             features = [area, perimeter, major_axis_length, minor_axis_length, eccentricity, convex_area, extent]
             features_list.append(features)

         return features_list

     # Define the main directory path
     main_directory_path = '/content/rice-image-dataset/Rice_Image_Dataset'

     # Get all class subdirectories
     classes = os.listdir(main_directory_path)
     all_features = []

     # Iterate through each class subdirectory
     for class_name in classes:
         class_path = os.path.join(main_directory_path, class_name)
         if os.path.isdir(class_path):
             # Get all image files in the class directory
             for filename in os.listdir(class_path):
                 if filename.endswith(".png") or filename.endswith(".jpg"):
                     image_path = os.path.join(class_path, filename)
                     features = extract_features(image_path)
                     if features:
                         all_features.extend(features)

     # Initialize DataFrame
     columns = ["Area", "Perimeter", "Major_Axis_Length", "Minor_Axis_Length", "Eccentricity", "Convex_Area", "Extent"]
     df = pd.DataFrame(all_features, columns=columns)

     # Drop rows with NaN values
     df = df.dropna()

     # Add additional columns
     df['Variable Name'] = ["Feature"] * len(df)
     df['Role'] = ["Feature"] * len(df)
     df['Type'] = ["Continuous"] * len(df)
     df['Description'] = [""] * len(df)
     df['Units'] = [""] * len(df)
     df['Missing Values'] = ["no"] * len(df)

     # Rearrange column order
     df = df[["Variable Name", "Role", "Type", "Description", "Units", "Missing Values", "Area", "Perimeter", "Major_Axis_Length", "Minor_Axis_Length", "Eccentricity", "Convex_Area", "Extent"]]

     # Display the DataFrame
     print(df)

     # Count the number of classes and images
     num_classes = len(classes)
     num_images = len(all_features)

     print(f'Number of classes: {num_classes}')
     print(f'Number of images: {num_images}')

     # Check if image sizes are uniform
     unique_sizes = set(df[["Major_Axis_Length", "Minor_Axis_Length"]].apply(tuple, axis=1))
     print(f'Unique image sizes: {unique_sizes}')
     ```

2. **Plot Example Classes of the Image**:
   - Plotted example images from each class to visualize the data.
   - Example code:
     ```python
     import os
     import cv2
     import matplotlib.pyplot as plt

     # Define the main directory path
     main_directory_path = '/content/rice-image-dataset/Rice_Image_Dataset'

     # Get all class subdirectories
     classes = os.listdir(main_directory_path)

     # Plot example images from each class
     fig, axs = plt.subplots(1, len(classes), figsize=(15, 5))

     for i, class_name in enumerate(classes):
         class_path = os.path.join(main_directory_path, class_name)
         if os.path.isdir(class_path):
             image_path = os.path.join(class_path, os.listdir(class_path)[0])
             image = cv2.imread(image_path)
             image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

             # Plot the image
             axs[i].imshow(image, cmap='gray')
             axs[i].set_title(class_name)
             axs[i].axis('off')

     plt.show()
     ```
![class_rice](https://raw.githubusercontent.com/Astraeven0502/CSE151A_Group_Project-Rice_Classification/main/data_picture/class_rice.png)

3. **Print the data description and check the number of missing values in each column**:
![data_description](https://raw.githubusercontent.com/Astraeven0502/CSE151A_Group_Project-Rice_Classification/main/data_picture/Data_description.png)

4. **Plot the correlation matrix heatmap**
![heatmap](https://raw.githubusercontent.com/Astraeven0502/CSE151A_Group_Project-Rice_Classification/main/data_picture/heatmap.png)

5. **Plot the Pairplot**
![pairplot](https://raw.githubusercontent.com/Astraeven0502/CSE151A_Group_Project-Rice_Classification/main/data_picture/pairplot.png) \\

All exploration steps are implemented in the Jupyter notebook and the code is available in the repository.

## Preprocess the data
* Load the image from the dataset to check the quality of the images.
* Check the size of the images and unify them to the same size.
* Convert labels to categories, `Arborio （1）-> Arborio`.
* Convert images to matrices.
* Compute feature matrix from image matrices.
* Manually add the categories to the feature matrix.
* Standardize the image matrix for later model building.
* Analyze feature data and remove data that affects model accuracy. Remove missing values, redundant features, unnecessary samples, outliers, duplicate records in the data to reduce redundancy.
