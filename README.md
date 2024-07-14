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
         image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
         if image is None:
             print(f"Failed to load image: {image_path}")
             return []

         _, thresh = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
         contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

         features_list = []
         for contour in contours:
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
                 else:
                     print(f"Invalid values for MA ({MA}) or ma ({ma}) in image {image_path}")
             hull = cv2.convexHull(contour)
             convex_area = cv2.contourArea(hull)
             extent = area / (w * h) if w * h > 0 else 0

             features = [area, perimeter, major_axis_length, minor_axis_length, eccentricity, convex_area, extent]
             features_list.append(features)

         return features_list

     main_directory_path = '/content/rice-image-dataset/Rice_Image_Dataset'
     classes = os.listdir(main_directory_path)
     all_features = []

     for class_name in classes:
         class_path = os.path.join(main_directory_path, class_name)
         if os.path.isdir(class_path):
             for filename in os.listdir(class_path):
                 if filename.endswith(".png") or filename.endswith(".jpg"):
                     image_path = os.path.join(class_path, filename)
                     features = extract_features(image_path)
                     if features:
                         all_features.extend(features)

     columns = ["Area", "Perimeter", "Major_Axis_Length", "Minor_Axis_Length", "Eccentricity", "Convex_Area", "Extent"]
     df = pd.DataFrame(all_features, columns=columns)

     # Drop rows with NaN values
     df = df.dropna()

     df['Variable Name'] = ["Feature"] * len(df)
     df['Role'] = ["Feature"] * len(df)
     df['Type'] = ["Continuous"] * len(df)
     df['Description'] = [""] * len(df)
     df['Units'] = [""] * len(df)
     df['Missing Values'] = ["no"] * len(df)

     df = df[["Variable Name", "Role", "Type", "Description", "Units", "Missing Values", "Area", "Perimeter", "Major_Axis_Length", "Minor_Axis_Length", "Eccentricity", "Convex_Area", "Extent"]]

     print(df)
     num_classes = len(classes)
     num_images = len(df)

     print(f'Number of classes: {num_classes}')
     print(f'Number of images: {num_images}')

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

     main_directory_path = '/content/rice-image-dataset/Rice_Image_Dataset'
     classes = os.listdir(main_directory_path)

     fig, axes = plt.subplots(1, len(classes), figsize=(15, 5))

     for i, class_name in enumerate(classes):
         class_path = os.path.join(main_directory_path, class_name)
         if os.path.isdir(class_path):
             example_image_path = os.path.join(class_path, os.listdir(class_path)[0])
             img = cv2.imread(example_image_path)
             img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
             
             axes[i].imshow(img)
             axes[i].set_title(class_name)
             axes[i].axis('off')

     plt.show()
     ```

All exploration steps are implemented in the Jupyter notebook and the code is available in the repository.
