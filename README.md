# CSE151A_Group_Project-Rice_Classification

#### Team member:
* Po-Yu Lai
* chaowen cao
* Xinheng Wang
* Jiawei Huang
* Zhenhan Hu
* Shiwei Yang
# Data exploration

We have 5 rice varieties in our image data, Arborio, Basmati, Ipsala, Jasmine, and Karacadag. The total number of the images is 75,000. Each of the varieties has 15,000 images from different angles and sizes. The image sizes are all 250x250 pixels, but the sizes of each rice are different. Every image is in a dark background with exactly one rice in the middle of the image. We will look at this image and use OpenCV to get the feature from those image datasets. The feature we will extract are:
|    | Variable Name     | Role    | Type       | Description                                                                                                     |
|---:|:------------------|:--------|:-----------|:----------------------------------------------------------------------------------------------------------------|
|  0 | Area              | Feature | Integer    | Returns the number of pixels within the boundaries of the rice grain                                            |
|  1 | Perimeter         | Feature | Continuous | Calculates the circumference by calculating the distance between pixels around the boundaries of the rice grain |
|  2 | Major_Axis_Length | Feature | Continuous | The longest line that can be drawn on the rice grain, i.e. the main axis distance                               |
|  3 | Minor_Axis_Length | Feature | Continuous | The shortest line that can be drawn on the rice grain, i.e. the small axis distance                             |
|  4 | Eccentricity      | Feature | Continuous | It measures how round the ellipse, which has the same moments as the rice grain                                 |
|  5 | Convex_Area       | Feature | Integer    | Returns the pixel count of the smallest convex shell of the region formed by the rice grain                     |
|  6 | Extent            | Feature | Continuous | Returns the ratio of the region formed by the rice grain to the bounding box pixels   

## Retrieve the data

```bash
!git lfs install
!git clone https://huggingface.co/datasets/nateraw/rice-image-dataset
!unzip /content/rice-image-dataset/rice-image-dataset.zip -d /content/rice-image-dataset/
```

## Library used

```python
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

## Data Exploration

We performed the following data exploration steps:

1. **Extract Features From Image Data**:
   - Example code:
    ```python
    # Define the main directory path
    main_directory_path = '/content/rice-image-dataset/Rice_Image_Dataset'

    # Get all class subdirectories
    classes = os.listdir(main_directory_path)
    classes.remove('Rice_Citation_Request.txt')
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
    ```

2. **Described the number of classes, number of images**.
    - Example code:
    ```python
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
    df = df[["Area", "Perimeter", "Major_Axis_Length", "Minor_Axis_Length", "Eccentricity", "Convex_Area", "Extent"]]

    # Display the DataFrame
    print(df)

    # Count the number of classes and images
    num_classes = len(classes)
    num_images = len(all_features)

    print(f'Number of classes: {num_classes}')
    print(f'Number of images: {num_images}')

    ```

3. **Plot Example Classes of the Image for each varieties**:
   - Example code:
    ```python
    # Plot three images for each variety
    fig, axs = plt.subplots(3, len(classes), figsize=(15, 10))

    for i, class_name in enumerate(classes):
        class_path = os.path.join(main_directory_path, class_name)
        if os.path.isdir(class_path):
            image_paths = os.listdir(class_path)[:3]
            for j, image_path in enumerate(image_paths):
                image_path = os.path.join(class_path, image_path)
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Plot the image
                axs[j, i].imshow(image, cmap='gray')
                axs[j, i].set_title(class_name)
                axs[j, i].axis('off')

    plt.show()
    ```

All exploration steps are implemented in the Jupyter notebook and the code is available in the repository.
