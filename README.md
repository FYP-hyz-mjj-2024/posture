# A Posture Analysis Model to Detect Cell-Phone Usages

# Methodology
&emsp; Before talking more about the project, we would like to share our methodology.
![methodology.png](project_documents%2FREADME_images%2Fmethodology.png)

# Schedules
## Step 0. Data Gathering & Feature Selection
### 0.0 Data Gathering
&emsp; From arbitrary sources, gather plenty of usable image data where 
the posture of a person is clearly given in the picture.

### 0.1 Feature Selection
&emsp; Decide which features is used to decide whether a person is using a phone.
A feature of a datapoint (a person) is one of the many numerical data of the person's
skeleton. E.g., an angle of a key joint.


## Step 1. Skeleton Extraction & Image Annotation
### 1.0 Extract Skeleton
&emsp; Extract the skeleton of the person with a pre-trained model and retrieve
the decided numerical data of the key feature. Mediapipe is used.

### 1.1 Annotate Image
&emsp; For each image of a person, after extracting the key features, store this person
into a vector of key features. Therefore, a key-feature vector represents a datapoint,
i.e., a person.

&emsp; An image is annotated into a json form as follows:
```json
[
    {
        "key": "Left_shoulder-Left_elbow-Left_wrist",
        "angle": 80.096
    },
    {
        "key": "Right_shoulder-Right_elbow-Right_wrist",
        "angle": 73.637
    },
    {
        "key": "Left_hip-Left_shoulder-Left_elbow",
        "angle": 19.399
    },
    {
        "key": "Right_hip-Right_shoulder-Right_elbow",
        "angle": 7.581
    },
    {
        "key": "Right_shoulder-Nose-Left_shoulder",
        "angle": 22.846
    },
    {
        "key": "Right_eye_outer-Right_shoulder-Nose",
        "angle": 11.771
    },
    {
        "key": "Left_eye_outer-Left_shoulder-Nose",
        "angle": 1.282
    }
]
```

## Step 2. Model Training
### 2.0 Gather Data
&emsp; Arrange annotated data into a single dataframe table to feed into the model.
Suppose that there are n datapoints (i.e., people) in the dataset, each has m features 
(e.g. joints of key angles). Therefore, the training data structure should be as follows:
```console
    {
        "feature_1": [...(type=float, len = n)],
        "feature_2": [...(type=float, len = n)],
        "feature_3": [...(type=float, len = n)],
        ....
        "feature_m": [...(type=float, len = n)],
        "labels": [...(type=int(boolean), len = n)]
    }
```

### 2.1 Train a Model
&emsp; Using Random Forest / SVR / CNN, a model is trained based on the datasets. To ensure
accuracy, random shuffle is applied to the dataset and the 100 separate model is trained at once.
After that, select the model with the accuracy closest to 0.85 as our target model.

#### Preliminary Test of Model Training Performance
| Num of Training Data | Accuracy Std Dev |
|----------------------|------------------|
| 11                   | 0.18             |
| 21                   | 0.13             |
| 31                   | 0.10             |
| 40                   | 0.09             |
