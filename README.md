# A Posture Analysis Model to Detect Cell-Phone Usages

## Start-Up
Run `pip install -r requirements.txt`

## Run
Run `python main.py`


# Schedules

## Phase 0. Data Preparation
### Step 0. Data Gathering & Feature Selection
#### 0.0 Data Gathering
&emsp; From arbitrary sources, gather plenty of usable image data where 
the posture of a person is clearly given in the picture.

#### 0.1 Feature Selection
&emsp; Decide which features is used to decide whether a person is using a phone.
A feature of a datapoint (a person) is one of the many numerical data of the person's
skeleton. E.g., an angle of a key joint.


### Step 1. Skeleton Extraction & Image Annotation
#### 1.0 Extract Skeleton
&emsp; Extract the skeleton of the person with a pre-trained model and retrieve
the decided numerical data of the key feature. Mediapipe is used.

#### 1.1 Annotate Image
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


## Phase 1. Model Training

## Notes
### Preliminary Test of Model Training Performance
| Num of Training Data | Accuracy Std Dev |
|----------------------|------------------|
| 11                   | 0.18             |
| 21                   | 0.13             |
| 31                   | 0.10             |
| 40                   | 0.09             |
