import os
import json
import random
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report


def batch_get_data(data_dir, label):
    people = []
    for root, _, files in os.walk(data_dir):
        for file_name in files:
            if not file_name.endswith(".json"):
                continue
            file_path = os.path.join(root, file_name)
            with open(file_path, "r") as f:
                data = json.load(f)
                key_angles = [dat['angle'] for dat in data]
                people.append((label, key_angles))

    return people


def get_training_data(dir_to_data, possible_labels):
    """
    Suppose there are n datapoints in the dataset. Each has m features.
    Training data structure:
    {
        "feature_1": [...(type=float, len = n)],
        "feature_2": [...(type=float, len = n)],
        "feature_3": [...(type=float, len = n)],
        ....
        "feature_m": [...(type=float, len = n)],
        "labels": [...(type=int(boolean), len = n)]
    }
    :return:
    """

    # Extract training data from json
    _train_data = []
    for label in possible_labels:
        _train_data += batch_get_data(os.path.join(dir_to_data, label), label)

    # Shuffle data
    random.shuffle(_train_data)

    # Create scikit-learn data
    train_data = {}
    labels = []
    for i, data in enumerate(_train_data):
        label, quadruple = data
        for fi, feature in enumerate(quadruple):
            feature_name = f"feature_{fi}"
            if feature_name not in train_data:
                train_data[feature_name] = []
            train_data[feature_name].append(feature)
        labels.append(possible_labels.index(label))
    train_data["labels"] = labels

    return train_data


def train_model():

    """ Get Training Data"""
    data = get_training_data("./data/train/angles/", ["using", "not_using"])
    for key in data.keys():
        print(f"{key} (len={len(data[key])}): {data[key]}")

    df = pd.DataFrame(data)

    # Split the data into features and labels
    x = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # Split data into train and test
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Standardize data
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # Train the Model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(x_train, y_train)

    # Evaluate the Model
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"Accuracy: {accuracy}")
    print(f"Report: {report}")


train_model()
