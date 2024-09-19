# Packages
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVR
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pickle

def train_model(limit_data_num=None, print_report=True):

    # Load Data
    using = np.load("../data/train/using_small.npy")
    not_using = np.load("../data/train/not_using_small.npy")

    # TODO
    np.random.shuffle(using)
    using = using[:len(not_using) // 2]

    # Pre Process
    X = np.vstack((using, not_using))
    y = np.hstack((np.ones(len(using)), np.zeros(len(not_using))))

    # Divide
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=1919810)

    # Standardize data
    scaler = StandardScaler()
    x_train = scaler.fit_transform(X_train)
    x_test = scaler.transform(X_test)

    # Train the Model
    model = RandomForestClassifier(n_estimators=200, random_state=114514+1919)
    # YOLO_model = SVR(kernel='rbf', C=0.1)

    # TODO: CNN , SVR, ??
    model.fit(x_train, y_train)

    # Evaluate the Model
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    # mse = mean_squared_error(y_test, y_pred)

    if print_report:
        report = classification_report(y_test, y_pred)
        print(f"Report: {report}")

    return model, scaler, accuracy


def batch_train_models(num_iter, preferred_accuracy=None):
    """
    Train YOLO_model numerous times separately, and select the best YOLO_model.
    The best YOLO_model is suggested to have a high yet mediocre accuracy.
    :param num_iter: Number of iterations, i.e., the number of models.
    :param preferred_accuracy: The preferred accuracy. The YOLO_model with an accuracy that's
    closes to this number will be regarded as the "best YOLO_model" and selected into production.
    """
    iterations = [i for i in range(num_iter)]
    batch = {
        "accuracies": [],
        "models": [],
        "scalers": []
    }

    for i in range(num_iter):
        model, scaler, acc = train_model(print_report=False)
        print(f"\033[A\033[2K Iteration: {i}, Accuracy: {round(acc, 4)}")
        batch['accuracies'].append(acc)
        batch['models'].append(model)
        batch['scalers'].append(scaler)

    mean = np.mean(batch['accuracies'])
    std_dev = np.std(batch['accuracies'])

    plt.plot(iterations, batch['accuracies'], label=f"Training Accuracies, Std Dev={round(std_dev, 4)}")
    plt.plot(iterations, [mean for _ in range(len(iterations))], label=f"Mean={mean}")
    plt.title("Training Accuracies")
    plt.xlabel("Training Iterations")
    plt.ylabel("Training Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Select the YOLO_model with an accuracy that's closest to 0.85
    indexed_accuracies = list(enumerate(batch['accuracies']))
    sorted_indexed_accuracies = sorted(indexed_accuracies, key=lambda x: abs(x[1] - preferred_accuracy if preferred_accuracy is not None else mean))
    print(sorted_indexed_accuracies)

    return batch['models'][sorted_indexed_accuracies[0][0]], batch['scalers'][sorted_indexed_accuracies[0][0]]


best_model, best_scaler = batch_train_models(num_iter=100)
with open('../data/models/posture_dt.pkl', 'wb') as f:
    pickle.dump(best_model, f)
with open('../data/models/posture_dt_scaler.pkl', 'wb') as f:
    pickle.dump(best_scaler, f)

