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


def get_training_df(paths) -> pd.DataFrame:
    """
    Load the csv files from the given paths.
    Merge and shuffle them to get a training dataframe.
    :param paths: Paths to the csv files.
    :return: Dataframe
    """
    start_df = None
    for path in paths:
        this_df = pd.read_csv(path)
        if start_df is None:
            start_df = this_df
        else:
            start_df = pd.concat([start_df, this_df])

    df = start_df.sample(frac=1).reset_index(drop=True)
    return df


def train_model(limit_data_num=None, print_report=True):

    """ Get Training Data"""
    df = get_training_df(["../data/train/using.csv", "../data/train/not_using.csv"])

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
    model = RandomForestClassifier(n_estimators=200, random_state=114514+1919)
    # model = SVR(kernel='rbf', C=0.1)

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


def batch_train_models(num_iter):
    iterations = [i for i in range(num_iter)]
    accuracies = []
    models = []
    scalers = []
    for i in range(num_iter):
        model, scaler, acc = train_model(print_report=False)
        print(f"\033[A\033[2K Iteration: {i}, Accuracy: {round(acc, 4)}")
        accuracies.append(acc)
        models.append(model)
        scalers.append(scaler)

    mean = np.mean(accuracies)
    std_dev = np.std(accuracies)

    plt.plot(iterations, accuracies, label=f"Training Accuracies, Std Dev={round(std_dev, 4)}")
    plt.plot(iterations, [mean for _ in range(len(iterations))], label=f"Mean={mean}")
    plt.title("Training Accuracies")
    plt.xlabel("Training Iterations")
    plt.ylabel("Training Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Select the model with an accuracy that's closest to 0.85
    indexed_accuracies = list(enumerate(accuracies))
    sorted_indexed_accuracies = sorted(indexed_accuracies, key=lambda x: abs(x[1] - 0.85))
    print(sorted_indexed_accuracies)

    return models[sorted_indexed_accuracies[0][0]], scalers[sorted_indexed_accuracies[0][0]]


best_model, best_scaler = batch_train_models(num_iter=100)
with open('../data/models/posture_classify.pkl', 'wb') as f:
    pickle.dump(best_model, f)
with open('../data/models/posture_classify_scaler.pkl', 'wb') as f:
    pickle.dump(best_scaler, f)

