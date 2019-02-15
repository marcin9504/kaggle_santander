from warnings import filterwarnings

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from keras import regularizers
from keras.layers import Dense
from keras.models import Sequential
from keras.models import model_from_json
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

pd.set_option("display.max_columns", None)
# pd.set_option("display.height", 1000)
# pd.set_option("display.max_rows", 500)
pd.set_option("display.width", 200)

from sklearn.exceptions import DataConversionWarning

filterwarnings(action="ignore", category=DataConversionWarning)


def main():
    train = pd.read_csv("train.csv", sep=",")
    print("Loaded train data")
    test = pd.read_csv("test.csv", sep=",")
    print("Loaded test data")
    # print(train.head())
    # print(train.describe())
    # return
    train_ids = train["ID_code"]
    test_ids = test["ID_code"]

    train = train.drop("ID_code", axis=1)
    test = test.drop("ID_code", axis=1)
    # print(train.shape)
    # print(test.shape)

    num_train = train.shape[0]
    num_test = test.shape[0]
    y_train = train.target.values

    all_data = pd.concat((train, test), sort=True).reset_index(drop=True)
    # print(all_data.head())
    all_data = all_data.drop(["target"], axis=1)

    train = all_data[:num_train]
    test = all_data[num_train:]

    # predictions = get_predictions_from_nn(test, train, y_train)

    compare_models(train, y_train)

    return

    model = models["cat_boost"]
    print("Training model")
    model.fit(train.values, y_train)
    print("Predicting target")
    predictions = model.predict(test.values)

    print("Saving output")
    final = pd.DataFrame()
    # print(test_ids, predictions)
    final["ID_code"] = test_ids
    final["target"] = predictions
    final.to_csv("submission.csv", index=False)


def compare_models(train, y_train):
    models = {
        "random_forest": RandomForestClassifier(n_estimators=1000),
        "mlp": MLPClassifier(max_iter=10000),
        "logistic": LogisticRegression(solver="lbfgs", max_iter=10000),
        "sgd": SGDClassifier(max_iter=10000, tol=1e-3),
        "knn": KNeighborsClassifier(3),
        "svm linear": SVC(kernel="linear", C=0.025),
        "svm rbf": SVC(gamma=2, C=1),
        "gaussian_process": GaussianProcessClassifier(1.0 * RBF(1.0)),
        "decision_tree": DecisionTreeClassifier(max_depth=5),
        "ada_boost": AdaBoostClassifier(),
        "naive_bayes": GaussianNB(),
        "qda": QuadraticDiscriminantAnalysis(),
        "cat_boost": CatBoostClassifier(loss_function="Logloss",
                                        eval_metric="AUC", verbose=0),
    }
    for model_name in models:
        print(model_name)
        model = models[model_name]
        score = cross_val_score(model, train, y_train, scoring=make_scorer(roc_auc_score), cv=3, n_jobs=-1)
        print(score.mean(), score.std())


def get_predictions_from_nn(test, train, y_train):
    nn_model = get_model(train.shape[1])
    # nn_model = load_model()
    train_model(nn_model, train, y_train)
    save_model(nn_model)
    print("Predicting target")
    predictions = [int(x) for x in np.rint(nn_model.predict(test.values))]
    return predictions


def save_model(model):
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("model.h5")
    print("Saved model to disk")


def load_model():
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")
    return loaded_model


def get_model(input_shape):
    model = Sequential()
    model.add(Dense(16, input_dim=input_shape, activation='relu', kernel_regularizer=regularizers.l2(0.005)))
    model.add(Dense(16, input_dim=input_shape, activation='relu', kernel_regularizer=regularizers.l2(0.005)))
    model.add(Dense(1, activation='sigmoid'))
    return model


def train_model(nn_model, train, y_train):
    X_train, X_val, y_train, y_val = train_test_split(train, y_train, test_size=0.1, stratify=y_train)
    nn_model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])
    epochs = 3
    nn_model.fit(X_train, y_train, epochs=epochs, validation_data=(X_val, y_val))


if __name__ == "__main__":
    main()
