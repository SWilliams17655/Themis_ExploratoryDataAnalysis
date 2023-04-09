import pickle
import matplotlib.style as style
import tensorflow as tf
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score


# *************************************************************************

def save_data_set(data, file_location):
    """
    Saves a data set to a .csv file.

    :param data : DataFrame representing the file to be saved.
    :param file_location : String representing the file location to save the information.
    """

    data.to_csv(file_location)


def split_training_test_set(data_set, attribute, training_file_loc, eval_file_loc):
    """
    Splits the test and training set into a 80/20 using the label as criteria

    :param data_set: DataFrame containing the training set to be split.
    :param attribute: Label for the attribute used to classify the data.
    :param training_file_loc: String representing the file location where training data will be saved.
    :param eval_file_loc: String representing the file location where test data will be saved.
    """

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(data_set, data_set[attribute]):
        train_set = data_set.loc[train_index]
        test_set = data_set.loc[test_index]

    train_set.to_csv(training_file_loc)
    test_set.to_csv(eval_file_loc)


def save_model(model, location):
    """
    Saves a model to a .pkl file

    :param model: The model to be saved.
    :param location: The location to save the file. Should end in .pkl.
    """
    pickle.dump(model, open(location, 'wb'))


def load_model(location):
    """
    Loads a model from a location

    :param location: The location to save the file. Should end in .pkl.
    """
    return pickle.load(open(location, 'rb'))


def display_neural_network(model):
    """
    Generates a image of the neural network using keras-visualizer.

    :param model: the neural network to be displayed.
    """
    visualizer(model, format='png', view=True)


def compare_roc(models, models_label, x_train, y_train):
    """
    Generates a ROC graph for a list of models.

    :param models: A list of models to added to the ROC curve.
    :param models_label: A  list of strings representing the title of the models.
    :param x_train: Training inputs for the model.
    :param y_train: The labels for the dataset.
    """

    print("Calculating ROC Curves now...\n")
    fig, (ax1) = plt.subplots(1, 1, figsize=(10, 9))
    i = 0
    for model in models:
        y_probas = cross_val_predict(model, x_train, y_train, cv=10, method="predict_proba")
        y_scores = y_probas[:, 1]
        fpr, tpr, thresholds = roc_curve(y_train, y_scores)
        ax1.plot(fpr, tpr, label=models_label[i])
        i += 1
    ax1.set_ylabel("True Positive Rate")
    ax1.set_xlabel("False Positive Rate")
    ax1.legend()
    ax1.set_title("Receiver Operating Characteristic (ROC)")

    plt.show()


def train_neural_network(x_train, y_train, x_eval_set, y_eval_set):
    """
    Trains a baseline neural network to classify data using test and training.

    :param x_train : DataFrame with attributes used for classification.
    :param y_train : Series with solutions for x_train attributes.
    :param x_eval_set : DataFrame with attributes used for verification.
    :param y_eval_set : Series with solutions for x_test attributes.
    :return: Returns the trained model.
    """
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    num_epoch = 75
    num_batch = 250
    drop_out_rate = .2
    input_shape = len(x_train.columns)
    initializer = tf.keras.initializers.HeUniform()
    num_hidden_layer_1 = 75
    num_hidden_layer_2 = 200
    num_hidden_layer_3 = 50
    num_output_layer = 1

    # creates the neural network model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(num_hidden_layer_1, input_shape=(input_shape,),
                              activation=tf.nn.relu,
                              kernel_initializer=initializer),
        tf.keras.layers.Dropout(drop_out_rate,
                                input_shape=(num_hidden_layer_1,)),
        tf.keras.layers.Dense(num_hidden_layer_2, activation=tf.nn.relu,
                              kernel_initializer=initializer),
        tf.keras.layers.Dense(num_hidden_layer_3, activation=tf.nn.relu,
                              kernel_initializer=initializer),
        tf.keras.layers.Dense(num_output_layer, activation=tf.nn.sigmoid,
                              # softmax and sparse cis best for non-binary, but use sigmoid and
                              kernel_initializer=initializer)
    ])
    # Softmax and loss=tf.keras.losses.SparseCategoricalCrossentropy() if it is a multi-classification or sigmoid and
    # loss=tf.keras.losses.BinaryCrossentropy() if it is binary with a single state.

    # compiles the model using accuracy as the metric and adam as the optimizer
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss=tf.keras.losses.BinaryCrossentropy(),  # loss=tf.keras.losses.SparseCategoricalCrossentropy()
                  metrics=['accuracy'])

    csv_logger = tf.keras.callbacks.CSVLogger("NN_log.csv")

    # Trains model using x_test and y_test.
    model.fit(x_train, y_train,
              epochs=num_epoch,
              batch_size=num_batch,
              validation_data=(x_eval_set, y_eval_set),
              callbacks=[csv_logger])

    return model


def train_random_forest(x_train, y_train, x_eval_set, y_eval_set, param_search_on=False):
    """
    Trains a baseline neural network to classify data using test and training

    :param x_train : DataFrame with attributes used for classification
    :param y_train : Series with solutions for x_train attributes
    :param x_eval_set : DataFrame with attributes used for classification
    :param y_eval_set : Series with solutions for x_train attributes
    :param param_search_on : Boolean to determine if a parameter search will be conducted
    """

    style.use('ggplot')

    model = RandomForestClassifier(bootstrap=True,
                                   criterion='gini',
                                   max_depth=None,
                                   max_features="sqrt",
                                   max_leaf_nodes=None,
                                   min_samples_leaf=1,
                                   min_samples_split=2,
                                   min_impurity_decrease=0.0,
                                   min_weight_fraction_leaf=0.0,
                                   n_estimators=400,
                                   n_jobs=-1,
                                   verbose=0
                                   )

    if param_search_on is True:
        print("\nConducting Grid Search for optimized parameters...\n")

        param_grid = [
            {'bootstrap': [True, False],
             'max_depth': [10, 20, None],
             'max_features': ["sqrt", "log2", None],
             'min_impurity_decrease': [0.0, 1.0, 3.0],
             'n_estimators': [400, 500, 700]}
        ]

        grid_search = GridSearchCV(model, param_grid, cv=3, scoring="accuracy")
        grid_search.fit(x_train.values, y_train.values)
        print("\nThe optimum parameters are: ", grid_search.best_estimator_, "\n")

    sol = cross_val_score(model, x_train, y_train, cv=10, n_jobs=-1, verbose=1, scoring="accuracy")
    print("The average accuracy was:", sol.mean())

    model.fit(x_train.values, y_train.values)
    score = model.score(x_eval_set.values, y_eval_set.values)
    print("The accuracy on the evaluation set was: " + str(score))

    return model


def train_sgd(x_train, y_train, x_eval_set, y_eval_set, param_search_on=False):
    """
    Trains a baseline neural network to classify data using test and training

    :param x_train : DataFrame with attributes used for classification
    :param y_train : Series with solutions for x_train attributes
    :param x_eval_set : DataFrame with attributes used for classification
    :param y_eval_set : Series with solutions for x_train attributes
    :param param_search_on : Boolean to determine if a parameter search will be conducted
    """

    style.use('ggplot')

    model = SGDClassifier(loss='log_loss',
                          alpha=0.01,
                          max_iter=500,
                          shuffle=True,
                          verbose=0,
                          n_jobs=-1,
                          learning_rate='optimal',
                          early_stopping=True,
                          eta0=0.0001,
                          )

    if param_search_on is True:
        print("\nConducting Grid Search for optimized parameters...\n")

        param_grid = [
            {'loss': ['log_loss', 'modified_huber'],  # there are a lot of options
             'alpha': [0.0001, 0.001, 0.01],
             'max_iter': [500, 1000, 2000],
             'learning_rate': ['constant', 'optimal', 'adaptive']}
        ]

        grid_search = GridSearchCV(model, param_grid, cv=3, scoring="accuracy")
        grid_search.fit(x_train, y_train)
        print("\nThe optimum parameters are: ", grid_search.best_params_, "\n")

    sol = cross_val_score(model, x_train, y_train, cv=10, n_jobs=-1, verbose=1, scoring="accuracy")
    print("The average accuracy was:", sol.mean())

    model.fit(x_train.values, y_train.values)
    score = model.score(x_eval_set.values, y_eval_set.values)
    print("The accuracy on the evaluation set was: " + str(score))

    return model


def train_svm(x_train, y_train, x_eval_set, y_eval_set, param_search_on=False):
    """
    Trains a baseline neural network to classify data using test and training.

    :param x_train : DataFrame with attributes used for classification.
    :param y_train : Series with solutions for x_train attributes.
    :param x_eval_set : DataFrame with attributes used for classification.
    :param y_eval_set : Series with solutions for x_train attributes.
    :param param_search_on : Boolean to determine if a parameter search will be conducted.
    """

    print("\nTraining SVM for classification...")

    model = SVC(kernel='linear',  # ‘rbf’, ‘poly’, ‘rbf’, ‘sigmoid’
                degree=3,
                gamma='scale',  # auto or scale
                coef0=1,
                probability=True,  # true may slow down processing
                C=2,  # smaller the number the wider the street
                max_iter=-1,
                verbose=True)

    if param_search_on is True:
        print("\nConducting Grid Search for optimized parameters...\n")

        param_grid = [{
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'degree': [3, 6, 10],
            'gamma': ['scale', 'auto'],
            'C': [2, 5, 10, 40]}]

        grid_search = GridSearchCV(model, param_grid, cv=3, scoring="accuracy")

        grid_search.fit(x_train, y_train)
        print("\nThe optimum parameters are: ", grid_search.best_params_, "\n")

    model.fit(x_train, y_train)
    score = model.score(x_eval_set, y_eval_set)
    print("The accuracy on the evaluation set was: " + str(score))

    return model


def make_one_prediction(model, input_data_series):
    print(input_data_series)
    input = np.array(input_data_series)
    input = np.reshape(input, (1, len(input_data_series)))
    ans = model.predict(input)
    return ans[0]
