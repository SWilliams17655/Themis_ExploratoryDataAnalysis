import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.style as style
import numpy as np


class AIDataset:
    def __init__(self, file_location=""):
        self.file_location = file_location
        self.data_file = pd.DataFrame()
        self.x = pd.DataFrame()
        self.y = pd.DataFrame()

    def load(self, os_loc=None, y_label=None):

        """
        Loads a data set for use in the machine learning program to test.

        :param os_loc: [Optional] String representing the file to be loaded. If None, the function
        will request user provide a location via the command prompt. Default = None.
        :param y_label: [Optional] String representing the label for y which will be split into y dataframe. Default = None.
        if it is None then bypasses splitting out the y array and lets user do it manually.
        :return: DataFrame representing the file loaded.
        """

        if os_loc is None:
            self.data_file = pd.read_csv(self.file_location)

        else:
            self.file_location = os_loc
            self.data_file = pd.read_csv(self.file_location)

        if y_label is not None:
            self.y = self.data_file[y_label]
            self.x = self.data_file.drop([y_label], axis=1)

        return self.data_file

    def show_table_in_html(self, filename):
        """
        Sends the dataset passed to an html file for viewing
        :param filename: String representing the filename where the html will be sent.
        """
        new_table = self.x
        new_table["Classifier"] = self.y
        new_table.to_html(filename)

    def show_data_description_in_html(self, filename):
        """
        Sends the dataset passed to an html file for viewing
        :param filename: String representing the filename where the html will be sent.
        """
        new_table = self.x
        new_table["Classifier"] = self.y
        new_table.describe().to_html(filename)

    def get_image_at(self, position=0):
        """
        Displays an image representing the character following processing.

        :param position: Position of the image in the array to be plotted.
        :return: 28x28 image representing the data at that location.
        """

        image = self.x.iloc[position]
        image = image.to_numpy().reshape((28, 28))
        return image

    def normalize(self, label, min_value, max_value):
        """
        Displays an image representing the character following processing.

        :param label: Label to be normalized.
        :param min_value: minimum value possible for data to be normalized.
        :param max_value: maximum value possible for data to be normalized.
        """
        self.x[str(label + "_norm")] = self.x[label].apply(lambda x: (x - min_value) / max_value)

    def show_hist(self, attribute, min_range=0, max_range=1, step=.25, y_axis_top=.75):
        """
        Plots multiple scatter plots showing relationship between attributes in
        attributes array. If mask is a value other than none, graph only shows correlating data for that value of Y.

        :param attribute: An array of strings with the title of the attributes to be correlated.
        :param min_range: Minimum range of the histogram x-axis.
        :param max_range: Maximum range of the histogram x-axis.
        :param step: Step used along the x-axis.
        :param y_axis_top: Top of the y-axis.
        """
        style.use('ggplot')
        # 'Solarize_Light2', '_classic_test_patch', '_mpl-gallery', '_mpl-gallery-nogrid', 'bmh', 'classic',
        # 'dark_background', 'fast', 'fivethirtyeight', 'ggplot', 'grayscale', 'seaborn-v0_8', 'seaborn-v0_8-bright',
        # 'seaborn-v0_8-colorblind', 'seaborn-v0_8-dark', 'seaborn-v0_8-dark-palette', 'seaborn-v0_8-darkgrid',
        # 'seaborn-v0_8-deep', 'seaborn-v0_8-muted', 'seaborn-v0_8-notebook', 'seaborn-v0_8-paper',
        # seaborn-v0_8-pastel', 'seaborn-v0_8-poster', 'seaborn-v0_8-talk', 'seaborn-v0_8-ticks',
        # 'seaborn-v0_8-white', 'seaborn-v0_8-whitegrid', 'tableau-colorblind10'

        bin_range = np.arange(min_range, max_range, step)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9))
        true_array = self.x.where((self.y == 1))
        false_array = self.x.where((self.y == 0))
        ax1.hist(true_array[attribute], bins=bin_range, density=True,
                 histtype='bar', rwidth=0.7, color="cadetblue",
                 label=attribute)
        ax2.hist(false_array[attribute], bins=bin_range, density=True,
                 histtype='bar', rwidth=0.7, color="darkkhaki",
                 label=attribute)

        ax1.set_ylabel("Density of Dataset")
        ax1.set_title("Classifier is True")
        ax1.legend()
        ax1.set_xticks(bin_range)
        ax1.set_ylim(top=y_axis_top)
        ax2.set_xlabel("Values")
        ax2.set_ylabel("Density of Dataset")
        ax2.set_title("Classifier is False")
        ax2.set_xticks(bin_range)
        ax2.set_ylim(top=y_axis_top)
        ax2.legend()

        plt.show()

    def show_corr_matrix(self):
        """
        Plots a correlation matrix
        """

        new_val = self.x.copy()
        new_val["Classifier"] = self.y
        corr_matrix = new_val.corr()
        print(corr_matrix["Classifier"].sort_values(ascending=False))
        print("\n")

        fig, (ax1) = plt.subplots(1, 1, figsize=(9, 9))
        color_axes = ax1.matshow(corr_matrix, cmap='Greens')
        fig.colorbar(color_axes)
        ax1.set_xticks(np.arange(len(corr_matrix)), corr_matrix.columns, rotation=90)
        ax1.set_yticks(np.arange(len(corr_matrix)), corr_matrix.columns)
        plt.show()

    def show_scatter(self, attribute1, attribute2):
        """
        Plots a scatter plot relationship between three variables with x being attribute1, y is attribute 2, and
        y classifier is color of information.

        :param attribute1: Variable along the x-axis.
        :param attribute2: Variable along the y-axis.

        """

        style.use('ggplot')

        print(plt.style.available)

        fig, (ax1) = plt.subplots(1, 1, figsize=(10, 9))
        true_array = self.x.where((self.y == 1))
        false_array = self.x.where((self.y == 0))

        ax1.scatter(true_array[attribute1], self.x[attribute2], color="cadetblue", label="Classifier: True")
        ax1.scatter(false_array[attribute1], self.x[attribute2], color="darkkhaki", label="Classifier: False")
        ax1.set_title("Compare: Classifier, " + attribute1 + ", & " + attribute2)
        ax1.legend()
        ax1.set_xlabel(attribute1)
        ax1.set_ylabel(attribute2)
        plt.show()

    def remove_nulls(self, attribute):
        """
        Removes null values of an attribute in a data_set by replacing with median.

        :param attribute: Label of the attribute to be adjusted.
        """

        median = self.x[attribute].median()
        self.x[attribute].fillna(median, inplace=True)

    def remove_zeros(self, attribute):
        """
        Removes zero values in datasets missing data

        :param attribute: Label of the attribute to be adjusted.
        """

        median = self.x[attribute].median()

        self.x[attribute] = self.x[attribute].replace(0, median)

    def map_text_to_category(self, attribute):
        """
        Removes zero values in datasets missing data

        :param attribute: Label of the attribute to be adjusted.
        """
        category = self.x[attribute]
        cat_encoded, categories = category.factorize()
        self.x[str(attribute + "_cat")] = cat_encoded
