import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.style as style
import numpy as np


class AIDataset:
    def __init__(self):
        self.file_location = ""
        self.data_file = pd.DataFrame()
        self.x = pd.DataFrame()
        self.y = pd.DataFrame()

    def load(self, os_loc=None, y_label=None):
        """ Loads a data set for use in the machine learning program to test.

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

    def show_histogram(self):
        """
        Plots multiple scatter plots showing relationship between attributes in
        attributes array. If mask is a value other than none, graph only shows correlating data for that value of Y.
        """
        style.use('dark_background')
        fig_array = []

        for attribute in self.x.columns:
            fig, (ax1) = plt.subplots(1, 1, figsize=(20, 9))
            true_array = self.x.where((self.y == 1))
            false_array = self.x.where((self.y == 0))

            ax1.hist([true_array[attribute], false_array[attribute]], bins=10, density=False,
                     stacked=False, histtype='bar', rwidth=0.7, color=["royalblue", "tomato"],
                     label=["True", "False"])

            ax1.set_ylabel("Density of Dataset", fontsize=12)
            ax1.set_title(attribute, fontsize=20)
            ax1.legend(fontsize=15)
            ax1.patch.set_alpha(0.0)
            fig.patch.set_alpha(0.0)
            fig_array.append(fig)

        return fig_array

    def show_correlation_matrix(self):
        """
        Plots a correlation matrix
        """
        style.use('dark_background')
        new_val = self.x.copy()
        new_val["Classifier"] = self.y
        corr_matrix = new_val.corr()
        print(corr_matrix["Classifier"].sort_values(ascending=False))
        print("\n")

        fig, (ax1) = plt.subplots(1, 1, figsize=(20, 9))
        color_axes = ax1.matshow(corr_matrix, cmap='Greens')
        fig.colorbar(color_axes)
        ax1.set_xticks(np.arange(len(corr_matrix)), corr_matrix.columns, rotation=90)
        ax1.set_yticks(np.arange(len(corr_matrix)), corr_matrix.columns)
        ax1.set_title("Correlation Matrix")
        ax1.patch.set_alpha(0.0)
        fig.patch.set_alpha(0.0)
        return fig, corr_matrix

    def show_scatter_plot(self, attribute1):
        """
        Plots a scatter plot relationship between three variables with x being attribute1, y is attribute 2, and
        y classifier is color of information.

        """
        fig_array = []
        for attribute2 in self.x.columns:
            if attribute1 != attribute2:
                style.use('dark_background')

                fig, (ax1) = plt.subplots(1, 1, figsize=(10, 9))
                true_array = self.x.where((self.y == 1))
                false_array = self.x.where((self.y == 0))

                ax1.scatter(true_array[attribute1], self.x[attribute2], color="royalblue", label="Classifier: True")
                ax1.scatter(false_array[attribute1], self.x[attribute2], color="tomato", label="Classifier: False")
                ax1.set_title("Compare " + attribute1 + " & " + attribute2)
                ax1.legend()
                ax1.set_xlabel(attribute1)
                ax1.set_ylabel(attribute2)
                ax1.patch.set_alpha(0.0)
                fig.patch.set_alpha(0.0)
                ax1.legend(fancybox=True, framealpha=0.2)
                fig_array.append(fig)
        return fig_array

    def normalize(self, attribute):
        """
        Displays an image representing the character following processing.

        :param attribute: Label to be normalized.
        """
        min_value = self.x[attribute].min()
        max_value = self.x[attribute].max()
        print(f"Normalizing with min value of {min_value} and max value of {max_value}")
        self.x[str(attribute + "_norm")] = self.x[attribute].apply(lambda x: (x - min_value) / max_value)

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
