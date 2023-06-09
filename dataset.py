import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.style as style
import numpy as np


class AIDataset:
    def __init__(self):
        self.data_file = pd.DataFrame()
        self.x = pd.DataFrame()
        self.y = pd.DataFrame()

    def load(self, os_loc=None, y_label=None):
        """
        Loads a data set for use in the machine learning program to test.
        :param os_loc: [Required] String representing the file to be loaded. If None, the function
        will request user provide a location via the command prompt. Default = None.
        :param y_label: [Optional] String representing the label for y which will be split. Default = None.
        if it is None then bypasses splitting out the y array and lets user do it manually.
        """
        try:
            self.data_file = pd.read_csv(os_loc)
        except:
            raise Exception("File did not load correctly.")

        try:
            self.y = self.data_file[y_label]
            self.x = self.data_file.drop([y_label], axis=1)
        except:
            raise Exception("Target attribute did not exist.")

    def show_histogram(self):
        """
        Plots multiple scatter plots showing relationship between attributes in attributes array. If mask is a
        value other than none, graph only shows correlating data for that value of Y.
        :return: An array of matplot figures with associated histograms for all variables.
        """
        style.use('dark_background')
        fig_array = []

        for attribute in self.x.columns:
            fig, (ax1) = plt.subplots(1, 1, figsize=(20, 9))
            true_array = self.x.where((self.y == 1))
            false_array = self.x.where((self.y == 0))
            x, bin_label, patch = ax1.hist([true_array[attribute], false_array[attribute]], bins=15, density=False,
                                           stacked=False, histtype='step', rwidth=0.7, align='mid',
                                           color=["royalblue", "tomato"],
                                           label=["True", "False"])

            ax1.set_xticks(bin_label)
            ax1.set_title(attribute, fontsize=20)
            ax1.patch.set_alpha(0.0)
            fig.patch.set_alpha(0.0)
            ax1.legend(fancybox=True, framealpha=0.1)
            fig_array.append(fig)

        return fig_array

    def show_correlation_matrix(self):
        """
        Plots a correlation matrix using y as the classifier.
        :return: A matplotlib figure representing the correlation matrix for the data.
        :return: A dataframe representing the correlation matrix for the data.
        """
        style.use('dark_background')
        new_val = self.x.copy()
        new_val["Classifier"] = self.y
        corr_matrix = new_val.corr()

        fig, (ax1) = plt.subplots(1, 1, figsize=(15, 9))
        color_axes = ax1.matshow(corr_matrix, cmap='Reds')
        fig.colorbar(color_axes)
        ax1.set_xticks(np.arange(len(corr_matrix)), corr_matrix.columns, fontsize=12, rotation=90)
        ax1.set_yticks(np.arange(len(corr_matrix)), corr_matrix.columns, fontsize=12)
        ax1.patch.set_alpha(0.0)
        fig.patch.set_alpha(0.0)
        return fig, corr_matrix

    def show_scatter_plot(self, attribute):
        """
        Plots a scatter plot relationship between three variables with x being attribute, y is attribute2, and
        y classifier is color of information.
        :param attribute: [Required] The first attribute to be plotted on the x-axis.
        :return: An array of matplot figures with associated scatterplots for all variables.
        """
        fig_array = []
        for attribute2 in self.x.columns:
            if attribute != attribute2:
                style.use('dark_background')

                fig, (ax1) = plt.subplots(1, 1, figsize=(20, 9))
                true_array = self.x.where((self.y == 1))
                false_array = self.x.where((self.y == 0))

                ax1.scatter(true_array[attribute], self.x[attribute2], color="royalblue", label="Classifier: True")
                ax1.scatter(false_array[attribute], self.x[attribute2], color="tomato", label="Classifier: False")
                ax1.set_title(attribute + " & " + attribute2, fontsize=20)
                ax1.set_xlabel(attribute, fontsize=12)
                ax1.set_ylabel(attribute2, fontsize=12)
                ax1.patch.set_alpha(0.0)
                fig.patch.set_alpha(0.0)
                ax1.legend(fontsize=15)
                ax1.legend(fancybox=True, framealpha=0.0)
                fig_array.append(fig)
        return fig_array

    def normalize(self, attribute):
        """
        Normalizes data using a max and min value.
        :param attribute: Label to be normalized then adds it as an additional column to the table.
        """
        min_value = self.x[attribute].min()
        max_value = self.x[attribute].max()
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

    def save(self, filename):
        """
        Saves data to a .csv file for later use. Target attribute is labeled in a Classifier column.
        :param filename: [Required] String representing the file to be saved.
        """
        new_val = self.x.copy()
        new_val["Classifier"] = self.y
        new_val.to_csv(filename)