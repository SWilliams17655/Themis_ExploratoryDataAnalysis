from dataset import AIDataset
import base64
from flask import Flask, flash, render_template, request
from io import BytesIO

from matplotlib.figure import Figure
# from sklearn.metrics import accuracy_score
import numpy as np
import pickle
# import matplotlib.pyplot as plt

from flask import Flask, flash, render_template, request

global train_set, attribute_selected

app = Flask(__name__)


@app.route("/")
def home_page():
    global train_set
    global attribute_selected

    attribute_selected = False

    train_set = AIDataset()
    train_set.load(os_loc="Data/train.csv", y_label='target')
    return render_template("index.html", dataset=train_set)


@app.route("/quality", methods=["GET", "POST"])
def get_quality():
    global train_set
    data = []
    fig_array = train_set.show_histogram()
    buf = BytesIO()  # Save image to a temporary buffer.
    for fig in fig_array:
        buf = BytesIO()  # Save image to a temporary buffer.
        fig.savefig(buf, format="png")
        data.append(base64.b64encode(buf.getbuffer()).decode("ascii"))

    return render_template("quality.html", data=data, dataset=train_set)


@app.route("/correlation", methods=["GET", "POST"])
def get_correlation():
    global train_set
    data = []

    fig_array, corr_matrix = train_set.show_correlation_matrix()
    print(corr_matrix["Classifier"])
    buf = BytesIO()  # Save image to a temporary buffer.
    fig_array.savefig(buf, format="png")
    data.append(base64.b64encode(buf.getbuffer()).decode("ascii"))

    if request.method == 'POST':
        print("In Post")
        fig_array = train_set.show_scatter_plot(request.form.get("selector"))
        print(corr_matrix["Classifier"])
        for fig in fig_array:
            buf = BytesIO()  # Save image to a temporary buffer.
            fig.savefig(buf, format="png")
            data.append(base64.b64encode(buf.getbuffer()).decode("ascii"))
        return render_template("correlation.html", data=data, corr_matrix=corr_matrix, dataset = train_set,
                               selection=True)

    else:
        print("Not In Post")
        return render_template("correlation.html", data=data, corr_matrix=corr_matrix, dataset=train_set,
                               selection=False)


@app.route("/view_data", methods=["GET", "POST"])
def get_view_data():
    return render_template("view_data.html", dataset=train_set)


@app.route("/save", methods=["GET", "POST"])
def get_save():
    return render_template("save.html", dataset=train_set)


if __name__ == "__main__":  # is the same thing as calling the run function
    app.run(host="0.0.0.0")
