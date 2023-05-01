from dataset import AIDataset
import base64
from flask import Flask, flash, render_template, request
from io import BytesIO

global train_set

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def home_page():
    """
    Using Flask and Jinja2, main page is loaded with option to load a page.
    :param methods: Uses POST method to pass the file being loaded and the classfier attribute from a form
    """
    global train_set

    if request.method == 'POST':  # Checks if the form has been filled out or not.
        try:
            # Creates the dataset then loads the selected file from the Data folder.
            train_set = AIDataset()
            train_set.load(os_loc=f"Data/{request.form.get('file')}", y_label=request.form.get("target_attribute"))
        except:
            train_set = None

    return render_template("index.html", dataset=train_set)

    # else:  # If it is successful then moves on to the quality analysis step.
    #     return render_template("quality.html", dataset=train_set)


@app.route("/quality")
def get_quality():
    """
    Using Flask and Jinja2, loads  page to conduct step 2 which is quality analysis of the data.
    :param: none
    """

    global train_set
    data = []  # a buffer array to hold all the figures that will be loaded on the page.
    fig_array = train_set.show_histogram()
    buf = BytesIO()  # Save image to a temporary buffer.
    for fig in fig_array:
        buf = BytesIO()
        fig.savefig(buf, format="png")    # Save image to a temporary buffer.
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
        return render_template("correlation.html", data=data, corr_matrix=corr_matrix, dataset=train_set,
                               selection=True)

    else:
        return render_template("correlation.html", data=data, corr_matrix=corr_matrix, dataset=train_set,
                               selection=False)


@app.route("/view_data", methods=["GET", "POST"])
def get_view_data():
    global train_set

    if request.method == 'POST':
        train_set.normalize(request.form.get("selector"))

    print(train_set.x.iloc[0])
    return render_template("view_data.html", dataset=train_set)


@app.route("/save", methods=["GET", "POST"])
def get_save():
    global train_set
    if request.method == 'POST':
        train_set.save(f"Data\{request.form.get('file')}")
    return render_template("save.html", dataset=train_set)


if __name__ == "__main__":  # is the same thing as calling the run function
    global train_set
    train_set = None
    app.run(host="0.0.0.0")
