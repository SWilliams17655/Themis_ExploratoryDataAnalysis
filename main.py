import ai_toolbox
from dataset import AIDataset
import numpy as np
from flask import Flask, flash, render_template, request

app = Flask(__name__)

train_set = AIDataset()
train_set.load(os_loc="Data/train.csv", y_label='target')

@app.route("/")
def home_page():
    return render_template("DataNannyHome.html")

if __name__ == "__main__":  # is the same thing as calling the run function
    app.run(host="0.0.0.0")
