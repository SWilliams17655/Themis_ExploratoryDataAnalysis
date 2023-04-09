import ai_toolbox
from dataset import AIDataset
import numpy as np

train_set = AIDataset("Data/training.csv")
train_set.load(y_label='Label')
train_set.remove_zeros("SkinThickness")
train_set.remove_zeros("Insulin")
train_set.normalize("Pregnancies", 0, 17)
train_set.normalize("Glucose", 0, 200)
train_set.normalize("BloodPressure", 0, 122)
train_set.normalize("SkinThickness", 0, 99)
train_set.normalize("Insulin", 0, 846)
train_set.normalize("BMI", 0, 70)
train_set.normalize("Age", 0, 90)

eval_set = AIDataset("Data/eval.csv")
eval_set.load(y_label='Label')
eval_set.remove_zeros("SkinThickness")
eval_set.remove_zeros("Insulin")
eval_set.normalize("Pregnancies", 0, 17)
eval_set.normalize("Glucose", 0, 200)
eval_set.normalize("BloodPressure", 0, 122)
eval_set.normalize("SkinThickness", 0, 99)
eval_set.normalize("Insulin", 0, 846)
eval_set.normalize("BMI", 0, 70)
eval_set.normalize("Age", 0, 90)

train_set.show_table_in_html("HTML/Table.html")
train_set.show_data_description_in_html("HTML/Describe.html")

attributes=["Pregnancies_norm", "Glucose_norm",	"BloodPressure_norm", "SkinThickness_norm",
            "Insulin_norm",	"BMI_norm",	"Age_norm"]

#model1 = ai_toolbox.train_random_forest(train_set.x[attributes], train_set.y, eval_set.x[attributes], eval_set.y)
#model2 = ai_toolbox.train_sgd(train_set.x[attributes], train_set.y, eval_set.x[attributes], eval_set.y)
#model3 = ai_toolbox.train_svm(train_set.x[attributes], train_set.y, eval_set.x[attributes], eval_set.y)
model4 = ai_toolbox.train_neural_network(train_set.x[attributes], train_set.y, eval_set.x[attributes], eval_set.y)

# ai_toolbox.save_model(model1, "file.pkl")
# model5 = ai_toolbox.load_model("file.pkl")
pred=ai_toolbox.make_one_prediction(model4, eval_set.x.iloc[100][attributes])
print(pred)
print(eval_set.y.iloc[100])
