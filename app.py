from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

with open("C:/Users/xbhi0/Downloads/model1.pkl", 'rb') as f:
    classification_model = pickle.load(f)

with open("C:/Users/xbhi0/Downloads/model2.pkl", 'rb') as f:
    clustering_model = pickle.load(f)

with open("C:/Users/xbhi0/Downloads/classification_accuracy.pkl", 'rb') as f:
    accuracy_classification = pickle.load(f)

with open("C:/Users/xbhi0/Downloads/clustering_accuracy.pkl", 'rb') as f:
    accuracy_clustering = pickle.load(f)

# with open("C:/Users/xbhi0/Downloads/X_train.pkl", 'rb') as f:
#     X_train = pickle.load(f)

# with open("C:/Users/xbhi0/Downloads/y_train.pkl", 'rb') as f:
#     y_train = pickle.load(f)

X_train = pd.read_csv("C:/Users/xbhi0/Downloads/X_train.csv")
y_train = pd.read_csv("C:/Users/xbhi0/Downloads/y_train.csv")

@app.route('/', methods=['GET', 'POST'])
def home():
    # if request.method == 'POST':
    #     input1 = float(request.form.get('fLength'))
    #     input2 = float(request.form.get('fWidth'))
    #     input3 = float(request.form.get('fSize'))
    #     input4 = float(request.form.get('fConc'))
    #     input5 = float(request.form.get('fM3Trans'))
    #     input6 = float(request.form.get('fAlpha'))
    #     input7 = float(request.form.get('fDist'))

    #     classification_prediction = classification_model.predict([[input1, input2, input3, input4, input5, input6, input7]])
    #     clustering_prediction = clustering_model.predict([[input1, input2, input3, input4, input5, input6, input7]])

    #     return render_template('index.html', accuracy_classification=accuracy_classification,
    #                            accuracy_clustering=accuracy_clustering,
    #                            classification_prediction=classification_prediction, clustering_prediction=clustering_prediction)

    return render_template('index.html', accuracy_classification=accuracy_classification,
                           accuracy_clustering=accuracy_clustering)


@app.route('/predict', methods=['POST'])
def predict():

    if request.method == 'POST':
        input1 = float(request.form.get('fLength'))
        input2 = float(request.form.get('fWidth'))
        input3 = float(request.form.get('fSize'))
        input4 = float(request.form.get('fConc'))
        input5 = float(request.form.get('fM3Trans'))
        input6 = float(request.form.get('fAlpha'))
        input7 = float(request.form.get('fDist'))

        classification_model.fit(X_train,y_train)
        classification_prediction = classification_model.predict([[input1, input2, input3, input4, input5, input6, input7]])
        clustering_prediction = clustering_model.predict([[input1, input2, input3, input4, input5, input6, input7]])

        return render_template('predict.html',classification_prediction=classification_prediction, clustering_prediction=clustering_prediction)


@app.route('/report', methods=['GET'])
def report():
    return render_template('report.html')


if __name__ == '__main__':
    app.run(debug=True)