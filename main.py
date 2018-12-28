from flask import Flask, render_template, request, url_for
import sklearn
import numpy
import pandas
import shap

app = Flask(__name__)

# @app.route('/', methods=['POST', 'GET'])
# def step2():	
# 	return render_template('index.html', dataset = dataset, model = model)

dataset = None
model = None

# @app.route('/step1', methods=['POST', 'GET'])
# def step1():
# 	if request.method == 'POST':
# 		dataset = request.form['dataset']
# 		return render_template('step2.html')
# 	return render_template('step1.html')

# @app.route('/step2', methods=['POST', 'GET'])
# def step2():
# 	if request.method == 'POST':
# 		model = request.form['model']
# 		message = "Dataset is " + dataset + " and model is " + model
# 		return render_template('step3.html', message = message)
# 	return render_template('step2.html')

# @app.route('/step3', methods=['GET'])
# def step3():
# 	message = "Dataset is " + dataset + " and model is " + model
# 	print (message)
# 	return render_template('step3.html', message = message)

def returnFeatures(dataset):
	if dataset == "boston":
		boston = load_boston()
		return boston.feature_names
	elif dataset == "IRIS":
		iris = load_iris()
		return iris.feature_names

@app.route('/', methods=['GET', 'POST'])
def index():
	global dataset
	global model
	if request.method == 'POST' and 'dataset' in request.form:
		dataset = request.form['dataset']
		return render_template('index.html', dataset=dataset, location = "#step-1")
	if request.method == 'POST' and 'model' in request.form:
		model = request.form['model']
		print ("Dataset is " + str(dataset) + " and Model is " +str(model))
		return render_template('index.html', dataset=dataset, model=model, location = "#step-2")
	if request.method == 'POST' and 'point' in request.form:
		datapoint = request.form['point']
		return render_template('index.html', dataset=dataset, model=model, location = "#step-3", datapoint = "Random")
	return render_template('index.html', location = "#")

if __name__ == '__main__':
    app.run(debug=True)