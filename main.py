from flask import Flask, render_template, request, url_for
import sklearn
import numpy as np
import pandas as pd
import shap
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import shap

app = Flask(__name__)

dataset = None
model = None

def makePrediction(dataset, model, datapoint):
	if dataset == 'IRIS':
		X_train,X_test,Y_train,Y_test = train_test_split(*shap.datasets.iris(), test_size=0.2, random_state=0)
	elif dataset == 'Boston':
		X_train,X_test,Y_train,Y_test = train_test_split(*shap.datasets.iris(), test_size=0.2, random_state=0)
	elif dataset == 'Mobile':
		df = pd.read_csv('../Data/train.csv')
		test = pd.read_csv('../Data/test.csv')
		df.isnull().sum().max()
		y_t = np.array(df['price_range'])
		X_t = df
		X_t = df.drop(['price_range'], axis=1)
		X_t = np.array(X_t)
		from sklearn.preprocessing import MinMaxScaler
		scaler = MinMaxScaler()
		X_t = scaler.fit_transform(X_t)
		X_train,X_test,Y_train,Y_test = train_test_split(X_t,y_t,test_size=.20,random_state=42)

		X_train = pd.DataFrame(X_train)
		X_train.columns = df.columns[:-1]
		X_train.index = X_train.index + 1

		X_test = pd.DataFrame(X_test)
		X_test.columns = df.columns[:-1]
		X_test.index = X_test.index + 1

		Y_train = pd.DataFrame(Y_train)
		Y_train.columns = Y_train.columns + 1
		Y_train.index = Y_train.index + 1

		Y_test = pd.DataFrame(Y_test)
		Y_test.columns = Y_test.columns + 1
		Y_test.index = Y_test.index + 1
	if model == "KNN":
		m = sklearn.neighbors.KNeighborsClassifier()
	elif model == "SVM":
		m = sklearn.svm.SVC(kernel='linear', probability=True)
	elif model == "RF":
		from sklearn.ensemble import RandomForestClassifier
		m = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=2, random_state=0)
	elif model == "NN":
		from sklearn.neural_network import MLPClassifier
		m = MLPClassifier(solver='lbfgs', alpha=1e-1, hidden_layer_sizes=(5, 2), random_state=0)

	m.fit(X_train, Y_train)
	explainer = shap.KernelExplainer(m.predict_proba, X_train)
	shap_values = explainer.shap_values(X_test.iloc[0,:])

	unique, counts = np.unique(Y_test, return_counts=True)
	frequencies = dict(zip(unique, counts))
	return m.predict([X_test.iloc[0,:]]), shap_values, frequencies, X_train.columns


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
		global category
		global shapvals
		global freq
		global colnames
		category, shapvals, freq, colnames = makePrediction(dataset, model, datapoint)
		return render_template('index.html', dataset=dataset, model=model, location = "#step-3", 
			datapoint = "Random", category = category[0], allclasses = list(freq.keys()) )
	if request.method == 'POST' and 'desiredcategory' in request.form:
		desiredcategory = request.form['desiredcategory']
		contrastive = "Why " + str(category[0]) + " not " + str(desiredcategory)
		return render_template('index.html', dataset=dataset, model=model, location = "#step-4", 
			datapoint = "Random", category = category[0], allclasses = list(freq.keys()), contrastive = contrastive )
	return render_template('index.html', location = "#")

if __name__ == '__main__':
    app.run(debug=True)