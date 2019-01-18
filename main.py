from flask import Flask, render_template, request, url_for
import sklearn, os
import numpy as np
import pandas as pd
import shap
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import shap
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from random import randint

app = Flask(__name__)


def returnModel(model):
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
	return m

def returnDataset(dataset):
	if dataset == 'IRIS':
		X_train,X_test,Y_train,Y_test = train_test_split(*shap.datasets.iris(), test_size=0.2, random_state=0)
	elif dataset == 'Boston':
		X_train,X_test,Y_train,Y_test = train_test_split(*shap.datasets.boston(), test_size=0.2, random_state=0)
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

	return X_train,X_test,Y_train,Y_test

def makePrediction(dataset, model, datapoint):
	X_train,X_test,Y_train,Y_test = returnDataset(dataset)
	algo = returnModel(model)
	algo.fit(X_train, Y_train)
	return int(algo.predict([X_test.iloc[datapoint,:]]))

def returnSHAP(dataset, model, datapoint):
	X_train,X_test,Y_train,Y_test = returnDataset(dataset)
	algo = returnModel(model)
	algo.fit(X_train, Y_train)
	explainer = shap.KernelExplainer(algo.predict_proba, X_train)
	shap_values = explainer.shap_values(X_test.iloc[datapoint,:])
	return shap_values

def returnColNames(dataset):
	X_train,X_test,Y_train,Y_test = returnDataset(dataset)
	return X_train.columns

def returnRandomDatapoint(dataset):
	X_train,X_test,Y_train,Y_test = returnDataset(dataset)
	# return randint(0, len(X_test)-1)
	return 29

def mergeTerms(terms):
	ans = ""
	if len(terms) > 1: 
		for tup in terms[:-1]:
			ans = ans + tup[0] + ", "
		ans = ans + " and " + terms[-1][0] + "."
	else:
		ans = terms[0][0] + "."
	return ans

def returnFrequencies(dataset):
	X_train,X_test,Y_train,Y_test = returnDataset(dataset)
	unique, counts = np.unique(Y_test, return_counts=True)
	return dict(zip(unique, counts))

def whyPnotQ(shap_values, category, colnames, anti):
	shapdict = dict(zip(colnames, shap_values[int(category)]))
	if anti:
		Pos = {k: v for k, v in shapdict.items() if v < 0}
		classification = "Anti"
	else:
		Pos = {k: v for k, v in shapdict.items() if v > 0}
		classification = "Pro"
	P = sorted(Pos.items(), key=lambda kv: kv[1], reverse=anti)
	n = len(P)//3
	if n > 0:
		newP = [P[i * n:(i + 1) * n] for i in range((len(P) + n - 1) // n )]
		ans = "Algorithms " + classification + " classification was primarily influenced by " + mergeTerms(newP[0])
		ans = ans + " Factors which moderately affected the outcome were " + mergeTerms(newP[1])
		ans = ans + " Factors which trivially affected the outcome were " + mergeTerms(newP[2])
	else:
		ans = "Algorithms " + classification + " classification was primarily influenced by " + mergeTerms(P)
	return ans

def plot_bar_x(shapvals, label):
    # this is for plotting purpose
    index = np.arange(len(label))
    plt.bar(index, shapvals, width=0.8)
    plt.xlabel('Feature', fontsize=8)
    plt.ylabel('Impact', fontsize=8)
    plt.xticks(index, label, fontsize=8, rotation=90)
    plt.title('Feature Impact')
    plt.tight_layout()
    plt.savefig("./static/img/bar.png")

def generateCounterfactual(dataset, model, noofneighbours, datapoint, shapvals, desiredcategory):
	X_train,X_test,Y_train,Y_test = returnDataset(dataset)
	algo = returnModel(model)
	algo.fit(X_train, Y_train)
	shapdict = dict(zip(returnColNames(dataset), shapvals[int(desiredcategory)]))
	MutateValues = {k: v for k, v in shapdict.items() if v < 0}
	neigh = NearestNeighbors(n_neighbors=noofneighbours)
	neigh.fit(X_train)
	out = neigh.kneighbors([X_test.iloc[datapoint]])
	newDatapoint = X_test.iloc[datapoint]
	origdatapoint = X_test.iloc[datapoint]
	result = []
	for point in out[1][0]:
	    for key, value in MutateValues.items():
	        newDatapoint[key] = X_train.iloc[point][key]
	    if int(algo.predict([newDatapoint])) == int(desiredcategory):
	        result.append(newDatapoint)
	if len(result) > 0:
		df = pd.DataFrame()
		df = df.append(result, ignore_index=True) # Collected Counterfactual Points
		df = df.drop_duplicates() # New Datapoints

		dp = pd.DataFrame() 
		dp = df.append([origdatapoint], ignore_index=True) # Original Datapoint
		df = pd.concat([dp, df])
		df = df.drop_duplicates(keep = False)
	else:
		df = pd.DataFrame()
	return df

@app.route('/', methods=['GET', 'POST'])
def index():
	if request.method == 'POST' and 'dataset' in request.form:
		try:
			os.remove("./static/img/bar.png") # Remove the previous image
		except:
			pass
		global dataset
		dataset = request.form['dataset']
		return render_template('index.html', dataset=dataset, location = "#step-1")
	if request.method == 'POST' and 'model' in request.form:
		global model
		model = request.form['model']
		print ("Dataset is " + str(dataset) + " and Model is " +str(model))
		return render_template('index.html', dataset=dataset, model=model, location = "#step-2")
	if request.method == 'POST' and 'point' in request.form:
		global datapoint
		# datapoint = request.form['point']
		global category
		global shapvals
		global freq
		global colnames
		datapoint = returnRandomDatapoint(dataset)
		category = makePrediction(dataset, model, datapoint)
		freq = returnFrequencies(dataset)
		colnames = returnColNames(dataset)
		shapvals = returnSHAP(dataset, model, datapoint)

		return render_template('index.html', dataset=dataset, model=model, location = "#step-3", 
			datapoint = "Random("+str(datapoint)+")", category = category, allclasses = list(freq.keys()) )
	if request.method == 'POST' and 'desiredcategory' in request.form:
		# Explanations
		desiredcategory = request.form['desiredcategory']
		contrastive = "Why " + str(category) + " not " + str(desiredcategory)
		print (contrastive)
		yP = whyPnotQ(shapvals, category, colnames, False)
		ynotQ = whyPnotQ(shapvals, int(desiredcategory), colnames, True)

		# Add Plot
		plot_bar_x(shapvals[int(category)], colnames)

		# Add Counterfactual points
		df = generateCounterfactual(dataset, model, 50, datapoint, shapvals, int(desiredcategory))
		if len(df) == 0:
			df = "Nothing to show"
		else:
			df = df.to_html()
		return render_template('index.html', dataset=dataset, model=model, location = "#step-4", 
			datapoint = "Random("+str(datapoint)+")", category = category, allclasses = list(freq.keys()), 
			contrastive = contrastive, yP=yP, ynotQ=ynotQ, desiredcategory=desiredcategory, df=df)
	return render_template('index.html', location = "#")

if __name__ == '__main__':
    app.run(debug=True)
