from flask import Flask, render_template, request, url_for
import sklearn, os
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from random import randint
import random

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
	elif dataset == 'Mobile':
		df = pd.read_csv('./Data/train.csv')
		test = pd.read_csv('./Data/test.csv')
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

def makePrediction(dataset, model, datapoint, random):
	X_train,X_test,Y_train,Y_test = returnDataset(dataset)
	algo = returnModel(model)
	algo.fit(X_train, Y_train)
	if random == True:
		return int(algo.predict([X_test.iloc[datapoint,:]]))
	return int(algo.predict([pd.Series(datapoint)]))

def returnSHAP(dataset, model, datapoint, random):
	X_train,X_test,Y_train,Y_test = returnDataset(dataset)
	algo = returnModel(model)
	algo.fit(X_train, Y_train)
	explainer = shap.KernelExplainer(algo.predict_proba, X_train)
	# print("datapoint----",str(datapoint),str(type(datapoint)))
	if random == True:
		# print("Random Datapoint is: ")
		# print (X_test.iloc[datapoint,:])
		shap_values = explainer.shap_values(X_test.iloc[datapoint,:])
	else:
		shap_values = explainer.shap_values(pd.Series(datapoint, index = X_test.columns))
	return shap_values

def returnColNames(dataset):
	X_train,X_test,Y_train,Y_test = returnDataset(dataset)
	return X_train.columns

def returnRandomDatapoint(dataset):
	if dataset == "IRIS":
		points = [10,8,24,7,20,13,11,14,16,21,27,1,0,17,12,9,28,3,4,6,15,18,22,23,25,26,28,29,2,19,5]
		return random.choice(points)
	else:
		return randint(0, 399)

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
	# print ("shap values----\n",str(shap_values))
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

def generateCounterfactual(dataset, model, noofneighbours, datapoint, shapvals, desiredcategory, random):
	X_train,X_test,Y_train,Y_test = returnDataset(dataset)
	algo = returnModel(model)
	algo.fit(X_train, Y_train)
	shapdict = dict(zip(returnColNames(dataset), shapvals[int(desiredcategory)]))
	MutateValues = {k: v for k, v in shapdict.items() if v < 0}
	neigh = NearestNeighbors(n_neighbors=noofneighbours)
	neigh.fit(X_train)
	if random == True:
		out = neigh.kneighbors([X_test.iloc[datapoint]])
		newDatapoint = X_test.iloc[datapoint]
		origdatapoint = newDatapoint
		df_original = X_test.iloc[[datapoint]]
	else:
		# print ("Datapoint: "+str(datapoint))
		out = neigh.kneighbors([pd.Series(datapoint, index = X_test.columns)])
		# print ("Neighbours: "+str(out))
		newDatapoint = pd.Series(datapoint, index = X_test.columns)
		origdatapoint = newDatapoint
		df_original = pd.DataFrame(newDatapoint)
	result = []
	for point in out[1][0]:
		newDatapoint = origdatapoint
		for key, value in MutateValues.items():
			newDatapoint[key] = X_train.iloc[point][key]
		if int(algo.predict([newDatapoint])) == int(desiredcategory):
			if newDatapoint.tolist() not in result:
				result.append(newDatapoint.tolist())
	df = pd.DataFrame(result, columns=X_test.columns)
	df = df.drop_duplicates() # New Datapoint
	df = df[~df.isin(df_original)]
	return df, df_original

def returnNearestNeighbours(dataset, model, datapoint, desiredcategory, nofneighbours):
	X_train,X_test,Y_train,Y_test = returnDataset(dataset)
	algo = returnModel(model)
	algo.fit(X_train, Y_train)
	neigh = NearestNeighbors(n_neighbors=nofneighbours)
	neigh.fit(X_train)
	out = neigh.kneighbors([pd.Series(datapoint, index = X_test.columns)])
	result = []
	for point in out[1][0]:
		if int(algo.predict([X_train.iloc[point]])) == int(desiredcategory):
			result.append(X_train.iloc[point])
	df = pd.DataFrame(result, columns=X_test.columns)
	return df

@app.route('/', methods=['GET', 'POST'])
def index():
	if request.method == 'POST' and 'dataset' in request.form:
		# try:
		# 	os.remove("./static/img/bar.png") # Remove the previous image
		# except:
		# 	pass
		global dataset
		dataset = request.form['dataset']
		return render_template('index.html', dataset=dataset, location = "#step-1")
	if request.method == 'POST' and 'model' in request.form:
		global model
		model = request.form['model']
		print ("Dataset is " + str(dataset) + " and Model is " +str(model))
		colstring = ', '.join(returnColNames(dataset))
		return render_template('index.html', dataset=dataset, model=model, location = "#step-2", col=colstring)
	if request.method == 'POST' and 'point' in request.form:
		global datapoint
		global category
		global shapvals
		global freq
		global colnames
		global israndom
		global pt
		global statement
		if request.form['point'] == "Random":
			print ("Got Random Value!")
			datapoint = returnRandomDatapoint(dataset)
			category = makePrediction(dataset, model, datapoint, True)
			shapvals = returnSHAP(dataset, model, datapoint, True)
			pt = "Random("+str(datapoint)+")"
			israndom = True
		else:
			print ("Got SpecificData")
			texta = request.form['SpecificData'].split(",")
			datapoint = [float(i) for i in texta]
			category = makePrediction(dataset, model, datapoint, False)
			shapvals = returnSHAP(dataset, model, datapoint, False)
			pt = str(datapoint)
			israndom = False
		freq = returnFrequencies(dataset)
		colnames = returnColNames(dataset)
		return render_template('index.html', dataset=dataset, model=model, location = "#step-3", 
			datapoint = pt, category = category, allclasses = list(freq.keys()) )
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
		df, original = generateCounterfactual(dataset, model, 50, datapoint, shapvals, int(desiredcategory), israndom)
		original.reset_index()
		original = original.to_html()
		if len(df) == 0:
			statement = "No mutated counterfactuals found, so displaying closest neighbours in class " + str(desiredcategory)
			nofneighbours = 0
			limit = 120 if datapoint == "IRIS" else 400
			df = []
			while len(df) == 0:
				nofneighbours = nofneighbours + int(limit/10)
				print ("nofneighbours: " + str(nofneighbours))
				if nofneighbours > limit:
					print ("If condition hit")
					break
				else:
					df = returnNearestNeighbours(dataset, model, datapoint, desiredcategory, nofneighbours)
		else:
			statement = "See table below"
		df = df.to_html()
		return render_template('index.html', dataset=dataset, model=model, location = "#step-4", 
			datapoint = pt, category = category, allclasses = list(freq.keys()), 
			contrastive = contrastive, yP=yP, ynotQ=ynotQ, desiredcategory=desiredcategory, df=df, original=original, statement = statement)
	return render_template('index.html', location = "#")

if __name__ == '__main__':
    app.run(debug=True)
