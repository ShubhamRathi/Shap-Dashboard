import sklearn, sys
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import shap
import time
import csv
from sklearn.neighbors import NearestNeighbors

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
	if dataset == "IRIS":
		X_train,X_test,Y_train,Y_test = train_test_split(*shap.datasets.iris(), test_size=0.2, random_state=0)
	elif dataset == "Mobile":
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
		X_train, X_test, Y_train, Y_test = train_test_split(X_t,y_t,test_size=.20,random_state=42)

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
 
def getClasses(dataset):
	X_train,X_test,Y_train,Y_test = returnDataset(dataset)
	unique, counts = np.unique(Y_test, return_counts=True)
	freq = dict(zip(unique, counts))
	return list(freq.keys())

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
	origdatapoint = newDatapoint
	df_original = X_test.iloc[[datapoint]]
	print("Original Datapoint outside is: \n" + str(origdatapoint))
	result = []
	for point in out[1][0]:
		newDatapoint = origdatapoint
		for key, value in MutateValues.items():
			newDatapoint[key] = X_train.iloc[point][key]
		if int(algo.predict([newDatapoint])) == int(desiredcategory):
			if newDatapoint.tolist() not in result:
				result.append(newDatapoint.tolist())
	df = pd.DataFrame(result, columns=df_original.columns)
	df = df.drop_duplicates() # New Datapoint
	print(df)
	return result

def send_mail(subject):
	import smtplib
	server = smtplib.SMTP('smtp.gmail.com', 587)
	server.starttls()
	#Next, log in to the server
	server.login("rathishubham1103", "Duucatibike123%")
	FROM= "script@shap-dashboard.com"
	SUBJECT= subject
	TEXT="See subject"
	TO=["shubhamiiitbackup@gmail.com"]
	#Send the mail
	msg = """From: %s\nTo: %s\nSubject: %s\n\n%s
	""" % (FROM, ", ".join(TO), SUBJECT, TEXT)
	server.sendmail("rathishubham1103@gmail.com", "shubhamiiitbackup@gmail.com", msg)
	server.quit()

def generateRanges(number):
	return [int(0.25 * number), int(0.5 * number), int(0.75 * number)]

def main():
	algos = [sys.argv[1]]
	ds = "IRIS"
	cols = ["Datapoint No.", "P", "Q", "Total Counterfactual Points"]
	statistics = []
	for algo in algos:
		X_train,X_test,Y_train,Y_test = returnDataset(ds)
		ranges = generateRanges(len(X_test))
		start = float(sys.argv[2]) 
		end = float(sys.argv[3])
		segment = str(sys.argv[4])
		# for datapoint in range(int(start * len(X_test)), int(end * len(X_test))):
		for datapoint in [15]:
			print ("Processing datapoint #" +str(datapoint))
			# if datapoint in ranges:
			# 	send_mail(str((datapoint/len(X_test))*100)+"% of CF Report for" + str(algo) + " done")
			category = makePrediction(ds, algo, datapoint)
			shapvals = returnSHAP(ds, algo, datapoint)
			columns = returnColNames(ds)
			classes = getClasses(ds)
			classes.remove(category)
			for desiredcategory in classes:
				print("Why "+str(category)+" not " +str(desiredcategory))
				df = generateCounterfactual(ds, algo, 50, datapoint, shapvals, desiredcategory)
				stat = [datapoint, category, desiredcategory, len(df)]
				statistics.append(stat)
				# if len(df) > 0:
				# 	send_mail("["+str(algo)+"] Found " + str(len(df)) + "CF points for # " + str(datapoint))
		report = pd.DataFrame(statistics, columns = cols)
		report.to_csv("./Results/CF/"+str(ds)+"/"+algo+str(segment)+".csv")

main()
