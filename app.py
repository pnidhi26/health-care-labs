from flask import Flask, render_template, request, redirect
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import time
import warnings
import os

app = Flask(__name__)

main_list = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean',
 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean', 'radius_se', 
 'perimeter_se', 'texture_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se', 
 'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst', 
 'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst']

def getContext(list_of_parameters, test_case = None):

	data = pd.read_csv('data.csv', index_col=False)
	#print(data)
	data['diagnosis'] = data['diagnosis'].apply(lambda x: '1' if x == 'M' else '0')
	data = data.set_index('id')
	#del data['Unnamed: 32']
	data = data.drop(list_of_parameters, axis=1)
	#print(data)

	my_path = os.path.dirname(__file__) 
	#print(my_path)
	my_file = 'static/plot1.png'
	i_s =0
	data.plot(kind='density', subplots=True, layout=(5,7), sharex=False, legend=False,fontsize=1)
	if os.path.isfile(os.path.join(my_path, my_file)):
		print("Yes")
		os.remove(os.path.join(my_path, my_file))
		i_s = 1
		my_file = 'static/plot13.png'
	plt.savefig(os.path.join(my_path, my_file))


	from matplotlib import cm as cm
	fig = plt.figure()
	ax1 = fig.add_subplot(111)
	cmap = cm.get_cmap('jet', 30)
	cax = ax1.imshow(data.corr(), interpolation="none", cmap=cmap)
	ax1.grid(True)
	plt.title('Breast Cancer Attributes Correlation')
	# Add colorbar, make sure to specify tick locations to match desired ticklabels
	fig.colorbar(cax, ticks=[.75,.8,.85,.90,.95,1])
	my_file = 'static/plot2.png'
	if os.path.isfile(os.path.join(my_path, my_file)):
		print("Yes")
		os.remove(os.path.join(my_path, my_file))
		my_file = 'static/plot24.png'
	plt.savefig(os.path.join(my_path, my_file))

	Y = data['diagnosis'].values
	X = data.drop(['diagnosis'], axis=1).values
	#print(X)
	X_train, X_test, Y_train, Y_test = train_test_split (X, Y, test_size = 0.20, random_state=11)



	# prepare the model
	with warnings.catch_warnings(): warnings.simplefilter("ignore")
	scaler = StandardScaler().fit(X_train)
	X_train_scaled = scaler.transform(X_train)
	model = SVC(C=2.0, kernel='rbf')
	start = time.time()
	model.fit(X_train_scaled, Y_train)
	end = time.time()
	#print( "Run Time: %f" % (end-start))

	# estimate accuracy on test dataset
	with warnings.catch_warnings(): warnings.simplefilter("ignore")
	
	X_test_scaled = scaler.transform(X_test)
	predictions = model.predict(X_test_scaled)
	if test_case is not None:
		test_case_scaled = scaler.transform(test_case)
		p = model.predict(test_case_scaled)
		#print("+++++++++++"+p)
		if p == '1':
			#print("Malignant")
			p = "Malignant"
		else:
			#print("Benign")
			p = "Benign"
	else:
		p=0

	#print("Accuracy score %f" % accuracy_score(Y_test, predictions))
	accu = round(accuracy_score(Y_test, predictions),4)*100
	x= classification_report(Y_test, predictions, output_dict=True)
	precision_1 = x['0']['precision']
	recall_1 = x['0']['recall']
	f1_1 = round(x['0']['f1-score'],2)
	sc_1 = x['0']['support']

	precision_2 = x['1']['precision']
	recall_2 = x['1']['recall']
	f1_2 = round(x['1']['f1-score'],2)
	sc_2 = x['1']['support']


	y= confusion_matrix(Y_test, predictions)
	v_11 = y[0][0]
	v_10 = y[0][1]
	v_01 = y[1][0]
	v_00 = y[1][1]

	return {
	'precision_1':round(precision_1,3),
	'precision_2':round(precision_2,3),
	'recall_1':round(recall_1,3),
	'recall_2':round(recall_2,3),
	'f1_1':f1_1,
	'f1_2':f1_2,
	'sc_1':sc_1,
	'sc_2':sc_2,
	'accuracy':accu,
	'v_11':v_11,
	'v_10':v_10,
	'v_01':v_01,
	'v_00':v_00,
	'i_s':i_s,
	'p':p,
	}

@app.route('/index.html')
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/team.html')
def team():
    return render_template('team.html')

@app.route('/pulse.html')
def pulse():
    return render_template('pulse.html')

@app.route('/contact.html')
def contact():
    return render_template('contact.html')

@app.route('/files.html', methods=['POST'])
def files():
	z = request.form.getlist('options')
	y=[]
	for i in z:
		if request.form.get(i):
			y.append(float(request.form.get(i)))
	print(y)

	x = list(set(main_list) - set(z))
	#x.append('diagnosis')
	#print(x)
	print("length of z: ",len(z))
	print("length of y: ",len(y))
	if len(y) != 0:
		if len(y) == len(z):
			print("++++++++++1")
			context = getContext(x,[y])
		else:
			print("++++++++++2")
			return redirect('pulse.html')
	else:
		print("++++++++++3")
		context = getContext(x)
	return render_template('files.html', context=context)

@app.after_request
def add_header(response):
    # response.cache_control.no_store = True
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response

if __name__ == "__main__":
	app.static_folder = 'static'
	app.config["CACHE_TYPE"] = "null"
	app.run()


