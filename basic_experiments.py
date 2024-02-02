
'''
Descript:	Script that tries basic classification algorithms
			for topic recognition of news headlines in Albanian.
Author:		Erion Çano
Language: 	Python 3.11
'''

import numpy as np
import pandas as pd
import os, sys, argparse, json, re, random
from nltk import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.tree import DecisionTreeClassifier

# function that tokenizes text
def core_tokenize(text):
	''' 
	Takes a text string and returns tokenized string using NLTK word_tokenize. Space, \n \t are lost. "" are replace by ``''
	'''
	# tokenize | _ ^ / ~ + = * that are not tokenized by word_tokenize
	text = text.replace("|", " | ") ; text = text.replace("_", " _ ")
	text = text.replace("^", " ^ ") ; text = text.replace("/", " / ")
	text = text.replace("+", " + ") ; text = text.replace("=", " = ")
	text = text.replace("~", " ~ ") ; text = text.replace("*", " * ") 
	tokens = word_tokenize(text)

	# put all tokens together
	text = ' '.join(tokens)
	# remove double+ spaces
	text = re.sub(r'\s{2,}', " ", text)
	# lowercase
	text = text.lower()
	return text

# for reproducibility of results
sd = 7 ; np.random.seed(sd) ; random.seed(sd)
os.environ['PYTHONHASHSEED'] = str(sd)

# for the fancy command line execution
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--classifier', choices=['lr', 'dt', 'svm', 'rf', 'gb', 'xgb'], help='Classification Model', required=True)
args = parser.parse_args()

train_file = "./data/train.csv"
test_file = "./data/test.csv"

## header=0 skips first row which is header
df_train = pd.read_csv(train_file, header=0, encoding='utf-8')
df_test = pd.read_csv(test_file, header=0, encoding='utf-8')

## encode the topic categories
label_encoder = LabelEncoder()
df_train['Topic'] = label_encoder.fit_transform(df_train['Topic'])
df_test['Topic'] = label_encoder.fit_transform(df_test['Topic'])

## train/test split
X_train = df_train["Text"].values.tolist()
X_test = df_test["Text"].values.tolist()
y_train = df_train["Topic"].values.tolist()
y_test = df_test["Topic"].values.tolist()

## lowercase and tokenize the text part of each sample
X_train = [core_tokenize(samp) for samp in X_train]
X_test = [core_tokenize(samp) for samp in X_test]

if __name__ == '__main__': 

	# TF-IDF vectorizer should work pretty well
	vect = TfidfVectorizer(lowercase=False)

	# trying some basic classifiers
	lr_model = LogisticRegression()
	dt_model = DecisionTreeClassifier()
	svm_model = SVC()

	# random forest that is a bagging ensemble learner
	rf_model = RandomForestClassifier(random_state=sd, n_jobs=-1)

	# and two more boosting learners
	gb_model = GradientBoostingClassifier(random_state=sd)
	xgb_model = xgb.XGBClassifier(random_state=sd)

	# selecting the classifier based on the command line option
	if args.classifier.lower() == "lr":
		model = lr_model
	elif args.classifier.lower() == "dt":
		model = dt_model
	elif args.classifier.lower() == "svm":
		model = svm_model	
	elif args.classifier.lower() == "rf":
		model = rf_model
	elif args.classifier.lower() == "gb":
		model = gb_model
	elif args.classifier.lower() == "xgb":
		model = xgb_model
	else:
		print("Wrong classifier...")
		sys.exit()

	# the vectorizer and classifier in a pipeline
	pipe_model = Pipeline([('vect', vect), ('clf', model)])

	# fit the model and get the accuracy score
	pipe_model.fit(X_train, y_train)
	score = pipe_model.score(X_test, y_test)
	# print the accuracy score
	print(f"Model accuracy: {score:.4f}")
