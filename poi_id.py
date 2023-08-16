#!/usr/bin/python

import os, numpy as np, pickle, sys

from pprint import pprint

from sklearn.feature_selection import f_classif, SelectKBest
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

sys.path.append(os.path.abspath(("../tools/")))
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

features_list = ['poi', 'salary', 'total_payments', 'bonus',
                 'total_stock_value', 'expenses', 
                 'exercised_stock_options', 'other',
                 'restricted_stock', 'proportion_from_poi',
                 'proportion_to_poi', 'proportion_shared_with_poi']


### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


### Task 2: Remove outliers
data_dict.pop('TOTAL') # Total line from financial data


### Task 3: Create new feature(s)

# adding proportion_from_poi
for k, v in data_dict.items():
    if v['to_messages'] != 'NaN' and v['from_poi_to_this_person'] != 'NaN':
        data_dict[k]['proportion_from_poi'] = \
            float(v['from_poi_to_this_person']) / float(v['to_messages'])
    else:
        data_dict[k]['proportion_from_poi'] = 'NaN'

# adding proportion_to_poi
for k, v in data_dict.items():
    if v['from_messages'] != 'NaN' and v['from_this_person_to_poi'] != 'NaN':
        data_dict[k]['proportion_to_poi'] = \
            float(v['from_this_person_to_poi']) / float(v['from_messages'])
    else:
        data_dict[k]['proportion_to_poi'] = 'NaN'

# adding proportion_shared_with_poi
for k, v in data_dict.items():
    if v['to_messages'] != 'NaN' and v['shared_receipt_with_poi']:
        data_dict[k]['proportion_shared_with_poi'] = \
            float(v['shared_receipt_with_poi']) / float(v['to_messages'])
    else:
        data_dict[k]['proportion_shared_with_poi'] = 'NaN'


### Store to my_dataset for easy export below.
my_dataset = data_dict


### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# First classifier
clf1 = DecisionTreeClassifier()

# Second Classifier
clf2 = SVC(gamma='scale')

# Final Classifier
clf = DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
                              max_features=None, max_leaf_nodes=None,
                              min_impurity_decrease=0.0, min_impurity_split=None,
                              min_samples_leaf=1, min_samples_split=3,
                              min_weight_fraction_leaf=0.0, presort=False, random_state=42,
                              splitter='best')


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

# Feature scaling for selection algorithm
scaler = MinMaxScaler()
features_train_scaled = scaler.fit_transform(features_train, labels_train)

# Feature Selection
selector = SelectKBest(f_classif, k = 'all')
selector.fit(features_train_scaled, labels_train)
print '\nFeature Scores:'
scores = {}
for i, score in enumerate(selector.scores_):
    print '\t%s: %.3f' % (features_list[i+1], round(score, 3))
    scores[features_list[i+1]] = score
scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)

# Algorithm Testing
print '\nIterative feature testing:'
for i, val in enumerate(scores):
    selector = SelectKBest(f_classif, k = i + 1)
    selector.fit(features_train_scaled, labels_train)
    print 'Using best %d feature(s):' % selector.k
    feat_list = ['poi'] + [score[0] for score in scores[:selector.k]]
    print 'Features Used:', [score[0] for score in scores[:selector.k]]
    print 'DecisionTreeClassifier Results:'
    test_classifier(clf1, my_dataset, feat_list)
    print 'SVC Results:'
    test_classifier(clf2, my_dataset, feat_list)

# Classifier Tuning
print "Classifier Tuning:"
features_list = ['poi','bonus','salary','proportion_to_poi']
param_grid = {'class_weight': (None,'balanced'),
              'criterion': ('gini','entropy'),
              'min_samples_split': range(2,19),
              'random_state': [42,],
              'splitter': ('best','random')}
estimator = DecisionTreeClassifier()
clf1 = GridSearchCV(estimator, param_grid, cv=10, iid=True)
clf1.fit(features_train, labels_train)
test_classifier(clf1.best_estimator_, my_dataset, features_list)

# Final classifier using testing script
print "Final Classifier:"
test_classifier(clf, my_dataset, features_list) 


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)