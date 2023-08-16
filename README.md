# Enron Submission Free-Response Questions

1. Summarize for us the goal of this project and how machine learning is useful in
   trying to accomplish it. As part of your answer, give some background on the dataset 
   and how it can be used to answer the project question. Were there any outliers in the 
   data when you got it, and how did you handle those?
	
	## Data Exploration
	This project aims to produce a data model which has been tuned to predict whether 
	a person within the enron dataset is a person of interest. 
	Machine learning is the perfect tool for developing this model becuase it can use 
	the underlying data to learn, test, and validate the patterns it finds.
	The dataset being used for modeling is a combination of financial and email data 
	from the Enron scandal(s). The features included in this datset should lend
	themselves nicely to a model which captures the financial and communication trends
	within Enron at the time of the scandals, and ultimately predict who was at fault.
	The Enron dataset consists of features for 146 different people. Of them, 18 were
	identified as persons of interest or POIs. The remaining 128 were not identified
	as POIs. There are a total of 20 features present about each of the people.
	Unfortunately, there were several features which could not be used for analysis
	because of the number of missing values:
	
	| Feature Name | NaN Count | POI NaN Count |
	| --- | --- | --- |
	| deferral_payments | 107 | 13 |
	| restricted_stock_deferred | 128 | 18 |
	| loan_advances | 142 | 17 |
	| director_fees | 129 | 18 |
	| deferred_income | 97 | 7 |
	| long_term_incentive | 80 | 6 |
	
	All six of these features were excluded from further analysis because either they
	did not contain enough data in general, or because they were missing data for over
	half of the POIs. It is possible that the missing data does indicate a trend.
	However it is difficult to know that and should not be assumed. The `email_address`
	was also removed from the feature list. This leaves a total of 13 remaining features.
	
	## Outlier Investigation	
	In addition to investigating missing values, outliers were investigated for each of
	the remaining 13 features. The majority of the identified outliers actually
	appeared to be natural trends in the data. For instance, there were four outliers
	found for th `total_salary` feature. Three of them were top level executives within
	Enron, making it sensible that they would make considerably more than the general
	Enron population. However, there was one outlier that stood out across all of the 
	features and that was TOTAL. After reviewing the PDF from which the financial data
	was sourced, it was clear that the summary line was being included as a person. 
	With this "person" removed from the dataset, there were 145 remaining.
	
---

2. What features did you end up using in your POI identifier, and what selection process
   did you use to pick them? Did you have to do any scaling? Why or why not? As part of
   the assignment, you should attempt to engineer your own feature that does not come
   ready-made in the dataset -- explain what feature you tried to make, and the
   rationale behind it. (You do not necessarily have to use it in the final analysis,
   only engineer and test it.) In your feature selection step, if you used an algorithm
   like a decision tree, please also give the feature importances of the features that
   you use, and if you used an automated feature selection function like SelectKBest,
   please report the feature scores and reasons for your choice of parameter values.
	
	## Create New Features
	Three new features were engineered: `proportion_from_poi`, `proportion_to_poi`, and
	`proportion_sared_with_poi`; using the following calculations:
	
	```python
	proportion_from_poi = from_poi_to_this_person / to_messages
	proportion_to_poi = from_this_person_to_poi / from_messages
	proportion_shared_with_poi = shared_recipient_with_poi / to_messages
	```
	
	These features are, in essence, scaled to the number of messages each person
	sends and receives since arbitrary email counts can't reliably be compared between
	individuals with differing email volumes. Because there is naturally a high
	correlation between the newly engineered features and the features used to 
	calculate them, the features used for calcualtion were omitted from the feature
	selection process. This leaves 11 features available for selection.
	
	## Intelligently Select Features
	To identify the features for selection, sklearn's `SelectKBest` function was used.
	The following lists the results of `SelectKBest` with `k='all'` to get a general
	understanding of the available features.
	
	| Feature Name | Feature Score |
	| --- | --- |
	| bonus | 30.652 |
	| salary | 15.806 |
	| proportion_to_poi | 13.791 |
	| total_stock_value | 10.815 |
	| proportion_shared_with_poi | 10.008 |
	| exercised_stock_options | 9.956 |
	| total_payments | 8.963 |
	| restricted_stock | 8.051 |
	| expenses | 4.314 |
	| other | 3.197 |
	| proportion_from_poi | 0.491 |
	
	The features were iteratively tested, using two different algorithms (discussed in
	a subsequent section), beginning with the feature of most importance, adding in the
	feature of next most importance with each iteration. The combination with the best
	metrics were selected. The specific metrics used are discussed later.
	
	The final features used were `bonus`, `salary`, and `proportion_to_poi`. After several
	iterations of algorithm tuning and feature selection, this combination of features
	offered the highest metric values while using the smallest number of features.
	
	## Properly Scale Features
	Prior to using sklearn's `selectKBest` function to obtain feature scores, I also
	utilized their `MinMaxScaler` function to scale each feature. This was to prevent
	features like `bonus` which can range up to the millions from overshadowing features
	like `proportion_to_poi` which is a proportion ranging from 0 to 1.
	
---

3. What algorithm did you end up using? What other one(s) did you try? How did model
   performance differ between algorithms?  
	
	## Pick an Algorithm
	Two different algorithms were evaluated for this model: `DecisionTreeClassifier`, and
	`SVC`. During the feature selection stage, both algorithms were compared without any 
	parameter tuning. Overall, the `SVC` algorithm had a higher accuracy score. However,
	the recall score for the `SVC` algorithm was terrible, reaching a maximum of '0.06'.
	The `DecisionTreeClassifier` algorithm, however, offered similar accuracy
	scores while maintaining more balanced precision and recall scores. For this reason,
	the `DecisionTreeClassifier` algorithm was selected for further tuning.
	
---

4. What does it mean to tune the parameters of an algorithm, and what can happen if you
   don’t do this well?  How did you tune the parameters of your particular algorithm?
   What parameters did you tune? (Some algorithms do not have parameters that you need
   to tune -- if this is the case for the one you picked, identify and briefly explain
   how you would have done it for the model that was not your final choice or a
   different model that does utilize parameter tuning, e.g. a decision tree classifier).
	
	## Discuss Parameter Tuning
	Parameter tuning is the act of systematically changing an algorithm's parameter
	values to achieve better metric outcomes. It allows you to alter the algorithm's
	behaviour so that it is able to better generalize to data it has not yet seen. If
	parameter tuning is not performed well, it can lead to an overfit model where the
	training scores are very high and the model fits well to the training set, but does
	not generalize well to the testing set. An overfit model is not able to predict events
	it has not explicitly seen before.
	
	## Tune the Algorithm
	While tuning the `DecisionTreeClassifier` algorithm, the `class_weight`, `criterion`,
	`min_samples_split`, and `splitter` were all evalutated using sklearn's `GridSearchCV`.
	The following parameter values were tested:
	
	| Parameter | Values |
	| --- | --- |
	| class_weight | None, 'balanced' |
	| criterion | 'gini', 'entropy' |
	| min_samples_split | 2 - 18 |
	| splitter | 'best', 'random' |
	
	Additionally, the `random_state` parameter was used to control the variation of output across
	runs. Using the `GridSearchCV`, `best_estimator_`, the best parameters were selected. The final
	algorithm after tuning was:
		
	```python
	DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
				max_features=None, max_leaf_nodes=None,
				min_impurity_decrease=0.0, min_impurity_split=None,
				min_samples_leaf=1, min_samples_split=3,
				min_weight_fraction_leaf=0.0, presort=False, random_state=42,
				splitter='best')
	```
	
	The only parameter which actually changed from the default was `min_samples_split`, changing from
	2 to 3. This increased the number of samples which are required in a single node to split into child
	nodes, meaning a single node of the tree must contain at least 3 samples to split into child nodes for
	POIs and non-POIs. By tuning this particular parameter in this way, the model is slightly less specific
	to the training data and more able to generalize to new items like the testing data.
	
---

5. What is validation, and what’s a classic mistake you can make if you do it wrong?
   How did you validate your analysis?
	
	## Discuss Validation
	Validation is achieved by splitting the data into testing and training sets. The training set is used
	to select features as well as select and tune the algorithm. The testing data is ONLY used to evaluate
	the performance of the algorithm. When done correctly, validation results in robust and reliable algorithms.
	However, there are a number of ways in which validation can go wrong. One of the most common ways is when
	the training and testing datasets get confused. If the testing dataset is used for ANYTHING other than
	evaluating the performance of the algorithm, the process becomes tainted and the results can no longer be
	trusted. This does not allow the algorithm to generalize beyond the testing/training sets. Another way
	validation can go wrong is due to the splitting of the training and testing samples. This issue is caused
	when the features are not proportionally split between the two datasets. For instance, All of the POIs
	could be present in the testing set, with none in the training set. If this were the case, the training
	would always perform terribly, but would test relatively well. Results such as these can be misleading.
	Cross Validation (discussed in the next section) is an excellent way of mitigating this issue.
	
	## Validation Strategy
	Because the number of POIs in this dataset is so low compared to the total sample size, Stratified Cross
	Validation was utilized. Cross validation is a method of training and testing algorithms on multiple
	versions of the data. This is done by taking repeated samples, splitting the data in different ways each 
	time, and testing the algorithms performance. The idea is that the more samples you are able to obtain,
	the more accurate the overall result will be becuase the errors will, in theory, not occur in every sample.
	With particularly small datasets such as this one, additional methods can be applied, such as
	`StratifiedShuffleSplit`. `StratifiedShuffleSplit` takes into account what the actual values of
	each feature are to achieve more balanced splits. For this analysis, `GridSearchCV` was used with 10-fold
	cross validation. All algorithm tests were evaluated using `StratifiedShuffleSplit` with 1000-fold 
	cross validation to account for the low POI count.
	
---

6. Give at least 2 evaluation metrics and your average performance for each of them.
   Explain an interpretation of your metrics that says something human-understandable
   about your algorithm’s performance.
	
	## Usage of Evaluation Metrics
	A total of three evaluation metrics were used throughout this analysis: `accuracy_score`,
	`precision_score`, and `recall_score`, with the most emphasis being placed on precision and recall.
	The overall accuracy of the final algorithm was 0.768. This means that of all the of the predictions the
	algorithm made, 76.8% of them correctly predicted that the person was either a POI or not a POI. This
	percentage was not the highest it could be, but was compromized to obtain more balanced precision and 
	recall scores. The precision of this algorithm was 0.35565. This means that of the times the algorithm
	predicted the person was a POI, it was correct 35.6% of the time. Additionally, the recall was 0.34,
	meaning that of the POIs for which predictions were made, 34% of them were correctly identifed.
	
	Now, these metrics may not seem all that great. And I would argue that they are not nearly enough
	to indict a new POI. However, it is clear that this algorithm is capturing the general patterns
	present in the data, and could be used to identify people who require a more thorough investigation
	to clear their name.
