#!/usr/bin/python2

#########################################################################################
### Data Exploration (related lesson: "Datasets and Question")
### Student response addresses the most important characteristics of the dataset and uses
### these characteristics to inform their analysis.
### Important characteristics include:
###     total number of data points
###     allocation across classes (POI/non-POI)
###     number of features used
###     are there any features with many missing values?
#########################################################################################

import joblib

enron_data = joblib.load(open("../final_project/final_project_dataset.pkl", "r"))

print "\nDATA EXPLORATION:"
print "total number of data points:", len(enron_data)
print "allocation accross classes:"
print "\tPOI:", sum([1 for v in enron_data.values() if v['poi']])
print "\tnon-POI:", sum(1 for v in enron_data.values() if not v['poi'])
print "number of features used:", max([len(v.keys()) for v in enron_data.values()])

features = enron_data.values()[0].keys()

print "are there features with many missing values?"
for feature in features:
    nan_count = sum([1 for v in enron_data.values() if v[feature] == "NaN"])
    poi_nan_count = sum([1 for v in enron_data.values() 
                            if v["poi"] and v[feature] == "NaN"])
    if nan_count >= 0.5 * len(enron_data):  # has NaN for half or more values
        print "\t%s: %d, %d POIs" % (feature, nan_count, poi_nan_count)

#   features with high NaN counts:
#       deferral_payments, restricted_stock_deferred, load_advances, director_fees,
#       deferred_income, long_term_incentive
#   features with high NaN counts for POIs:
#       deferral_payments, restricted_stock_deferred, loan_advances, director fees


#########################################################################################
### Outlier Investigation (related lesson: "Outliers")
### Student response identifies outlier(s) in the financial data, and explains how they
### are removed or otherwise handled.
#########################################################################################

from matplotlib import pyplot as plt
import seaborn as sns

print "\nOUTLIER INVESTIGATION:"

# salary
salary = [v["salary"] for v in enron_data.values() if v["salary"] != "NaN"]
#plt.hist(salary)
#plt.show()
print "salary outliers:"
for k, v in enron_data.items():
    if v["salary"] != "NaN" and v["salary"] >= 1000000:
        print "\t%s: %d" % (k, v["salary"])

# total_payments:
total_payments = [v["total_payments"] for v in enron_data.values()
                    if v["total_payments"] != "NaN"]
#plt.hist(total_payments)
#plt.show()
print "total_payments outliers:"
for k, v in enron_data.items():
    if v["total_payments"] != "NaN" and v["total_payments"] > 0.8e8:
        print "\t%s: %d" % (k, v["total_payments"])

# exercised_stock_options:
exercised_stock_options = [v["exercised_stock_options"] for v in enron_data.values()
                            if v["exercised_stock_options"] != "NaN"]
#plt.hist(exercised_stock_options)
#plt.show()
print "exercised_stock_options outliers:"
for k, v in enron_data.items():
    if v["exercised_stock_options"] != "NaN" and v["exercised_stock_options"] > 2.5e7:
        print "\t%s: %d" % (k, v["exercised_stock_options"])

# bonus:
bonus = [v["bonus"] for v in enron_data.values() if v["bonus"] != "NaN"]
#plt.hist(bonus)
#plt.show()
print "bonus outliers:"
for k, v in enron_data.items():
    if v["bonus"] != "NaN" and v["bonus"] > 6000000:
        print "\t%s: %d" % (k, v["bonus"])

# restricted_stock:
restricted_stock = [v["restricted_stock"] for v in enron_data.values() 
                        if v["restricted_stock"] != "NaN"]
#plt.hist(restricted_stock)
#plt.show()
print "restricted_stock outliers:"
for k, v in enron_data.items():
    if v["restricted_stock"] != "NaN" and v["restricted_stock"] > 1.25e7:
        print "\t%s: %d" % (k, v["restricted_stock"])

# total_stock_value:
total_stock_value = [v["total_stock_value"] for v in enron_data.values() 
                        if v["total_stock_value"] != "NaN"]
#plt.hist(total_stock_value)
#plt.show()
print "total_stock_value outliers:"
for k, v in enron_data.items():
    if v["total_stock_value"] != "NaN" and v["total_stock_value"] > 4e7:
        print "\t%s: %d" % (k, v["total_stock_value"])

# expenses:
expenses = [v["expenses"] for v in enron_data.values() if v["expenses"] != "NaN"]
#plt.hist(expenses)
#plt.show()
print "expenses outliers:"
for k, v in enron_data.items():
    if v["expenses"] != "NaN" and v["expenses"] > 200000:
        print "\t%s: %d" % (k, v["expenses"])

# other:
other = [v["other"] for v in enron_data.values() if v["other"] != "NaN"]
#plt.hist(other)
#plt.show()
print "other outliers:"
for k, v in enron_data.items():
    if v["other"] != "NaN" and v["other"] > 0.6e7:
        print "\t%s: %d" % (k, v["other"])

# remove 'TOTAL', which is the sum of all values; all others were real employees of Enron
enron_data.pop('TOTAL')


#########################################################################################
### Create new features (related lesson: "Feature Selection")
### At least one new feature is implemented. Justification for that feature is provided
### in the written response. The effect of that feature on the final algorithm
### performance is tested or its strength is compared to other features in feature
### selection. THe student is not required to include their new feature in their final
### feature set.
#########################################################################################

# adding proportion_from_poi
for k, v in enron_data.items():
    if v['to_messages'] != 'NaN' and v['from_poi_to_this_person'] != 'NaN':
        enron_data[k]['proportion_from_poi'] = \
            float(v['from_poi_to_this_person']) / float(v['to_messages'])
    else:
        enron_data[k]['proportion_from_poi'] = 'NaN'

# adding proportion_to_poi
for k, v in enron_data.items():
    if v['from_messages'] != 'NaN' and v['from_this_person_to_poi'] != 'NaN':
        enron_data[k]['proportion_to_poi'] = \
            float(v['from_this_person_to_poi']) / float(v['from_messages'])
    else:
        enron_data[k]['proportion_to_poi'] = 'NaN'

# adding proportion_shared_with_poi
for k, v in enron_data.items():
    if v['to_messages'] != 'NaN' and v['shared_receipt_with_poi']:
        enron_data[k]['proportion_shared_with_poi'] = \
            float(v['shared_receipt_with_poi']) / float(v['to_messages'])
    else:
        enron_data[k]['proportion_shared_with_poi'] = 'NaN'
