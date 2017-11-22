#!/usr/bin/python

import sys
import pickle
import matplotlib.pyplot as plt
sys.path.append("../tools/")
#%%
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data,test_classifier
#%%
### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list=['poi','bonus','salary','to_messages', 'deferral_payments', 
               'expenses',
                'deferred_income', 
               'long_term_incentive',
               'restricted_stock_deferred', 
               'shared_receipt_with_poi', 'loan_advances',
               'from_messages', 'other', 'director_fees', 
               'total_stock_value', 'from_poi_to_this_person',
               'from_this_person_to_poi', 'restricted_stock',  
               'total_payments','exercised_stock_options','email_address']             
### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
#%%
key=list(data_dict.keys())
count=0
poi_count=0
for i in range(len(key)):
    for j in features_list:
        if data_dict[key[i]][j]=='NaN':
            count+=1
    if data_dict[key[i]]['poi']==1:
        poi_count+=1
print "Number of missing data points in the giving dataset :",count   
print "Number of employees found gulilty :",poi_count,"(out of 146)"         
#%%
## required Functions

def computeFraction( poi_messages, all_messages ):
    if poi_messages=='NaN' or all_messages=='NaN':
        return 'NaN'
    else:
        return float(poi_messages)/all_messages
def data_points_of(feature):    
    data_points=[]
    names=list(my_dataset.keys())   
    for i in range(len(names)):
        if my_dataset[names[i]][feature] == 'NaN' :
            data_points.append(0)
        else:
            data_points.append(my_dataset[names[i]][feature])
    return data_points
#%%
### Task 2: Removing the outliers
data_dict.pop("TOTAL",0)
data_dict.pop("THE TRAVEL AGENCY IN THE PARK",0)
### Task 3: Create new feature(s)

#Engineering 4 new features: 
for keys,features in data_dict.items():
    x=features['from_this_person_to_poi']
    y=features['to_messages']
    a=features['from_poi_to_this_person']
    b=features['from_messages']
    c=features['total_payments']
    d=features['total_stock_value']
    e=features['bonus']
    f=features['salary']
#fraction_from/to_poi is the fraction of poi_messages and from/to_messages
    features['fraction_from_poi']=computeFraction(a,b)
    features['fraction_to_poi']=computeFraction(x,y) 
#bonus_salary_ratio is the ratio of bonus to salary
    if e=='NaN' or f=='NaN':
        features['bonus_salary_ratio']='NaN'
    else :
        features['bonus_salary_ratio']=float(e) /float(f)   
# total net worth is the sum of tatal payments and total stock value
    if c=='NaN' or d=='NaN':
        features['total_net_worth']='NaN'
    else:
        features['total_net_worth']=c+d   

#adding new features to features_list
features_list+=['total_net_worth']+['bonus_salary_ratio']+\
['fraction_from_poi']+['fraction_to_poi']
# eleminating the feature 'email_address'
features_list.remove('email_address')
print "\ntotal number of features in including new features:",\
len(features_list),"\n"
#%%
### Store to my_dataset for easy export below.
my_dataset = data_dict
### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
#%%

## selecting 7 best features excluding 'poi'
import operator
from sklearn.feature_selection import SelectKBest,f_classif
features_selected=[]
clf = SelectKBest(f_classif,k=7)
selected_features = clf.fit_transform(features,labels)
for i in clf.get_support(indices=True):
    features_selected.append(features_list[i+1])
features_score = zip(features_list[1:25],clf.scores_[:24])
features_score = sorted(features_score,key=operator.itemgetter(1),reverse=True)
#%%
features_list=['poi']+features_selected
print "Scores of the features :\n"
for i in features_score:
    print i
print " \n THE BEST 8 Features including 'poi' are :"
print features_list
#%%
### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report,accuracy_score,recall_score,\
precision_score
#%%
# Provided to give you a starting point. Try a variety of classifiers.
print "\nGaussianNB classifier(Default) :"
NB = GaussianNB()
test_classifier(NB,my_dataset,features_list,folds = 1000)
#%%
print "\nDecission Tree classifier(default) :"
dt = DecisionTreeClassifier()
test_classifier(dt,my_dataset,features_list,folds = 1000)
#%%
print "\nKneighbour classifier(default)"
kn=KNeighborsClassifier()
test_classifier(kn,my_dataset,features_list,folds = 1000)
#%%
print "\n PCA with decisiontree \n"
param_grid = {
         'pca__n_components':[1,2,3,4,5,6],
         'tree__min_samples_split':[2,5,10,100],
         'tree__criterion':['gini'],
         'tree__splitter':['best']
          }
estimators = [('pca',PCA()),('tree',DecisionTreeClassifier())]
pipe = Pipeline(estimators)
gs = GridSearchCV(pipe, param_grid,n_jobs=1,scoring = 'f1')
gs.fit(features,labels)
clf = gs.best_estimator_
test_classifier(clf,my_dataset,features_list,folds = 1000)
#%%
### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

#%%
# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split, StratifiedShuffleSplit
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)
shuffle= StratifiedShuffleSplit(labels_train,n_iter = 25,test_size = 0.5,
                                random_state = 0)
print "\n Best classifier : PCA with GaussianNb \n"
param_grid = {
         'pca__n_components':[1,2,3,4,5,6]
          }
estimators = [('pca',PCA()),('gaussian',GaussianNB())]
pipe = Pipeline(estimators)
gs = GridSearchCV(pipe, param_grid,n_jobs=1,scoring = 'f1',cv=shuffle)
gs.fit(features_train,labels_train)
pred=gs.predict(features_test)
clf = gs.best_estimator_
test_classifier(clf,my_dataset,features_list,folds = 1000)
print "\n\nbest parameters ",gs.best_params_
print '\n Accuracy:',accuracy_score(pred,labels_test),\
"\n Precision:",precision_score(pred,labels_test),\
"\nRecall",recall_score(pred,labels_test)
#%%
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
