# coding: utf-8

# # CS 7641 HW1 Code - Unsupervised Learning

# This file will provide analysis clustering and dimensionality reduction techniques for for two datasets.
# 
# Datasets: Phishing Websites, Bank Marketing.
# 
# Clustering Techniques: k-Means, Expectation Maximization.
# Dimensionality Reduction Techniques: PCA, ICA, RP, RFC

# # 1. Data Load and Preprocessing

# First we load the data! Please save the datasets to your local machine and change the current directory to a file where you have the data stored.

# In[2]:


import os
import pandas as pd
import numpy as np
import random

# get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
pd.set_option('display.max_columns', None)
# os.chdir(r"C:\GT\2022 Fall\7641\hw3\Assignment 3 Unsupervised Learning") #change this to your current working directory
#os.chdir(r"C:\Users\kwest18\Desktop\ML Code") #change this to your current working directory

seed = 9
# ### Load the Phishing Website Data

# In[3]:


## Download the Phishing Data from OpenML https://www.openml.org/d/4534

df_phish = pd.read_csv('PhishingWebsitesData.csv').astype('category')
print("Data has",len(df_phish),"rows and", len(df_phish.columns),"columns.")
if df_phish.isnull().values.any():
    print("Warning: Missing Data")
#df_phish.head()
#df_phish.describe(include='all')


# Now that the phishing data is loaded, we need to do some preprocessing. Several of the columns are categorical with the levels {-1,0,1} and the rest are all binary with levels {-1,1}. For the 3-level columns we will use one-hot encoding to create additional features with level {0,1}. Finally, we will edit the binary features so that the new levels are all {0,1}. We will have more features now, but they will all be binary.

# In[4]:


col_1hot = ['URL_Length','having_Sub_Domain','SSLfinal_State','URL_of_Anchor','Links_in_tags','SFH','web_traffic','Links_pointing_to_page']
df_1hot = df_phish[col_1hot]
df_1hot = pd.get_dummies(df_1hot)
df_others = df_phish.drop(col_1hot,axis=1)
df_phish = pd.concat([df_1hot,df_others],axis=1)
df_phish = df_phish.replace(-1,0).astype('category')
column_order = list(df_phish)
column_order.insert(0, column_order.pop(column_order.index('Result')))
df_phish = df_phish.loc[:, column_order]  #move the target variable 'Result' to the front
#df_phish.describe(include='all')


df_phish.to_csv("PhishingWebsitesData_preprocessed.csv")


# We now have a file with no missing data in the format [y, X] where all features are binary {0,1}. The phishing data is ready to go! Now we move on to loading the Bank Marketing data.

# ### Load the Bank Marketing Data

# In[5]:


## Load the Bank Marketing Data from OpenML https://www.openml.org/d/1461

# names = pd.read_csv('adult.names',sep=',',header=None)
df_default = pd.read_csv('default.csv',sep=',')

# df_default = pd.read_csv('DefaultCreditData.csv')
print("Data has",len(df_default),"rows and", len(df_default.columns),"columns.")
if df_default.isnull().values.any():
    print("Warning: Missing Data")
# #%%
# # remove missing
# df_default = df_default.drop(columns=['stalk-root'])


#%%
# df_default.head()
# df_default.describe(include='all')
df_default.rename(columns={"default payment next month": "Result"},inplace=True)
# df_default['diagnosis'] = np.where(df_default.diagnosis=='M',1,0)

cols = df_default.columns.tolist()
cols = cols[-1:] + cols[:-1]
df_default = df_default[cols]
df_default = (df_default-df_default.min())/(df_default.max()-df_default.min())
# This dataset needs some preprocessing love too. We will convert all categorical columns using one hot encoding. Additionally, we will standardize all of the numeric features and we will convert the target variable from {no,yes} to {0,1}. It should be noted that the feature 'pdays' is numeric but contains values that are '999' if the customer was not called before. It may be worth while to create a new feature that defines whether or not {0,1} a customer had been called before. In the current state the '999' values may be outliers. Finally we will standardize all numeric columns.

# In[6]:

# no need to do one_hot encoding
# col_1hot = ['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']
# df_1hot = df_default[col_1hot]
# df_1hot = pd.get_dummies(df_1hot).astype('category')
# df_others = df_default.drop(col_1hot,axis=1)
# df_default = pd.concat([df_others,df_1hot],axis=1)
# column_order = list(df_default)
# column_order.insert(0, column_order.pop(column_order.index('y')))
# df_default = df_default.loc[:, column_order]
# df_default['y'].replace("no",0,inplace=True)
# df_default['y'].replace("yes",1,inplace=True)
# df_default['y'] = df_default['y'].astype('category')

# numericcols = ['age','duration','campaign','pdays','previous','emp.var.rate','cons.price.idx','cons.conf.idx','euribor3m','nr.employed']
# df_num = df_default[numericcols]
# df_stand =(df_num-df_num.min())/(df_num.max()-df_num.min())
# df_default_categorical = df_default.drop(numericcols,axis=1)
# df_default = pd.concat([df_default_categorical,df_stand],axis=1)
# #df_default.describe(include='all')

# df_default.to_csv("Default Credit Card_preprocessed.csv")


# Now we have successfully loaded and processed both datasets. We are ready to start the ML!

# # 2. Helper Functions

# ### Data Loading and Function Prep

# Before we get into the algorithms, let's define some helper functions that will be used across all of the models and both datasets. We will define a function to load the data (not really necessary in a Jupyter notebook, but good if this is exported as a .py for later use). We will also define a function that plots the learning curve (training and cross validation score as a function of training examples) of an estimator (classification model). Finally, we define functions to output final model scores using an untouched test dataset.

# In[7]:


import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate, train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import itertools
import timeit
from collections import Counter
from sklearn.metrics.pairwise import pairwise_distances

plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['font.size'] = 12

def import_data():

    X1 = np.array(df_phish.values[:,1:],dtype='int64')
    Y1 = np.array(df_phish.values[:,0],dtype='int64')
    X2 = np.array(df_default.values[:,1:],dtype='float')
    Y2 = np.array(df_default.values[:,0],dtype='int64')
    return X1, Y1, X2, Y2


def plot_learning_curve(clf, X, y, title="Title Here"):
    
    n = len(y)
    train_mean = []; train_std = [] #model performance score (f1)
    cv_mean = []; cv_std = [] #model performance score (f1)
    fit_mean = []; fit_std = [] #model fit/training time
    pred_mean = []; pred_std = [] #model test/prediction times
    train_sizes=(np.linspace(.1, 1.0, 10)*n).astype('int')  
    
    for i in train_sizes:
        idx = np.random.randint(X.shape[0], size=i)
        X_subset = X[idx,:]
        y_subset = y[idx]
        scores = cross_validate(clf, X_subset, y_subset, cv=5, scoring='f1', n_jobs=-1, return_train_score=True)
        
        train_mean.append(np.mean(scores['train_score'])); train_std.append(np.std(scores['train_score']))
        cv_mean.append(np.mean(scores['test_score'])); cv_std.append(np.std(scores['test_score']))
        fit_mean.append(np.mean(scores['fit_time'])); fit_std.append(np.std(scores['fit_time']))
        pred_mean.append(np.mean(scores['score_time'])); pred_std.append(np.std(scores['score_time']))
    
    train_mean = np.array(train_mean); train_std = np.array(train_std)
    cv_mean = np.array(cv_mean); cv_std = np.array(cv_std)
    fit_mean = np.array(fit_mean); fit_std = np.array(fit_std)
    pred_mean = np.array(pred_mean); pred_std = np.array(pred_std)
    
    plot_LC(train_sizes, train_mean, train_std, cv_mean, cv_std, title)
    plot_times(train_sizes, fit_mean, fit_std, pred_mean, pred_std, title)
    
    return train_sizes, train_mean, fit_mean, pred_mean
    

def plot_LC(train_sizes, train_mean, train_std, cv_mean, cv_std, title):
    
    plt.figure()
    plt.title("Learning Curve: "+ title)
    plt.xlabel("Training Examples")
    plt.ylabel("Model F1 Score")
    plt.fill_between(train_sizes, train_mean - 2*train_std, train_mean + 2*train_std, alpha=0.1, color="b")
    plt.fill_between(train_sizes, cv_mean - 2*cv_std, cv_mean + 2*cv_std, alpha=0.1, color="r")
    plt.plot(train_sizes, train_mean, 'o-', color="b", label="Training Score")
    plt.plot(train_sizes, cv_mean, 'o-', color="r", label="Cross-Validation Score")
    plt.legend(loc="best")
    plt.show()
    
    
def plot_times(train_sizes, fit_mean, fit_std, pred_mean, pred_std, title):
    
    plt.figure()
    plt.title("Modeling Time: "+ title)
    plt.xlabel("Training Examples")
    plt.ylabel("Training Time (s)")
    plt.fill_between(train_sizes, fit_mean - 2*fit_std, fit_mean + 2*fit_std, alpha=0.1, color="b")
    plt.fill_between(train_sizes, pred_mean - 2*pred_std, pred_mean + 2*pred_std, alpha=0.1, color="r")
    plt.plot(train_sizes, fit_mean, 'o-', color="b", label="Training Time (s)")
    plt.plot(train_sizes, pred_std, 'o-', color="r", label="Prediction Time (s)")
    plt.legend(loc="best")
    plt.show()
    
    
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Oranges):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(2), range(2)):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    
def final_classifier_evaluation(clf,X_train, X_test, y_train, y_test):
    
    start_time = timeit.default_timer()
    clf.fit(X_train, y_train)
    end_time = timeit.default_timer()
    training_time = end_time - start_time
    
    start_time = timeit.default_timer()    
    y_pred = clf.predict(X_test)
    end_time = timeit.default_timer()
    pred_time = end_time - start_time
    
    auc = roc_auc_score(y_test, y_pred)
    f1 = f1_score(y_test,y_pred)
    accuracy = accuracy_score(y_test,y_pred)
    precision = precision_score(y_test,y_pred)
    recall = recall_score(y_test,y_pred)
    cm = confusion_matrix(y_test,y_pred)
    output_dict = dict()

    output_dict['training time'] = training_time
    output_dict['prediction time'] = pred_time
    output_dict['f1 score'] = f1
    output_dict['accuracy'] = accuracy
    output_dict['precision'] = precision
    output_dict['recall'] = recall
    output_dict['auc'] = auc
    # output_dict['scenario'] = scenario
    outputdf = pd.DataFrame.from_dict(output_dict, orient='index')

    print("Model Evaluation Metrics Using Untouched Test Dataset")
    print("*****************************************************")
    print("Model Training Time (s):   "+"{:.5f}".format(training_time))
    print("Model Prediction Time (s): "+"{:.5f}\n".format(pred_time))
    print("F1 Score:  "+"{:.2f}".format(f1))
    print("Accuracy:  "+"{:.2f}".format(accuracy)+"   AUC:     "+"{:.2f}".format(auc))
    print("Precision: "+"{:.2f}".format(precision)+"   Recall:  "+"{:.2f}".format(recall))
    print("*****************************************************")
    plt.figure()
    plot_confusion_matrix(cm, classes=["0","1"], title='Confusion Matrix')
    plt.show()
    return outputdf

def cluster_predictions(Y,clusterLabels):
    assert (Y.shape == clusterLabels.shape)
    pred = np.empty_like(Y)
    for label in set(clusterLabels):
        mask = clusterLabels == label
        sub = Y[mask]
        target = Counter(sub).most_common(1)[0][0]
        pred[mask] = target
#    assert max(pred) == max(Y)
#    assert min(pred) == min(Y)    
    return pred

def pairwiseDistCorr(X1,X2):
    assert X1.shape[0] == X2.shape[0]
    
    d1 = pairwise_distances(X1)
    d2 = pairwise_distances(X2)
    return np.corrcoef(d1.ravel(),d2.ravel())[0,1]


# ## Neural Network Classifier

# This section will build a forward-feed neural network which computes weights via backpropagation (a multilayer perceptron). The main hyperparameter will be number of hidden nodes in a network defined by a single hidden layer, while others that could be searched over in grid search are activation function, and learning rate. This will be used later when we compare neural networks built from different combinations of features after clustering and dimensionality reduction

# In[8]:


from sklearn.neural_network import MLPClassifier

def hyperNN(X_train, y_train, X_test, y_test, title):

    f1_test = []
    f1_train = []
    hlist = np.linspace(1,150,30).astype('int')
    for i in hlist:
            clf = MLPClassifier(hidden_layer_sizes=(i,), solver='adam', activation='logistic', 
                                learning_rate_init=0.05, random_state=seed)
            clf.fit(X_train, y_train)
            y_pred_test = clf.predict(X_test)
            y_pred_train = clf.predict(X_train)
            f1_test.append(accuracy_score(y_test, y_pred_test))
            f1_train.append(accuracy_score(y_train, y_pred_train))
      
    plt.plot(hlist, f1_train, 'o-', color = 'b', label='Train Accuracy')
    plt.plot(hlist, f1_test, 'o-', color='r', label='Test Accuracy')
    plt.ylabel('Model Accuracy')
    plt.xlabel('No. Hidden Units')
    
    plt.title(title)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()
    
    
def NNGridSearchCV(X_train, y_train):
    #parameters to search:
    #number of hidden units
    #learning_rate
    h_units = [5, 10, 20, 30, 40, 50, 60]
    param_grid = {'hidden_layer_sizes': h_units}

    net = GridSearchCV(estimator = MLPClassifier(solver='adam',activation='logistic',learning_rate_init=0.05,random_state=seed),
                       param_grid=param_grid, cv=10)
    net.fit(X_train, y_train)
    print("Per Hyperparameter tuning, best parameters are:")
    print(net.best_params_)
    return net.best_params_['hidden_layer_sizes']


# # 3. Clustering

# ## k-Means Clustering

# This section will implement k-means clustering for both datasets. Our objectives are to:
# 1. Determine the best number of clusters for each dataset by using the elbow inspection method on silhouette score.
# 2. Describe the attributes which make up each cluster.
# 3. Score each cluster with an accuracy since technically we do have labels available for these datasets (labels are not used when determining clusters).
# 
# Since k-Means is susceptible to get stuck in local optima due to the random selection of initial cluster centers, I will report the average metrics over 5 models for each number of k clusters.

# In[9]:


from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, f1_score, homogeneity_score, completeness_score
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import v_measure_score

# When all samples in cluster k have the same label c, the homogeneity equals 1.
# When all samples of kind c have been assigned to the same cluster k, the completeness equals 1.


np.random.seed(seed)

def run_kmeans(X,y,title):
    plot_range = np.arange(0,20,2)
    kclusters = list(np.arange(2,20,2))
    silhouette_scores = []; f1_scores = []; homo_scores = []; train_times = []
    v_m_scores = []
    comp_scores = []

    for k in kclusters:
        start_time = timeit.default_timer()
        km = KMeans(n_clusters=k, n_init=5,random_state=seed).fit(X)
        end_time = timeit.default_timer()
        train_times.append(end_time - start_time)
        silhouette_scores.append(silhouette_score(X, km.labels_))
        y_mode_vote = cluster_predictions(y,km.labels_)
        f1_scores.append(f1_score(y, y_mode_vote))
        homo_scores.append(homogeneity_score(y, km.labels_))
        v_m_scores.append(v_measure_score(y, km.labels_))
        comp_scores.append(completeness_score(y, km.labels_))


    # elbow curve for silhouette score
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(kclusters, silhouette_scores)
    plt.grid(True)
    plt.xticks(plot_range)
    plt.xlabel('No. Clusters')
    plt.ylabel('Avg Silhouette Score')
    plt.title('Silhouette Score for KMeans: '+ title)
    plt.show()
   
    # plot homogeneity scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(kclusters, homo_scores, label="Homogeneity Score")
    ax.plot(kclusters, v_m_scores, label="V-measure Score")
    ax.plot(kclusters, comp_scores, label="Completeness Score")
    plt.grid(True)
    plt.legend(loc="best")

    plt.xticks(plot_range)
    plt.xlabel('No. Clusters')
    plt.ylabel('Score')
    plt.title('V-measure Scores KMeans: '+ title)
    plt.show()


#     # plot f1 scores
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     ax.plot(kclusters, f1_scores)
#     plt.grid(True)
#     plt.xlabel('No. Clusters')
#     plt.ylabel('F1 Score')
#     plt.title('F1 Scores KMeans: '+ title)
#     plt.show()

    # plot model training time
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(kclusters, train_times)
    plt.grid(True)
    plt.xlabel('No. Clusters')
    plt.ylabel('Training Time (s)')
    plt.title('KMeans Training Time: '+ title)
    plt.show()
    
def evaluate_kmeans(km, X, y, scenario=''): # here the km should be fitted with train data and used for testing
    start_time = timeit.default_timer()
    km.fit(X, y)
    end_time = timeit.default_timer()
    training_time = end_time - start_time
    
    y_mode_vote = cluster_predictions(y,km.labels_)
    auc = roc_auc_score(y, y_mode_vote)
    f1 = f1_score(y, y_mode_vote)
    accuracy = accuracy_score(y, y_mode_vote)
    precision = precision_score(y, y_mode_vote)
    recall = recall_score(y, y_mode_vote)
    cm = confusion_matrix(y, y_mode_vote)
    output_dict = dict()

    output_dict['training time'] = training_time
    output_dict['No. iter'] = km.n_iter_
    output_dict['f1 score'] = f1
    output_dict['accuracy'] = accuracy
    output_dict['precision'] = precision
    output_dict['recall'] = recall
    output_dict['auc'] = auc
    output_dict['scenario'] = scenario
    outputdf = pd.DataFrame.from_dict(output_dict, orient='index')

    # print("Model Evaluation Metrics Using Mode Cluster Vote")
    # print("*****************************************************")
    # print("Model Training Time (s):   "+"{:.2f}".format(training_time))
    # print("No. Iterations to Converge: {}".format(km.n_iter_))
    # print("F1 Score:  "+"{:.2f}".format(f1))
    # print("Accuracy:  "+"{:.2f}".format(accuracy)+"   AUC:     "+"{:.2f}".format(auc))
    # print("Precision: "+"{:.2f}".format(precision)+"   Recall:  "+"{:.2f}".format(recall))
    # print("*****************************************************")
    plt.figure()
    plot_confusion_matrix(cm, classes=["0","1"], title='Confusion Matrix')
    plt.show()
    return outputdf


# In[ ]:


phishX,phishY,bankX,bankY = import_data()
X_train, X_test, y_train, y_test = train_test_split(np.array(phishX),np.array(phishY), test_size=0.3)
run_kmeans(X_train,y_train,'Phishing Data')

#%%

km_p = KMeans(n_clusters=2,n_init=5,random_state=seed)

km_output_phish = evaluate_kmeans(km_p,X_test,y_test)

df = pd.DataFrame(km_p.cluster_centers_)
df.to_csv("Phishing kMeans Cluster Centers.csv")


# In[ ]:

phishX,phishY,bankX,bankY = import_data()
X_train, X_test, y_train, y_test = train_test_split(np.array(bankX),np.array(bankY), test_size=0.3)

run_kmeans(X_train,y_train,'Default Credit Card Data')

#%%
km_b = KMeans(n_clusters=8,n_init=5,random_state=seed)
km_output_default = evaluate_kmeans(km_b,X_test,y_test)
df = pd.DataFrame(km_b.cluster_centers_)
df.to_csv("Default Credit Card kMeans Cluster Centers.csv")


# ## Expectation Maximization

# This section will implement k-means clustering for both datasets. The same 3 objectives from k-means apply here.

# In[10]:


from sklearn.mixture import GaussianMixture as EM
from sklearn.metrics import silhouette_score, f1_score, homogeneity_score, completeness_score
from sklearn.metrics.cluster import v_measure_score
import matplotlib.pyplot as plt


np.random.seed(seed)

def run_EM(X,y,title):

    #kdist =  [2,3,4,5]
    #kdist = list(range(2,51))
    kdist = list(np.arange(2,30,2))
    silhouette_scores = []; f1_scores = []; homo_scores = []; train_times = []; aic_scores = []; bic_scores = []
    v_m_scores = []
    comp_scores = []
    range_step = np.arange(0, 30, 2)

    for k in kdist:
        start_time = timeit.default_timer()
        em = EM(n_components=k,covariance_type='diag',n_init=1,warm_start=True,random_state=seed).fit(X)
        end_time = timeit.default_timer()
        train_times.append(end_time - start_time)
        
        labels = em.predict(X)
        silhouette_scores.append(silhouette_score(X, labels))
        y_mode_vote = cluster_predictions(y,labels)
        f1_scores.append(f1_score(y, y_mode_vote))
        homo_scores.append(homogeneity_score(y, labels))
        v_m_scores.append(v_measure_score(y, labels))
        comp_scores.append(completeness_score(y, labels))
        aic_scores.append(em.aic(X))
        bic_scores.append(em.bic(X))

        
    # elbow curve for silhouette score
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(kdist, silhouette_scores)
    plt.grid(True)
    plt.xticks(range_step)

    plt.xlabel('No. Distributions')
    plt.ylabel('Avg Silhouette Score')
    plt.title('Silhouette Score for EM: '+ title)
    plt.show()
   
    # plot homogeneity scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(kdist, homo_scores, label='Homogeneity score')
    ax.plot(kdist, v_m_scores, label='V-measure score')
    ax.plot(kdist, comp_scores, label='Completeness score')
    plt.legend(loc="best")
    plt.grid(True)
    plt.xticks(range_step)

    plt.xlabel('No. Distributions')
    plt.ylabel('Score')
    plt.title('V-measure Scores EM: '+ title)
    plt.show()

    # # plot f1 scores
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.plot(kdist, f1_scores)
    # plt.grid(True)
    # plt.xticks(range_step)

    # plt.xlabel('No. Distributions')
    # plt.ylabel('F1 Score')
    # plt.title('F1 Scores EM: '+ title)
    # plt.show()

    # plot model AIC and BIC
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(kdist, aic_scores, label='AIC')
    ax.plot(kdist, bic_scores,label='BIC')
    plt.grid(True)
    plt.xticks(range_step)

    plt.xlabel('No. Distributions')
    plt.ylabel('Model Complexity Score')
    plt.title('EM Model Complexity: '+ title)
    plt.legend(loc="best")
    plt.show()

    # plot model training time
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(kdist, train_times)
    plt.grid(True)
    plt.xticks(range_step)
    plt.xlabel('No. Clusters')
    plt.ylabel('Training Time (s)')
    plt.title('EM Training Time: '+ title)
    plt.show()
    
def evaluate_EM(em, X, y, scenario=''):
    start_time = timeit.default_timer()
    em.fit(X, y)
    end_time = timeit.default_timer()
    training_time = end_time - start_time
    
    labels = em.predict(X)
    y_mode_vote = cluster_predictions(y,labels)
    auc = roc_auc_score(y, y_mode_vote)
    f1 = f1_score(y, y_mode_vote)
    accuracy = accuracy_score(y, y_mode_vote)
    precision = precision_score(y, y_mode_vote)
    recall = recall_score(y, y_mode_vote)
    cm = confusion_matrix(y, y_mode_vote)

    output_dict = dict()

    output_dict['training time'] = training_time
    output_dict['No. iter'] = em.n_iter_
    output_dict['f1 score'] = f1
    output_dict['accuracy'] = accuracy
    output_dict['precision'] = precision
    output_dict['recall'] = recall
    output_dict['auc'] = auc
    output_dict['scenario'] = scenario
    outputdf = pd.DataFrame.from_dict(output_dict, orient='index')

    # print("Model Evaluation Metrics Using Mode Cluster Vote")
    # print("*****************************************************")
    # print("Model Training Time (s):   "+"{:.2f}".format(training_time))
    # print("No. Iterations to Converge: {}".format(em.n_iter_))
    # print("Log-likelihood Lower Bound: {:.2f}".format(em.lower_bound_))
    # print("F1 Score:  "+"{:.2f}".format(f1))
    # print("Accuracy:  "+"{:.2f}".format(accuracy)+"     AUC:       "+"{:.2f}".format(auc))
    # print("Precision: "+"{:.2f}".format(precision)+"     Recall:    "+"{:.2f}".format(recall))
    # print("*****************************************************")
    plt.figure()
    plot_confusion_matrix(cm, classes=["0","1"], title='Confusion Matrix')
    plt.show()
    return outputdf


# In[ ]:


phishX,phishY,bankX,bankY = import_data()
X_train, X_test, y_train, y_test = train_test_split(np.array(phishX),np.array(phishY), test_size=0.3)
run_EM(X_train,y_train,'Phishing Data')

#%%

em = EM(n_components=4,covariance_type='diag',n_init=1,warm_start=True,random_state=seed)
# em = EM(n_components=24,covariance_type='diag',n_init=1,warm_start=True,random_state=seed)
em_phish_output = evaluate_EM(em,X_test,y_test, 'Phishing Data')

df = pd.DataFrame(em.means_)
df.to_csv("Phishing EM Component Means.csv")


# In[ ]:

phishX,phishY,bankX,bankY = import_data()
X_train, X_test, y_train, y_test = train_test_split(np.array(bankX),np.array(bankY), test_size=0.3)
run_EM(X_train,y_train,'Default Credit Card Data')
# run_EM(bankX,bankY,'Default Credit Card Data')

#%%
em = EM(n_components=12,covariance_type='diag',n_init=1,warm_start=True,random_state=seed)
em_default_output = evaluate_EM(em,X_test,y_test, 'Default Credit Data')
df = pd.DataFrame(em.means_)
df.to_csv("Default Credit Card EM Component Means.csv")


# 
# # 4. Dimensionality Reduction

# This section will implement 4 different dimensionality reduction techniques on both the phishing and the Default Credit Dataset. Then, k-means and EM clustering will be performed for each (dataset * dim_reduction) combination to see how the clustering compares with using the All featuress. The 4 dimensionality reduction techniques are:
# - Principal Components Analysis (PCA). Optimal number of PC chosen by inspecting % variance explained and the eigenvalues.
# - Independent Components Analysis (ICA). Optimal number of IC chosen by inspecting kurtosis.
# - Random Components Analysis (RP) (otherwise known as Randomized Projections). Optimal number of RC chosen by inspecting reconstruction error.
# - Random Forest Classifier (RFC). Optimal number of components chosen by feature importance.

# In[11]:


from sklearn.decomposition import PCA, FastICA as ICA
from sklearn.random_projection import GaussianRandomProjection as GRP, SparseRandomProjection as RP
from sklearn.ensemble import RandomForestClassifier as RFC
from itertools import product
from collections import defaultdict

step = 4

def run_PCA(X,y,title):

    plot_range = np.arange(0,X.shape[1]+1,5)
    pca = PCA(random_state=seed).fit(X) #for all components
    cum_var = np.cumsum(pca.explained_variance_ratio_)

    fig, ax1 = plt.subplots()
    ax1.plot(list(range(len(pca.explained_variance_ratio_))), cum_var, 'b-')
    ax1.set_xlabel('Principal Components')
    # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel('Cumulative Explained Variance Ratio', color='b')
    ax1.tick_params('y', colors='b')
    plt.grid(True)

    ax2 = ax1.twinx()
    ax2.plot(list(range(len(pca.singular_values_))), pca.singular_values_, 'r-')
    ax2.set_ylabel('Eigenvalues', color='r')
    ax2.tick_params('y', colors='r')
    plt.grid(True)
    plt.xticks(plot_range)

    plt.title("PCA Explained Variance and Eigenvalues: "+ title)
    fig.tight_layout()
    plt.show()
    
def run_ICA(X,y,title):
    plot_range = np.arange(0,X.shape[1]+1,5)
    dims = list(np.arange(2,X.shape[1], step))
    dims.append(X.shape[1])
    ica = ICA(random_state=seed, max_iter=300,tol=0.005)
    kurt = []

    for dim in dims:
        ica.set_params(n_components=dim)
        tmp = ica.fit_transform(X)
        tmp = pd.DataFrame(tmp)
        tmp = tmp.kurt(axis=0)
        kurt.append(tmp.abs().mean())

    plt.figure()
    plt.title("ICA Kurtosis: "+ title)
    plt.xlabel("Independent Components")
    plt.ylabel("Avg Kurtosis Across IC")
    plt.plot(dims, kurt, 'b-')
    plt.grid(True)
    plt.xticks(plot_range)

    plt.show()

def run_RP(X,y,title):
    
    plot_range = np.arange(0,X.shape[1]+1,5)

    dims = list(np.arange(2,X.shape[1],step))
    dims.append(X.shape[1])
    tmp = defaultdict(dict)
    times = 3

    for i,dim in product(range(times),dims):
        rp = RP(random_state=i, n_components=dim)  # random_state = i?
        tmp[dim][i] = pairwiseDistCorr(rp.fit_transform(X), X)
    tmp = pd.DataFrame(tmp).T
    mean_recon = tmp.mean(axis=1).tolist()
    std_recon = tmp.std(axis=1).tolist()


    fig, ax1 = plt.subplots()
    ax1.plot(dims,mean_recon, 'b-')
    ax1.set_xlabel('Random Components')
    # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel('Mean Reconstruction Correlation', color='b')
    ax1.tick_params('y', colors='b')
    plt.grid(True)

    ax2 = ax1.twinx()
    ax2.plot(dims,std_recon, 'r-')
    ax2.set_ylabel('STD Reconstruction Correlation', color='r')
    ax2.tick_params('y', colors='r')
    plt.grid(True)
    plt.xticks(plot_range)

    plt.title(f"Random Components for {times} Restarts: "+ title)
    fig.tight_layout()
    plt.show()
    
def run_RFC(X,y,df_original):
    rfc = RFC(n_estimators=300,max_depth=3,random_state=seed,n_jobs=-1)
    imp = rfc.fit(X,y).feature_importances_ 
    imp = pd.DataFrame(imp,columns=['Feature Importance'],index=df_original.columns[1::])
    imp.sort_values(by=['Feature Importance'],inplace=True,ascending=False)
    imp['Cum Sum'] = imp['Feature Importance'].cumsum()
    imp = imp[imp['Cum Sum']<=0.95]
    top_cols = imp.index.tolist()
    return imp, top_cols


# In[56]:

phishX,phishY,bankX,bankY = import_data()
X_train, X_test, y_train, y_test = train_test_split(np.array(phishX),np.array(phishY), test_size=0.5)

run_PCA(X_train,y_train,"Phishing Data")
run_ICA(X_train,y_train,"Phishing Data")
run_RP(X_train,y_train,"Phishing Data")
imp_phish, topcols_phish = run_RFC(X_train,y_train,df_phish)


# Recreating Clustering Experiment (k-means and EM) for phishing data.

# In[71]:

# phishX,phishY,bankX,bankY = import_data()
# imp_phish, topcols_phish = run_RFC(phishX,phishY,df_phish)

pca_phish = PCA(n_components=25,random_state=seed).fit_transform(X_test)
ica_phish = ICA(n_components=38,random_state=seed).fit_transform(X_test)
rca_phish = RP(n_components=35,random_state=seed).fit_transform(X_test)
X_test = pd.DataFrame(X_test, columns = df_phish.columns[1:])
rfc_phish = X_test[topcols_phish]
rfc_phish = np.array(rfc_phish.values,dtype='int64')


# In[63]:

run_kmeans(pca_phish,y_test,'PCA Phishing Data')
run_kmeans(ica_phish,y_test,'ICA Phishing Data')
run_kmeans(rca_phish,y_test,'RP Phishing Data')
run_kmeans(rfc_phish,y_test,'RFC Phishing Data')


# In[70]:

# ideally here should use test dataset, but here used total dataset
pca_phish_kmeans = evaluate_kmeans(KMeans(n_clusters=2,n_init=5,random_state=seed),pca_phish,y_test,'pca_phish_kmeans')
ica_phish_kmeans = evaluate_kmeans(KMeans(n_clusters=8,n_init=5,random_state=seed),ica_phish,y_test,'ica_phish_kmeans')
rca_phish_kmeans = evaluate_kmeans(KMeans(n_clusters=4,n_init=5,random_state=seed),rca_phish,y_test,'rca_phish_kmeans')
rfc_phish_kmeans = evaluate_kmeans(KMeans(n_clusters=2,n_init=5,random_state=seed),rfc_phish,y_test,'rfc_phish_kmeans')


# In[66]:


run_EM(pca_phish,y_test,'PCA Phishing Data')
run_EM(ica_phish,y_test,'ICA Phishing Data')
run_EM(rca_phish,y_test,'RP Phishing Data')
run_EM(rfc_phish,y_test,'RFC Phishing Data')


# In[71]:


pca_phish_em = evaluate_EM(EM(n_components=20,covariance_type='diag',n_init=1,warm_start=True,random_state=seed),pca_phish,y_test,'pca_phish_em')
ica_phish_em = evaluate_EM(EM(n_components=22,covariance_type='diag',n_init=1,warm_start=True,random_state=seed),ica_phish,y_test,'ica_phish_em')
rca_phish_em = evaluate_EM(EM(n_components=10,covariance_type='diag',n_init=1,warm_start=True,random_state=seed),rca_phish,y_test,'rca_phish_em')
rfc_phish_em = evaluate_EM(EM(n_components=2,covariance_type='diag',n_init=1,warm_start=True,random_state=seed),rfc_phish,y_test,'rfc_phish_em')

phish_em_o = pd.concat([pca_phish_em, ica_phish_em, rca_phish_em, rfc_phish_em], axis=1)

# Recreating Clustering Experiment (k-means and EM) for Default Credit Data.


# In[13]:

phishX,phishY,bankX,bankY = import_data()
X_train, X_test, y_train, y_test = train_test_split(np.array(bankX),np.array(bankY), test_size=0.5)
run_PCA(X_train,y_train,"Default Credit Card Data")
run_ICA(X_train,y_train,"Default Credit Card Data")
run_RP(X_train,y_train,"Default Credit Card Data")
imp_bank, topcols_bank = run_RFC(X_train,y_train,df_default)

#%%

pca_bank = PCA(n_components=10,random_state=seed).fit_transform(X_test)
ica_bank = ICA(n_components=22,random_state=seed).fit_transform(X_test)
rca_bank = RP(n_components=15,random_state=seed).fit_transform(X_test)
X_test = pd.DataFrame(X_test, columns = df_default.columns[1:])
rfc_bank = X_test[topcols_bank]
rfc_bank = np.array(rfc_bank.values,dtype='float64')


# In[1]:


run_kmeans(pca_bank,y_test,'PCA Default Credit Data')
run_kmeans(ica_bank,y_test,'ICA Default Credit Data')
run_kmeans(rca_bank,y_test,'RP Default Credit Data')
run_kmeans(rfc_bank,y_test,'RFC Default Credit Data')


# In[ ]:


pca_default_kmeans = evaluate_kmeans(KMeans(n_clusters=10,n_init=5,random_state=seed,),pca_bank,y_test,'pca_default_kmeans')
ica_default_kmeans = evaluate_kmeans(KMeans(n_clusters=12,n_init=5,random_state=seed,),ica_bank,y_test,'ica_default_kmeans')
rca_default_kmeans = evaluate_kmeans(KMeans(n_clusters=10,n_init=5,random_state=seed,),rca_bank,y_test,'rca_default_kmeans')
rfc_default_kmeans = evaluate_kmeans(KMeans(n_clusters=12,n_init=5,random_state=seed,),rfc_bank,y_test,'rfc_default_kmeans')

default_kmeans_o = pd.concat([pca_default_kmeans, ica_default_kmeans, rca_default_kmeans, rfc_default_kmeans], axis=1)


# In[ ]:


run_EM(pca_bank,y_test,'PCA Default Credit Data')
run_EM(ica_bank,y_test,'ICA Default Credit Data')
run_EM(rca_bank,y_test,'RP Default Credit Data')
run_EM(rfc_bank,y_test,'RFC Default Credit Data')


# In[ ]:


pca_default_em = evaluate_EM(EM(n_components=12,covariance_type='diag',n_init=1,warm_start=True,random_state=seed),pca_bank,y_test,'pca_default_em')
ica_default_em = evaluate_EM(EM(n_components=12,covariance_type='diag',n_init=1,warm_start=True,random_state=seed),ica_bank,y_test,'ica_default_em')
rca_default_em = evaluate_EM(EM(n_components=10,covariance_type='diag',n_init=1,warm_start=True,random_state=seed),rca_bank,y_test,'rca_default_em')
rfc_default_em = evaluate_EM(EM(n_components=12,covariance_type='diag',n_init=1,warm_start=True,random_state=seed),rfc_bank,y_test,'rfc_default_em')


default_em_o = pd.concat([pca_default_em, ica_default_em, rca_default_em, rfc_default_em], axis=1)


# # 5. Training Neural Network on Projected Data

# This section will train a neural network on the 4 projected datasets for only the default data. We will examine the learning curves on the training data as well as the final network performance on the test dataset.

# In[30]:


# Original, All features
X_train, X_test, y_train, y_test = train_test_split(np.array(bankX),np.array(bankY), test_size=0.30)
full_est = MLPClassifier(hidden_layer_sizes=(50,), solver='adam', activation='logistic', learning_rate_init=0.01, random_state=seed)
train_samp_full, NN_train_score_full, NN_fit_time_full, NN_pred_time_full = plot_learning_curve(full_est, X_train, y_train,title="Neural Net Default Credit: Full")
full_output = final_classifier_evaluation(full_est, X_train, X_test, y_train, y_test)


#%%
pca_bank = PCA(n_components=10,random_state=seed).fit_transform(bankX)
ica_bank = ICA(n_components=22,random_state=seed).fit_transform(bankX)
rca_bank = RP(n_components=14,random_state=seed).fit_transform(bankX)
# X_test = pd.DataFrame(X_test, columns = df_default.columns[1:])
imp_bank, topcols_bank = run_RFC(bankX,bankY,df_default)
rfc_bank = df_default[topcols_bank]
rfc_bank = np.array(rfc_bank.values,dtype='float64')



# In[32]:


X_train, X_test, y_train, y_test = train_test_split(np.array(pca_bank),np.array(bankY), test_size=0.30)
pca_est = MLPClassifier(hidden_layer_sizes=(50,), solver='adam', activation='logistic', learning_rate_init=0.01, random_state=seed)
train_samp_pca, NN_train_score_pca, NN_fit_time_pca, NN_pred_time_pca = plot_learning_curve(pca_est, X_train, y_train,title="Neural Net Default Credit: PCA")
pca_output = final_classifier_evaluation(pca_est, X_train, X_test, y_train, y_test)


# In[34]:


X_train, X_test, y_train, y_test = train_test_split(np.array(ica_bank),np.array(bankY), test_size=0.30)
ica_est = MLPClassifier(hidden_layer_sizes=(50,), solver='adam', activation='logistic', learning_rate_init=0.01, random_state=seed)
train_samp_ica, NN_train_score_ica, NN_fit_time_ica, NN_pred_time_ica = plot_learning_curve(ica_est, X_train, y_train,title="Neural Net Default Credit: ICA")
ica_output = final_classifier_evaluation(ica_est, X_train, X_test, y_train, y_test)


# In[35]:


X_train, X_test, y_train, y_test = train_test_split(np.array(rca_bank),np.array(bankY), test_size=0.30)
rca_est = MLPClassifier(hidden_layer_sizes=(50,), solver='adam', activation='logistic', learning_rate_init=0.01, random_state=seed)
train_samp_rca, NN_train_score_rca, NN_fit_time_rca, NN_pred_time_rca = plot_learning_curve(rca_est, X_train, y_train,title="Neural Net Default Credit: RP")
rca_output = final_classifier_evaluation(rca_est, X_train, X_test, y_train, y_test)


# In[36]:


X_train, X_test, y_train, y_test = train_test_split(np.array(rfc_bank),np.array(bankY), test_size=0.30)
rfc_est = MLPClassifier(hidden_layer_sizes=(50,), solver='adam', activation='logistic', learning_rate_init=0.01, random_state=seed)
train_samp_rfc, NN_train_score_rfc, NN_fit_time_rfc, NN_pred_time_rfc = plot_learning_curve(rfc_est, X_train, y_train,title="Neural Net Default Credit: RFC")
rfc_output = final_classifier_evaluation(rfc_est, X_train, X_test, y_train, y_test)


# # 6. Model Comparison Plots

# Let's define and call a function that will plot training times and learning rates for the 4 different NN models so that we can compare across the classifiers for the same dataset.

# In[28]:


def compare_fit_time(n,full_fit,pca_fit,ica_fit,rca_fit,rfc_fit,title):
    
    plt.figure()
    plt.title("Model Training Times: " + title)
    plt.xlabel("Training Examples")
    plt.ylabel("Model Training Time (s)")
    plt.plot(n, full_fit, '-', color="k", label="All features")
    plt.plot(n, pca_fit, '-', color="b", label="PCA")
    plt.plot(n, ica_fit, '-', color="r", label="ICA")
    plt.plot(n, rca_fit, '-', color="g", label="RP")
    plt.plot(n, rfc_fit, '-', color="m", label="RFC")
    plt.legend(loc="best")
    plt.show()
    
def compare_pred_time(n,full_pred, pca_pred, ica_pred, rca_pred, rfc_pred, title):
    
    plt.figure()
    plt.title("Model Prediction Times: " + title)
    plt.xlabel("Training Examples")
    plt.ylabel("Model Prediction Time (s)")
    plt.plot(n, full_pred, '-', color="k", label="All features")
    plt.plot(n, pca_pred, '-', color="b", label="PCA")
    plt.plot(n, ica_pred, '-', color="r", label="ICA")
    plt.plot(n, rca_pred, '-', color="g", label="RP")
    plt.plot(n, rfc_pred, '-', color="m", label="RFC")
    plt.legend(loc="best")
    plt.show()


def compare_learn_time(n,full_learn, pca_learn, ica_learn, rca_learn, rfc_learn, title):
    
    plt.figure()
    plt.title("Model F1 score: " + title)
    plt.xlabel("Training Examples")
    plt.ylabel("Model F1 Score")
    plt.plot(n, full_learn, '-', color="k", label="All features")
    plt.plot(n, pca_learn, '-', color="b", label="PCA")
    plt.plot(n, ica_learn, '-', color="r", label="ICA")
    plt.plot(n, rca_learn, '-', color="g", label="RP")
    plt.plot(n, rfc_learn, '-', color="m", label="RFC")
    plt.legend(loc="best")
    plt.show() 


# In[37]:


compare_fit_time(train_samp_full, NN_fit_time_full, NN_fit_time_pca, NN_fit_time_ica, 
                 NN_fit_time_rca, NN_fit_time_rfc, 'Default Credit Dataset')              
compare_pred_time(train_samp_full, NN_pred_time_full, NN_pred_time_pca, NN_pred_time_ica, 
                 NN_pred_time_rca, NN_pred_time_rfc, 'Default Credit Dataset')   
compare_learn_time(train_samp_full, NN_train_score_full, NN_train_score_pca, NN_train_score_ica, 
                 NN_train_score_rca, NN_train_score_rfc, 'Default Credit Dataset')  


# # 7. Training Neural Network on Projected Data with Cluster Labels

# This section will train a neural network on the 4 projected datasets for only the phishing data. The difference in this section is that we now add cluster labels from both k-means and EM (after 1-hot encoding) to the reduced datasets. We will examine the learning curves on the training data as well as the final network performance on the test dataset.

# In[54]:


def addclusters(X,km_labels,em_labels):
    
    df = pd.DataFrame(X)
    df['KM Cluster'] = km_labels
    df['EM Cluster'] = em_labels
    col_1hot = ['KM Cluster', 'EM Cluster']
    df_1hot = df[col_1hot]
    df_1hot = pd.get_dummies(df_1hot).astype('category')
    df_others = df.drop(col_1hot,axis=1)
    df = pd.concat([df_others,df_1hot],axis=1)
    new_X = np.array(df.values,dtype='float64')
    
    return new_X


# In[57]:


km = KMeans(n_clusters=8,n_init=5,random_state=seed).fit(bankX) # yes
km_labels = km.labels_
em = EM(n_components=12,covariance_type='diag',n_init=1,warm_start=True,random_state=seed).fit(bankX) # yes
em_labels = em.predict(bankX)

clust_full = addclusters(bankX,km_labels,em_labels)
clust_pca = addclusters(pca_bank,km_labels,em_labels)
clust_ica = addclusters(ica_bank,km_labels,em_labels)
clust_rca = addclusters(rca_bank,km_labels,em_labels)
clust_rfc = addclusters(rfc_bank,km_labels,em_labels)


# In[70]:

# Add the features and rerun NN
# Original, All features
X_train, X_test, y_train, y_test = train_test_split(np.array(clust_full),np.array(bankY), test_size=0.30)
full_est = MLPClassifier(hidden_layer_sizes=(50,), solver='adam', activation='logistic', learning_rate_init=0.01, random_state=seed)
train_samp_full, NN_train_score_full, NN_fit_time_full, NN_pred_time_full = plot_learning_curve(full_est, X_train, y_train,title="Neural Net Default Credit with Clusters: Full")
full_out_add = final_classifier_evaluation(full_est, X_train, X_test, y_train, y_test)


# In[69]:


X_train, X_test, y_train, y_test = train_test_split(np.array(clust_pca),np.array(bankY), test_size=0.30)
pca_est = MLPClassifier(hidden_layer_sizes=(50,), solver='adam', activation='logistic', learning_rate_init=0.01, random_state=seed)
train_samp_pca, NN_train_score_pca, NN_fit_time_pca, NN_pred_time_pca = plot_learning_curve(pca_est, X_train, y_train,title="Neural Net Default Credit with Clusters: PCA")
pca_out_add = final_classifier_evaluation(pca_est, X_train, X_test, y_train, y_test)


# In[68]:


X_train, X_test, y_train, y_test = train_test_split(np.array(clust_ica),np.array(bankY), test_size=0.30)
ica_est = MLPClassifier(hidden_layer_sizes=(50,), solver='adam', activation='logistic', learning_rate_init=0.01, random_state=seed)
train_samp_ica, NN_train_score_ica, NN_fit_time_ica, NN_pred_time_ica = plot_learning_curve(ica_est, X_train, y_train,title="Neural Net Default Credit with Clusters: ICA")
ica_out_add = final_classifier_evaluation(ica_est, X_train, X_test, y_train, y_test)


# In[67]:


X_train, X_test, y_train, y_test = train_test_split(np.array(clust_rca),np.array(bankY), test_size=0.30)
rca_est = MLPClassifier(hidden_layer_sizes=(50,), solver='adam', activation='logistic', learning_rate_init=0.01, random_state=seed)
train_samp_rca, NN_train_score_rca, NN_fit_time_rca, NN_pred_time_rca = plot_learning_curve(rca_est, X_train, y_train,title="Neural Net Default Credit with Clusters: RP")
rca_out_add = final_classifier_evaluation(rca_est, X_train, X_test, y_train, y_test)


# In[66]:


X_train, X_test, y_train, y_test = train_test_split(np.array(clust_rfc),np.array(bankY), test_size=0.30)
rfc_est = MLPClassifier(hidden_layer_sizes=(50,), solver='adam', activation='logistic', learning_rate_init=0.01, random_state=seed)
train_samp_rfc, NN_train_score_rfc, NN_fit_time_rfc, NN_pred_time_rfc = plot_learning_curve(rfc_est, X_train, y_train,title="Neural Net Default Credit with Clusters: RFC")
rfc_out_add = final_classifier_evaluation(rfc_est, X_train, X_test, y_train, y_test)

#%%

feature_add_comparison = pd.concat([full_out_add, pca_out_add, ica_out_add, rca_out_add,rfc_out_add], axis=1)
feature_add_comparison.columns =  ['All features','PCA', 'ICA', 'RP', 'RFC']

# Evaluate new datasets with cluster labels added.

# In[65]:


compare_fit_time(train_samp_full, NN_fit_time_full, NN_fit_time_pca, NN_fit_time_ica, 
                 NN_fit_time_rca, NN_fit_time_rfc, 'Default Credit Dataset')              
compare_pred_time(train_samp_full, NN_pred_time_full, NN_pred_time_pca, NN_pred_time_ica, 
                 NN_pred_time_rca, NN_pred_time_rfc, 'Default Credit Dataset')   
compare_learn_time(train_samp_full, NN_train_score_full, NN_train_score_pca, NN_train_score_ica, 
                 NN_train_score_rca, NN_train_score_rfc, 'Default Credit Dataset')  


# %%
