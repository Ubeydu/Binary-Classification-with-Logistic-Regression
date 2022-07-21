# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 12:08:51 2022

@author: ubeydullah.keles
"""

# libraries
import numpy as np # used for handling numbers
import pandas as pd # used for handling the dataset
from sklearn.preprocessing import LabelEncoder # used for encoding categorical data
from sklearn.model_selection import train_test_split # used for splitting training and testing data
from sklearn.preprocessing import RobustScaler # used for feature scaling
import matplotlib.pyplot as mp
import seaborn as sb

# read-in the data
vrphomework = pd.read_csv (r'C:\Users\ubeydullah.keles\Documents\şahsi\Data Science assignment Deniz Pekşen\Cleaned up project - for submission -\raw csv delivered by Veripark\VeriparkDSHomework.csv')   
#read the csv file (put 'r' before the path string to address any special characters in the path, such as '\'). 
#Don't forget to put the file name
# at the end of the path + ".csv"

#############################################   EDA and pre-processing  #########################################

results_describe=vrphomework.describe() 
file_name='results_of_describe_method_raw_data.xlsx'
results_describe.to_excel(file_name) 
# Just exported the results to an excel file so I can take a look at the statistical parameters all of the columns.
# I did this just to look at some basic statistical parameters of the features. 
# It was good to see the results in an excel sheet, because, that way I saw that the following features had exactly the same
# statistical parameters: Feature_0, Feature_40	,Feature_41,Feature_42,Feature_43,Feature_44,Feature_45,Feature_46,
#Feature_47,Feature_48,Feature_49  (For instace, the mean values is the same for all of them: 14.12729174)
# This is good hint. Probably, all these 11 columns have absolutely the same values so I can drop 10 of them 
#without losing predictive power.
# But, I will do this later in the code when I have more concrete evidence for the sameness of these columns. (Line: ???)

# Now, I will call info method on my dataframe just to see a summary of it
print(vrphomework.info()) 
# The things that I see by looking at the result:
    # 1- There are 569 entries (rows)
    # 2- There are 52 features (columns)
    # 3- These are data types we have: dtypes: float64(48), int64(1), object(3)
    # 4- There are 3 features that are "object" datatype: Feature_1, Feature_2, Feature_50
    # 5- In the "Non-Null Count" column, we see that; there are 10 "0 non-null" columns, that means these columns
    # are completely empty. So, we can drop them!
    # 6- Also, in the same column, we see that all the rest of the columns have "569 non-null" values. And, since this
    # is equal to the number of entries (rows) that tells that none of the columns have any missing value. 
    # So, we don't need to call any imputer method on our data. 
    
# But, to be %100 percent sure that we don't need to call any imputer method, we can do the following:
print(vrphomework.isnull().sum()/len(vrphomework)*100) #percentage of missing values in each variable
# As you can see, there are 10 columns that have 100 percent missing values, in other words, they are empty. 
# These features are: Feature_30, Feature_31, Feature_32,, Feature_33, Feature_34,
# Feature_35, Feature_36, Feature_37, Feature_38, Feature_39

#Since, these empty columns possess no predictive value, I can drop them from my dataset. 
#I will do it by the following code:
    
vrphomework.dropna(how='all', axis=1, inplace=True) #I set the inplace parameter to 'True' because I want to 
#change/edit my dataset. I don't want to return a new dataset. 

# Great. Now, you can see in the variable explorer pane that the size of vrphomework is (569, 42)

# As we noticed above when we called the info method on our dataframe, we have three 
# features that are object datatype. They are the following:
#  #   Column      Non-Null Count  Dtype    
#  1   Feature_1   569 non-null    object 
#  2   Feature_2   569 non-null    object
#  40  Feature_50  569 non-null    object

# They are categorical variables because their data type is "object".
# I need to convert them into numerical values so I can apply classification algorithms on my dataset.
# Classification algorithms work with numerical values, not with objects(strings)
# But, which encoding (conversion) method should I use: Label encoding or one-hot encoding?
# I will use label encoding because our data is ordinal.Somewhat, they imply hierarchy. As in, Low is smaller than Medium.
# 1   Feature_1   569 non-null    object : Low, Medium, High
# 2   Feature_2   569 non-null    object : Good, Bad
# 40  Feature_50  569 non-null    object : Low

# I can use the following code snippet to find all the features with data type 'object'
objList = vrphomework.select_dtypes(include = "object").columns
print (objList)

# Now I am going to loop-through my list 'objList' to convert the values of these columns 
#into numeric values using LabelEncoder

#Label Encoding for object to numeric conversion
le = LabelEncoder()

print("Encoding Start") # Label'ları excelde kontrol et. 
for feat in objList:
    print(vrphomework[feat])
    vrphomework[feat] = le.fit_transform(vrphomework[feat].astype(str))
    print(vrphomework[feat])
print("Encoding End")

#Great, I converted all values in my dataset to numerical values. 

# Let's call "var" method on the dataframe and take a look at the variances of my features
print(vrphomework.var())  #function call to check variances of all columns in the dataset.
# I noticed something really useful here. Feature_50 has "0" variance. That shows me that this column has identical values.
# All the data points are the same value. So, I can drop this column because it doesn't have any predictive value. 

vrphomework.drop(['Feature_50'], inplace=True,axis = 1)
# Great, now the dataframe has 41 columns. (see the variable explorer)

# Another important thing I noticed after checking the variances of all columns is that we have 11 features
# that have exactly the same variance (var=12.418920). That tells me they are probably identical. 
# To be %100 sure, I can do a couple of things. 
# First. I will create a pairplot and see how these features correlate. If we see that
# they are perfectly correlated with each other then we can safely assume that they are equal.
# And, that way we can safely drop these features except one of them. I don't want to lose all of them
# because that will mean losing data. 
# But, if you think creating a pairplot and looking at their correlations is not enough, then I have a more precise solution, 
#which is to use a Pandas method that drops identical(duplicate) columns. --> df.T.drop_duplicates().T
#First, I will create a pairlplot and then apply the Pandas method to drop identical columns. 


mp.figure()
sb.pairplot(vrphomework,vars=['Feature_0','Feature_40','Feature_41','Feature_42', 'Feature_43', 'Feature_44', 
'Feature_45','Feature_46', 'Feature_47', 'Feature_48', 'Feature_49', 'TARGET'])
mp.show()

# As I have suspected, all of these 11 features are identical because the plot shows perfect correlation between them.
# You may check out the png file titled "11 features with same variance plot.png" in the folder.
# Now, I will go ahead and use the Pandas method to drop the identical columns. 
# Right now, we have 41 columns in 'vrphomework', if my method call works the way I expect, it'll drop 
# 10 out of 11 identical columns and my dataset will become a 31-column dataset. Let's try.

vrphomework=vrphomework.T.drop_duplicates(keep='first').T
#Worked perfectly! The method dropped the identical (and so useless) features except for the first one (Feature_0)

# Before using scikit-learn's train_test_split() to split my data into train-test subsets, I will 
# check the magnitudes of my features to see if I need to do feature scaling. This is a very critical issue
# because if I don't scale my features, then, the features with high values (magnitudes) will dominate 
# my classification model. This is due to the Euclidean distance concept that works behind.
# I don't need to run any methdo or function to inspect this, because, the excel file that
# I created running the code in the lines 25-26-27 show me that there is a real serious magnitude 
# difference between my features. 
# For instance, 'Feature_3' has a mean of 654.8891037 while 'Feature_4' has a mean of 0.096360281, wow!
# That is a huge gap.


# But, before scaling my data, I'd like to see the situation with outliers. It'll be helpful to know this,
# because, for instance, if some of the features have extreme outliers, then it'll be good to use
# RobustScaler, as opposed to min-max scaler or standard scaler.
# To analyze my features in regard to the outliers, I will go ahead and group them into features that have 
# close values (magnitude wise)
# This is really easy to see. I will call the describe method on the dataframe and get the 
# the results to an excel file and inspect the 'mean' values of the features. 
results_describeV2=vrphomework.describe() 
file_name='results_of_describe_method_V2.xlsx'
results_describeV2.to_excel(file_name) 

#Now, I will create variable groups and then use these groups to create box plots, so I can visually analyze 
# outliers. 

df_1 = vrphomework[['Feature_0', 'Feature_13','Feature_20', 'Feature_21']]
df_2 = vrphomework[['Feature_3', 'Feature_22', 'Feature_23']]
df_3 = vrphomework[['Feature_1','Feature_2','Feature_4','Feature_5','Feature_6','Feature_7','Feature_8',
                    'Feature_9','Feature_10','Feature_11','Feature_12','Feature_14','Feature_15','Feature_16',
                    'Feature_17','Feature_18','Feature_19','Feature_24',
                    'Feature_25','Feature_26','Feature_27','Feature_28','Feature_29']]

box_plot_df_1 = sb.boxplot(data=df_1, orient="h", palette="Set2") # We have some serious outliers in 
# Feature_13
box_plot_df_2 = sb.boxplot(data=df_2, orient="h", palette="Set2") # We have some serious outliers in 
# Feature_3 and Feature_23
box_plot_df_3 = sb.boxplot(data=df_3, orient="h", palette="Set2") # We have some serious outliers in 
# Feature_12. This boxplot is harder to analyze visually since there are a bit too many features.
# For the level of complexity we want to achieve, this boxplot analysis is sufficient. 
# If it is absolutely required, we could for some other quantitative techniques to further analyze. 
# But, our analysis is already sufficient because we can see our data has univariate outliers in it.
# And, this helps us decide which type of scaler method we need to apply. We need a method that is good 
# with data with outliers, and that is RobustScaler. 


# Now, it is time to scale the data. 
# But, right before this, I will split the attributes into independent and dependent attributes
X = vrphomework.iloc[:, :-1].values  # attributes to determine dependent variable -independent variables- / Class
Y = vrphomework.iloc[:, -1].values # dependent variable / Class
#By executing the two lines above, I split the dataset into the dependent variable (Y) 
#and the independent variables -attributes- (X) 
# I did this because I want to apply the scaler method to my independent variables only (features)

# Let's apply RobustScaler method
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
X_scaled.shape
#print(X_scaled) # You can see the data scaled if you uncomment this print line. 
# Done with scaling!

#########################   Splitting the data into train and test subsets  ############################

x_train, x_test, y_train, y_test = train_test_split(X_scaled, Y, test_size=0.3, random_state=42) # I used random_state parameter
# so that I get same random output for each function call.  DONE!

#########################   Fitting / Training the model  ############################

# In this stage, I will try an approach that is mentioned in this article:
    # https://towardsdatascience.com/top-machine-learning-algorithms-for-classification-2197870ff501
    # The thing that I liked about this approach is that, instead of relying on assumptions for 
    # choosing a classification algorithm, it creates a model list and then appends the default state
    # of 6 algorithms and iterates through all of them. That way, we get to compare the results
    # of all them! After, seeing the results and choosing one or two best performing models,
    # I can go ahead and get more metrics if I want to. Let's see what we will get.


# First, let's import the models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

#Now, I will create a model list so I can iterate through all of the models to train, test, predict and evaluate.

model_pipeline = []
model_pipeline.append(LogisticRegression())
# I decided to remove 'liblinear' from the solver parameter, because it is a bit too advanced for me. 
# But, I may come back to it and try another solver depending on performance metrics. 
model_pipeline.append(SVC())
model_pipeline.append(KNeighborsClassifier())
model_pipeline.append(DecisionTreeClassifier())
model_pipeline.append(RandomForestClassifier())
model_pipeline.append(GaussianNB())

# Next, I will go ahead and import metrics module from sklearn so I can assess prediction errors of different sorts.
from sklearn import metrics

#And now, I will import the following; classification report,precision_score, recall_score, f1_score.
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
# And now, I will create a list of my models to use it later to create plots of confusion matrix heatmaps
# as well as to create a table to look at accuracy and AUC scores.

model_list = ['Logistic Regr.', 'SVM', 'KNN', 'Decision Tree','Random Forest', 'Naive Bayes']
# And, some other lists so that later I can append all my scores into them and print to compare.
acc_list = [] # for accuracy scores
auc_list = [] # For AUC scores (the areas under the ROC curves)
cm_list = [] # Ratios in the confusion matrices
prec_score_list = []  # List for saving precision scores
recall_score_list = [] # list for saving recall scores
f1_score_list = []  # list for f1 scores.


# Now, I will write a for loop to train all my models (in other words, to fit my data to all the 6 algorithms):

for model in model_pipeline:
    model.fit(x_train, y_train)  # fitting my training data to the model
    y_pred = model.predict(x_test) # predicting target classes
    acc_list.append(metrics.accuracy_score(y_test, y_pred)) # Filling my acc_list with accuracy scores
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred) # Calculating false positive rate and true positive rate
    auc_list.append(round(metrics.auc(fpr,tpr),2)) # Getting the AUC values from our models
    cm_list.append(confusion_matrix(y_test, y_pred))  # Getting the values for confusion matrix into a list
    prec_score_list.append(precision_score(y_test, y_pred))
    recall_score_list.append(recall_score(y_test, y_pred))
    f1_score_list.append(f1_score(y_test, y_pred))
    
# Now, we are going to visualize the confusion matrix values by using 'cm_list'
# plotting the confusion matrices for all 6 algorithms by iterating through my score list 
fig = mp.figure(figsize = (18,10))
for i in range(len(cm_list)):
    cm = cm_list[i]
    model = model_list[i]
    sub = fig.add_subplot(2, 3, i+1).set_title(model)
    cm_plot = sb.heatmap(cm, annot=True, cmap = 'Blues_r', fmt='g')
    cm_plot.set_xlabel('Predicted Values')
    cm_plot.set_ylabel('Actual Values')

# COOOLL! I got the matrices. A quick inspection of the plot show me that the best performing one
# is the Logistic Regression model, because total of FN and FP are 3. That is the smallest value of all 6 matrices.
# After logistic regression comes the random forest and SVM algorithms, their FN + FP values are equal to 6, that is the 
# smallest second value out of the 6. 
# But, I will go ahead and look at other metrics too.

# Now, let's look at several score metrics:

result_df = pd.DataFrame({'Model': model_list, 'Accuracy': acc_list, 'AUC': auc_list,
                          'Precision': prec_score_list, 'Recall': recall_score_list, 'f1': f1_score_list})
print(result_df)
# Same thing; logistic regression algorithm have the highest scores on ALL metrics. By far, it is the
# top performing algorithm. 

################################# CONCLUSION ABOUT ALGORITHM PERFORMANCE ####################

# As we have observed above, the logistic regression algorithm performed the best 
# with our data. It has the highest predictive power. 

# Now, finally we can fit our data to logistic regression model and get our model variable.

logresmodel = LogisticRegression()
logresmodel.fit(x_train, y_train)

# Now, let's get the predictions

y_pred = logresmodel.predict(x_test)

# Calculate the confusion matrix

conf_matrix = confusion_matrix(y_test, y_pred)

# Let's print it using matplotlib

fig, ax = mp.subplots(figsize=(5, 5))
ax.matshow(conf_matrix, cmap=mp.cm.Oranges, alpha=0.3)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
 
mp.xlabel('Predictions', fontsize=18)
mp.ylabel('Actuals', fontsize=18)
mp.title('Confusion Matrix', fontsize=18)
mp.show()
     
# Nice! We can easily see that this confusion matrix is exactly the same like the one we saw in the previous plot. 
# It is the same with the 1st plot of the big plot above. (Logistic Regression) 

# And, finally we can also print out the performance metrics of our model.

print(metrics.classification_report(y_test, y_pred))  # Great performance indeed! f1 score for '0' is 0.98 and 0.99 for '1'.
# Precision, recall, and accuracy values are really high too.   


# And, finally, let's plot the ROC

sb.set_style('darkgrid')
preds_train = logresmodel.predict(x_train)
 # calculate prediction probability
prob_train = np.squeeze(logresmodel.predict_proba(x_train)[:,1].reshape(1,-1))
prob_test = np.squeeze(logresmodel.predict_proba(x_test)[:,1].reshape(1,-1))
 # false positive rate, true positive rate, thresholds
fpr1, tpr1, thresholds1 = metrics.roc_curve(y_test, prob_test)
fpr2, tpr2, thresholds2 = metrics.roc_curve(y_train, prob_train)
 # auc score
auc1 = metrics.auc(fpr1, tpr1)
auc2 = metrics.auc(fpr2, tpr2)
mp.figure(figsize=(8,8))
 # plot auc 
mp.plot(fpr1, tpr1, color='blue', label='Test ROC curve area = %0.2f'%auc1)
mp.plot(fpr2, tpr2, color='green', label='Train ROC curve area = %0.2f'%auc2)
mp.plot([0,1],[0,1], 'r--')
mp.xlim([-0.1, 1.1])
mp.ylim([-0.1, 1.1])
mp.xlabel('False Positive Rate', size=14)
mp.ylabel('True Positive Rate', size=14)
mp.legend(loc='lower right')
mp.show()    


import pickle
#dump : put the data of the object in a file
pickle.dump(logresmodel, open(r"C:\Users\ubeydullah.keles\Documents\şahsi\Data Science assignment Deniz Pekşen\Cleaned up project - for submission -\pickle file\logres.pickle", "wb")) 


logresmodel2=pickle.load(open(r"C:\Users\ubeydullah.keles\Documents\şahsi\Data Science assignment Deniz Pekşen\Cleaned up project - for submission -\pickle file\logres.pickle", "rb"))
print(logresmodel2)
# logresmodel.close()







