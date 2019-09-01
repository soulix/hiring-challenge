# -*- coding: utf-8 -*-
"""
Created on Fri May 24 10:04:54 2019
@author: subha
"""

import numpy as np #linear algebra
import pandas as pd #import pandas
import matplotlib.pyplot as plt #eda
#from scipy import stats #imputing missing values
import seaborn as sns # the commonly used alias for seaborn is sns
from sklearn.model_selection import train_test_split
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.feature_selection import RFECV
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score

# =============================================================================
# Solution approach and rationale:-
# Cross Industry Standard Process for Data Mining (CRISP–DM) framework.It involves a series of steps:
# 	1.Business understanding
# 	2.Data understanding
# 	3.Data Preparation & EDA
# 	4.Model Building
# 	5.Model Evaluation
#  6.Model Validation
# 	7.Model Deployment
# 
# =============================================================================
######### Business Understanding #############
#Profiling each candidate to  predict candidates hiring likelihood.
#With 15 predictor variables we need to predict whether a particular candidate will get hired or not.

############# Data Understanding ###################
##### imporitng the dataset ########
hire = pd.read_csv("Hiring_Challenge.csv")

hire.head()
hire.tail()
hire.describe()
hire.info()

#hired is our dependent variable and rest of the variables are independent predictor variables.

############ Data Preparation #####################

#replacing "?" with nan , it will help in imputing missing values.
hire = hire.replace('?', np.nan)

#converting object/ character/ string  data type to numeric wherever required.
hire.describe()
cols_num = ['C2','C14']
hire[cols_num] = hire[cols_num].apply(pd.to_numeric, errors='coerce', axis=1)
hire.describe()

#converting all the categorical values in data frame to lower case
hire = hire.apply(lambda x: x.str.lower() if(x.dtype == 'object') else x)

#checking for duplicate observations and droping duplicates
dup_hire = hire.duplicated() #will return a sereis of boolean ; no duplicated value present
#hire = hire.drop_duplicates()

# Total Number of missing values or NAN in each column 
hire.isnull().sum()
#percentage of missing values each column
round((hire.isnull().sum()/len(hire))*100,2)


#----------------------------------------------------------------------------------------------------------------------------------------------------------------
#outlier treatment for numbmeric values with a UDF centile
def centile(df,num_var):
    var_quantile =  df[num_var].quantile(np.arange(0,1.01,0.01))
    print(var_quantile)

def outlier_treatment(df,num_outlier_treat_var, value):
    df[num_outlier_treat_var] = np.where(df[num_outlier_treat_var] > value ,value ,df[num_outlier_treat_var])
    
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
#set a seaborn style of your taste
sns.set_style("whitegrid")
#outlier treatment & handeling missing values
#C3 : No missing values
sns.boxplot(hire['C3'])
centile(hire,'C3')
outlier_treatment(hire,'C3',15.665) #values are graduatlly increasing 97%ile values comes within 15.665 so we are taking it as a max cuttoff. rest are outlier.



#outlier treatment & handeling missing values
#C8 : No missing values
sns.boxplot(hire['C8'])
centile(hire,'C8')
outlier_treatment(hire,'C8',6.00) #values are graduatlly increasing 90 %ile values comes within 6.00 so we are taking it as a max cuttoff. rest are outlier.


#outlier treatment & handeling missing values
#C11 : No missing values
sns.boxplot(hire['C11'])
centile(hire,'C11')
outlier_treatment(hire,'C11',11.00) #values are graduatlly increasing 95 %ile values comes within 11.00 so we are taking it as a max cuttoff. rest are outlier.

hire.describe()
hire.isnull().sum()
hire.info()

#C15 : No missing values
sns.boxplot(hire['C15'])
centile(hire,'C15')
outlier_treatment(hire,'C15',1489.78) #values are graduatlly increasing 89 %ile values comes within 1489.78 so we are taking it as a max cuttoff. rest are outlier.
hire.describe()


#C14 : Has missing values : Imputing through forward fill and backward fill
sns.boxplot(hire['C14'])
centile(hire,'C14')
outlier_treatment(hire,'C14', 560.00) #values are graduatlly increasing 98 %ile values comes within 560.00 so we are taking it as a max cuttoff. rest are outlier.
hire["C14"] = hire["C14"].ffill().bfill()
hire.isnull().sum()
hire.describe()

#C2 : : Has missing values : Imputing through forward fill and backward fill
sns.boxplot(hire['C2'])
centile(hire,'C2')
outlier_treatment(hire,'C2',59.2427) #values are graduatlly increasing 97 %ile values comes within 60.00 so we are taking it as a max cuttoff. rest are outlier.
hire["C2"] = hire["C2"].ffill().bfill()
hire.isnull().sum()
hire.describe()


#categorical missing value imputation

#C7
sns.countplot(x= hire['C7'], data = hire)
((hire['C7'].value_counts())/len(hire))*100
#57.826087 % values from category 'v' so high propablity that missing value would be the same.
hire['C7'] = hire.fillna(hire['C7'].value_counts().index[0])
hire.isnull().sum()


#C6
sns.countplot(x= hire['C6'], data = hire)
((hire['C6'].value_counts())/len(hire))*100
# c,q,w,i,aa,ff,k,cc has fiar bit of  percentage of distribution, difficult to understand from which category missing values belongs so we are taking using forward fill or backward fill method.
hire['C6'] = hire['C6'].ffill().bfill()
hire.isnull().sum()


#C5
sns.countplot(x= hire['C5'], data = hire)
((hire['C5'].value_counts())/len(hire))*100
# above 75% values from category 'g' so high propablity that missing value would be the same.
hire['C5'] = hire.fillna(hire['C5'].value_counts().index[0])
hire.isnull().sum()


#C4
sns.countplot(x= hire['C4'], data = hire)
((hire['C4'].value_counts())/len(hire))*100
# above 75% values from category 'u' so high propablity that missing value would be the same.
hire['C4'] = hire.fillna(hire['C4'].value_counts().index[0])
hire.isnull().sum()

#C1
sns.countplot(x= hire['C1'], data = hire)
((hire['C1'].value_counts())/len(hire))*100
# only two categories are present with a high percentage of distribution, once again we are using forward fill and backward fill method.
hire['C1'] = hire['C1'].ffill().bfill()
hire.isnull().sum()


########### explorotary data analysis and derived metrics  ###########

hiring_rate = (np.sum(hire['Hired'])/len(hire))*100
hiring_rate

#lets define a User defined Fnction to plot response_rate across categorical variables
def plot_cat(df,cat_var):  
    sns.set_style("whitegrid")
    hire_rate_perc =  pd.DataFrame(((df.groupby(cat_var).Hired.sum())/ np.sum(df['Hired']) )*100)
    hire_rate_perc.columns = ['hiring_rate']
    plt.title('Hired Percentage Across Category')
    sns.barplot(hire_rate_perc.index , y='hiring_rate', data= hire_rate_perc)
    plt.show()


hire.info()
plot_cat(hire,'C1')
plot_cat(hire,'C4')
plot_cat(hire,'C5')
plot_cat(hire,'C6')
plot_cat(hire,'C7')
plot_cat(hire,'C10')
plot_cat(hire,'C9')
plot_cat(hire,'C12')
plot_cat(hire,'C13')


#binning of numeric variables and hiring percentage of each binned group

hire.describe()

num_features = ['C2','C3','C8','C11','C14','C15','Hired']
df_num = hire[num_features]
df_num['C2_bin'] = pd.cut(df_num.loc[:,'C2'], 5, labels=range(1, 6)) #creating five groupd because the range is small here.
df_num['C3_bin'] = pd.cut(df_num.loc[:,'C3'], 5, labels=range(1, 6))  #creating five groupd because the range is small here.
df_num['C8_bin'] = pd.cut(df_num.loc[:,'C8'], 5, labels=range(1, 6)) #creating five groupd because the range is small here.
df_num['C11_bin'] = pd.cut(df_num.loc[:,'C11'], 5, labels=range(1, 6)) #creating five groupd because the range is small here.
df_num['C14_bin'] = pd.cut(df_num.loc[:,'C11'], 5, labels=range(1, 6))  #creating 5 groupd because the range is small here.
df_num['C15_bin'] = pd.cut(df_num.loc[:,'C15'], 5, labels=range(1, 6))  #creating 5 groupd because the range is small here.


plot_cat(df_num,'C2_bin')
plot_cat(df_num,'C3_bin')
plot_cat(df_num,'C8_bin')
plot_cat(df_num,'C11_bin')
plot_cat(df_num,'C14_bin')
plot_cat(df_num,'C15_bin')


#corelation matrix between numeric variables & corelation between numeric variables
# using mendel.corr()
cor = pd.DataFrame(hire[num_features].corr())
round(cor, 2)
# figure size # heatmap
plt.figure(figsize=(10,8))
sns.heatmap(cor, cmap="YlGnBu", annot=True)
# pairplot
sns.pairplot(cor)
plt.show()


############# Scaling , Sampling and Dummy variables creation and feature selection for model#####################################
#standadizing with z-score 

num_feature_std = ['C2','C3','C8','C11','C14','C15']
num_hire =  hire[num_feature_std]
num_hire = pd.DataFrame(stats.zscore(num_hire))
num_hire.columns = ['C2','C3','C8','C11','C14','C15']

#selecting catgorical  variables of more than two levels
cols_cat = ['C1','C4','C5','C6','C7','C9','C10', 'C12','C13']
cat_hire = hire.loc[:,cols_cat]

# Converting t to 1 and f to 0
cat_hire['C9'] = cat_hire['C9'].map({'t': 1, 'f': 0})
cat_hire['C10'] = cat_hire['C10'].map({'t': 1, 'f': 0})
cat_hire['C12'] = cat_hire['C12'].map({'t': 1, 'f': 0})

cat_hire_two_lvl = cat_hire.loc[:,['C9','C10','C12']]

cols_cat_multi_lvl = ['C1','C4','C5','C6','C7','C13']
cat_mul_lvl_cols = cat_hire.loc[:,cols_cat_multi_lvl]


#Creating dummy varibles for categorical variables
# we can use drop_first = True to drop the first column from dummy dataframe.
cat_hire_dummy = pd.get_dummies(cat_mul_lvl_cols,drop_first=True)

hired= pd.DataFrame(hire['Hired'])
#creating the final data set which wre going to throw to the model
#adding dummy data frames of categorical varibles and standadized data frames of numeric varibles
hire_final_df = pd.concat([cat_hire_dummy,cat_hire_two_lvl,num_hire,hired],axis=1)
hire_final_df.info()



#spliting the data set into test and train
#random_state is the seed used by the random number generator, it can be any integer.
# Putting feature variables to X
X = hire_final_df.loc[:, hire_final_df.columns !='Hired']
# Putting dependent or response  variable to y
y = hire_final_df['Hired']
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7 ,test_size = 0.3, random_state= 81)


hire_final_df.info()



###################### Model Building ############################



# Logistic regression model
##Adding a constant column to our dataframe
logm_1 = sm.GLM(y_train,(sm.add_constant(X_train)), family = sm.families.Binomial())
logm_1.fit().summary()

logr = LogisticRegression()
rfe_cv = RFECV(logr, step= 1 , cv = 5)
rfe_cv = rfe_cv.fit(X_train, y_train)

print(rfe_cv.support_)                      # Printing the boolean results
print(rfe_cv.ranking_)  
print(rfe_cv.n_features_)
rfe_cv.grid_scores_


# Creating X_test dataframe with RFECV selected variables
col_rfe_cv = X_train.columns[rfe_cv.support_]
X_train_rfe_cv = X_train[col_rfe_cv]
print(col_rfe_cv)


#Adding a constant column to our dataframe
X_train_rfe_cv = sm.add_constant(X_train_rfe_cv)   
# create a first fitted model
#model 1
logm_1 = sm.GLM(y_train,X_train_rfe_cv, family = sm.families.Binomial())
#Let's see the summary of our first linear model
logm_1.fit().summary()

#--------------------------------------------------------------------------------------------------
# UDF for calculating vif value
def vif_cal(input_data, dependent_col):
    vif_df = pd.DataFrame( columns = ['Var', 'Vif'])
    x_vars=input_data.drop([dependent_col], axis=1)
    xvar_names=x_vars.columns
    for i in range(0,xvar_names.shape[0]):
        y=x_vars[xvar_names[i]] 
        x=x_vars[xvar_names.drop(xvar_names[i])]
        rsq=sm.OLS(y,x).fit().rsquared  
        vif=round(1/(1-rsq),2)
        vif_df.loc[i] = [xvar_names[i], vif]
    return vif_df.sort_values(by = 'Vif', axis=0, ascending=False, inplace=False)



df_rfe_cv =  pd.concat([X_train_rfe_cv, y_train], axis = 1)
df_rfe_cv.info()

#----------------------------------------------------------------------------------------------------


#we are droping the constant bcz we have added constant for stat models.bcz stat models doesnot automatically add a consatnt.
vif_cal(input_data= df_rfe_cv.drop(["const"], axis=1) ,  dependent_col = "Hired")
# VIF is within 3 so no need to eliminate. based on VIF


#droping 'C6_i'
X_train_rfe_cv = X_train_rfe_cv.drop('C6_i', axis = 1)

#model 2
logm_2 = sm.GLM(y_train,X_train_rfe_cv, family = sm.families.Binomial())
logm_2.fit().summary()


#droping 'C12'
X_train_rfe_cv = X_train_rfe_cv.drop('C12', axis = 1)

#model 3
logm_3 = sm.GLM(y_train,X_train_rfe_cv, family = sm.families.Binomial())
logm_3.fit().summary()

#droping 'C6_ff'
X_train_rfe_cv = X_train_rfe_cv.drop('C6_ff', axis = 1)

#model 4
logm_4 = sm.GLM(y_train,X_train_rfe_cv, family = sm.families.Binomial())
logm_4.fit().summary()

#droping 'C6_w'
X_train_rfe_cv = X_train_rfe_cv.drop('C6_w', axis = 1)

#model 5

logm_5 = sm.GLM(y_train,X_train_rfe_cv, family = sm.families.Binomial())
logm_5.fit().summary()

#droping 'C6_c'
X_train_rfe_cv = X_train_rfe_cv.drop('C6_c', axis = 1)

#model 6
logm_6 = sm.GLM(y_train,X_train_rfe_cv, family = sm.families.Binomial())
logm_6.fit().summary()


#droping 'C6_q'
X_train_rfe_cv = X_train_rfe_cv.drop('C6_q', axis = 1)
#model 7
logm_7 = sm.GLM(y_train,X_train_rfe_cv, family = sm.families.Binomial())
logm_7.fit().summary()


#droping 'C6_cc'
X_train_rfe_cv = X_train_rfe_cv.drop('C6_cc', axis = 1)
#model 8
final_model = sm.GLM(y_train,X_train_rfe_cv, family = sm.families.Binomial())
final_model.fit().summary()

############ Making Prediction ##################

# Now let's use our model to make predictions.
# Creating X_test_6 dataframe by dropping variables from X_test
# Let's run the model using the selected variables with sklearn
logsk = LogisticRegression()
X_train_rfe_cv = X_train_rfe_cv.drop('const', axis = 1)
logsk.fit(X_train_rfe_cv, y_train)

X_test_rfe_cv = X_test[col_rfe_cv]
X_test_rfe_cv = X_test_rfe_cv.drop(["C6_i","C12","C6_ff","C6_w","C6_c","C6_q","C6_cc"], axis = 1)

# Predicted probabilities
y_pred = logsk.predict_proba(X_test_rfe_cv)
print(y_pred)

#y_pred.info()
# Converting y_pred to a dataframe which is an array
y_pred_df = pd.DataFrame(y_pred)
# Converting to column dataframe
y_pred_1 = y_pred_df.iloc[:,[1]]

# Let's see the head
y_pred_1.head()

# Converting y_test to dataframe
y_test_df = pd.DataFrame(y_test)

# Removing index for both dataframes to append them side by side 
y_pred_1.reset_index(drop=True, inplace=True)
y_test_df.reset_index(drop=True, inplace=True)
# Appending y_test_df and y_pred_1
y_pred_final = pd.concat([y_test_df,y_pred_1],axis=1)

# Renaming the column 
y_pred_final= y_pred_final.rename(columns={ 1 : 'hiring_prob'})

# Creating new column 'predicted' with 1 if hiring_prob>0.5 else 0
y_pred_final['predicted'] = y_pred_final.hiring_prob.map( lambda x: 1 if x > 0.5 else 0)

# Let's see the head
y_pred_final.head()


######################### Model Evaluation #########################
help(metrics.confusion_matrix)
# Confusion matrix 
confusion = metrics.confusion_matrix( y_pred_final.Hired, y_pred_final.predicted )
confusion

                          #Predicted: not_hired   predicted :hired
# Actual :   # not_hired            107                  9                precision:  TPR = TP/TP+FP
# Actual :   # hired                11                   80                TNR = TN/TN+FN, FPR = 1 - TNR




#Let's check the overall accuracy.
metrics.accuracy_score( y_pred_final.Hired, y_pred_final.predicted)
TN = confusion[0,0] # true Negatives 
TP = confusion[1,1] # true positives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives


# Let's see the sensitivity of our logistic regression model
TP / float(TP+FN) # number of candidates correctly predicted hired/ total number of Actual Hired

# Let us calculate specificity
TN / float(TN+FP) ## number of candidates correctly predicted not hired/ total number of Actual not Hired

# positive predictive value : precision
print (TP / float(TP+FP))

# Negative predictive value
print (TN / float(TN+ FN))



###############ROC Curve #################################

#An ROC curve demonstrates several things:
#It shows the tradeoff between sensitivity and specificity (any increase in sensitivity will be accompanied by a decrease in specificity).
#The closer the curve follows the left-hand border and then the top border of the ROC space, the more accurate the test.
#The closer the curve comes to the 45-degree diagonal of the ROC space, the less accurate the test.

def draw_roc( actual, probs ):
    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,
                                              drop_intermediate = False )
    auc_score = metrics.roc_auc_score( actual, probs )
    plt.figure(figsize=(6, 4))
    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    return fpr, tpr, thresholds


draw_roc(y_pred_final.Hired, y_pred_final.predicted)

#####Finding Optimal Cutoff Point################

######Optimal cutoff probability is that prob where we get balanced sensitivity and specificity


# Let's create columns with different probability cutoffs 
numbers = [float(x)/10 for x in range(10)]
for i in numbers:
    y_pred_final[i]= y_pred_final.hiring_prob.map( lambda x: 1 if x > i else 0)
y_pred_final.head()


# Now let's calculate accuracy sensitivity and specificity for various probability cutoffs.
cutoff_df = pd.DataFrame( columns = ['prob','accuracy','sensi','speci'])
num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for i in num:
    cm1 = metrics.confusion_matrix( y_pred_final.Hired, y_pred_final[i] )
    total1=sum(sum(cm1))
    accuracy = (cm1[0,0]+cm1[1,1])/total1
    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]
print(cutoff_df)



# Let's plot accuracy sensitivity and specificity for various probabilities.
cutoff_df.plot.line(x='prob', y=['accuracy','sensi','speci'])

#from the curve above prob = 0.49 is the Optimal cutoff probability

y_pred_final['final_predicted'] = y_pred_final.hiring_prob.map( lambda x: 1 if x > 0.49 else 0)

y_pred_final.head()

#final accuracy
#Let's check the overall accuracy. #final accuracy 0.9033816425120773
metrics.accuracy_score( y_pred_final.Hired, y_pred_final.final_predicted) 
metrics.confusion_matrix( y_pred_final.Hired, y_pred_final.final_predicted)
draw_roc(y_pred_final.Hired, y_pred_final.final_predicted)
roc_auc_score(y_pred_final.Hired, y_pred_final.final_predicted) #0.9007673361121636

auc = 0.9007673361121636 #getting from the auc curve
print(2*auc -1) #gini score : 0.8015346722243273

print(classification_report(y_pred_final.Hired, y_pred_final.final_predicted))

#             precision    recall  f1-score   support
#
#          0       0.91      0.92      0.91       116
#          1       0.90      0.88      0.89        91
#
#avg / total       0.90      0.90      0.90       207











