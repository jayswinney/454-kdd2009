##  upselling_svm_jdion

##################################################################################################
###       COMMENT FOR SUPPORT VECTOR MACHINE MODEL FOR UPSELLING
##################################################################################################
#
#
#The variable selection process started with the variables selected by the random forrest as being
#the strongest and then do to performance issues, was reduced to only the 4 strongest variables.
#
#Training the model was also taking as long as an hour for the training data set as well as
#even a small subset of variables, so, a separate dataset of 20% of the observations was created
#for the SVM model.
#
#When the results were applied to the test dataset, the ROC curve shows results that are slightly
#better than a random model. The model herein relies on the radial kernel with a cost of 10.
#linear, polynomial, radial and sigmoid methods were all attempted without signficant improvement.
#An automate method was explored for identifying the appropriate level of C (cost) and gamma
#however due to the size of the data set was not successful.  Cost was manually adjusted using a range
#from 1 indicating the narrowest margin on the hyperplane to 10000 indicating a very wide hyperplane
#with many missclassifications.
#
##################################################################################################
##################################################################################################

###ADD LIBRARIES AND FUNCTION
library(kernlab)
library(e1071)
library(ROCR)

rocplot = function (pred, truth, ...){
  predob = prediction(pred,truth)
  perf = performance (predob, "tpr", "fpr")
  plot(perf,...)}


###   SET DIRECTORY PATH:
dirs <- c('c:/Users/jay/Dropbox/pred_454_team',
          'c:/Users/uduak/Dropbox/pred_454_team',
          'C:/Users/Sandra/Dropbox/pred_454_team',
          '~/Manjari/Northwestern/R/Workspace/Predict454/KDDCup2009/Dropbox',
          'C:/Users/JoeD/Dropbox/pred_454_team'
          )

for (d in dirs){
  if(dir.exists(d)){
    setwd(d)
  }
}

###   READ DATA FILES:


###   CREATE TRAINING, TEST AND TINY DATA SETS (TINY INCREASED TO 20% AND USED FOR TRAINING)
source('data_transformations/impute_0.r')



### SVM MODEL TRAINED WITH TINY (20% OF DATASET)

svmfit.opt = svm(upselling ~ Var126
+ Var226 + Var204 + Var28
#+ Var113 + Var206 + Var153 + Var57 + Var216
#+ Var81 + Var133 + Var125 + Var197 + Var163 + Var38 + Var119 + Var6 + Var76 + Var228 + Var25 + Var134 + Var73
#+ Var22 + Var94 + Var149 + Var13 + Var160 + Var140 + Var21 + Var211
, data = tiny, kernel = 'radial', cachesize = 2000, cost=10, decision.values = T)

fitted = attributes(predict(svmfit.opt, test, decision.values = TRUE))$decision.values

rocplot(fitted, test$upselling , main = "ROC Curve for Upselling")

##  appetency_svm_jdion

##################################################################################################
###       COMMENT FOR SUPPORT VECTOR MACHINE MODEL FOR CHURN
##################################################################################################
#
#
#The variable selection process started with the variables selected by the random forrest as being
#the strongest and then do to performance issues, was reduced to only the 5 strongest variables.
#
#Training the model was also taking as long as an hour for the training data set as well as
#even a small subset of variables, so, a separate dataset of 20% of the observations was created
#for the SVM model.
#
#When the results were applied to the test dataset, the ROC curve shows results that don't differ
#signifcantly from a random model. The model herein relies on the radial kernel with a cost of 10.
#linear, polynomial, radial and sigmoid methods were all attempted without signficant improvement.
#
##################################################################################################
##################################################################################################

###ADD LIBRARIES AND FUNCTION
library(kernlab)
library(e1071)
library(ROCR)

rocplot = function (pred, truth, ...){
  predob = prediction(pred,truth)
  perf = performance (predob, "tpr", "fpr")
  plot(perf,...)}


###   SET DIRECTORY PATH:
dirs <- c('c:/Users/jay/Dropbox/pred_454_team',
          'c:/Users/uduak/Dropbox/pred_454_team',
          'C:/Users/Sandra/Dropbox/pred_454_team',
          '~/Manjari/Northwestern/R/Workspace/Predict454/KDDCup2009/Dropbox',
          'C:/Users/JoeD/Dropbox/pred_454_team'
          )

for (d in dirs){
  if(dir.exists(d)){
    setwd(d)
  }
}


source('data_transformations/impute_0.r')


###   CREATE TRAINING, TEST AND TINY DATA SETS (TINY INCREASED TO 20% AND USED FOR TRAINING)




### SVM MODEL TRAINED WITH TINY (20% OF DATASET)

svmfit.opt = svm(appetency ~ Var204 + Var226 + Var126 + Var57 + Var113
#+ Var6 + Var81 + Var125 + Var153 + Var28 + Var216 + Var119 + Var134
#+ Var123 + Var94 + Var76 + Var133 + Var22 + Var206 + Var140 + Var197 + Var73 + Var160 + Var25 + Var13
#+ Var38 + Var21 + Var163 + Var112 + Var83
, data = tiny, kernel = 'radial', cachesize = 2000, cost=10, decision.values = T)


fitted = attributes(predict(svmfit.opt, test, decision.values = TRUE))$decision.values
rocplot(fitted, test$appetency , main = "ROC Curve for Appetency")

###SAVE RData File

save(list = c('svmfit.opt'),
 file = "models/upsell/upsell_svm_jdion.RData")
