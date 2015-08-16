##  churn_cp_3_sandra-B

library(knitr)
library(lattice)
library(plyr)
library(dplyr)
library(tidyr)
library(grid)
library(gridExtra)
library(ROCR)
library(e1071)
library(ggplot2)
library(data.table)
library(glmnet)
library(randomForest)
library(rpart)
library(rpart.plot)
library(rattle)
library(GA)

##  PACKAGES NEEDED FOR THE CONFUSION MATRIX
#install.packages("lme4")
library(lme4)
#install.packages("caret")
library(caret)

# choose a script to load and transform the data as of 8/12/2015 by Jay
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
#head(df)

#Data set containing predictors and churn only
train <- select(train, -appetency, -upsell)
test <- select(test, -appetency, -upsell)
tiny <- select(tiny, -appetency, -upsell)



########################################
### RANDOM FOREST FITTING - FULL MODEL
########################################

set.seed(123)
churn_rf_fit_cp3 <- randomForest(factor(churn) ~.,
                                 data = train,
                                 ntree = 10, nodesize = 10, importance = TRUE)

summary(churn_rf_fit_cp3)

print(churn_rf_fit_cp3)  # USE

plot(churn_rf_fit_cp3,main="Random Forest Error Rate for churn_rf_fit_cp3")
legend("top", colnames(churn_rf_fit_cp3$err.rate),col=1:4,cex=0.8,fill=1:4)

varImpPlot(churn_rf_fit_cp3, sort=TRUE, n.var=min(30, nrow(churn_rf_fit_cp3$importance)),
           type=NULL, class=NULL, scale=TRUE,
           main=deparse(substitute(churn_rf_fit_cp3)))

churn_rf_fit_cp3.varImp <- varImpPlot(churn_rf_fit_cp3, sort=TRUE, n.var=min(30, nrow(churn_rf_fit_cp3$importance)),
           type=NULL, class=NULL, scale=TRUE,
           main=deparse(substitute(churn_rf_fit_cp3)))
#churn_rf_fit_cp3.varImp
#setwd("C:/Users/Sandra/Dropbox/PREDICT 454 - Advanced Modeling - Bhatti/KDD 2009 Cup CRM - Project Files/Checkpoint 3 Files")
#write.csv(churn_rf_fit_cp3.varImp,"churn_rf_fit_cp3.varImp.csv")

### EVALUATE ON TRAIN DATA:
# Confustion Matrix:
train$Pred <- predict(churn_rf_fit_cp3, type="class", newdata=train)
churn.pred.rp.conf.matrix.train <- confusionMatrix(data=train$Pred, reference=train$churn, positive="1")
print(churn.pred.rp.conf.matrix.train)
# AUC:
rf.churn.pred.train.prob <- predict(churn_rf_fit_cp3,train,type = 'prob')[,2]
rf.churn.pred.train.prob.prediction <- prediction(rf.churn.pred.train.prob, train$churn)
#pred.train
perfAUC.train <- performance(rf.churn.pred.train.prob.prediction,'auc')
#perfAUC.train
AUC.Train <- as.numeric(perfAUC.train@y.values)
AUC.Train

### EVALUATE ON TEST DATA:
# Confustion Matrix:
test$Pred <- predict(churn_rf_fit_cp3, type="class", newdata=test)
churn.pred.rp.conf.matrix.test <- confusionMatrix(data=test$Pred, reference=test$churn, positive="1")
print(churn.pred.rp.conf.matrix.test)
# AUC:
rf.churn.pred.test.prob <- predict(churn_rf_fit_cp3,test,type = 'prob')[,2]
rf.churn.pred.test.prob.prediction <- prediction(rf.churn.pred.test.prob, test$churn)
#pred.test
perfAUC.test <- performance(rf.churn.pred.test.prob.prediction,'auc')
#perfAUC.test
AUC.Test <- as.numeric(perfAUC.test@y.values)
AUC.Test

# save .RData model and preidctions output
save(list = c('churn_rf_fit_cp3', 'rf.churn.pred.test.prob.prediction'),
     file = 'models/churn/churn_rf_fit_cp3_sandra.RData')



########################################
### RANDOM FOREST FITTING - WITH 47 IMPORTANT VARIABLES MODEL
########################################

set.seed(123)
churn_rf_fit_cp3.sub <- randomForest(factor(churn) ~ Var226 +
                                   Var204 +
                                   Var113 +
                                   Var57 +
                                   Var216 +
                                   Var119 +
                                   Var76 +
                                   Var126 +
                                   Var28 +
                                   Var38 +
                                   Var13 +
                                   Var133 +
                                   Var6 +
                                   Var134 +
                                   Var206 +
                                   Var25 +
                                   Var228 +
                                   Var73 +
                                   Var153 +
                                   Var81 +
                                   Var197 +
                                   Var125 +
                                   Var94 +
                                   Var160 +
                                   Var21 +
                                   Var205 +
                                   Var112 +
                                   Var24 +
                                   Var109 +
                                   Var83 +
                                   Var163 +
                                   Var149 +
                                   Var140 +
                                   Var22 +
                                   Var189 +
                                   Var207 +
                                   Var218 +
                                   Var123 +
                                   Var74 +
                                   Var132 +
                                   Var85 +
                                   Var229 +
                                   Var199 +
                                   Var51_missing +
                                   Var198 +
                                   Var195 +
                                   Var40,
                                 data = train,
                                 ntree = 10, nodesize = 10, importance = TRUE)

summary(churn_rf_fit_cp3.sub)

print(churn_rf_fit_cp3.sub)  # USE

plot(churn_rf_fit_cp3.sub,main="Random Forest Error Rate for churn_rf_fit_cp3.sub")
legend("top", colnames(churn_rf_fit_cp3.sub$err.rate),col=1:4,cex=0.8,fill=1:4)

varImpPlot(churn_rf_fit_cp3.sub, sort=TRUE, n.var=min(30, nrow(churn_rf_fit_cp3.sub$importance)),
           type=NULL, class=NULL, scale=TRUE,
           main=deparse(substitute(churn_rf_fit_cp3.sub)))

churn_rf_fit_cp3.varImp.sub <- varImpPlot(churn_rf_fit_cp3.sub, sort=TRUE, n.var=min(30, nrow(churn_rf_fit_cp3.sub$importance)),
                                      type=NULL, class=NULL, scale=TRUE,
                                      main=deparse(substitute(churn_rf_fit_cp3.sub)))
#churn_rf_fit_cp3.varImp.sub
#setwd("C:/Users/Sandra/Dropbox/PREDICT 454 - Advanced Modeling - Bhatti/KDD 2009 Cup CRM - Project Files/Checkpoint 3 Files")
#write.csv(churn_rf_fit_cp3.varImp.sub,"churn_rf_fit_cp3.varImp.sub.csv")

### EVALUATE ON TRAIN DATA:
# Confustion Matrix:
train$Predsub <- predict(churn_rf_fit_cp3.sub, type="class", newdata=train)
churn.pred.rp.conf.matrix.train.sub <- confusionMatrix(data=train$Predsub, reference=train$churn, positive="1")
print(churn.pred.rp.conf.matrix.train.sub)
# AUC:
rf.churn.pred.train.prob.sub <- predict(churn_rf_fit_cp3.sub,train,type = 'prob')[,2]
rf.churn.pred.train.prob.prediction.sub <- prediction(rf.churn.pred.train.prob.sub, train$churn)
#rf.churn.pred.train.prob.sub
perfAUC.train.sub <- performance(rf.churn.pred.train.prob.prediction.sub,'auc')
#perfAUC.train.sub
AUC.Train.sub <- as.numeric(perfAUC.train.sub@y.values)
AUC.Train.sub

### EVALUATE ON TEST DATA:
# Confustion Matrix:
test$Predsub <- predict(churn_rf_fit_cp3.sub, type="class", newdata=test)
churn.pred.rp.conf.matrix.test.sub <- confusionMatrix(data=test$Predsub, reference=test$churn, positive="1")
print(churn.pred.rp.conf.matrix.test.sub)
# AUC:
rf.churn.pred.test.prob.sub <- predict(churn_rf_fit_cp3.sub,test,type = 'prob')[,2]
rf.churn.pred.test.prob.prediction.sub <- prediction(rf.churn.pred.test.prob.sub, test$churn)
#rf.churn.pred.test.prob.prediction.sub
perfAUC.test.sub <- performance(rf.churn.pred.test.prob.prediction.sub,'auc')
#perfAUC.test.sub
AUC.Test.sub <- as.numeric(perfAUC.test.sub@y.values)
AUC.Test.sub

#plot multiple ROC curves
library(ROCR)
perf1 <- performance(rf.churn.pred.test.prob.prediction, "tpr", "fpr")
perf2 <- performance(rf.churn.pred.test.prob.prediction.sub, "tpr", "fpr")

# if both full model and sub or partial models have been run:
plot(perf1,col='red', main="Random Forest ROC for churn. Full model red line; partial model blue line")
plot(perf2, add = TRUE, col='blue')
abline(0,1,lty=8,col='grey')

# if only the sub or partial model has been run:
plot(perf2,col='red', main="Random Forest ROC for churn. Partial model with 47 perdictors red line")
abline(0,1,lty=8,col='grey')


# save .RData model and preidctions output
save(list = c('churn_rf_fit_cp3.sub', 'rf.churn.pred.test.prob.prediction.sub'),
     file = 'models/churn/churn_rf_fit_cp3_sandra.sub.RData')


##########################################################
###   SUPPORT VECTOR MACHINE
##########################################################
#install.packages("e1071")
require(e1071)
library("ROCR")

rocplot = function (pred, truth, ...){
  predob = prediction(pred,truth)
  perf = performance (predob, "tpr", "fpr")
  plot(perf,...)}

###  FITTING MODEL WITH 47 IMPORTANT VARIABLES
# FIT MODEL WITH PROBABILITY
set.seed(123)
svm.churn.fit.prob <- svm(as.factor(churn) ~ Var226 +
                            Var204 +
                            Var113 +
                            Var57 +
                            Var216 +
                            Var119 +
                            Var76 +
                            Var126 +
                            Var28 +
                            Var38 +
                            Var13 +
                            Var133 +
                            Var6 +
                            Var134 +
                            Var206 +
                            Var25 +
                            Var228 +
                            Var73 +
                            Var153 +
                            Var81 +
                            Var197 +
                            Var125 +
                            Var94 +
                            Var160 +
                            Var21 +
                            Var205 +
                            Var112 +
                            Var24 +
                            Var109 +
                            Var83 +
                            Var163 +
                            Var149 +
                            Var140 +
                            Var22 +
                            Var189 +
                            Var207 +
                            Var218 +
                            Var123 +
                            Var74 +
                            Var132 +
                            Var85 +
                            Var229 +
                            Var199 +
                            Var51_missing +
                            Var198 +
                            Var195 +
                            Var40,
                          data=train,probability=TRUE)  # works
svm.churn.pred.prob.train <- predict(svm.churn.fit.prob,train,probability=TRUE)  # works
svm.churn.pred.prob.test <- predict(svm.churn.fit.prob,test,probability=TRUE)  # works

# FIT MODEL WITH DECISION VALUES
svm.churn.fit.val <- svm(as.factor(churn) ~ Var226 +
                           Var204 +
                           Var113 +
                           Var57 +
                           Var216 +
                           Var119 +
                           Var76 +
                           Var126 +
                           Var28 +
                           Var38 +
                           Var13 +
                           Var133 +
                           Var6 +
                           Var134 +
                           Var206 +
                           Var25 +
                           Var228 +
                           Var73 +
                           Var153 +
                           Var81 +
                           Var197 +
                           Var125 +
                           Var94 +
                           Var160 +
                           Var21 +
                           Var205 +
                           Var112 +
                           Var24 +
                           Var109 +
                           Var83 +
                           Var163 +
                           Var149 +
                           Var140 +
                           Var22 +
                           Var189 +
                           Var207 +
                           Var218 +
                           Var123 +
                           Var74 +
                           Var132 +
                           Var85 +
                           Var229 +
                           Var199 +
                           Var51_missing +
                           Var198 +
                           Var195 +
                           Var40,
                         data=train,decision.values=TRUE) # works

svm.churn.pred.val.train <- predict(svm.churn.fit.val,train,decision.values=TRUE) # works
svm.churn.pred.val.test <- predict(svm.churn.fit.val,test,decision.values=TRUE) # works


## EVALUATE SVM WITH TRAIN DATA:
# Confusion Matrix:
confusionMatrix(svm.churn.pred.prob.train, sample(train$churn), positive="1")
confusionMatrix(svm.churn.pred.val.train, sample(train$churn), positive="1")

# AUC:
svm.churn.train.pred.dec.vals = attributes(svm.churn.pred.val.train)$decision.values
rocplot(svm.churn.train.pred.dec.vals, train$churn , main = "ROC Curve for svm.churn.train.pred.dec.vals")
svm.churn.auc.train.perf <- performance(prediction(svm.churn.train.pred.dec.vals,train$churn),'auc')
svm.churn.auc.train.perf
AUC.Train <- as.numeric(svm.churn.auc.train.perf@y.values)
AUC.Train

## EVALUATE SVM WITH TEST DATA:
# Confusion Matrix:
confusionMatrix(svm.churn.pred.prob.test, sample(test$churn), positive="1")
confusionMatrix(svm.churn.pred.val.test, sample(test$churn), positive="1")

# AUC:
svm.churn.test.pred.dec.vals = attributes(svm.churn.pred.val.test)$decision.values
rocplot(svm.churn.test.pred.dec.vals, test$churn , main = "ROC Curve for svm.churn.test.pred.dec.vals")
svm.churn.auc.test.perf <- performance(prediction(svm.churn.test.pred.dec.vals,test$churn),'auc')
svm.churn.auc.test.perf
AUC.Test <- as.numeric(svm.churn.auc.test.perf@y.values)
AUC.Test

# save .RData model and preidctions output
save(list = c('svm.churn.fit.prob', 'svm.churn.pred.prob.test'),
     file = 'models/churn/churn_svm_fit_cp3_sandra.prob.RData')

# save .RData model and preidctions output
save(list = c('svm.churn.fit.val', 'svm.churn.pred.val.test'),
     file = 'models/churn/churn_svm_fit_cp3_sandra.val.RData')



#########################################################################
###  LOGISTIC REGRESSION FITTING MODEL WITH 47 IMPORTANT VARIABLES:
#########################################################################

glm.churn.fit.sub <- glm(as.factor(churn) ~ Var226 +
                           Var204 +
                           Var113 +
                           Var57 +
                           Var216 +
                           Var119 +
                           Var76 +
                           Var126 +
                           Var28 +
                           Var38 +
                           Var13 +
                           Var133 +
                           Var6 +
                           Var134 +
                           Var206 +
                           Var25 +
                           Var228 +
                           Var73 +
                           Var153 +
                           Var81 +
                           Var197 +
                           Var125 +
                           Var94 +
                           Var160 +
                           Var21 +
                           Var205 +
                           Var112 +
                           Var24 +
                           Var109 +
                           Var83 +
                           Var163 +
                           Var149 +
                           Var140 +
                           Var22 +
                           Var189 +
                           Var207 +
                           Var218 +
                           Var123 +
                           Var74 +
                           Var132 +
                           Var85 +
                           Var229 +
                           #Var199 +
                           Var51_missing +
                           Var198 +
                           Var195 +
                           Var40,
                         train,
                         family='binomial')

summary(glm.churn.fit.sub)


##################
### EVALUATE PERFORMANCE WITH TRAIN DATA:
##################
glm.churn.pred.sub.train <- predict(glm.churn.fit.sub,train,type="response")

glm.churn.pred.sub.prediction.train <- prediction(glm.churn.pred.sub.train, train$churn)

glm.churn.pred.sub.perf.tpr.fpr.train <- performance(glm.churn.pred.sub.prediction.train,"tpr","fpr")

# plot the ROC curve:
#plot(glm.churn.pred.sub.perf.tpr.fpr.train,
#     main=paste("ROC Curve for Logistic Regression for Churn with TRAIN dataset - 47 predictors"),
#     col=2,lwd=2)
#abline(a=0,b=1,lwd=2,lty=2,col="gray")

# get the AUC performance measure:
glm.churn.perfAUC.train.sub <- performance(glm.churn.pred.sub.prediction.train,'auc')
glm.churn.AUC.Train.sub <- as.numeric(glm.churn.perfAUC.train.sub@y.values)
glm.churn.AUC.Train.sub

# plot the LIFT performance measure:
glm.churn.perf.lift.train.sub <- performance(glm.churn.pred.sub.prediction.train,"lift")
plot(glm.churn.perf.lift.train.sub,
     main="Lift for Logistic Regression for Churn with TRAIN dataset - 47 variables",
     col=2,lwd=2)

## precision/recall curve (x-axis: recall, y-axis: precision)
glm.churn.perf.prec.rec.train.sub <- performance(glm.churn.pred.sub.prediction.train, "prec", "rec")
plot(glm.churn.perf.prec.rec.train.sub, main="Precision/Recall for Logistic Regression for Churn with TRAIN dataset - 47 predictors")


## sensitivity/specificity curve (x-axis: specificity,
## y-axis: sensitivity)
glm.churn.perf.sens.spec.train.sub <- performance(glm.churn.pred.sub.prediction.train, "sens", "spec")
plot(glm.churn.perf.sens.spec.train.sub, main="Sensitivity/Specificity for Logistic Regression for Churn with TRAIN dataset - 47 predictors")


##################
### Evaluate Performance Measures with the TEST Data:
##################
glm.churn.pred.sub.test <- predict(glm.churn.fit.sub,test,type="response")

glm.churn.pred.sub.prediction.test <- prediction(glm.churn.pred.sub.test, test$churn)

glm.churn.pred.sub.perf.tpr.fpr.test <- performance(glm.churn.pred.sub.prediction.test,"tpr","fpr")

# plot the ROC curve:
plot(glm.churn.pred.sub.perf.tpr.fpr.test,
     main=paste("ROC Curve for Logistic Regression for Churn with TEST dataset - 47 predictors"),
     col=2,lwd=2)
abline(a=0,b=1,lwd=2,lty=2,col="gray")

# get the AUC performance measure:
glm.churn.perfAUC.test.sub <- performance(glm.churn.pred.sub.prediction.test,'auc')
glm.churn.AUC.Test.sub <- as.numeric(glm.churn.perfAUC.test.sub@y.values)
glm.churn.AUC.Test.sub

# plot the LIFT performance measure:
glm.churn.perf.lift.test.sub <- performance(glm.churn.pred.sub.prediction.test,"lift")
plot(glm.churn.perf.lift.test.sub,
     main="Lift for Logistic Regression for Churn with TEST dataset - 47 variables",
     col=2,lwd=2)

## precision/recall curve (x-axis: recall, y-axis: precision)
glm.churn.perf.prec.rec.test.sub <- performance(glm.churn.pred.sub.prediction.test, "prec", "rec")
plot(glm.churn.perf.prec.rec.test.sub, main="Precision/Recall for Logistic Regression for Churn with TEST dataset - 47 predictors")

## sensitivity/specificity curve (x-axis: specificity,
## y-axis: sensitivity)
glm.churn.perf.sens.spec.test.sub <- performance(glm.churn.pred.sub.prediction.test, "sens", "spec")
plot(glm.churn.perf.sens.spec.test.sub, main="Sensitivity/Specificity for Logistic Regression for Churn with TEST dataset - 47 predictors")


#plot multiple ROC curves
library(ROCR)
perf1 <- performance(glm.churn.pred.sub.prediction.train,"tpr","fpr")
perf2 <- performance(glm.churn.pred.sub.prediction.test,"tpr","fpr")

plot(perf1,col='red', main="Logistic Regression ROC for churn. In-Sample Evaluation red line; Out-of-Sample Evaluation blue line")
plot(perf2, add = TRUE, col='blue')
abline(0,1,lty=8,col='grey')

# save .RData model and preidctions output
save(list = c('glm.churn.fit.sub', 'glm.churn.pred.sub.test'),
     file = 'models/churn/churn_glm_fit_cp3_sub_sandra.RData')
