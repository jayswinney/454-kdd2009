#Check point 3 code : Upsell
#upsell_cp_3_manjari.r

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

#dirs <- c('c:/Users/jay/Dropbox/pred_454_team',
#          'c:/Users/uduak/Dropbox/pred_454_team',
#          'C:/Users/Sandra/Dropbox/pred_454_team',
#          '~/Manjari/Northwestern/R/Workspace/Predict454/KDDCup2009/Dropbox',
#          'C:/Users/JoeD/Dropbox/pred_454_team'
#)

#for (d in dirs){
#  if(dir.exists(d)){
#    setwd(d)
#  }
#}


setwd("~/Manjari/Northwestern/R/Workspace/Predict454/KDDCup2009/Dropbox")
# choose a script to load and transform the data
source('data_transformations/impute_0.r')
# Converting data to matrix form
source('kdd_tools.r')
df_mat <- make_mat(df)

#Data set containing predictors and upsell only
train_upsell <- select(train, -appetency, -churn)

table(train$upsell)
#######################################################
#    Random Forest Model
#######################################################

#####Full Model with all variables ####################

set.seed(512356)
upsell_rf_full_fit_cp3 <- randomForest(factor(train_upsell$upsell)~ ., data=train_upsell,
                                       ntree = 50, nodesize = 10, importance = TRUE)
plot(upsell_rf_full_fit_cp3)
upsell.varImp <- importance(upsell_rf_full_fit_cp3)
upsell.varImp
upsell.varImplot <- varImpPlot(upsell_rf_full_fit_cp3, type=1, main='Variable Importance Full RF Model')
#barchart(upsell.varImp[,4], type=2, sort=TRUE, xlab="Mean decerease in Gini")
#write.csv(upsell.varImp,"upsell_rf_fit_cp3.varImp.csv")
# make predictions

### EVALUATE ON TRAIN DATA:
# Confustion Matrix:
train_upsell$Pred <- predict(upsell_rf_full_fit_cp3, type="class", newdata=train_upsell)
upsell.pred.rp.conf.matrix.train <- confusionMatrix(train_upsell$Pred,train_upsell$upsell)
print(upsell.pred.rp.conf.matrix.train)
# AUC:
rf.upsell.pred.train.prob <- predict(upsell_rf_full_fit_cp3,train,type = 'prob')[,2]
rf.upsell.pred.train.prob.prediction <- prediction(rf.upsell.pred.train.prob, train_upsell$upsell)
#pred.train
perfAUC.train <- performance(rf.upsell.pred.train.prob.prediction,'auc')
#perfAUC.train
AUC.Train <- as.numeric(perfAUC.train@y.values)
AUC.Train


### EVALUATE ON TEST DATA:
# Confustion Matrix:
test$Pred <- predict(upsell_rf_full_fit_cp3, type="class", newdata=test)
upsell.pred.rp.conf.matrix.test <- confusionMatrix(test$Pred,test$upsell)
print(upsell.pred.rp.conf.matrix.test)
# AUC:
rf.upsell.pred.test.prob <- predict(upsell_rf_full_fit_cp3,test,type = 'prob')[,2]
rf.upsell.pred.test.prob.prediction <- prediction(rf.upsell.pred.test.prob, test$upsell)
#pred.test
perfAUC.test <- performance(rf.upsell.pred.test.prob.prediction,'auc')
#perfAUC.test
AUC.Test <- as.numeric(perfAUC.test@y.values)
AUC.Test

# save .RData model and preidctions output

save(list = c('upsell_rf_full_fit_cp3', 'rf.upsell.pred.test.prob.prediction'),
     file = 'models/upsell/upsell_rf_fullvars_fit_cp3_manjari.RData')


#####Random Forest Model with top 25 variables ####################

set.seed(512356)
upsell.selVars <- names(sort(upsell.varImp[,1],decreasing=T))[1:25]
str(upsell.selVars)
upsell_rf_top_25_cp3 <- randomForest(x=train[,upsell.selVars], y=factor(train$upsell) ,
                                         ntree = 50, nodesize = 10, importance = TRUE)

plot(upsell_rf_top_25_cp3)
upsell.varImp <- importance(upsell_rf_top_25_cp3)
upsell.varImp
upsell.varImplot <- varImpPlot(upsell_rf_top_25_cp3, type=1)


# make predictions

### EVALUATE ON TRAIN DATA:
# Confustion Matrix:
train$pred.upsell.rf.sub<- predict(upsell_rf_top_25_cp3, type="class", newdata=train)
upsell.pred.rp.conf.matrix.train.sub <- confusionMatrix(train$pred.upsell.rf.sub,train$upsell)
print(upsell.pred.rp.conf.matrix.train.sub)
# AUC:
train$pred.upsell.rf.sub.prob <- predict(upsell_rf_top_25_cp3,train,type = 'prob')[,2]
rf.upsell.pred.train.prob.prediction.sub <- prediction(train$pred.upsell.rf.sub.prob, train$upsell)
#pred.train
perfAUC.train.sub <- performance(rf.upsell.pred.train.prob.prediction.sub,'auc')
#perfAUC.train
AUC.Train <- as.numeric(perfAUC.train.sub@y.values)
AUC.Train


### EVALUATE ON TEST DATA:
# Confustion Matrix:
test$pred.upsell.rf.sub <- predict(upsell_rf_top_25_cp3, type="class", newdata=test)

upsell.pred.rp.conf.matrix.test.sub <- confusionMatrix(test$pred.upsell.rf.sub,test$upsell)
print(upsell.pred.rp.conf.matrix.test.sub)
# AUC:
test$pred.upsell.rf.sub.prob <- predict(upsell_rf_top_25_cp3,test,type = 'prob')[,2]
rf.upsell.pred.test.prob.prediction.sub <- prediction(test$pred.upsell.rf.sub.prob, test$upsell)
#pred.test
perfAUC.test.sub <- performance(rf.upsell.pred.test.prob.prediction.sub,'auc')
#perfAUC.test
AUC.Test.sub <- as.numeric(perfAUC.test.sub@y.values)
AUC.Test.sub

#ROC curve
#performance in terms of true and false positive rates
d.rf.perf = performance(rf.upsell.pred.test.prob.prediction.sub,"tpr","fpr")

#plot the curve
plot(d.rf.perf,main="ROC Curve for Random Forest of top 25 model in test data",col=2,lwd=2)
abline(a=0,b=1,lwd=2,lty=2,col="gray")


# save .RData model and preidctions output

save(list = c('upsell_rf_top_25_cp3', 'rf.upsell.pred.test.prob.sub'),
     file = 'models/upsell/upsell_rf_top25_fit_cp3_manjari.RData')

#####Random Forest Model with Oversampling upsell ####################

# over sample the positive instances of churn
train_oversample <- rbind(train, train[train$upsell == 1,],
               train[train$upsell == 1,],
               train[train$upsell == 1,])

train_oversample <- select(train_oversample, -churn, -appetency)
str(train_oversample)

set.seed(512356)

upsell_rf_top_25_over_cp3 <- randomForest(x=train_oversample[,upsell.selVars], y=factor(train_oversample$upsell) ,
                                     ntree = 50, nodesize = 10, importance = TRUE)


plot(upsell_rf_top_25_over_cp3)
upsell.varImp <- importance(upsell_rf_top_25_over_cp3)
upsell.varImp
upsell.varImplot <- varImpPlot(upsell_rf_top_25_over_cp3, type=1)


# make predictions

### EVALUATE ON TRAIN DATA:
# Confustion Matrix:
train$Pred.sub.over <- predict(upsell_rf_top_25_over_cp3, type="class", newdata=train)
upsell.pred.rp.conf.matrix.train.sub.over <- confusionMatrix(train$Pred.sub.over,train$upsell)
print(upsell.pred.rp.conf.matrix.train.sub.over)
# AUC:
rf.upsell.pred.train.prob.sub.over <- predict(upsell_rf_top_25_over_cp3,train,type = 'prob')[,2]
rf.upsell.pred.train.prob.prediction.sub.over <- prediction(rf.upsell.pred.train.prob.sub.over, train$upsell)
#pred.train
perfAUC.train.sub <- performance(rf.upsell.pred.train.prob.prediction.sub.over,'auc')
#perfAUC.train
AUC.Train <- as.numeric(perfAUC.train.sub@y.values)
AUC.Train


### EVALUATE ON TEST DATA:
# Confustion Matrix:
test$Pred.sub.over <- predict(upsell_rf_top_25_over_cp3, type="class", newdata=test)
upsell.pred.rp.conf.matrix.test.sub.over <- confusionMatrix(test$Pred.sub.over,test$upsell)
print(upsell.pred.rp.conf.matrix.test.sub.over)
# AUC:
rf.upsell.pred.test.prob.sub.over <- predict(upsell_rf_top_25_over_cp3,test,type = 'prob')[,2]
rf.upsell.pred.test.prob.prediction.sub.over <- prediction(rf.upsell.pred.test.prob.sub.over, test$upsell)
#pred.test
perfAUC.test.sub.over <- performance(rf.upsell.pred.test.prob.prediction.sub.over,'auc')
#perfAUC.test
AUC.Test.sub.over <- as.numeric(perfAUC.test.sub.over@y.values)
AUC.Test.sub.over

#ROC curve
#performance in terms of true and false positive rates
d.rf.perf = performance(rf.upsell.pred.test.prob.prediction.sub.over,"tpr","fpr")

#plot the curve
plot(d.rf.perf,main="ROC Curve for SVM of top 25 model in test data",col=2,lwd=2)
abline(a=0,b=1,lwd=2,lty=2,col="gray")

# save .RData model and predictions output

#save(list = c('upsell_rf_top_25_over_cp3', 'rf.upsell.pred.test.prob.sub.over'),
#     file = 'models/upsell/upsell_rf_top25_oversampling_cp3_manjari.RData')



#####Random Forest Model with sampling option of 50:50 ####################
# over sample the positive instances of churn
train_oversample <- rbind(train, train[train$upsell == 1,],
                          train[train$upsell == 1,],
                          train[train$upsell == 1,])

train_oversample <- select(train_oversample, -churn, -appetency)
str(train_oversample)
#table(train$upsell)
set.seed(512356)
upsell.selVars <- names(sort(upsell.varImp[,1],decreasing=T))[1:25]
print(upsell.selVars)

# with selected vars only
upsell_rf_equalsampling_cp3 <- randomForest(x=train_oversample[,upsell.selVars], y=factor(train_oversample$upsell) ,
                                     strata = factor(train_oversample$upsell),
                                     sampsize = c(1000, 1000),
                                     ntree = 50, nodesize = 10, importance = TRUE)
# with full model
#upsell_rf_equalsampling_cp3 <- randomForest(factor(train_oversample$upsell) ~ . , data = train_oversample,
#                                            strata = factor(train_oversample$upsell),
#                                            sampsize = c(1000, 1000),
#                                            ntree = 50, nodesize = 10, importance = TRUE)

plot(upsell_rf_equalsampling_cp3)
upsell.varImp.equalsampling <- importance(upsell_rf_equalsampling_cp3)
upsell.varImp.equalsampling
upsell.varImplot.equalsampling <- varImpPlot(upsell_rf_equalsampling_cp3, type=1)


# make predictions

### EVALUATE ON TRAIN DATA:
# Confustion Matrix:
train$pred.upsell.rf.equalsampling <- predict(upsell_rf_equalsampling_cp3, type="class", newdata=train)
upsell.pred.rp.conf.matrix.train.equalsampling <- confusionMatrix(train$pred.upsell.rf.equalsampling,train$upsell)
print(upsell.pred.rp.conf.matrix.train.equalsampling)
# AUC:
train$pred.upsell.rf.equalsampling.prob <- predict(upsell_rf_equalsampling_cp3,train,type = 'prob')[,2]
rf.upsell.pred.train.prob.prediction.equalsampling <- prediction(train$pred.upsell.rf.equalsampling.prob, train$upsell)
#pred.train
perfAUC.train.equalsampling <- performance(rf.upsell.pred.train.prob.prediction.equalsampling,'auc')
#perfAUC.train
AUC.Train <- as.numeric(perfAUC.train.equalsampling@y.values)
AUC.Train


### EVALUATE ON TEST DATA:
# Confustion Matrix:
test$pred.upsell.rf.equalsampling <- predict(upsell_rf_equalsampling_cp3, type="class", newdata=test)

upsell.pred.rp.conf.matrix.test.equalsampling <- confusionMatrix(test$pred.upsell.rf.equalsampling,test$upsell)
print(upsell.pred.rp.conf.matrix.test.equalsampling)
# AUC:
test$pred.upsell.rf.equalsampling.prob <- predict(upsell_rf_equalsampling_cp3,test,type = 'prob')[,2]
rf.upsell.pred.test.prob.prediction.equalsampling <- prediction(test$pred.upsell.rf.equalsampling.prob, test$upsell)
#pred.test
perfAUC.test.equalsampling <- performance(rf.upsell.pred.test.prob.prediction.equalsampling,'auc')
#perfAUC.test
AUC.Test.equalsampling <- as.numeric(perfAUC.test.equalsampling@y.values)
AUC.Test.equalsampling

#ROC curve
#performance in terms of true and false positive rates
d.rf.perf = performance(rf.upsell.pred.test.prob.prediction.sub,"tpr","fpr")

#plot the curve
plot(d.rf.perf,main="ROC Curve for Random Forest of top 25 model in test data",col=2,lwd=2)
abline(a=0,b=1,lwd=2,lty=2,col="gray")

rf.upsell.pred.test.equalsampling<- test$pred.upsell.rf.equalsampling

# save .RData model and preidctions output

#predictions output on ensemble data
#rf.upsell.pred.ensemble.equalsampling <- predict(upsell_rf_equalsampling_cp3, type="class", newdata=ensemble_test)
rf.upsell.pred.ensemble.equalsampling <- predict(upsell_rf_top_25_cp3,
                                                 ensemble_test,
                                                 type = 'prob')[,2]

rf.upsell.pred.test.equalsampling <- predict(upsell_rf_top_25_cp3,
                                                 test,
                                                 type = 'prob')[,2]

save(list = c('upsell_rf_top_25_cp3', 'rf.upsell.pred.test.equalsampling',
              'rf.upsell.pred.ensemble.equalsampling'),
     file = 'models/upsell/upsell_rf_top25_equalsampling_manjari.RData')

#######################################################
#    Support Vector Machine
#######################################################

#Using top 25 variables based on variable importance from Random forest full model
set.seed(512356)
upsell.selVars <- names(sort(upsell.varImp[,1],decreasing=T))[1:25]
str(upsell.selVars)
print(upsell.selVars)
###  FITTING MODEL WITH 25 IMPORTANT VARIABLES
# FIT MODEL WITH PROBABILITY
set.seed(123)
train.subset <-
str(train)
svm.upsell.top25fit <- svm(factor(upsell)~ Var126+Var28+Var206+Var21+Var153+Var22+Var6+Var133+Var160+Var123+
                             Var113+Var81+Var216+Var76+Var38+Var109+Var83+Var149+Var204+Var226+
                             Var112+Var25+Var119+Var212+Var144,
                           data=train, kernel ="radial", gamma=1, cost=10,
                           probability=TRUE, decision.values=TRUE)

### In Train
train$pred.upsell.svm <- predict(svm.upsell.top25fit,train,probability=TRUE)
confusionMatrix(train$pred.svm ,train$upsell)

#AUC
svm.upsell.fit.val <-predict(svm.upsell.top25fit,train,decision.values=TRUE)
ROCR <-prediction(attributes(svm.upsell.fit.val)$decision.values, train$upsell)
#performance in terms of true and false positive rates
d.rf.perf = performance(ROCR,"tpr","fpr")

#plot the curve
plot(d.rf.perf,main="ROC Curve for SVM model in train",col=2,lwd=2)
abline(a=0,b=1,lwd=2,lty=2,col="gray")

as.numeric(performance(ROCR , "auc")@y.values)

### In test

svm.upsell.pred.prob.test <- predict(svm.upsell.top25fit,test,probability=TRUE)
confusionMatrix(svm.upsell.pred.prob.test,test$upsell)

#AUC
svm.upsell.fit.val.test <-predict(svm.upsell.top25fit,test,decision.values=TRUE)
ROCR <-prediction(attributes(svm.upsell.fit.val.test)$decision.values, train$upsell)
#performance in terms of true and false positive rates
d.rf.perf = performance(ROCR,"tpr","fpr")

#plot the curve
plot(d.rf.perf,main="ROC Curve for SVM model in test",col=2,lwd=2)
abline(a=0,b=1,lwd=2,lty=2,col="gray")

as.numeric(performance(ROCR , "auc")@y.values)


save(list = c('svm.upsell.top25fit', 'svm.upsell.pred.prob.test'),
     file = 'models/upsell/upsell_svm_fit_cp3_manjari_prob.RData')

save(list = c('svm.upsell.top25fit', 'svm.upsell.fit.val.test'),
     file = 'models/upsell/upsell_svm_fit_cp3_manjari_val.RData')

###############################################################
#    Logistic Regression- Fitting model with top 25 variables
###############################################################

glm.upsell.top25.fit <- glm(as.factor(upsell) ~ Var126+Var28+Var206+Var21+Var153+Var22+Var6+Var133+Var160+Var123+
                              Var113+Var81+Var216+Var76+Var38+Var109+Var83+Var149+Var204+Var226+
                              Var112+Var25+Var119+Var212+Var144,
                            train,
                            family='binomial')


summary(glm.upsell.top25.fit)
par(mfrow=c(2,2))
plot(glm.upsell.top25.fit)

#####In train
upsell.predict.train = predict(glm.upsell.top25.fit,newdata=train)

ROCR = prediction(upsell.predict.train, train$upsell)
#performance in terms of true and false positive rates
d.rf.perf = performance(ROCR,"tpr","fpr")

#plot the curve
plot(d.rf.perf,main="ROC Curve for Logistic regression model on training data",col=2,lwd=2)
abline(a=0,b=1,lwd=2,lty=2,col="gray")

as.numeric(performance(ROCR , "auc")@y.values)
str(train)

train$pred.upsell.glm = rep(0,7500)
train$pred.upsell.glm[upsell.predict.train>0.5] = 1

table(train$pred.upsell.glm,train$upsell)
# Accuracy =(6956+5)/(6956+5+3+536) = 0.9281

#####In test

upsell.predict.test = predict(glm.upsell.top25.fit,newdata=test)

ROCR = prediction(upsell.predict.test, test$upsell)
#performance in terms of true and false positive rates
d.rf.perf = performance(ROCR,"tpr","fpr")

#plot the curve
plot(d.rf.perf,main="ROC Curve for Logistic regression model on test data",col=2,lwd=2)
abline(a=0,b=1,lwd=2,lty=2,col="gray")

as.numeric(performance(ROCR , "auc")@y.values)
str(test)

test$pred.upsell.glm = rep(0,7500)
test$pred.upsell.glm[upsell.predict.test>0.5] = 1

table(test$pred.upsell.glm,test$upsell)
# Accuracy =(6963+1)/(6963+1+3+533) = 0.92853


save(list = c('glm.upsell.top25.fit', 'upsell.predict.test'),
     file = 'models/upsell/upsell_glm_fit_cp3_top25_manjari.RData')

########################################################
# Ensemble model : using predictions from RF top 25 , logistic regression and SVM
#############################################################
#RF top 25
#train$pred.upsell.rf.sub
#test$pred.upsell.rf.sub

set.seed(512356)

upsell_ensemble_rf <- randomForest(factor(upsell) ~ pred.upsell.rf.sub
                                   # + pred.upsell.svm
                                   #+ pred.upsell.glm
                                   + pred.upsell.rf.equalsampling
                                   , data = train,
                                   ntree = 50, nodesize = 10, importance = TRUE)

upsell_ensemble_rf_predictions <- predict(upsell_ensemble_rf, test,
                                          type = 'prob')[,2]

pred <- prediction(upsell_ensemble_rf_predictions, test$upsell)
perf <- performance(pred,'auc')
perf@y.values
