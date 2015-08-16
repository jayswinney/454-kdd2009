# this model is a random forest model to predict churn

library(randomForest)
library(dplyr)

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


# choose a script to load and transform the data
source('data_transformations/impute_0.r')

# the data needs to be in matrix form, so I'm using make_mat()
# from kdd_tools.r
source('kdd_tools.r')
df_mat <- make_mat(df)


train_colnames <- colnames(select(train,-churn, -upsell, -appetency))

set.seed(512356)
churn_rf_manjari <- randomForest(x=train[,train_colnames], y=factor(train$churn) ,
                                     ntree = 50, nodesize = 10, importance = TRUE)
plot(churn_rf_manjari)
churn.varImp <- importance(churn_rf_manjari)
#churn.varImp
varImpPlot(churn_rf_manjari, type=1)
# make predictions

churn_rf_manjari_predictions <- predict(churn_rf_manjari, newdata=test)
# Confusion Matrix
#Confusion Matrix
table(test$churn, churn_rf_manjari_predictions)
#Accuracy = 0.924

#Creating Random forest with top 50 variables based on variable importance reduced accuracy of the
# model . So we will be using the full model itself.
selVars <- names(sort(churn.varImp[,1],decreasing=T))[1:50]
set.seed(123)
churn_rf_top_50_manjari <- randomForest(x=train[,selVars], y=factor(train$churn) ,
                                        ntree = 50, nodesize = 10, importance = TRUE)
# AUC
# On train data
churn_rf_top_50_manjari_predictions_train <- predict(churn_rf_top_50_manjari, newdata=train,s = 'lambda.min')
# Confusion Matri#Confusion Matrix
table(train$churn, churn_rf_top_50_manjari_predictions_train)
# On test data
churn_rf_top_50_manjari_predictions_test <- predict(churn_rf_top_50_manjari,
                                                    newdata = test,
                                                    type = 'prob')[,2]
# Confusion Matri#Confusion Matrix

table(test$churn, churn_rf_top_50_manjari_predictions_test)
#Accuracy =0.924

### Determining AUC in train data set

library('ROCR')

PredTrain = predict(churn_rf_top_50_manjari,newdata=train, type="prob")[, 2]

ROCR = prediction(PredTrain, train$churn)
#performance in terms of true and false positive rates
d.rf.perf = performance(ROCR,"tpr","fpr")

#plot the curve
plot(d.rf.perf,main="ROC Curve for Random Forest Churn in train data",col=2,lwd=2)
abline(a=0,b=1,lwd=2,lty=2,col="gray")

as.numeric(performance(ROCR , "auc")@y.values)
# AUC = 0.99686

### Determining AUC in test data set

PredTrain_test = predict(churn_rf_top_50_manjari,newdata=test, type="prob")[, 2]

ROCR = prediction(PredTrain_test, test$churn)
#performance in terms of true and false positive rates
d.rf.perf = performance(ROCR,"tpr","fpr")

#plot the curve
plot(d.rf.perf,main="ROC Curve for Random Forest Churn in test data",col=2,lwd=2)
abline(a=0,b=1,lwd=2,lty=2,col="gray")

as.numeric(performance(ROCR , "auc")@y.values)
# AUC = 0.68832


churn_rf_manjari <- churn_rf_top_50_manjari
churn_rf_manjari_predictions <- churn_rf_top_50_manjari_predictions_test
# save the output
save(list = c('churn_rf_manjari', 'churn_rf_manjari_predictions'),
     file = 'models/churn/churn_rf_manjari.RData')
