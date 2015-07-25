# this model is a random forest model to predict churn

library(randomForest)
library(dplyr)

setwd("~/Manjari/Northwestern/R/Workspace/Predict454/KDDCup2009/Dropbox")


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
churn_rf_top_50_manjari_predictions_test <- predict(churn_rf_top_50_manjari, newdata=select(test, -appetency, -upsell),s = 'lambda.min', type='Class')
# Confusion Matri#Confusion Matrix

table(test$churn, churn_rf_top_50_manjari_predictions_test)
#Accuracy =0.924


churn_rf_manjari <- churn_rf_top_50_manjari
churn_rf_manjari_predictions <- churn_rf_top_50_manjari_predictions_test
# save the output
save(list = c('churn_rf_manjari', 'churn_rf_manjari_predictions'),
     file = 'models/churn/churn_rf_manjari.RData')
