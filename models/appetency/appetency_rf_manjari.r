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


set.seed(512356)
train_colnames <- colnames(select(train,-churn, -upsell, -appetency))


appetency_rf_full_manjari <- randomForest(x=train[,train_colnames], y=factor(train$appetency) ,
                                     ntree = 10, nodesize = 10, importance = TRUE)
plot(appetency_rf_full_manjari)
appetency.varImp <- importance(appetency_rf_full_manjari)
#appetency.varImp
varImpPlot(appetency_rf_full_manjari, type=1)
# make predictions

appetency_rf_full_manjari_predictions <- predict(appetency_rf_full_manjari, newdata=test,s = 'lambda.min')
# Confusion Matrix 
#Confusion Matrix
table(test$appetency, appetency_rf_full_manjari_predictions)
#Accuracy = 0.9816 , The full model did not catch any of the appetency cases in test.This is not an acceptable model

#Creating Random forest with top 25 variables based on variable importance reduced accuracy of the
# model . So we will be using the full model itself.
appetency.selVars <- names(sort(appetency.varImp[,1],decreasing=T))[1:25]

appetency_rf_top_25_manjari <- randomForest(x=train[,appetency.selVars], y=factor(train$appetency) ,
                                        ntree = 10, nodesize = 10, importance = TRUE)
# AUC 
appetency_rf_top_25_manjari_predictions <- predict(appetency_rf_top_25_manjari, newdata=test,s = 'lambda.min')
# Confusion Matri#Confusion Matrix
table(test$appetency, appetency_rf_top_25_manjari_predictions)
#Accuracy = (11541+9)=0.9816 , The model did not catch any of the appetency cases

appetency_rf_manjari <- appetency_rf_top_25_manjari
appetency_rf_manjari_predictions <- appetency_rf_top_25_manjari_predictions
# save the output
save(list = c('appetency_rf_manjari', 'appetency_rf_manjari_predictions'),
     file = 'models/appetency/appetency_rf_manjari.RData')
