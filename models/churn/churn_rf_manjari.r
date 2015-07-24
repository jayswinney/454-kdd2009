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


churn_rf_manjari <- randomForest(x=train[,train_colnames], y=factor(train$churn) ,
                                     ntree = 10, nodesize = 10, importance = TRUE)
plot(churn_rf_manjari)
varImp <- importance(churn_rf_manjari)
varImp
varImpPlot(churn_rf_manjari, type=1)
# make predictions

churn_rf_manjari_predictions <- predict(churn_rf_manjari, newdata=test,s = 'lambda.min')
# Confusion Matrix 
#Confusion Matrix
table(test$churn, churn_rf_manjari_predictions)
#Accuracy = 0.924

#Creating Random forest with top 25 variables based on variable importance reduced accuracy of the
# model . So we will be using the full model itself.
#selVars <- names(sort(varImp[,1],decreasing=T))[1:25]

#churn_rf_top_25_manjari <- randomForest(x=train[,selVars], y=factor(train$churn) ,
#                                        ntree = 10, nodesize = 10, importance = TRUE)
# AUC 
#churn_rf_top_25_manjari_predictions <- predict(churn_rf_top_25_manjari, newdata=test,s = 'lambda.min')
# Confusion Matri#Confusion Matrix
#table(test$churn, churn_rf_top_25_manjari_predictions)
#Accuracy = (11541+9)=0.922

# save the output
save(list = c('churn_rf_manjari', 'churn_rf_manjari_predictions'),
     file = 'models/churn/churn_rf_manjari.RData')
