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
upsell_rf_full_manjari <- randomForest(x=train[,train_colnames], y=factor(train$upsell) ,
                                     ntree = 10, nodesize = 10, importance = TRUE)
plot(upsell_rf_full_manjari)
upsell.varImp <- importance(upsell_rf_full_manjari)
#upsell.varImp
varImpPlot(upsell_rf_full_manjari, type=1)
# make predictions

upsell_rf_full_manjari_predictions <- predict(upsell_rf_full_manjari, newdata=test,s = 'lambda.min')
# Confusion Matrix 
#Confusion Matrix
table(test$upsell, upsell_rf_full_manjari_predictions)
#Accuracy = 0.93696

#Creating Random forest with top 25 variables based on variable importance reduced accuracy of the
# model . So we will be using the full model itself.
set.seed(512356)
upsell.selVars <- names(sort(upsell.varImp[,1],decreasing=T))[1:25]

upsell_rf_top_25_manjari <- randomForest(x=train[,upsell.selVars], y=factor(train$upsell) ,
                                        ntree = 10, nodesize = 10, importance = TRUE)
# AUC 
upsell_rf_top_25_manjari_predictions <- predict(upsell_rf_top_25_manjari, newdata=test,s = 'lambda.min')
# Confusion Matri#Confusion Matrix
table(test$upsell, upsell_rf_top_25_manjari_predictions)
#Accuracy = 0.9472
#upsell.selVars <- names(sort(upsell.varImp[,1],decreasing=T))[1:50]
#upsell_rf_top_50_manjari <- randomForest(x=train[,upsell.selVars], y=factor(train$upsell) ,
                                         ntree = 10, nodesize = 10, importance = TRUE)
# AUC 
#upsell_rf_top_50_manjari_predictions <- predict(upsell_rf_top_25_manjari, newdata=test,s = 'lambda.min')
# Confusion Matri#Confusion Matrix
#table(test$upsell, upsell_rf_top_50_manjari_predictions)
#Accuracy = 0.94668

# Choosing the model with top 50 variables based on importance may lead to overfitting , thus we will go with top 25 variables instead
upsell_rf_manjari <- upsell_rf_top_25_manjari
upsell_rf_manjari_predictions <- upsell_rf_top_25_manjari_predictions

# save the output
save(list = c('upsell_rf_manjari', 'upsell_rf_manjari_predictions'),
     file = 'models/upsell/upsell_rf_manjari.RData')
