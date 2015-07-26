library(randomForest)
library(dplyr)
library(ROCR)

setwd('c:/Users/Jay/Dropbox/pred_454_team')


# choose a script to load and transform the data
source('data_transformations/impute_0.r')

# the data needs to be in matrix form, so I'm using make_mat()
# from kdd_tools.r
source('kdd_tools.r')

# over sample the possitive instances of churn
train <- rbind(train, train[train$churn == 1,],
               train[train$churn == 1,],
               train[train$churn == 1,])

train <- select(train, -upsell, -appetency)

churn_rf_jay <- randomForest(factor(churn) ~ ., data = train,
                                 nodesize = 4, ntree = 250)


churn_rf_jay_predictions <- predict(churn_rf_jay, test,
                                        type = 'prob')[,2]

pred <- prediction(churn_rf_jay_predictions, test$churn)
perf <- performance(pred,'auc')

save(list = c('churn_rf_jay_predictions'),
     file = 'models/churn/rf_jay.RData')
