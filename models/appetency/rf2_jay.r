library(randomForest)
library(dplyr)
library(ROCR)

setwd('c:/Users/Jay/Dropbox/pred_454_team')


# choose a script to load and transform the data
source('data_transformations/impute_0.r')

# the data needs to be in matrix form, so I'm using make_mat()
# from kdd_tools.r
source('kdd_tools.r')

# over sample the possitive instances of appetency
train <- rbind(train, train[train$appetency == 1,],
               train[train$appetency == 1,],
               train[train$appetency == 1,])

train <- select(train, -upsell, - churn)

appetency_rf_jay <- randomForest(factor(appetency) ~ ., data = train,
                                 nodesize = 4, ntree = 250)


appetency_rf_jay_predictions <- predict(appetency_rf_jay, test,
                                        type = 'prob')[,2]

pred <- prediction(appetency_rf_jay_predictions, test$appetency)
perf <- performance(pred,'auc')

save(list = c('appetency_rf_jay_predictions'),
     file = 'models/appetency/rf_jay.RData')

