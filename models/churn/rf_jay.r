library(randomForest)
library(dplyr)
library(ROCR)

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

# over sample the possitive instances of churn
train <- select(train, -upsell, -appetency)

set.seed(2651)
churn_rf_jay <- randomForest(factor(churn) ~ ., data = train,
                             nodesize = 4, ntree = 1000,
                             strata = factor(train$churn),
                             sampsize = c(2500, 2500)
                             )

churn_rf_jay_predictions <- predict(churn_rf_jay, test,
                                        type = 'prob')[,2]

pred <- prediction(churn_rf_jay_predictions, test$churn)
perf <- performance(pred,'auc')
perf@y.values

save(list = c('churn_rf_jay_predictions'),
     file = 'models/churn/rf_jay.RData')
