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


train <- select(train, -upsell, - churn)

set.seed(287)
app_rf_jay <- randomForest(factor(appetency) ~ ., data = train,
                           nodesize = 4, ntree = 1000,
                           strata = factor(train$appetency),
                           sampsize = c(608, 608)
                           )


app_rf_jay_predictions <- predict(app_rf_jay, test,
                                        type = 'prob')[,2]

pred <- prediction(app_rf_jay_predictions, test$appetency)
perf <- performance(pred,'auc')
perf@y.values

app_ens_rf_jay_pred <- predict(app_rf_jay, ensemble_test,
                               type = 'prob')[,2]

save(list = c('app_rf_jay_predictions','app_ens_rf_jay_pred'),
     file = 'models/appetency/rf_jay.RData')
