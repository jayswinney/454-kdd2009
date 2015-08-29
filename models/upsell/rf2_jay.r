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

# over sample the possitive instances of upsell
train <- rbind(train, train[train$upsell == 1,],
               train[train$upsell == 1,],
               train[train$upsell == 1,])

train <- select(train, -appetency, - churn)
set.seed(434)
upsell_rf_jay <- randomForest(factor(upsell) ~ ., data = train,
                              nodesize = 4, ntree = 1000,
                              strata = factor(train$upsell),
                              sampsize = c(1000, 1000),
                              importance = TRUE
                              )

upsell.varImp <- importance(upsell_rf_jay)
upsell.selVars <- names(sort(upsell.varImp[,1],decreasing=T))[1:50]
set.seed(434)
upsell_rf_jay <- randomForest(x=train[,upsell.selVars], y=factor(train$upsell),
                              nodesize = 1, ntree = 1000,
                              strata = factor(train$upsell),
                              sampsize = c(1000, 1000),
                              importance = TRUE
                              )


upsell_rf_jay_predictions <- predict(upsell_rf_jay, test,
                                     type = 'prob')[,2]

upsell_ens_rf_jay_predictions <- predict(upsell_rf_jay, ensemble_test,
                                         type = 'prob')[,2]

pred <- prediction(upsell_rf_jay_predictions, test$upsell)
perf <- performance(pred,'auc')
perf@y.values

save(list = c('upsell_rf_jay_predictions', 'upsell_ens_rf_jay_predictions'),
     file = 'models/upsell/rf_jay.RData')
