# this model is a regularized logisitc regression model to predict appetency
# the regularization parameter is selected automatically using cross validation

library(glmnet)

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

app_lreg_jay <- cv.glmnet(df_mat[train_ind,],
                           factor(train$appetency), family = "binomial",
                           nfolds = 4, type.measure = 'auc')


# make predictions
app_lreg_jay_predictions <- predict(app_lreg_jay, df_mat[test_ind,],
                                      type = 'response', s = 'lambda.min')[,1]
# save the output
save(list = c('app_lreg_jay', 'app_lreg_jay_predictions'),
     file = 'models/appetency/app_lreg_jay.RData')
