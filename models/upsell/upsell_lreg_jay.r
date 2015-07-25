# this model is a regularized logisitc regression model to predict upsell
# the regularization parameter is selected automatically using cross validation

library(glmnet)

setwd('c:/Users/Jay/Dropbox/pred_454_team')


# choose a script to load and transform the data
source('data_transformations/impute_0.r')

# the data needs to be in matrix form, so I'm using make_mat()
# from kdd_tools.r
source('kdd_tools.r')
df_mat <- make_mat(df)

upsell_lreg_jay <- cv.glmnet(df_mat[train_ind,],
                           factor(train$upsell), family = "binomial",
                           nfolds = 4, type.measure = 'auc')

# make predictions
upsell_lreg_jay_predictions <- predict(upsell_lreg_jay, df_mat[-train_ind,],
                                      type = 'response', s = 'lambda.min')
# save the output
save(list = c('upsell_lreg_jay', 'upsell_lreg_jay_predictions'),
     file = 'models/upsell/upsell_lreg_jay.RData')
