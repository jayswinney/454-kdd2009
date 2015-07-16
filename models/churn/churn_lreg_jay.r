# this model is a regularized logisitc regression model to predict churn
# the regularization parameter is selected automatically using cross validation

library(glmnet)
library(ggplot2)

setwd('c:/Users/Jay/Dropbox/pred_454_team')


# choose a script to load and transform the data
source('data_transformations/impute_0.r')

# the data needs to be in matrix form, so I'm using make_mat()
# from kdd_tools.r
source('kdd_tools.r')
df_mat <- make_mat(df)

churn_lreg_jay <- cv.glmnet(df_mat[train_ind,],
                           factor(train$churn), family = "binomial",
                           nfolds = 10, type.measure = 'auc')

save(churn_lreg_jay, file = 'models/churn/churn_lreg_jay.RData')

# for code that would be useful for the paper highlight it like this
# the line below lets with the dashes helps knitr find a particular chunk of
# code to be inseterted into a r markdown document.
# ---- churn_lreg_jay_plot ----
# library(ggplot2)
plot_df <- data.frame(cvm = churn_lreg_jay$cvm, cvup = churn_lreg_jay$cvup,
                      cvlo = churn_lreg_jay$cvlo, lambda = churn_lreg_jay$lambda)

ggplot(data = plot_df, aes(x = log(lambda), y = cvm)) +
  geom_line(colour = '#d9544d', size = 1) +
  geom_ribbon(aes(x = log(lambda), ymin = cvlo, ymax = cvup),
              alpha=0.2, fill = '#d9544d') +
  geom_vline(xintercept = log(churn_lreg_jay$lambda.min),
             linetype = 3, size = 1) +
  geom_vline(xintercept = log(churn_lreg_jay$lambda.1se),
             linetype = 3, size = 1) +
  ylab('AUC') + ggtitle('Cross Validation Curve Logistic Regression')
# ----
# line above with the dashes ends the code chunk
