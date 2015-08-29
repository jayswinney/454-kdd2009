library(ROCR)
library(ggplot2)
library(randomForest)
library(dplyr)
library(tidyr)


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


source('kdd_tools.r')
source('data_transformations/impute_0.r')
load("data_transformations/response.RData")

my_colors <- c('#1f78b4', '#33a02c', '#e31a1c', '#ff7f00',
               '#6a3d9a', '#b15928', '#a6cee3', '#b2df8a',
               '#fb9a99', '#fdbf6f', '#cab2d6', '#ffff99')
# load the predictions for everyone
load("models/appetency/appetency_nb_sandra.RData")
load("models/appetency/app_lreg_jay.RData")
# load("models/appetency/appetency_rf_manjari.RData")
load("models/appetency/appetency.knnTestPred.RData")
load("models/appetency/rf_jay.RData")
load("models/appetency/app_lreg_udy.RData")



app_vote <- rowSums(data.frame(
  # scale_vec(app_ens_rf_jay_pred),
  scale_vec(app_ens_lreg_udy_pred),
  # scale_vec(app_ens_nb_sandra_pred),
  scale_vec(app_ens_lreg_jay_predictions)
  ))/2

# dataframe to train neural network ensemble on

app_train <- data.frame(
  appetency = test_response$appetency,
  random_forest2 = app_rf_jay_predictions,
  logistic_regression = app_lreg_udy_pred,
  naive_bayes = appetency_nb_sandra_predictions,
  logistic_regression2 = app_lreg_jay_predictions
  )

app_train <- cbind(app_train, select(test, -upsell, -churn))

set.seed(61)
rf_stack <- randomForest(factor(appetency) ~ ., data = app_train,
                         nodesize = 5, ntree = 100,
                         strata = factor(test$appetency),
                         sampsize = c(500, 100))

app_df <- data.frame(
  appetency = ensemble_test$appetency,
  random_forest2 = app_ens_rf_jay_pred,
  logistic_regression = app_ens_lreg_udy_pred,
  naive_bayes = app_ens_nb_sandra_pred,
  # app_rf_manjari_pred = app_ens_rf_manjari_pred,
  logistic_regression2 = app_ens_lreg_jay_predictions,
  vote_ensemble = app_vote
  )

rf_stack_pred <- predict(rf_stack,
                         cbind(ensemble_test, app_df),
                         type = 'prob')[,2]

app_df$stacked_random_forest <- rf_stack_pred

app_df2 <- gather(app_df, appetency, 'prediction')
names(app_df2) <- c('true_value', 'algorithm', 'prediction')


app_roc_df <- make_roc(app_df2, ens_response$appetency)
# plot results
ggplot(data = app_roc_df, aes(x = FPR, y = TPR, group = algorithm,
                              colour = algorithm)) +
  geom_line(size = 1) +
  scale_color_manual(values = my_colors) +
  ggtitle('ROC Curves Appetency Models')

make_auc(app_df2, ens_response$appetency, 0.7435)

save(list = c('rf_stack_pred', 'app_vote'), file = 'ensembles/app.RData')
