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

load("models/upsell/upsell_lreg_jay.RData")
load("models/upsell/rf_jay.RData")
load("models/upsell/upsell_nb_sandra.RData")
# load('models/upsell/upsell_lreg_udy.RData')

#load('models/upsell/upsell_rf_top25_oversampling_cp3_manjari.RData')

load('models/upsell/upsell_rf_top25_equalsampling_manjari.RData')

# str(upsell_ens_lreg_jay_predictions)
# str(rf.upsell.pred.ensemble.equalsampling)
# str(ensemble_test)
upsell_vote <- rowSums(data.frame(
  scale_vec(upsell_ens_rf_jay_predictions),
  scale_vec(upsell_ens_lreg_jay_predictions),
  # scale_vec(upsell_ens_nb_sandra_predictions),
  scale_vec(rf.upsell.pred.ensemble.equalsampling)
))/3

# str(ensemble_test$upsell)
# str(upsell_ens_rf_jay_predictions)
# str(rf.upsell.pred.ensemble.equalsampling)

upsell_df <- data.frame(
  upsell = ensemble_test$upsell,
  random_forest2 = upsell_ens_rf_jay_predictions,
  naive_bayes = upsell_ens_nb_sandra_predictions,
  random_forest = rf.upsell.pred.ensemble.equalsampling,
  logistic_regression = upsell_ens_lreg_jay_predictions,
  vote_ensemble = upsell_vote
)

upsell_train <- data.frame(
  upsell = test$upsell,
  random_forest2 = upsell_rf_jay_predictions,
  naive_bayes = upsell_nb_sandra_predictions,
  random_forest = rf.upsell.pred.test.equalsampling,
  logistic_regression = upsell_lreg_jay_predictions
)

lreg_combiner <- glm(factor(upsell) ~ ., data = upsell_train,
                     family = 'binomial')

up_logistic_ens <- predict(lreg_combiner, upsell_df)
upsell_df$logistic_ensemble <- predict(lreg_combiner, upsell_df)

upsell_df2 <- gather(upsell_df, upsell, 'prediction')
names(upsell_df2) <- c('true_value', 'algorithm', 'prediction')

upsell_roc_df <- make_roc(upsell_df2, ens_response$upsell)
# plot results
ggplot(data = upsell_roc_df, aes(x = FPR, y = TPR, group = algorithm,
                                 colour = algorithm)) +
  geom_line(size = 1) +
  scale_color_manual(values = my_colors) +
  ggtitle('ROC Curves upsell Models')

make_auc(upsell_df2, ens_response$upsell, 0.8975)

save(list = c('upsell_vote', 'up_logistic_ens'),
     file = 'ensembles/upsell.RData')
