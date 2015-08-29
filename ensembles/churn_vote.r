
library(ROCR)
library(ggplot2)
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

# load the predictions for everyone
load("models/churn/churn_lreg_jay.RData")
load("models/churn/churn_nb_sandra.RData")
load("models/churn/churn_rf_manjari.RData")
load("models/churn/churn_lreg_udy.RData")
load("models/churn/rf_jay.RData")

source('kdd_tools.r')
load("data_transformations/response.RData")

my_colors <- c('#1f78b4', '#33a02c', '#e31a1c', '#ff7f00',
               '#6a3d9a', '#b15928', '#a6cee3', '#b2df8a',
               '#fb9a99', '#fdbf6f', '#cab2d6', '#ffff99')

# find the mean of the scaled predictions as simple ensemble
# for some reason the niave bayes modelis giving the same output for all
# so I did not include it in the ensemble
churn_vote <- rowSums(data.frame(
  scale_vec(churn_ens_rf_jay_predictions),
  scale_vec(churn_ens_lreg_udy_predictions),
  # scale_vec(churn_ens_nb_sandra_predictions),
  scale_vec(churn_ens_rf_manjari_predictions),
  scale_vec(churn_ens_lreg_jay_predictions)
  ))/4

# combine all predictions
churn_df <- data.frame(churn = ens_response$churn,
                       logistic_regression = churn_ens_rf_jay_predictions,
                       logistic_regression2 = churn_ens_lreg_udy_predictions,
                       naive_bayes = churn_ens_nb_sandra_predictions,
                       random_forest = churn_ens_rf_manjari_predictions,
                       random_forest2 = churn_ens_lreg_jay_predictions,
                       vote_ensemble = churn_vote)


churn_df2 <- gather(churn_df, churn, 'prediction')
names(churn_df2) <- c('true_value', 'algorithm', 'prediction')


churn_roc_df <- make_roc(churn_df2, ens_response$churn)
# plot results
ggplot(data = churn_roc_df, aes(x = FPR, y = TPR, group = algorithm,
                              colour = algorithm)) +
  geom_line(size = 1) +
  scale_color_manual(values = my_colors) +
  ggtitle('ROC Curves Churn Models')

make_auc(churn_df2, ens_response$churn, 0.7435)
