
# ---- libraries ----
library(ROCR)
library(ggplot2)
library(tidyr)
library(dplyr)
library(xtable)

# ---- rdata ----
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

load("models/appetency/appetency_nb_sandra.RData")
load("models/appetency/app_lreg_jay.RData")
load("models/appetency/appetency_rf_manjari.RData")
load("models/appetency/appetency.knnTestPred.RData")
load("models/appetency/rf_jay.RData")
load("models/appetency/app_lreg_udy.RData")
load('ensembles/app.RData')

load("models/churn/churn_lreg_jay.RData")
load("models/churn/churn_nb_sandra.RData")
load("models/churn/churn_rf_manjari.RData")
load("models/churn/churn.knnTestPred.RData")
load("models/churn/churn_lreg_udy.RData")
load("models/churn/rf_jay.RData")

load("models/upsell/upsell_lreg_jay.RData")
load("models/upsell/rf_jay.RData")
load("models/upsell/upsell_nb_sandra.RData")
load("models/upsell/upsell_rf_manjari.RData")
load("models/upsell/upsell.knnTestPred.RData")
load('models/upsell/upsell_rf_fullvars_fit_cp3_manjari.RData')
load('models/upsell/upsell_rf_top25_fit_cp3_manjari.RData')
load('models/upsell/upsell_rf_top25_oversampling_cp3_manjari.RData')
load('models/upsell/upsell_svm_fit_cp3_manjari_prob.RData')
load('models/upsell/upsell_svm_fit_cp3_manjari_val.RData')
load('models/upsell/upsell_glm_fit_cp3_top25_manjari.RData')

load("data_transformations/response.RData")


my_colors <- c('#1f78b4', '#33a02c', '#e31a1c', '#ff7f00',
               '#6a3d9a', '#b15928', '#a6cee3', '#b2df8a',
               '#fb9a99', '#fdbf6f', '#cab2d6', '#ffff99')



source('kdd_tools.r')
# ---- combine predictions ----
# appetency
app_df <- data.frame(
  appetency = ens_response$appetency,
  random_forest2 = app_ens_rf_jay_pred,
  logistic_regression = app_ens_lreg_udy_pred,
  naive_bayes = app_ens_nb_sandra_pred,
  # app_rf_manjari_pred = app_ens_rf_manjari_pred,
  logistic_regression2 = app_ens_lreg_jay_predictions,
  vote_ensemble = app_vote,
  stacked_rf = rf_stack_pred
  )

app_df2 <- gather(app_df, appetency, 'prediction')
names(app_df2) <- c('true_value', 'algorithm', 'prediction')

# just to clean up the workspace, remove prediction vectors

rm( list = c('app_lreg_jay_predictions',
             'app_lreg_udy_pred',
             'appetency_nb_sandra_predictions',
             'appetency_rf_manjari_predictions',
             'app_rf_jay_predictions',
             'appetency.knnTestPred'))

# churn
# create the vote ensemble
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

rm(list = c('churn_lreg_udy_predictions',
            'churn_lreg_jay_predictions',
            'churn_nb_sandra_predictions',
            'churn_rf_manjari_predictions',
            'churn_rf_jay_predictions',
            'churn.knnTestPred'))

# upsell
upsell_df <- data.frame(upsell = test_response$upsell
                        , logistic_regression = upsell_lreg_jay_predictions
                        , naive_bayes = upsell_nb_sandra_predictions
                      # , random_forest = upsell_rf_manjari_predictions
                        , random_forest2 = upsell_rf_jay_predictions
                        , random_forest = rf.upsell.pred.test.prob.sub)
                      # , random_forest5 = rf.upsell.pred.test.prob.sub.over
                      # , svm = attr(svm.upsell.pred.prob.test,
                      #               'probabilities')[,1])
                      # , knn = upsell.knnTestPred)

upsell_df2 <- gather(upsell_df, upsell, 'prediction')
names(upsell_df2) <- c('true_value', 'algorithm', 'prediction')

rm(list = c('upsell_lreg_jay_predictions',
            'upsell_nb_sandra_predictions',
            'upsell_rf_manjari_predictions',
            'upsell_rf_jay_predictions',
            'upsell.knnTestPred',
            'rf.upsell.pred.test.prob.prediction',
            'rf.upsell.pred.test.prob.sub',
            'rf.upsell.pred.test.prob.sub.over',
            'svm.upsell.pred.prob.test'))

# ---- app ROC ----

app_roc_df <- make_roc(app_df2, ens_response$appetency)
# plot results
ggplot(data = app_roc_df, aes(x = FPR, y = TPR, group = algorithm,
                              colour = algorithm)) +
  geom_line(size = 1) +
  scale_color_manual(values = my_colors) +
  ggtitle('ROC Curves Appetency Models')

# ---- app AUC ----
print(xtable(make_auc(app_df2, ens_response$appetency, 0.8522)))

# ---- Churn ROC ----

churn_roc_df <- make_roc(churn_df2, ens_response$churn)
# plot results
ggplot(data = churn_roc_df, aes(x = FPR, y = TPR, group = algorithm,
                              colour = algorithm)) +
  geom_line(size = 1) +
  scale_color_manual(values = my_colors) +
  ggtitle('ROC Curves Churn Models')

# ---- churn AUC ----
print(xtable(make_auc(churn_df2, ens_response$churn, 0.7435)))

# ---- Upsell ROC ----

upsell_roc_df <- make_roc(upsell_df2, test_response$upsell)
# plot results
ggplot(data = upsell_roc_df, aes(x = FPR, y = TPR, group = algorithm,
                              colour = algorithm)) +
  geom_line(size = 1) +
  scale_color_manual(values = my_colors) +
  ggtitle('ROC Curves Up-Sell Models')
  # theme(legend.position="top")

# ---- upsell AUC ----
print(xtable(make_auc(upsell_df2, test_response$upsell, 0.8975)))
