
# ---- libraries ----
library(ROCR)
library(ggplot2)
library(tidyr)
library(dplyr)

# ---- rdata ----
setwd('c:/Users/Jay/Dropbox/pred_454_team')
load("models/appetency/appetency_nb_sandra.RData")
load("models/appetency/app_lreg_jay.RData")
load("models/appetency/appetency_rf_manjari.RData")
load("models/appetency/appetency.knnTestPred.RData")
load("models/appetency/rf_jay.RData")
load("models/appetency/app_lreg_udy.RData")

load("models/churn/churn_lreg_jay.RData")
load("models/churn/churn_nb_sandra.RData")
load("models/churn/churn_rf_manjari.RData")
load("models/churn/churn.knnTestPred.RData")
load("models/churn/churn_lreg_udy.RData")
load("models/churn/rf_jay.RData")

load("models/upsell/upsell_lreg_jay.RData")
load("models/upsell/upsell_nb_sandra.RData")
load("models/upsell/upsell_rf_manjari.RData")
load("models/upsell/upsell.knnTestPred.RData")

load("data_transformations/response.RData")


my_colors <- c('#1f78b4', '#33a02c', '#e31a1c', '#ff7f00',
               '#6a3d9a', '#b15928', '#a6cee3', '#b2df8a',
               '#fb9a99', '#fdbf6f', '#cab2d6', '#ffff99')

# ---- combine predictions ----

make_roc <- function(df, response_vec){
  # transform data frame to be ready to plot roc curves

  df_list = list()
  # cycle through algorithms and append them to the dataframe
  for (alg in unique(df$algorithm)){
    pred <- prediction(df[df$algorithm == alg, 'prediction'], response_vec)
    perf <- performance(pred, measure = "tpr", x.measure = "fpr")
    df_list[[alg]] <- data.frame(
      algorithm = rep(alg, length(perf@y.values)),
      TPR = unlist(perf@y.values),
      FPR = unlist(perf@x.values))
  }
  # rbind all the dataframes
  return(do.call("rbind", df_list))
}

make_auc <- function(df, response_vec, in_house){
  # funciton to help me build AUC comparison tables
  auc_table = data.frame(In_House = c(in_house))
  # cycle through algorithms and append them to the dataframe
  for (alg in unique(df$algorithm)){
    pred <- prediction(df[df$algorithm == alg, 'prediction'], response_vec)
    perf <- performance(pred, measure = "auc")
    auc_table[,alg] <- perf@y.values[[1]]
  }
  auc_table <- t(auc_table)
  colnames(auc_table) <- c("AUC")
  return(auc_table)
}


# appetency
appetency_df <- data.frame(appetency = test_response$appetency,
                           logistic_regression = app_lreg_jay_predictions,
                           logistic_regression2 = app_lreg_udy_predictions,
                           naive_bayes = appetency_nb_sandra_predictions,
                           random_forest = appetency_rf_manjari_predictions,
                           random_forest2 = appetency_rf_jay_predictions,
                           knn = appetency.knnTestPred)

app_df2 <- gather(appetency_df, appetency, 'prediction')
names(app_df2) <- c('true_value', 'algorithm', 'prediction')

# churn
churn_df <- data.frame(churn = test_response$churn,
                       logistic_regression = churn_lreg_udy_predictions,
                       logistic_regression2 = churn_lreg_jay_predictions,
                       naive_bayes = churn_nb_sandra_predictions,
                       random_forest = churn_rf_manjari_predictions,
                       random_forest2 = churn_rf_jay_predictions,
                       knn = churn.knnTestPred)

churn_df2 <- gather(churn_df, churn, 'prediction')
names(churn_df2) <- c('true_value', 'algorithm', 'prediction')

# upsell
upsell_df <- data.frame(upsell = test_response$upsell,
                        logistic_regression = upsell_lreg_jay_predictions,
                        naive_bayes = upsell_nb_sandra_predictions,
                        random_forest = upsell_rf_manjari_predictions,
                        knn = upsell.knnTestPred)

upsell_df2 <- gather(upsell_df, upsell, 'prediction')
names(upsell_df2) <- c('true_value', 'algorithm', 'prediction')

# ---- app ROC ----

app_roc_df <- make_roc(app_df2, test_response$appetency)
# plot results
ggplot(data = app_roc_df, aes(x = FPR, y = TPR, group = algorithm,
                              colour = algorithm)) +
  geom_line(size = 1) +
  scale_color_manual(values = my_colors) +
  ggtitle('ROC Curves Appetency Models')

# ---- app AUC ----
print(xtable(make_auc(app_df2, test_response$appetency, 0.8522)))

# ---- Churn ROC ----

churn_roc_df <- make_roc(churn_df2, test_response$churn)
# plot results
ggplot(data = churn_roc_df, aes(x = FPR, y = TPR, group = algorithm,
                              colour = algorithm)) +
  geom_line(size = 1) +
  scale_color_manual(values = my_colors) +
  ggtitle('ROC Curves Churn Models')

# ---- churn AUC ----
print(xtable(make_auc(churn_df2, test_response$churn, 0.7435)))

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
