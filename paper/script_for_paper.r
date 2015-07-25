
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
load("models/churn/churn_lreg_jay.RData")
load("models/churn/churn_nb_sandra.RData")
load("models/churn/churn_rf_manjari.RData")
load("models/upsell/upsell_lreg_jay.RData")
load("models/upsell/upsell_nb_sandra.RData")
load("models/upsell/upsell_rf_manjari.RData")
load("data_transformations/response.RData")

# ---- combine predictions ----

# appetency
appetency_df <- data.frame(appetency = test_response$appetency,
                           logistic_regression = app_lreg_jay_predictions,
                           niave_bayes = appetency_nb_sandra_predictions,
                           random_forest = appetency_rf_manjari_predictions)

app_df2 <- gather(appetency_df, appetency, 'prediction')
names(app_df2) <- c('true_value', 'algorithm', 'prediction')

# churn
churn_df <- data.frame(churn = test_response$churn,
                       logistic_regression = churn_lreg_jay_predictions,
                       niave_bayes = churn_nb_sandra_predictions,
                       random_forest = churn_rf_manjari_predictions)

churn_df2 <- gather(churn_df, churn, 'prediction')
names(churn_df2) <- c('true_value', 'algorithm', 'prediction')

# upsell
upsell_df <- data.frame(upsell = test_response$upsell,
                        logistic_regression = app_lreg_jay_predictions,
                        niave_bayes = upsell_nb_sandra_predictions,
                        random_forest = upsell_rf_manjari_predictions)

upsell_df2 <- gather(upsell_df, upsell, 'prediction')
names(upsell_df2) <- c('true_value', 'algorithm', 'prediction')

# ---- app ROC ----

# prediction objects
pred_lr <- prediction(app_lreg_jay_predictions, factor(test_response$appetency))
pred_nb <- prediction(appetency_nb_sandra_predictions,
                      factor(test_response$appetency))

pred_rf <- prediction(appetency_rf_manjari_predictions,
                      factor(test_response$appetency))
# performance objects
perf_lr <- performance(pred_lr, measure = "tpr", x.measure = "fpr")
perf_nb <- performance(pred_nb, measure = "tpr", x.measure = "fpr")
perf_rf <- performance(pred_rf, measure = "tpr", x.measure = "fpr")

# create dataframe for roc curves
app_roc_df <- rbind(
  data.frame(algorithm = rep('logistic_regression', length(perf_lr@y.values)),
             TPR = unlist(perf_lr@y.values),
             FPR = unlist(perf_lr@x.values)),
  data.frame(algorithm = rep('random_forest', length(perf_rf@y.values)),
             TPR = unlist(perf_rf@y.values),
             FPR = unlist(perf_rf@x.values)),
  data.frame(algorithm = rep('naive_bayes', length(perf_nb@y.values)),
             TPR = unlist(perf_nb@y.values),
             FPR = unlist(perf_nb@x.values))
)

# plot results
ggplot(data = app_roc_df, aes(x = FPR, y = TPR, group = algorithm, colour = algorithm)) +
  geom_line(size = 1) + scale_color_manual(values = c('#3b5b92', '#d9544d', '#39ad48'))





























