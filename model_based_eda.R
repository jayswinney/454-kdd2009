# ---- libraries ----
library(lattice)
library(plyr)
library(dplyr)
library(tidyr)
library(grid)
library(gridExtra)
library(ROCR)
library(e1071)
library(knitr)
library(ggplot2)
library(data.table)
# ----

# ---- read_data ----
# read in the data to R
# I'm using na.stings = '' to replace blanks with na
# this also helps R read the numerical varaibles as numerical
setwd('C:/Users/Jay/Documents/Northwestern/predict_454/KDD_Cup_2009/')
df <- read.csv('orange_small_train.data', header = TRUE,
               sep = '\t', na.strings = '')
# read the target variables
churn_ <- read.csv('orange_small_train_churn.labels', header = FALSE)
appetency_ <- read.csv('orange_small_train_appetency.labels', header = FALSE)
upsell_ <- read.csv('orange_small_train_upselling.labels', header = FALSE)

churn_[churn_$V1 < 0,] <- 0
appetency_[appetency_$V1 < 0,] <- 0
upsell_[upsell_$V1 < 0,] <- 0
# ----

# ---- impute ----
# impute mising data with zeros and "missing"
# also creates missing variable column
for (i in names(df)){
  vclass <- class(df[,i])
  if(vclass == 'logical'){
    # some of the variables are 100% missing, they are the only logical class vars
    # so we can safely remove all logical class vars
    df[,i] <- NULL
  }else if(vclass %in% c('integer', 'numeric')){
    #first check that there are missing variables
    if(sum(is.na(df[,i])) == 0) next
    # create a missing variable column
    df[,paste(i,'_missing',sep='')] <- as.integer(is.na(df[,i]))
    # fill missing variables with 0
    df[is.na(df[,i]),i] <- 0
  }else{
    # gather infrequent levels into 'other'
    levels(df[,i])[xtabs(~df[,i])/dim(df)[1] < 0.015] <- 'other'
    # replace NA with 'missing'
    levels(df[,i]) <- append(levels(df[,i]), 'missing')
    df[is.na(df[,i]), i] <- 'missing'
  }
}
# ----

# ---- target_vars ----
# add the target variables to the data frame
df$churn <- churn_$V1
df$appetency <- appetency_$V1
df$upsell <- upsell_$V1
# ----



# ---- train_test_mat ----
# get the index for training/testing data
set.seed(123)
smp_size <- floor(0.75 * nrow(df))
train_ind <- sample(seq_len(nrow(df)), size = smp_size)
# making a "tiny" data set so I cn quickly test r markdown and graphical paramters
# this will be removed in the submitted version
tiny_ind <- sample(seq_len(nrow(df)), size = floor(0.01 * nrow(df)))
# split the data
train <- df[train_ind, ]
test <- df[-train_ind, ]
tiny <- df[tiny_ind, ]


df_mat <- select(df, -churn, -appetency, -upsell)

for (i in names(df_mat)){
  if (class(df_mat[,i]) == 'factor'){
    for(level in unique(df_mat[,i])){
      df_mat[sprintf('%s_dummy_%s', i, level)] <- ifelse(df_mat[,i] == level, 1, 0)
    }
    df_mat[,i] <- NULL
  } else {
    # scale numeric variables
    # this is important for regularized logistic regression and KNN
    df_mat[,i] <- scale(df_mat[,i])
  }
}

df_mat <- data.matrix(df_mat)
# ----

# Churn



## Logistic Regression with Elastic-Net Penalty

# ---- lreg_churn ----
library(glmnet)
# regularized logistic regression with cross validation
# this takes a while, try using nfolds < 10 to reduce time
churn_lreg.cv <- cv.glmnet(df_mat[train_ind,],
                     factor(train$churn), family = "binomial",
                     nfolds = 10, type.measure = 'auc')
# ----

# view the Area Under the Curve for different values of lambda.
plot(churn_lreg.cv)
title('Cross Validation Curve Logistic Regression',line =+2.8)



cv_coefs <- data.table(variable = row.names(coef(churn_lreg.cv))[
  abs(as.vector(coef(churn_lreg.cv))) > 1e-5],
  coeficient = coef(churn_lreg.cv)[abs(coef(churn_lreg.cv)) > 1e-5])


kable(cv_coefs[variable %like% '26'],
      caption = "Variables Selected by Elastic-Net")




# KNN
# library(FNN)
#
# auc_vec <- rep(0, 20)
#
# for(i in 1:20){
#   #print(sprintf('trying k = %d', i))
#   yhat <- knn(df_mat[train_ind,], df_mat[-train_ind,],
#               cl = factor(train$churn), k = i, prob = TRUE)
#   pred <- prediction((as.numeric(yhat[1:dim(df_mat[-train_ind,])[1]]) - 1) * attr(yhat,'prob'),
#                      factor(test$churn))
#   # the following commented out code is for use with the tiny data set
#   # yhat <- knn(df_mat[tiny_ind,], df_mat[tiny_ind,],
#   #             cl = factor(tiny$churn), k = i, prob = TRUE)
#   # pred <- prediction((as.numeric(yhat[1:dim(df_mat[tiny_ind,])[1]]) - 1) * attr(yhat,'prob'),
#   #                    factor(tiny$churn))
#   perf <- performance(pred, measure = "tpr", x.measure = "fpr")
#   #print(sprintf('AUC: %f',
#   #              attributes(performance(pred, 'auc'))$y.values[[1]]))
#   auc_vec[i] <- attributes(performance(pred, 'auc'))$y.values[[1]]
# }
#
# p <- qplot(y = auc_vec, color = 'AUC') + geom_line() +
#   xlab('k = x') + ylab('AUC') + ggtitle('K-NN')
# p



## Decision Tree

# ---- dt_churn ----
library(rpart)
library(rpart.plot)

churn_tree <- rpart(factor(churn)~.,
                    data = select(train, -appetency, -upsell),
                    method = 'class',
                    control=rpart.control(minsplit=40, minbucket=10, cp=0.001))
# ----

# ---- plot_tree_churn ----
rpart.plot(churn_tree, main = 'Churn Decision Tree')
# ----


## Random Forest

# ---- rf_churn ----
library(randomForest)
set.seed(123)
churn_rf <- randomForest(factor(churn)~.,
                         data = select(train, -appetency, -upsell),
                         ntree = 10, nodesize = 10, importance = TRUE)
# ----
# ---- plot_rf_churn ----
varImpPlot(churn_rf, type = 2, main = 'Variable Importance Churn')
# ----

#
# ---- roc_churn ----
yhat <- predict(churn_rf, select(test, -appetency, -upsell), type = 'prob')

pred <- prediction(yhat[,2], factor(test$churn))
perf <- performance(pred, measure = "tpr", x.measure = "fpr")
p <- ggplot(data.frame(TPR = unlist(perf@y.values),
                       FPR = unlist(perf@x.values)),
            aes(FPR, TPR)) + geom_line(color = '#3b5b92', size = 1) +
  xlab('False Positvie Rate') + ylab('True Positive Rate') +
  ggtitle('Random Forest ROC Curve') +
  theme(plot.title = element_text(lineheight=.8, face="bold")) +
  annotate("text", x = 0.75, y = 0.20, label = sprintf('AUC: %f',
                                                       attributes(performance(pred, 'auc'))$y.values[[1]]))
p
# ----




# ## Principal Components
# pca <- princomp(df_mat)
#
#
# library(ggbiplot)
#
# p <- ggbiplot(pca, groups = factor(df$churn), ellipse = FALSE,
#               var.axes = FALSE) +
#   ggtitle('First 2 Principal Components') +
#   xlim(-3, 3) + ylim(-3, 3) +
#   scale_fill_discrete(name = 'Churn')
#
# p


# Appetency


## Logistic Regression with Elastic-Net Penalty
# ---- lreg_app ----
app_lreg.cv <- cv.glmnet(df_mat[train_ind,], factor(train$appetency), family = "binomial",
                     nfolds = 8, type.measure = 'auc')
# ----

# view the bionmial deviance (log loss) of differnt values of lambda
plot(app_lreg.cv)
title('Cross Validation Curve Logistic Regression', line =+2.8)


cv_coefs <- data.frame(
  coeficient = coef(app_lreg.cv, s = 'lambda.1se')[abs(coef(app_lreg.cv,
                                                        s = 'lambda.1se')) > 1e-3])

row.names(cv_coefs) <- row.names(coef(app_lreg.cv,
                                      s = 'lambda.1se'))[abs(as.vector(coef(app_lreg.cv, s = 'lambda.1se'))) > 1e-3]

kable(cv_coefs, caption = "Variables Selected by Elastic-Net")




yhat <- predict(app_lreg.cv, df_mat[-train_ind,], type = 'response')

pred <- prediction(yhat, factor(test$appetency))
perf <- performance(pred, measure = "tpr", x.measure = "fpr")

p <- ggplot(data.frame(TPR = unlist(perf@y.values),
                       FPR = unlist(perf@x.values)),
            aes(FPR, TPR)) + geom_line(color = '#3b5b92', size = 1) +
  xlab('False Positvie Rate') + ylab('True Positive Rate') +
  ggtitle('Logistic Regression ROC Curve') +
  theme(plot.title = element_text(lineheight=.8, face="bold")) +
  annotate("text", x = 0.75, y = 0.20, label = sprintf('AUC: %f',
                                                       attributes(performance(pred, 'auc'))$y.values[[1]]))
p




## Decision Tree

# ---- dt_app ----
app_tree <- rpart(factor(appetency)~.,
                  data = select(train, -churn, -upsell),
                  method = 'class',
                  control=rpart.control(minsplit=40, minbucket=10, cp=0.001))
# ----
rpart.plot(app_tree, main = 'Appetency Decision Tree')
app_tree



## Random Forest

# ---- rf_app ----
app_rf <- randomForest(factor(appetency)~.,
                             data = select(train, -churn, -upsell),
                             ntree = 10, nodesize = 10, importance = TRUE)
# ----
varImpPlot(app_rf, type = 2, main = 'Variable Importance Appetency')


yhat <- predict(app_rf, select(test, -churn, -upsell), type = 'prob')

pred <- prediction(yhat[,2], factor(test$appetency))
perf <- performance(pred, measure = "tpr", x.measure = "fpr")
p <- ggplot(data.frame(TPR = unlist(perf@y.values),
                       FPR = unlist(perf@x.values)),
            aes(FPR, TPR)) + geom_line(color = '#3b5b92', size = 1) +
  xlab('False Positvie Rate') + ylab('True Positive Rate') +
  ggtitle(' Random Forest ROC Curve') +
  theme(plot.title = element_text(lineheight=.8, face="bold")) +
  annotate("text", x = 0.75, y = 0.20, label = sprintf('AUC: %f',
                                                       attributes(performance(pred, 'auc'))$y.values[[1]]))
p



# Up-Sell


## Logistic Regression with Elastic-Net Penalty
# ---- lreg_upsell ----
upsell_lreg.cv <- cv.glmnet(df_mat[train_ind,], factor(train$upsell), family = "binomial",
                            nfolds = 8, type.measure = 'auc')
# ----
# view the bionmial deviance (log loss) of differnt values of lambda
plot(upsell_lreg.cv)
title('Cross Validation Curve Logistic Regression',line =+2.8)




cv_coefs <- data.frame( coeficient = coef(upsell_lreg.cv, s = 'lambda.1se')[
  abs(coef(upsell_lreg.cv, s = 'lambda.1se')) > 1e-3])

row.names(cv_coefs) <- row.names(coef(upsell_lreg.cv, s = 'lambda.1se'))[
  abs(as.vector(coef(upsell_lreg.cv, s = 'lambda.1se'))) > 1e-3]


## Decision Tree
# ---- dt_upsell ----
upsell_tree <- rpart(factor(upsell)~.,
                     data = select(train, -appetency, -churn),
                     method = 'class',
                     control=rpart.control(minsplit=100, minbucket=10, cp=0.001))
# ----
rpart.plot(upsell_tree, main = 'Up-Sell Decision Tree')




summary(upsell_tree)


## Random Forest
# ---- rf_upsell ----
upsell_rf <- randomForest(factor(upsell)~.,
                          data = select(train, -appetency, -churn),
                          ntree = 10, nodesize = 10, importance = TRUE)
# ----
varImpPlot(upsell_rf, type = 2, main = 'Variable Importance Up-Sell')

save(list = c('churn_lreg.cv', 'churn_rf', 'churn_tree',
              'app_lreg.cv', 'app_rf', 'app_tree',
              'upsell_lreg.cv', 'upsell_rf', 'upsell_tree'),
     file = 'C:/Users/Jay/Documents/Northwestern/predict_454/team_project/eda.RData')
