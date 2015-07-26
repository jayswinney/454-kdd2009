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
library(caret)
# ----

# ---- read_data ----
# read in the data to R
# I'm using na.stings = '' to replace blanks with na
# this also helps R read the numerical varaibles as numerical
setwd('c:/Users/Uduak/Dropbox/pred_454_team/data')
# choose a script to load and transform the data
# source('data_transformations/impute_0.r')
df <- read.csv('orange_small_train.data', header = TRUE,
               sep = '\t', na.strings = '')
# read the target variables
churn_ <- read.csv('orange_small_train_churn.labels', header = FALSE)
appetency_ <- read.csv('orange_small_train_appetency.labels', header = FALSE)
upsell_ <- read.csv('orange_small_train_upselling.labels', header = FALSE)

churn_[churn_$V1 < 0,] <- 0
appetency_[appetency_$V1 < 0,] <- 0
upsell_[upsell_$V1 < 0,] <- 0

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

#Make the responses factors
churn <- factor(df$churn)
appetency <- factor(df$appetency)
upsell <- factor(df$upsell)

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

#check for factor variables
f <- sapply(df, is.factor)
which(f)

df_mat <- select(df, -churn, -appetency, -upsell)


#Creating separate variables for the different factors
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

#Create input matrix
df_mat <- data.matrix(df_mat)
# #Convert to data.frame
# df_mat.frame <- data.frame(df_mat)

# ----

#Exploratory decision Tree (Classification) for churn, appetency and upsell
require(tree)
ChurnTree_exp <- tree(churn~.-appetency-upsell,data = df)
summary(ChurnTree_exp)
plot(ChurnTree_exp)
text(ChurnTree_exp,pretty = 0)
ChurnTree_exp

#Focus on Var 126
summary(Var126)
logVar126 <- log(abs(Var126))
ggplot(df) + geom_bar(aes(x=Var126,group=churn,fill=churn),position="dodge")


# CHURN
# #Convert to data.frame
df_mat.frame <- data.frame(df_mat)
df_mat.frame$churn <- df$churn
dim(df_mat.frame)
anyNA(df_mat.frame)
## Decision Tree

# ---- dt_churn ----
library(rpart)
library(rpart.plot)

churn_tree <- rpart(churn~.,
                    method = 'class',data = df_mat.frame,
                    control=rpart.control(minsplit=40, minbucket=10, cp=0.001))

churn_tree
summary(churn_tree)
# ---- plot_tree_churn ----
rpart.plot(churn_tree, main = 'Churn Decision Tree')
# ----

#GOF logistic regression using variables from Tree selection
lrTree.Churn <- glm(churn~Var126+Var217_dummy_missing+Var217_dummy_other+
                      Var218_dummy_missing+Var211_dummy_L84s+Var211_dummy_Mtgm+
                      Var73+Var126_missing+Var229_dummy_missing+Var113+
                      Var221_dummy_zCkv+Var22_missing+Var200_dummy_missing+
                      Var200_dummy_other+Var214_dummy_missing+Var214_dummy_other+
                      Var227_dummy_6fzt+Var28+Var125_missing+Var13_missing+
                      Var65+Var65_missing+Var7_missing+Var74_missing+
                      Var198_dummy_other+Var207_dummy_Kxdu+Var220_dummy_other+
                      Var222_dummy_other,data = df_mat.frame, family = binomial)


#Refitted with statitically significant variables
lrTree.ChurnRe <- glm(churn~Var126+Var217_dummy_missing+Var211_dummy_L84s+
                        Var73+Var126_missing+Var229_dummy_missing+
                        Var113+Var22_missing+Var65,
                      data = df_mat.frame,family = binomial)

summary(lrTree.ChurnRe)
par(mfrow=c(1,1))
fit <- lrTree.ChurnRe$fitted
fit
hist(fit)
par(mfrow=c(2,2))
plot(lrTree.ChurnRe)
lrTree.ChurnRe
par(mfrow=c(1,1))

#################################################
# Chi-square goodness of fit test
#################################################
# Calculate residuals across all individuals
r.tree <- (df_mat.frame$churn - fit)/(sqrt(fit*(1-fit)))
# Sum of squares of these residuals follows a chi-square
sum(r.tree^2)
#Calculate the p-value from the test
1- pchisq(50209.03, df=49990)

#Exploratory logistic Regression with LASSO - Udy's code changes
# ---- lreg_churn ----
library(glmnet)
# regularized logistic regression with LASSO
churn_lreg <- glmnet(df_mat[train_ind,],
                     factor(train$churn), alpha=1, family = "binomial")

plot(churn_lreg,xvar="lambda")
grid()

# Show number of selected variables
churn_lreg

#Use cross validation to select the best lambda (minlambda) and lambda.1se

cv.churn_lreg <- cv.glmnet(df_mat[train_ind,],
                           train$churn, alpha=1)
plot(cv.churn_lreg)
coef(cv.churn_lreg) # Gives coefficients lambda.1se
best_lambda <- cv.churn_lreg$lambda.min
best_lambda

attach(df_mat.frame)

#GOF logistic regression models LASSO Variables selected
lrLASSO.churn <- glm(churn~Var7+Var73+Var113+Var126+Var22_missing+
                       Var28_missing+Var205_dummy_sJzTlal+Var206_dummy_IYzP+
                       Var210_dummy_g5HH+Var212_dummy_NhsEn4L+
                       Var217_dummy_other+Var218_dummy_cJvF+
                       Var218_dummy_missing+Var229_dummy_missing,
                     data = df_mat.frame, family = binomial)
summary(lrLASSO.churn)
par(mfrow=c(2,2))
plot(lrLASSO.churn)
par(mfrow=c(1,1))

#Refitted with statitically significant variables
lrLASSO.churnRe <- glm(churn~Var7+Var73+Var113+Var126+
                       Var205_dummy_sJzTlal+Var210_dummy_g5HH+
                       Var212_dummy_NhsEn4L+Var217_dummy_other+
                       Var218_dummy_cJvF+Var229_dummy_missing,
                     data = df_mat.frame, family = binomial)

summary(lrLASSO.churnRe)
par(mfrow=c(1,1))
fit.LASSO <- lrLASSO.churnRe$fitted
fit.LASSO
hist(fit)
par(mfrow=c(2,2))
plot(lrLASSO.churnRe)
lrLASSO.churnRe
par(mfrow=c(1,1))

#################################################
# Chi-square goodness of fit test for LASSO Variables
#################################################
# Calculate residuals across all individuals
r.LASSO <- (df_mat.frame$churn - fit.LASSO)/(sqrt(fit.LASSO*(1-fit.LASSO)))
# Sum of squares of these residuals follows a chi-square
sum(r.LASSO^2)
#Calculate the p-value from the test
1- pchisq(49888.49, df=49989)


# ----


# # view the Area Under the Curve for different values of lambda.
# plot(churn_lreg.cv)
# title('Cross Validation Curve Logistic Regression',line =+2.8)
#
#
#
# cv_coefs <- data.table(variable = row.names(coef(churn_lreg.cv))[
#   abs(as.vector(coef(churn_lreg.cv))) > 1e-5],
#   coeficient = coef(churn_lreg.cv)[abs(coef(churn_lreg.cv)) > 1e-5])
#
#
# kable(cv_coefs[variable %like% '26'],
#       caption = "Variables Selected by Elastic-Net")




## Random Forest
?randomForest
# ---- rf_churn ----
library(randomForest)
set.seed(123)
churn_rf <- randomForest(churn~.,
                         data = df_mat.frame,
                         ntree = 10, nodesize = 10, importance = TRUE)
# ----
# ---- plot_rf_churn ----
varImpPlot(churn_rf, type = 2, main = 'Variable Importance Churn')

#GOF logistic regression using variables from Tree selection
lrRf.Churn <- glm(churn~Var113+Var126+Var57+Var81+Var28+Var153+Var73+
                    Var6+Var133+Var125+Var76+Var134+Var38+Var119+Var94+Var13+
                    Var140+Var163+Var149+Var189+Var160+Var25+Var123+Var22+
                    Var112+Var21+Var74+Var109+Var85+Var83,
                  data = df_mat.frame, family = binomial)

summary(lrRf.Churn)
par(mfrow=c(2,2))
plot(lrRf.Churn)
tree.churn
par(mfrow=c(1,1))

#Refitted with statitically significant variables
lrRf.ChurnRe <- glm(churn~Var113+Var126+Var81+Var28+Var73
                    +Var94+Var189+Var74,
                    data = df_mat.frame,family = binomial)

summary(lrRf.ChurnRe)
par(mfrow=c(1,1))
fit.Rf <- lrRf.ChurnRe$fitted
fit.Rf
hist(fit.Rf)
par(mfrow=c(2,2))
plot(lrRf.ChurnRe)
lrRf.ChurnRe
par(mfrow=c(1,1))

#################################################
# Chi-square goodness of fit test for RandomForest Variables
#################################################
# Calculate residuals across all individuals
r.Rf <- (df_mat.frame$churn - fit.Rf)/(sqrt(fit.Rf*(1-fit.Rf)))
# Sum of squares of these residuals follows a chi-square
sum(r.Rf^2)
# Sum of squares of these residuals follows a chi-square
1- pchisq(50650.24, df=49991)

# ----

# Create train and test data set for data frame

set.seed(123)
smp_size <- floor(0.75 * nrow(df_mat.frame))
train_ind <- sample(seq_len(nrow(df_mat.frame)), size = smp_size)
# making a "tiny" data set so I cn quickly test r markdown and graphical paramters
# this will be removed in the submitted version
tiny_ind <- sample(seq_len(nrow(df_mat.frame)), size = floor(0.01 * nrow(df)))
# split the data
train.frame <- df_mat.frame[train_ind, ]
test.frame <- df_mat.frame[-train_ind, ]
tiny.frame <- df_mat.frame[tiny_ind, ]

#Logistic regression prediction training data set

#Selected
# lrfit <-glm(churn~Var126+Var73+Var113+Var229_dummy_missing+
#               Var217_dummy_missing,data=train.frame,family = binomial)

#Decision Tree Variables:
lrfit <-glm(churn~Var126+Var217_dummy_missing+Var211_dummy_L84s+
              Var73+Var126_missing+Var229_dummy_missing+Var113+
              Var22_missing+Var65,data=train.frame,family = binomial)
#LASSO Variables
# lrfit <-glm(churn~Var7+Var73+Var113+Var126+Var205_dummy_sJzTlal+
#               Var210_dummy_g5HH+Var212_dummy_NhsEn4L+
#               Var217_dummy_other+Var218_dummy_cJvF+
#               Var229_dummy_missing,data=train.frame,family = binomial)
#RF variables
# lrfit <-glm(churn~Var113+Var126+Var81+Var28+Var73+
#               +Var189+Var74,data=train.frame,family = binomial)
summary(lrfit)
head(train.frame)

#Checking prediction quality on training
Plogit <- predict(lrfit,newdata =train.frame,type = "response")
p.churn <- round(Plogit)

require(e1071)
require(caret)
confusionMatrix(p.churn,train.frame$churn)

#Checking prediction quality on test
PlogitTest <- predict(lrfit,newdata=test.frame,type = "response")
p.churnTest <- round(PlogitTest)
confusionMatrix(p.churnTest,test.frame$churn)

#How good is the Logistic model in-sample:
logit.scores <- prediction(Plogit,train.frame$churn)
plot(performance(logit.scores,'tpr','fpr'),col='red')
abline(0,1,lty=8,col='grey')
logit.auc <- performance(logit.scores,'auc')
logit.auc

#How good is the Logistic model out-sample:
logit.scores.test <- prediction(PlogitTest,test.frame$churn)
#ROC plot for logistic regression
plot(performance(logit.scores.test,'tpr','fpr'),col='red')
abline(0,1,lty=8,col='grey')
#AUC value
logit.auc.test <- performance(logit.scores.test,'auc')
logit.auc.test

# make logsitc regression predictions
churn_lreg_udy_predictions <- predict(lrfit, df_mat.frame[-train_ind,],
                                      type = 'response')

# churn_svm_udy_predictions <- predict(lrfit, df_mat[-train_ind,],
#                                       type = 'response')


# save the output
setwd('c:/Users/Uduak/Dropbox/pred_454_team')
save(list = c('lrfit', 'churn_lreg_udy_predictions'),
     file = 'models/churn/churn_lreg_udy.RData')
# save(list = c('churn_svm_udy', 'churn_svm_udy_predictions'),
#      file = 'models/churn/churn_svm_udy.RData')



###############################
# SVM:
###############################
class(churn)
svmfit <- svm(as.factor(churn)~Var126+Var217_dummy_missing+Var211_dummy_L84s+
                Var73+Var126_missing+Var229_dummy_missing+Var113+
                Var22_missing+Var65,data=train.frame,type="C-classification")

#Checking prediction quality on training set
Psvm.churn <- predict(svmfit,train.frame)
table(Psvm.churn,train.frame$churn)
confusionMatrix(Psvm.churn,train.frame$churn)
class(Psvm.churn)
class(churn)
Psvm.churn <- as.integer(Psvm.churn)
train.frame$churn <- as.integer(train.frame$churn)


#How good is the Logistic model in-sample - ROC and AUC
svm.scores <- prediction(Psvm.churn,train.frame$churn)
plot(performance(svm.scores,'tpr','fpr'),col='red')
abline(0,1,lty=8,col='grey')
svm.auc <- performance(svm.scores,'auc')
svm.auc


 #Checking prediction quality on test set
test.frame$churn <- as.integer(test.frame$churn)
Psvm.churn.test <- predict(svmfit,test.frame)
confusionMatrix(Psvm.churn.test,test.frame$churn)

#How good is the Logistic model out-sample - ROC and AUC
Psvm.churn.test <- as.integer(Psvm.churn.test)
svm.scores.test <- prediction(Psvm.churn.test,test.frame$churn)
#ROC plot for logistic regression
plot(performance(svm.scores.test,'tpr','fpr'),col='red')
abline(0,1,lty=8,col='grey')
#AUC value
svm.auc.test <- performance(svm.scores.test,'auc')
svm.auc.test
