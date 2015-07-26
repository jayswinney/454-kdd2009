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
library(rpart)
library(rpart.plot)
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

# # ----
# 
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
# tiny <- df[tiny_ind, ]

#check for factor variables
names(df)
f <- sapply(df, is.factor)
which(f)


#Data set containing predictors only
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
df_mat.frame <- data.frame(df_mat)
df_mat.frame$appetency <- df$appetency

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

# ----
#Exploratory decision Tree (Classification) for appetency
train.frame$appetency <- factor(train.frame$appetency)
AppTree <- rpart(appetency~.,method="class",data = train.frame,
                 control=rpart.control(minsplit=100, minbucket=10, cp=0.001))
AppTree
printcp(AppTree)
plot(AppTree, uniform=TRUE)
text(AppTree, all=TRUE,cex=0.75, splits=TRUE, use.n=TRUE, xpd=TRUE)
?predict
p <- predict(AppTree,newdata=test.frame,type="class")
table(actual=test.frame$appetency,predicted=p)


#Graphical analysis of DT selected variables
summary(Var126)
ggplot(df) + geom_bar(aes(x=Var126,group=appetency,
                          fill=appetency))
table(Var204_dummy_RVjC)
ggplot(df) + geom_bar(aes(x=Var204_dummy_RVjC,group=appetency,
                          fill=appetency))
table(Var218_dummy_cJvF)
ggplot(df) + geom_bar(aes(x=Var218_dummy_cJvF,group=appetency,
                          fill=appetency))
summary(Var25)
ggplot(df) + geom_bar(aes(x=Var25,group=appetency,
                          fill=appetency))
summary(Var38)
ggplot(df) + geom_bar(aes(x=Var38,group=appetency,
                          fill=appetency))
summary(Var57)
ggplot(df) + geom_bar(aes(x=Var57,group=appetency,
                          fill=appetency))


#GOF logistic regression using variables from Tree selection
lrApp.tree <- glm(appetency~Var126+Var204_dummy_RVjC+Var218_dummy_cJvF+Var25+
                    Var38+Var57,data = df_mat.frame, family = binomial)

summary(lrApp.tree)

#Refitted with statitically significant variables
lrApp.treeRe <- glm(appetency~Var126+Var218_dummy_cJvF+
                    Var38,data = df_mat.frame, family = binomial)

summary(lrApp.treeRe)

anova(lrApp.treeRe,lrApp.tree,test="Chisq")


par(mfrow=c(1,1))
fit <- lrApp.treeRe$fitted
hist(fit)
par(mfrow=c(2,2))
lrApp.treeRe
par(mfrow=c(1,1))

#################################################
# Chi-square goodness of fit test
#################################################
# Calculate residuals across all individuals
r.tree <- (df_mat.frame$appetency - fit)/(sqrt(fit*(1-fit)))
# Sum of squares of these residuals follows a chi-square
sum(r.tree^2)
#Calculate the p-value from the test
1- pchisq(65795.97, df=49996)

#Exploratory logistic Regression with LASSO - Udy's code changes
# ---- LASSO_appetency ----
library(glmnet)
# regularized logistic regression with LASSO
app.LASSO <- glmnet(df_mat[train_ind,],
                     factor(train$appetency), alpha=1, family = "binomial")

plot(app.LASSO,xvar="lambda",label=TRUE)
grid()


#Use cross validation to select the best lambda (minlambda) and lambda.1se
#Cross Validated LASSO
appCVlasso <- cv.glmnet(df_mat[train_ind,],train$appetency)
#plot(fit.lasso,xvar="dev",label=TRUE)
plot(appCVlasso)
coef(appCVlasso)

# Show number of selected variables
app.LASSO


plot(appCVlasso)
coef(appCVlasso, s="lambda.min") # Gives coefficients lambda.min
best_lambda <- appCVlasso$lambda.min
best_lambda

attach(df_mat.frame)

#GOF logistic regression models LASSO Variables selected
lrAppLASSO <- glm(appetency~Var28+Var34+Var38+Var64+Var67+Var81+Var125+
                    Var126+Var14+Var152+Var153+Var171+Var126_missing+
                    Var194_dummy_SEuy+Var197_dummy_487l+Var197_dummy_TyGl+
                    Var204_dummy_15m3+
                    Var204_dummy_m_h1+Var205_dummy_VpdQ+Var206_dummy_43pnToF+
                    Var208_dummy_kIsH+Var210_dummy_uKAI+Var212_dummy_NhsEn4L+
                    Var216_dummy_other+Var216_dummy_7WwuNea+Var216_dummy_kZJtVhC+
                    Var218_dummy_cJvF+Var218_dummy_UYBR+Var223_dummy_M_8D+
                    Var226_dummy_xb3V+Var226_dummy_fKCe+Var226_dummy_FSa2+
                    Var226_dummy_uWr3,data = df_mat.frame, family = binomial)

summary(lrAppLASSO)
# par(mfrow=c(2,2))
# plot(lrAppLASSO)
# par(mfrow=c(1,1))

#Refitted with statitically significant variables
lrAppLASSORe1 <- glm(appetency~Var34+Var64+Var67+Var126+Var171+Var194_dummy_SEuy+                         Var197_dummy_TyGl+Var204_dummy_m_h1+Var205_dummy_VpdQ+
                      Var208_dummy_kIsH+Var210_dummy_uKAI+Var212_dummy_NhsEn4L+
                      Var216_dummy_7WwuNea+Var216_dummy_kZJtVhC+Var218_dummy_cJvF+
                      Var226_dummy_xb3V,data = df_mat.frame, family = binomial)

summary(lrAppLASSORe1)

#Dropping Var64 because it was statistically insignificant in refitted model
lrAppLASSORe <- glm(appetency~Var34+Var67+Var126+Var171+Var194_dummy_SEuy+                         Var197_dummy_TyGl+Var204_dummy_m_h1+Var205_dummy_VpdQ+
                       Var208_dummy_kIsH+Var210_dummy_uKAI+Var212_dummy_NhsEn4L+
                       Var216_dummy_7WwuNea+Var216_dummy_kZJtVhC+Var218_dummy_cJvF+
                       Var226_dummy_xb3V,data = df_mat.frame, family = binomial)


summary(lrAppLASSORe)


#Test for better fit between nested models:
anova(lrAppLASSORe1,lrAppLASSO,test="Chisq")


par(mfrow=c(1,1))
fit.LASSO <- lrAppLASSO$fitted
hist(fit)
par(mfrow=c(2,2))
plot(lrAppLASSO)
# lrAppLASSO
# par(mfrow=c(1,1))

#################################################
# Chi-square goodness of fit test for LASSO Variables
#################################################
# Calculate residuals across all individuals
r.LASSO <- (df_mat.frame$appetency - fit.LASSO)/(sqrt(fit.LASSO*(1-fit.LASSO)))
# Sum of squares of these residuals follows a chi-square
sum(r.LASSO^2)
#Calculate the p-value from the test
1- pchisq(60207.87, df=49966)


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
set.seed(101)
appRf <- randomForest(appetency~.,
                         data = df_mat.frame,
                         ntree = 100, nodesize = 1, importance = TRUE)
# ----
# ---- plot_rf_churn ----
varImpPlot(appRf, type = 1, main = 'Variable Importance appetency')
varImpPlot(appRf, type = 2, main = 'Variable Importance appetency')

# write the variable importance to a file that can be read into excel
fo <- file("rf.txt", "w")
imp <- importance(appRf)
write.table(imp, fo, sep="\t")
flush(fo)
close(fo)
#GOF logistic regression using variables from Tree selection
lrAppRF <- glm(appetency~Var126+Var119+Var28+Var6+Var109+Var81+Var133+Var153+
                 Var25+Var113+Var83+Var73+Var160+Var85+Var218_dummy_UYBR+Var22+
                 Var38+Var24+Var134+Var144+Var21+Var218_dummy_cJvF+
                 Var216_dummy_kZJtVhC+Var123+Var163+Var225_dummy_kG3k+
                 Var211_dummy_Mtgm+Var140,
               data = df_mat.frame, family = binomial)

summary(lrAppRF)
par(mfrow=c(2,2))
plot(lrRf.Churn)
par(mfrow=c(1,1))

#Refitted with statitically significant variables
lrAppRFRe <- glm(appetency~Var126+Var73+Var218_dummy_cJvF+
                      Var216_dummy_kZJtVhC+Var225_dummy_kG3k+Var140,
                      data = df_mat.frame, family = binomial)


summary(lrAppRFRe)

#Test for better fit between nested models:
anova(lrAppRFRe,lrAppRF,test="Chisq")

par(mfrow=c(1,1))
fit.Rf <- lrAppRF$fitted
hist(fit.Rf)

#################################################
# Chi-square goodness of fit test for RandomForest Variables
#################################################
# Calculate residuals across all individuals
r.Rf <- (df_mat.frame$appetency - fit.Rf)/(sqrt(fit.Rf*(1-fit.Rf)))
# Sum of squares of these residuals follows a chi-square
sum(r.Rf^2)
# Sum of squares of these residuals follows a chi-square
1- pchisq(66473.61, df=49971)

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

############################
#Logistic Models
############################


#Decision Tree Variables:
lrfitDT <-glm(appetency~Var126+Var218_dummy_cJvF+Var38,data=train.frame,
              family = binomial)

summary(lrfitDT)

#Checking prediction quality on training
Plogit <- predict(lrfitDT,newdata =train.frame,type = "response")
p.app <- round(Plogit)

require(e1071)
require(caret)
confusionMatrix(p.app,train.frame$appetency)

#Checking prediction quality on test
PlogitTest <- predict(lrfitDT,newdata=test.frame,type = "response")
p.AppTest <- round(PlogitTest)
confusionMatrix(p.AppTest,test.frame$appetency)

#How good is the Logistic model in-sample (AUC)
DT.scores <- prediction(Plogit,train.frame$appetency)
plot(performance(DT.scores,'tpr','fpr'),col='red')
abline(0,1,lty=8,col='grey')
DT.auc <- performance(DT.scores,'auc')
DT.auc

#How good is the Logistic model out-sample (AUC)
DT.scores.test <- prediction(PlogitTest,test.frame$appetency)
#ROC plot for logistic regression
plot(performance(DT.scores.test,'tpr','fpr'),col='red')
abline(0,1,lty=8,col='grey')
#AUC value
DT.auc.test <- performance(DT.scores.test,'auc')
DT.auc.test

#LASSO
lrfitLASSO <-glm(appetency~Var28+Var34+Var38+Var64+Var67+Var81+Var125+Var126+Var14+Var152+Var153+Var171+Var126_missing+Var194_dummy_SEuy+Var197_dummy_487l+Var197_dummy_TyGl+Var204_dummy_15m3+Var204_dummy_m_h1+Var205_dummy_VpdQ+Var206_dummy_43pnToF+Var208_dummy_kIsH+Var210_dummy_uKAI+Var212_dummy_NhsEn4L+Var216_dummy_other+Var216_dummy_7WwuNea+Var216_dummy_kZJtVhC+Var218_dummy_cJvF+Var218_dummy_UYBR+Var223_dummy_M_8D+Var226_dummy_xb3V+Var226_dummy_fKCe+Var226_dummy_FSa2+Var226_dummy_uWr3,data=train.frame,family = binomial)

summary(lrfitLASSO)

#Checking prediction quality on training
logitLASSO.train <- predict(lrfitLASSO,newdata =train.frame,type = "response")
pLASSO.app <- round(logitLASSO.train)

require(e1071)
require(caret)
confusionMatrix(pLASSO.app,train.frame$appetency)

#Checking prediction quality on test
logitLASSO.test <- predict(lrfitLASSO,newdata=test.frame,type = "response")
p.AppTest <- round(logitLASSO.test)
confusionMatrix(p.AppTest,test.frame$appetency)

#How good is the Logistic model in-sample:
LASSO.scores <- prediction(logitLASSO.train,train.frame$appetency)
plot(performance(LASSO.scores,'tpr','fpr'),col='red')
abline(0,1,lty=8,col='grey')
LASSO.auc <- performance(LASSO.scores,'auc')
LASSO.auc

#How good is the Logistic model out-sample:
LASSO.scores.test <- prediction(logitLASSO.test,test.frame$appetency)
#ROC plot for logistic regression
plot(performance(LASSO.scores.test,'tpr','fpr'),col='red')
abline(0,1,lty=8,col='grey')
#AUC value
LASSO.auc.test <- performance(LASSO.scores.test,'auc')
LASSO.auc.test


#Random Forest
lrfitRf <-glm(appetency~Var126+Var119+Var28+Var6+Var109+Var81+Var133+Var153+Var25+Var113+Var83+Var73+Var160+ Var85+Var218_dummy_UYBR+Var22+Var38+Var24+Var134+Var144+Var21+Var218_dummy_cJvF+Var216_dummy_kZJtVhC+ Var123+Var163+Var225_dummy_kG3k+Var211_dummy_Mtgm+Var140,data=train.frame,family = binomial)

summary(lrfitRf)

#Checking prediction quality on training
logitRf.train <- predict(lrfitRf,newdata =train.frame,type = "response")
pRf.app <- round(logitRf.train)

require(e1071)
require(caret)
confusionMatrix(pRf.app,train.frame$appetency)

#Checking prediction quality on test
logitRf.test <- predict(lrfitRf,newdata=test.frame,type = "response")
p.AppTest <- round(logitRf.test)
confusionMatrix(p.AppTest,test.frame$appetency)

#How good is the Logistic model in-sample:
Rf.scores <- prediction(logitRf.train,train.frame$appetency)
plot(performance(Rf.scores,'tpr','fpr'),col='red')
abline(0,1,lty=8,col='grey')
Rf.auc <- performance(Rf.scores,'auc')
Rf.auc

#How good is the Logistic model out-sample:
Rf.scores.test <- prediction(logitRf.test,test.frame$appetency)
#ROC plot for logistic regression
plot(performance(Rf.scores.test,'tpr','fpr'),col='red')
abline(0,1,lty=8,col='grey')
#AUC value
Rf.auc.test <- performance(Rf.scores.test,'auc')
Rf.auc.test

#plot multiple ROC curves
?plot.performance
library(ROCR)
pred1 <- prediction(logitRf.test,test.frame$appetency)
pred2 <- prediction(logitLASSO.test,test.frame$appetency)
pred3 <- prediction(PlogitTest,test.frame$appetency)
  
perf1 <- performance(pred1, "tpr", "fpr")
perf2 <- performance(pred2, "tpr", "fpr")
perf3 <- performance(pred3, "tpr", "fpr")

plot(perf1, colorize = TRUE)
plot(perf2, add = TRUE, colorize = TRUE)
plot(perf3, add = TRUE, colorize = TRUE)
abline(0,1,lty=8,col='grey')


# make logsitc regression predictions
app_lreg_udy_predictions <- predict(lrfitDT, df_mat.frame[-train_ind,],
                                      type = 'response')

# churn_svm_udy_predictions <- predict(lrfit, df_mat[-train_ind,],
#                                       type = 'response')


# save the output
setwd('c:/Users/Uduak/Dropbox/pred_454_team')
save(list = c('lrfitDT', 'app_lreg_udy_predictions'),
     file = 'models/appetency/app_lreg_udy.RData')
# save(list = c('churn_svm_udy', 'churn_svm_udy_predictions'),
#      file = 'models/churn/churn_svm_udy.RData')
