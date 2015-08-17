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
# choose a script to load and transform the data
source('data_transformations/impute_0.r')


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
names(df_mat.frame)
dim(df_mat.frame)

# Create train and test data set for data frame

set.seed(123)
smp_size <- floor(0.70 * nrow(df_mat.frame))
test_ind <- seq_len(nrow(df_mat.frame))
train_ind <- sample(test_ind, size = smp_size)
# remove train observations from test
test_ind <- test_ind[! test_ind %in% train_ind]
# create an ensemble test set
set.seed(123)
smp_size <- floor(0.15 * nrow(df_mat.frame))
ens_ind <- sample(test_ind, size = smp_size)
# remove ensemble observations from test
test_ind <- test_ind[! test_ind %in% ens_ind]
# partition the data
ensemble_test <- df_mat.frame[ens_ind, ]
train <- df_mat.frame[train_ind, ]
test <- df_mat.frame[test_ind, ]

# ----
#Exploratory decision Tree (Classification) for appetency
library(rpart)
library(rpart.plot)
df_mat.frame$appetency <- factor(df_mat.frame$appetency)
AppTree <- rpart(appetency~.,method="class",data = df_mat.frame,
                 control=rpart.control(minsplit=10, minbucket=10, cp=0.001))
AppTree
printcp(AppTree)
plot(AppTree, uniform=TRUE)
text(AppTree, all=TRUE,cex=0.75, splits=TRUE, use.n=TRUE, xpd=TRUE)
?predict
p <- predict(AppTree,newdata=df_mat.frame,type="class")
table(actual=df_mat.frame$appetency,predicted=p)

par(mar = c(5, 10, 4, 2) + 0.1)
barplot(AppTree$variable.importance,horiz=T,las=1, cex.names = 0.75)
par(mar = c(5, 4, 4, 2) + 0.1)

# #Graphical analysis of DT selected variables
# summary(Var126)
# ggplot(df) + geom_bar(aes(x=Var126,group=appetency,
#                           fill=appetency))
# table(Var204_dummy_RVjC)
# ggplot(df) + geom_bar(aes(x=Var204_dummy_RVjC,group=appetency,
#                           fill=appetency))
# table(Var218_dummy_cJvF)
# ggplot(df) + geom_bar(aes(x=Var218_dummy_cJvF,group=appetency,
#                           fill=appetency))
# summary(Var25)
# ggplot(df) + geom_bar(aes(x=Var25,group=appetency,
#                           fill=appetency))
# summary(Var38)
# ggplot(df) + geom_bar(aes(x=Var38,group=appetency,
#                           fill=appetency))
# summary(Var57)
# ggplot(df) + geom_bar(aes(x=Var57,group=appetency,
#                           fill=appetency))


#GOF logistic regression using variables from Tree selection
lrApp.tree <- glm(appetency~Var112+Var113+Var126+Var140+
                    Var153+Var21+Var216_dummy_beK4AFX+Var218_dummy_cJvF+Var24+
                    Var38+Var6+Var76+Var81,data = train, family = binomial)

summary(lrApp.tree)

#Refitted with statitically significant variables
lrApp.treeRe <- glm(appetency~Var126+Var140+Var218_dummy_cJvF
                      ,data = train, family = binomial)

summary(lrApp.treeRe)

anova(lrApp.treeRe,lrApp.tree,test="Chisq")


par(mfrow=c(1,1))
fit <- lrApp.tree$fitted
hist(fit)
par(mfrow=c(1,1))

# #################################################
# # Chi-square goodness of fit test
# #################################################
# # Calculate residuals across all individuals
# r.tree <- (df_mat.frame$appetency - fit)/(sqrt(fit*(1-fit)))
# # Sum of squares of these residuals follows a chi-square
# sum(r.tree^2)
# #Calculate the p-value from the test
# 1- pchisq(65795.97, df=49996)

#Exploratory logistic Regression with LASSO - Udy's code changes
# ---- LASSO_appetency ----
library(glmnet)
# regularized logistic regression with LASSO

train.glm <- as.matrix(train[-552])
dim(train.glm)
app.LASSO <- glmnet(train.glm,train$appetency,alpha=1, family = "binomial")

plot(app.LASSO,xvar="lambda",label=TRUE)
grid()
# Show number of selected variables
app.LASSO

#Use cross validation to select the best lambda (minlambda) and lambda.1se
#Cross Validated LASSO
appCVlasso <- cv.glmnet(train.glm,train$appetency)
#plot(fit.lasso,xvar="dev",label=TRUE)
plot(appCVlasso)
coef(appCVlasso)

coef(appCVlasso, s="lambda.min") # Gives coefficients lambda.min
best_lambda <- appCVlasso$lambda.min
best_lambda

attach(df_mat.frame)

#GOF logistic regression models LASSO Variables selected
names(train)
lrAppLASSO <- glm(appetency~Var28+Var34+Var38+Var44+Var58+Var64+Var67+Var75+
                    Var81+Var84+Var95+Var124+Var125+
                    Var126+Var140+Var144+Var152+Var162+Var171+Var177+
                    Var181+Var126_missing+Var194_dummy_SEuy+
                    Var197_dummy_0Xwj+Var197_dummy_487l+Var197_dummy_TyGl+
                    Var197_dummy_z32l+Var204_dummy_YULl+Var204_dummy_15m3+
                    Var204_dummy_4N0K+Var204_dummy_m_h1+Var205_dummy_VpdQ+
                    Var205_dummy_sJzTlal+Var206_dummy_zm5i+Var206_dummy_43pnToF+
                    Var208_dummy_kIsH+Var210_dummy_uKAI+Var210_dummy_other+
                    Var211_dummy_L84s+Var211_dummy_Mtgm+
                    Var212_dummy_NhsEn4L+Var216_dummy_other+
                    Var216_dummy_7WwuNea+Var216_dummy_kZJtVhC+
                    Var218_dummy_cJvF+Var218_dummy_UYBR+Var219_dummy_OFWH+
                    Var223_dummy_M_8D+Var226_dummy_xb3V+Var226_dummy_fKCe+
                    Var226_dummy_Aoh3+Var226_dummy_uWr3+Var226_dummy_7P5s+
                    Var228_dummy_R4y5gQQWY8OodqDV,data = train, family = binomial)

summary(lrAppLASSO)
# par(mfrow=c(2,2))
# plot(lrAppLASSO)
# par(mfrow=c(1,1))

#Refitted with statitically significant variables
lrAppLASSORe1 <- glm(appetency~Var34+Var44+Var64+Var67+Var84+Var125+Var126+
                       Var140+Var144+Var177+Var204_dummy_m_h1+Var205_dummy_VpdQ+
                       Var211_dummy_L84s+Var212_dummy_NhsEn4L+
                       Var216_dummy_7WwuNea+Var216_dummy_kZJtVhC+Var218_dummy_cJvF+
                       Var226_dummy_xb3V,data = train, family = binomial)

summary(lrAppLASSORe1)

# variables 44 and 64 dropped because they are not statistically insignificant.Model refitted
lrAppLASSORe <- glm(appetency~Var34+Var67+Var84+Var125+Var126+
                      Var140+Var144+Var177+Var204_dummy_m_h1+Var205_dummy_VpdQ+
                      Var211_dummy_L84s+Var212_dummy_NhsEn4L+
                      Var216_dummy_7WwuNea+Var216_dummy_kZJtVhC+Var218_dummy_cJvF+
                      Var226_dummy_xb3V,data = train, family = binomial)


summary(lrAppLASSORe)


#Test for better fit between nested models: chi-sq
anova(lrAppLASSORe,lrAppLASSO,test="Chisq")
#Test for better fit between nested models: AIC
AIC(lrAppLASSO,lrAppLASSORe)


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
?randomForest
appRf <- randomForest(appetency~.,data = train,importance = TRUE)
rf.important <- importance(appRf, type = 1)
barplot(rf.important)
# ----
# ---- plot_rf_churn ----
varImpPlot(appRf, type = 1, main = 'Variable Importance appetency')
varImpPlot(appRf, type = 2, main = 'Variable Importance appetency')

par(mar = c(5, 10, 4, 2) + 0.1)
barplot(rf.important,horiz=T)
par(mar = c(5, 4, 4, 2) + 0.1)

# write the variable importance to a file that can be read into excel
fo <- file("rf.txt", "w")
imp <- importance(appRf)
write.table(imp, fo, sep="\t")
flush(fo)
close(fo)
#GOF logistic regression using variables from Tree selection
lrAppRF <- glm(appetency~Var126+Var6+Var81+Var113+Var119+Var28+Var25+
                 Var85+Var218_dummy_UYBR+Var22+Var73+Var153+Var83+
                 Var133+Var123+Var109+Var160+Var125+Var218_dummy_cJvF+
                 Var21+ Var134+Var74+Var112+Var211_dummy_Mtgm+
                 Var225_dummy_kG3k+Var211_dummy_L84s+Var38+Var163+Var76+Var140,
               data = train, family = binomial)

summary(lrAppRF)
par(mfrow=c(2,2))
plot(lrRf.Churn)
par(mfrow=c(1,1))

#Refitted with statitically significant variables
lrAppRFRe <- glm(appetency~Var126+Var73+Var125+Var218_dummy_cJvF+
                   Var211_dummy_Mtgm+Var140,
                 data = train, family = binomial)


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

# set.seed(123)
# smp_size <- floor(0.75 * nrow(df_mat.frame))
# train_ind <- sample(seq_len(nrow(df_mat.frame)), size = smp_size)
# # making a "tiny" data set so I cn quickly test r markdown and graphical paramters
# # this will be removed in the submitted version
# tiny_ind <- sample(seq_len(nrow(df_mat.frame)), size = floor(0.01 * nrow(df)))
# # split the data
# train.frame <- df_mat.frame[train_ind, ]
# test.frame <- df_mat.frame[-train_ind, ]
# tiny.frame <- df_mat.frame[tiny_ind, ]

############################
#Logistic Models
############################


#Decision Tree Variables:
lrfitDT <-glm(appetency~Var112+Var113+Var126+Var140+Var153+Var21+
                Var216_dummy_beK4AFX+Var218_dummy_cJvF+Var24+Var38+
                Var6+Var76+Var81,data=train,family = binomial)

summary(lrfitDT)

#Checking prediction quality on training
Plogit <- predict(lrfitDT,newdata =train,type = "response")
p.app <- round(Plogit)

require(e1071)
require(caret)
confusionMatrix(p.app,train$appetency)

#Checking prediction quality on test
PlogitTest <- predict(lrfitDT,newdata=test,type = "response")
p.AppTest <- round(PlogitTest)
confusionMatrix(p.AppTest,test$appetency)

#How good is the Logistic model in-sample (AUC)
DT.scores <- prediction(Plogit,train$appetency)
plot(performance(DT.scores,'tpr','fpr'),col='red')
abline(0,1,lty=8,col='grey')
DT.auc <- performance(DT.scores,'auc')
DT.auc

#How good is the Logistic model out-sample (AUC)
DT.scores.test <- prediction(PlogitTest,test$appetency)
#ROC plot for logistic regression
plot(performance(DT.scores.test,'tpr','fpr'),col='red')
abline(0,1,lty=8,col='grey')
#AUC value
DT.auc.test <- performance(DT.scores.test,'auc')
DT.auc.test

#LASSO
lrfitLASSO <-glm(appetency~Var28+Var34+Var38+Var44+Var58+Var64+Var67+Var75+
                   Var81+Var84+Var95+Var124+Var125+
                   Var126+Var140+Var144+Var152+Var162+Var171+Var177+
                   Var181+Var126_missing+Var194_dummy_SEuy+
                   Var197_dummy_0Xwj+Var197_dummy_487l+Var197_dummy_TyGl+
                   Var197_dummy_z32l+Var204_dummy_YULl+Var204_dummy_15m3+
                   Var204_dummy_4N0K+Var204_dummy_m_h1+Var205_dummy_VpdQ+
                   Var205_dummy_sJzTlal+Var206_dummy_zm5i+Var206_dummy_43pnToF+
                   Var208_dummy_kIsH+Var210_dummy_uKAI+Var210_dummy_other+
                   Var211_dummy_L84s+Var211_dummy_Mtgm+
                   Var212_dummy_NhsEn4L+Var216_dummy_other+
                   Var216_dummy_7WwuNea+Var216_dummy_kZJtVhC+
                   Var218_dummy_cJvF+Var218_dummy_UYBR+Var219_dummy_OFWH+
                   Var223_dummy_M_8D+Var226_dummy_xb3V+Var226_dummy_fKCe+
                   Var226_dummy_Aoh3+Var226_dummy_uWr3+Var226_dummy_7P5s+
                   Var228_dummy_R4y5gQQWY8OodqDV,data=train,family = binomial)

summary(lrfitLASSO)

#Checking prediction quality on training
logitLASSO.train <- predict(lrfitLASSO,newdata =train,type = "response")
pLASSO.app <- round(logitLASSO.train)

require(e1071)
require(caret)
confusionMatrix(pLASSO.app,train$appetency)

#Checking prediction quality on test
logitLASSO.test <- predict(lrfitLASSO,newdata=test,type = "response")
p.AppTest <- round(logitLASSO.test)
confusionMatrix(p.AppTest,test$appetency)

#How good is the Logistic model in-sample:
LASSO.scores <- prediction(logitLASSO.train,train$appetency)
plot(performance(LASSO.scores,'tpr','fpr'),col='red')
abline(0,1,lty=8,col='grey')
LASSO.auc <- performance(LASSO.scores,'auc')
LASSO.auc

#How good is the Logistic model out-sample:
LASSO.scores.test <- prediction(logitLASSO.test,test$appetency)
#ROC plot for logistic regression
plot(performance(LASSO.scores.test,'tpr','fpr'),col='red')
abline(0,1,lty=8,col='grey')
#AUC value
LASSO.auc.test <- performance(LASSO.scores.test,'auc')
LASSO.auc.test


#Random Forest
lrfitRf <-glm(appetency~Var126+Var6+Var81+Var113+Var119+Var28+Var25+Var85+
                Var218_dummy_UYBR+Var22+Var73+Var153+Var83+
                Var133+Var123+Var109+Var160+Var125+Var218_dummy_cJvF+Var21+
                Var134+Var74+Var211_dummy_Mtgm+Var225_dummy_kG3k+
                Var38+Var163+Var76+Var140,
              data=train,family = binomial)

summary(lrfitRf)

#Checking prediction quality on training
logitRf.train <- predict(lrfitRf,newdata =train,type = "response")
pRf.app <- round(logitRf.train)
confusionMatrix(pRf.app,train$appetency)

#Checking prediction quality on test
logitRf.test <- predict(lrfitRf,newdata=test,type = "response")
p.AppTest <- round(logitRf.test)
confusionMatrix(p.AppTest,test$appetency)

#How good is the Logistic model in-sample:
Rf.scores <- prediction(logitRf.train,train$appetency)
plot(performance(Rf.scores,'tpr','fpr'),col='red')
abline(0,1,lty=8,col='grey')
Rf.auc <- performance(Rf.scores,'auc')
Rf.auc

#How good is the Logistic model out-sample:
Rf.scores.test <- prediction(logitRf.test,test$appetency)
#ROC plot for logistic regression
plot(performance(Rf.scores.test,'tpr','fpr'),col='red')
abline(0,1,lty=8,col='grey')
#AUC value
Rf.auc.test <- performance(Rf.scores.test,'auc')
Rf.auc.test

#plot multiple ROC curves
?plot.performance
library(ROCR)

predapp <- prediction(logitLASSO.test,test$appetency)
perfapp <- performance(predapp, "tpr", "fpr")
plot(perfapp)
abline(0,1,lty=8,col='grey')


# make logsitic regression predictions
app_lreg_udy_predictions <- predict(lrfitLASSO, test,
                                      type = 'response')

# churn_svm_udy_predictions <- predict(lrfit, df_mat[-train_ind,],
#                                       type = 'response')


# save the output
save(list = c('lrfitLASSO', 'app_lreg_udy_predictions'),
     file = 'models/appetency/app_lreg_udy.RData')
