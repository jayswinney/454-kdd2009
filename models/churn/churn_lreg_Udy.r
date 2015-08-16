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
          'C:/Users/JoeD/Dropbox/pred_454_team')

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
df_mat.frame$churn <- df$churn
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
df_mat.frame$churn <- factor(df_mat.frame$churn)
churnTree <- rpart(churn~.,method="class",data = df_mat.frame,
                 control=rpart.control(minsplit=10, minbucket=10, cp=0.001))
churnTree
printcp(churnTree)
plot(churnTree, uniform=TRUE)
text(churnTree, all=TRUE,cex=0.75, splits=TRUE, use.n=TRUE, xpd=TRUE)
?predict
p <- predict(churnTree,newdata=df_mat.frame,type="class")
table(actual=df_mat.frame$churn,predicted=p)

par(mar = c(5, 10, 4, 2) + 0.1)
barplot(churnTree$variable.importance,horiz=T,las=1, cex.names = 0.75)
par(mar = c(5, 4, 4, 2) + 0.1)


#GOF logistic regression using variables from Tree selection
lrchurn.tree <- glm(churn~Var126+Var126_missing+Var217_dummy_other+
                      Var221_dummy_zCkv+Var229_dummy_missing+Var28+
                      Var65+Var73,data = train, family = binomial)

summary(lrchurn.tree)

#Refitted with statitically significant variables
lrchurn.treeRe <- glm(churn~Var126+Var126_missing+Var217_dummy_other+
                      Var229_dummy_missing+Var28+
                      Var73,data = train, family = binomial)
summary(lrchurn.treeRe)

anova(lrchurn.treeRe,lrchurn.tree,test="Chisq")


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
churn.LASSO <- glmnet(train.glm,train$churn,alpha=1, family = "binomial")

plot(churn.LASSO,xvar="lambda",label=TRUE)
grid()
# Show number of selected variables
churn.LASSO

#Use cross validation to select the best lambda (minlambda) and lambda.1se
#Cross Validated LASSO
churnCVlasso <- cv.glmnet(train.glm,train$churn)
#plot(fit.lasso,xvar="dev",label=TRUE)
plot(churnCVlasso)
coef(churnCVlasso)

# coef(churnCVlasso, s="lambda.1se") # Gives coefficients lambda.1se
# best_lambda <- churnCVlasso$lambda.1se
# best_lambda

#GOF logistic regression models LASSO Variables selected
names(train)
lrchurnLASSO <- glm(churn~Var7+Var73+Var113+Var126+Var22_missing+
                    Var25_missing+Var28_missing+Var126_missing+
                    Var205_dummy_sJzTlal+Var206_dummy_IYzP+Var210_dummy_g5HH+
                    Var212_dummy_NhsEn4L+Var217_dummy_other+Var218_dummy_cJvF+
                    Var218_dummy_missing+Var229_dummy_missing,data = train,
                  family = binomial)

summary(lrchurnLASSO)
# par(mfrow=c(2,2))
# plot(lrAppLASSO)
# par(mfrow=c(1,1))

#Refitted after dropping variables that could not be estimated
lrchurnLASSO1 <- glm(churn~Var7+Var73+Var113+Var126+Var22_missing+
                       Var28_missing+Var126_missing+
                       Var205_dummy_sJzTlal+Var206_dummy_IYzP+Var210_dummy_g5HH+
                       Var212_dummy_NhsEn4L+Var217_dummy_other+Var218_dummy_cJvF+
                       Var229_dummy_missing,data = train,
                     family = binomial)

summary(lrchurnLASSO1)

#Refitted with significant variables
lrchurnLASSO2 <- glm(churn~Var7+Var73+Var113+Var126+Var28_missing+Var126_missing+
                       Var205_dummy_sJzTlal+Var210_dummy_g5HH+
                       Var212_dummy_NhsEn4L+Var217_dummy_other+Var218_dummy_cJvF+
                       Var229_dummy_missing,data = train,
                     family = binomial)

summary(lrchurnLASSO2)


#Test for better fit between nested models: chi-sq
anova(lrchurnLASSO2,lrchurnLASSO1,test="Chisq")
#Test for better fit between nested models: AIC
AIC(lrchurnLASSO1,lrchurnLASSO2)


par(mfrow=c(1,1))
fit.LASSO.churn <- lrchurnLASSO2$fitted
hist(fit.LASSO.churn)
par(mfrow=c(2,2))
plot(lrchurnLASSO2)
par(mfrow=c(1,1))

## Random Forest
?randomForest
# ---- rf_churn ----
library(randomForest)
set.seed(101)
churnRf <- randomForest(churn~.,data = train,importance = TRUE)
churn.important <- importance(churnRf, type = 1, scale = FALSE)
varImp(churnRf)
# ----
# ---- plot_rf_churn ----
varImpPlot(churnRf, type = 1, main = 'Variable Importance churn')
varImpPlot(churnRf, type = 2, main = 'Variable Importance churn')

# write the variable importance to a file that can be read into excel
fo <- file("rf.txt", "w")
imp <- importance(churnRf)
write.table(imp, fo, sep="\t")
flush(fo)
close(fo)
#GOF logistic regression using variables from Tree selection
lrchurnRF <- glm(churn~Var113+Var73+Var126+Var6+Var119+Var153+Var25+Var160+
                   Var133+Var81+Var112+Var13+Var125+Var109+Var38+
                   Var210_dummy_g5HH+Var21+Var24+Var123+Var83+Var28+
                   Var22+Var144+Var85+Var140+Var74+Var134+Var126_missing+
                   Var76+Var211_dummy_L84s,data = train, family = binomial)

summary(lrchurnRF)
par(mfrow=c(2,2))
plot(lrRf.Churn)
par(mfrow=c(1,1))

#Refitted with statitically significant variables
lrchurnRF1 <- glm(churn~Var113+Var73+Var126+Var6+Var81+Var210_dummy_g5HH+
                   Var28+Var74+Var126_missing+Var211_dummy_L84s,
                 data = train, family = binomial)

summary(lrchurnRF1)

#Test for better fit between nested models:
anova(lrchurnRF1,lrchurnRF,test="Chisq")
AIC(lrchurnRF1,lrchurnRF)



#Decision Tree Variables:
lrfitchurnDT <-glm(churn~Var126+Var126_missing+Var217_dummy_other+
                Var221_dummy_zCkv+Var229_dummy_missing+Var28+Var65+
                Var73,data=train,family = binomial)

summary(lrfitchurnDT)

#Checking prediction quality on training
Pchurn <- predict(lrfitchurnDT,newdata =train,type = "response")
p.churn <- round(Pchurn)

require(e1071)
require(caret)
confusionMatrix(p.churn,train$churn)

#Checking prediction quality on test
PchurnTest <- predict(lrfitchurnDT,newdata=test,type = "response")
p.churnTest <- round(PchurnTest)
confusionMatrix(p.churnTest,test$churn)

#How good is the Logistic model in-sample (AUC)
DT.churnscores <- prediction(Pchurn,train$churn)
plot(performance(DT.churnscores,'tpr','fpr'),col='red')
abline(0,1,lty=8,col='grey')
DT.churnauc <- performance(DT.churnscores,'auc')
DT.churnauc

#How good is the Logistic model out-sample (AUC)
DT.churnscores.test <- prediction(PchurnTest,test$churn)
#ROC plot for logistic regression
plot(performance(DT.churnscores.test,'tpr','fpr'),col='red')
abline(0,1,lty=8,col='grey')
#AUC value
DT.churnauc.test <- performance(DT.churnscores.test,'auc')
DT.churnauc.test

#LASSO
lrchurnLASSO <-glm(churn~Var7+Var73+Var113+Var126+Var28_missing+
          Var126_missing+Var205_dummy_sJzTlal+Var210_dummy_g5HH+
          Var212_dummy_NhsEn4L+Var217_dummy_other+Var218_dummy_cJvF+
          Var229_dummy_missing,data=train,family = binomial)

summary(lrchurnLASSO)

#Checking prediction quality on training
pchurnLASSO <- predict(lrchurnLASSO,newdata =train,type = "response")
pLASSO.churn <- round(pchurnLASSO)

require(e1071)
require(caret)
confusionMatrix(pLASSO.churn,train$churn)

#Checking prediction quality on test
churnLASSO.test <- predict(lrchurnLASSO,newdata=test,type = "response")
p.churnTest <- round(churnLASSO.test)
confusionMatrix(p.churnTest,test$churn)

#How good is the Logistic model in-sample:
LASSO.churnscores <- prediction(pchurnLASSO,train$churn)
plot(performance(LASSO.churnscores,'tpr','fpr'),col='red')
abline(0,1,lty=8,col='grey')
LASSO.churnauc <- performance(LASSO.churnscores,'auc')
LASSO.churnauc

#How good is the Logistic model out-sample:
LASSO.churnscores.test <- prediction(churnLASSO.test,test$churn)
#ROC plot for logistic regression
plot(performance(LASSO.churnscores.test,'tpr','fpr'),col='red')
abline(0,1,lty=8,col='grey')
#AUC value
LASSO.churnauc.test <- performance(LASSO.churnscores.test,'auc')
LASSO.churnauc.test


#Random Forest
lrchurnRf <-glm(churn~Var113+Var73+Var126+Var6+Var81+Var210_dummy_g5HH+
                  Var28+Var74+Var126_missing+Var211_dummy_L84s,data=train,
                family = binomial)

summary(lrchurnRf)

#Checking prediction quality on training
churnRf.train <- predict(lrchurnRf,newdata =train,type = "response")
pchurnRf.app <- round(churnRf.train)
confusionMatrix(pchurnRf.app,train$churn)

#Checking prediction quality on test
churnRf.test <- predict(lrchurnRf,newdata=test,type = "response")
p.churnTest <- round(churnRf.test)
confusionMatrix(p.churnTest,test$churn)

#How good is the Logistic model in-sample:
churnRf.scores <- prediction(churnRf.train,train$churn)
plot(performance(churnRf.scores,'tpr','fpr'),col='red')
abline(0,1,lty=8,col='grey')
Rf.churnauc <- performance(churnRf.scores,'auc')
Rf.churnauc

#How good is the Logistic model out-sample:
Rf.churnscores.test <- prediction(churnRf.test,test$churn)
#ROC plot for logistic regression
plot(performance(Rf.churnscores.test,'tpr','fpr'),col='red')
abline(0,1,lty=8,col='grey')
#AUC value
Rf.churnauc.test <- performance(Rf.churnscores.test,'auc')
Rf.churnauc.test

#plot ROC curves
?plot.performance
library(ROCR)

predchurn <- prediction(churnLASSO.test,test$churn)

perfchurn <- performance(predchurn, "tpr", "fpr")

plot(perfchurn, main="ROC of churn")

abline(0,1,lty=8,col='blue')


# make logsitic regression predictions
churn_lreg_udy_predictions <- predict(lrchurnLASSO, test,
                                    type = 'response')

# churn_svm_udy_predictions <- predict(lrfit, df_mat[-train_ind,],
#                                       type = 'response')


# save the output
save(list = c('lrchurnLASSO', 'churn_lreg_udy_predictions'),
     file = 'models/appetency/churn_lreg_udy.RData')
