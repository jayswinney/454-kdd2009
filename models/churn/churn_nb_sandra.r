##  churn_nb_sandra

##################################################################################################
###       COMMENT FOR NAIVE MODEL FOR CHURN
##################################################################################################
#The Na�ve Bayes technique was applied in a computational EDA manner to obtain the highest AUC
#score for churn.
#
#The variable selection process was based on the smallest deviance of each variable.
#This variable selection process resulted in 47 variables out of 230 with deviance of 291.862
#based on the Calibration data set.
#
#The Calibration data set is a 10% random selection of observations from the original data set.

#The resulting Naive Bayes model using the selected variables shows that the model is
#overfitting the data because the AUC Score with the Train data is 0.9315 but the AUC Score
#with the Test data is 0.6622, which is about a 27-point difference.  However, the AUC for
#the Test is significantly above 0.50 of a random guess, so we could consider the Naive Bayes
#model for churn to be reasonably accurate.
#
##################################################################################################
##################################################################################################

###   SET DIRECTORY PATH:
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

###   READ DATA FILES:
d <- read.table('data/orange_small_train.data',
                header=T,
                sep='\t',
                na.strings=c('NA',''))

churn <- read.table('data/orange_small_train_churn.labels',
                    header=F,sep='\t')
d$churn <- churn$V1

upselling <- read.table('data/orange_small_train_upselling.labels',
                        header=F,sep='\t')
d$upselling <- upselling$V1

appetency <- read.table('data/orange_small_train_appetency.labels',
                        header=F,sep='\t')
d$appetency <- appetency$V1


###   CREATING TRAIN, CALIBRATION, AND TEST DATA SETS
# this portion of the code should be copied exactly
# in every data transformation script
# that way we will all be using the same training/testing data
set.seed(123)
smp_size <- floor(0.70 * nrow(d))
test_ind <- seq_len(nrow(d))
train_ind <- sample(test_ind, size = smp_size)
# remove train observations from test
test_ind <- test_ind[! test_ind %in% train_ind]
# create an ensemble test set
set.seed(123)
smp_size <- floor(0.15 * nrow(d))
ens_ind <- sample(test_ind, size = smp_size)
# remove ensemble observations from test
test_ind <- test_ind[! test_ind %in% ens_ind]
# partition the data
ensemble_test <- d[ens_ind, ]
train <- d[train_ind, ]
test <- d[test_ind, ]

tiny_ind <- sample(seq_len(nrow(d)), size = floor(0.10 * nrow(d)))
dCal <- d[tiny_ind, ]



###  SETTING OTHER VARIABLES:
outcomes=c('churn','appetency','upselling')

vars <- setdiff(colnames(train), c(outcomes,'rgroup'))

catVars <- vars[sapply(train[,vars],class) %in% c('factor','character')]

numericVars <- vars[sapply(train[,vars],class) %in% c('numeric','integer')]

rm(list=c('d','churn','appetency','upselling'))

outcome <- 'churn'

pos <- '1'


# Title: Function to build single-variable models for categorical variables
# example 6.4
mkPredC <- function(outCol,varCol,appCol) {
  pPos <- sum(outCol==pos)/length(outCol)
  naTab <- table(as.factor(outCol[is.na(varCol)]))
  pPosWna <- (naTab/sum(naTab))[pos]
  vTab <- table(as.factor(outCol),varCol)
  pPosWv <- (vTab[pos,]+1.0e-3*pPos)/(colSums(vTab)+1.0e-3)
  pred <- pPosWv[appCol]
  pred[is.na(appCol)] <- pPosWna
  pred[is.na(pred)] <- pPos
  pred
}

# Title: Applying single-categorical variable models to all of our datasets
# example 6.5
for(v in catVars) {
  pi <- paste('pred',v,sep='')
  train[,pi] <- mkPredC(train[,outcome],train[,v],train[,v])
  dCal[,pi] <- mkPredC(train[,outcome],train[,v],dCal[,v])
  test[,pi] <- mkPredC(train[,outcome],train[,v],test[,v])
}


# Title: Scoring categorical variables by AUC
# example 6.6
library('ROCR')

calcAUC <- function(predcol,outcol) {
  perf <- performance(prediction(predcol,outcol==pos),'auc')
  as.numeric(perf@y.values)
}

for(v in catVars) {
  pi <- paste('pred',v,sep='')
  aucTrain <- calcAUC(train[,pi],train[,outcome])
  if(aucTrain>=0.8) {
    aucCal <- calcAUC(dCal[,pi],dCal[,outcome])
    aucTest <- calcAUC(test[,pi],test[,outcome])
    print(sprintf("%s, trainAUC: %4.3f calibrationAUC: %4.3f  testAUC: %4.3f",
                  pi,aucTrain,aucCal,aucTest))
  }
}


# Title: Scoring numeric variables by AUC
mkPredN <- function(outCol,varCol,appCol) {
  cuts <- unique(as.numeric(quantile(varCol,probs=seq(0, 1, 0.1),na.rm=T)))
  varC <- cut(varCol,cuts)
  appC <- cut(appCol,cuts)
  mkPredC(outCol,varC,appC)
}

for(v in numericVars) {
  pi <- paste('pred',v,sep='')
  train[,pi] <- mkPredN(train[,outcome],train[,v],train[,v])
  test[,pi] <- mkPredN(train[,outcome],train[,v],test[,v])
  dCal[,pi] <- mkPredN(train[,outcome],train[,v],dCal[,v])
  aucTrain <- calcAUC(train[,pi],train[,outcome])
  if(aucTrain>=0.55) {
    aucCal <- calcAUC(dCal[,pi],dCal[,outcome])
    aucTest <- calcAUC(test[,pi],test[,outcome])
    print(sprintf("%s, trainAUC: %4.3f calibrationAUC: %4.3f TestAUC: %4.3f",
                  pi,aucTrain,aucCal,aucTest))
  }
}


# Title: Basic variable selection
# example 6.11
logLikelyhood <- function(outCol,predCol) {
  sum(ifelse(outCol==pos,log(predCol),log(1-predCol)))
}

selVars <- c() # this variable stores the selected variable
minStep <- 5
baseRateCheck <- logLikelyhood(dCal[,outcome],
                               sum(dCal[,outcome]==pos)/length(dCal[,outcome]))

for(v in catVars) {
  pi <- paste('pred',v,sep='')
  liCheck <- 2*((logLikelyhood(dCal[,outcome],dCal[,pi]) -
                   baseRateCheck))
  if(liCheck>minStep) {
    print(sprintf("%s, calibrationScore: %g",
                  pi,liCheck))
    selVars <- c(selVars,pi)
  }
}


for(v in numericVars) {
  pi <- paste('pred',v,sep='')
  liCheck <- 2*((logLikelyhood(dCal[,outcome],dCal[,pi]) -
                   baseRateCheck))
  if(liCheck>=minStep) {
    print(sprintf("%s, calibrationScore: %g",
                  pi,liCheck))
    selVars <- c(selVars,pi)
  }
}



# Title: Plotting the receiver operating characteristic curve
# example 6.21
plotROC <- function(predcol,outcol) {
  perf <- performance(prediction(predcol,outcol==pos),'tpr','fpr')
  pf <- data.frame(
    FalsePositiveRate=perf@x.values[[1]],
    TruePositiveRate=perf@y.values[[1]])
  ggplot() +
    geom_line(data=pf,aes(x=FalsePositiveRate,y=TruePositiveRate)) +
    geom_line(aes(x=c(0,1),y=c(0,1)))
}


# Title: Using a Naive Bayes package
# example 6.24
library('e1071')
# with only the selVars the variables:
ff <- paste('as.factor(',outcome,'>0) ~ ', paste(selVars,collapse=' + '),sep='')

nbmodel <- naiveBayes(as.formula(ff),data=train)

train$nbpred <- predict(nbmodel,newdata=train,type='raw')[,'TRUE']
dCal$nbpred <- predict(nbmodel,newdata=dCal,type='raw')[,'TRUE']
test$nbpred <- predict(nbmodel,newdata=test,type='raw')[,'TRUE']

calcAUC(train$nbpred,train[,outcome])
## [1] 0.9315453  # with selVars
calcAUC(dCal$nbpred,dCal[,outcome])
## [1] 0.8739238  # with selVars
calcAUC(test$nbpred,test[,outcome])
## [1] 0.6622495  # with selVars

# install.packages("ggplot2")
require("ggplot2")
print(plotROC(train$nbpred,train[,outcome]))
print(plotROC(dCal$nbpred,dCal[,outcome]))
print(plotROC(test$nbpred,test[,outcome]))

#### Model Output .RData for Project:
churn_nb_sandra_model <- naiveBayes(as.formula(ff),data=train)
churn_nb_sandra_predictions <-predict(churn_nb_sandra_model,
                                      newdata=test,type='raw')[,'TRUE']

churn_ens_nb_sandra_predictions <-predict(churn_nb_sandra_model,
                                          ensemble_test,type='raw')[,'TRUE']

# save the output
save(list = c('churn_nb_sandra_model', 'churn_nb_sandra_predictions',
              'churn_ens_nb_sandra_predictions'),
     file = 'models/churn/churn_nb_sandra.RData')
