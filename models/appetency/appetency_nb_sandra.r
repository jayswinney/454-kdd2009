##  appetency_nb_sandra

################################################################################
###       COMMENT FOR NAIVE MODEL FOR APPETENCY
#################################################################################
#The Na�ve Bayes technique was applied in a computational EDA manner to obtain the highest
#AUC score for appetency.
#
#The variable selection process was based on the smallest deviance of each variable.
#This variable selection process resulted in 31 variables out of 230 with deviance of 186.294
#based on the Calibration data set.
#
#The Calibration data set is a 10% random selection of observations from the original data set.
#
#The resulting Naive Bayes model using the selected variables shows that the model is overfitting
#the data because the AUC Score with the Train data is 0.9619 but the AUC Score with the Test data
#is 0.7624, which is about a 20-point difference.  However, the AUC for the Test is significantly
#above 0.50 of a random guess, so we could consider the Naive Bayes model for appetency
#to be reasonably accurate.
#
#################################################################################
#################################################################################

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
# get the index for training/testing data
set.seed(123)
smp_size <- floor(0.85 * nrow(df))
train_ind <- sample(seq_len(nrow(df)), size = smp_size)
# making a "tiny" data set so I can quickly
# test r markdown and graphical paramters
# this will be removed in the submitted version
tiny_ind <- sample(seq_len(nrow(df)), size = floor(0.01 * nrow(df)))
# split the data
train <- df[train_ind, ]
test <- df[-train_ind, ]
tiny <- df[tiny_ind, ]

# create a validation set
set.seed(123)
smp_size <- 7500
train_ind <- sample(seq_len(nrow(train)), size = smp_size)

ensemble_test <- train[-train_ind, ]
train <- df[train_ind, ]


###  SETTING OTHER VARIABLES:
outcomes=c('churn','appetency','upselling')

vars <- setdiff(colnames(dTrain), c(outcomes,'rgroup'))

catVars <- vars[sapply(dTrain[,vars],class) %in% c('factor','character')]

numericVars <- vars[sapply(dTrain[,vars],class) %in% c('numeric','integer')]

rm(list=c('d','churn','appetency','upselling'))

outcome <- 'appetency'

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
  dTrain[,pi] <- mkPredC(dTrain[,outcome],dTrain[,v],dTrain[,v])
  dCal[,pi] <- mkPredC(dTrain[,outcome],dTrain[,v],dCal[,v])
  dTest[,pi] <- mkPredC(dTrain[,outcome],dTrain[,v],dTest[,v])
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
  aucTrain <- calcAUC(dTrain[,pi],dTrain[,outcome])
  if(aucTrain>=0.8) {
    aucCal <- calcAUC(dCal[,pi],dCal[,outcome])
    aucTest <- calcAUC(dTest[,pi],dTest[,outcome])
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
  dTrain[,pi] <- mkPredN(dTrain[,outcome],dTrain[,v],dTrain[,v])
  dTest[,pi] <- mkPredN(dTrain[,outcome],dTrain[,v],dTest[,v])
  dCal[,pi] <- mkPredN(dTrain[,outcome],dTrain[,v],dCal[,v])
  aucTrain <- calcAUC(dTrain[,pi],dTrain[,outcome])
  if(aucTrain>=0.55) {
    aucCal <- calcAUC(dCal[,pi],dCal[,outcome])
    aucTest <- calcAUC(dTest[,pi],dTest[,outcome])
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

nbmodel <- naiveBayes(as.formula(ff),data=dTrain)

dTrain$nbpred <- predict(nbmodel,newdata=dTrain,type='raw')[,'TRUE']
dCal$nbpred <- predict(nbmodel,newdata=dCal,type='raw')[,'TRUE']
dTest$nbpred <- predict(nbmodel,newdata=dTest,type='raw')[,'TRUE']

calcAUC(dTrain$nbpred,dTrain[,outcome])
## [1] 0.9618698  # with selVars
calcAUC(dCal$nbpred,dCal[,outcome])
## [1] 0.9128786? with selVars
calcAUC(dTest$nbpred,dTest[,outcome])
## [1] 0.7624391 with selVars

print(plotROC(dTrain$nbpred,dTrain[,outcome]))
print(plotROC(dCal$nbpred,dCal[,outcome]))
print(plotROC(dTest$nbpred,dTest[,outcome]))

#### Model Output .RData for Project:
appetency_nb_sandra_model <- naiveBayes(as.formula(ff),data=dTrain)
appetency_nb_sandra_predictions <-predict(
  appetency_nb_sandra_model,newdata=dTest,type='raw')[,'TRUE']

# set the director path were the file will be placed
###   SET DIRECTORY PATH:
setwd("C:/Users/Sandra/Dropbox/pred_454_team/models/appetency")


# save the output
save(list = c('appetency_nb_sandra_model', 'appetency_nb_sandra_predictions'),
     file = 'appetency_nb_sandra.RData')
