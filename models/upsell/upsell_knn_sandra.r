##  upsell_knn_sandra

###   SET DIRECTORY PATH:
setwd("C:/Users/Sandra/Dropbox/pred_454_team/data")

###   READ DATA FILES:
d <- read.table('orange_small_train.data',  	
                header=T,
                sep='\t',
                na.strings=c('NA','')) 	

churn <- read.table('orange_small_train_churn.labels',
                    header=F,sep='\t') 
d$churn <- churn$V1 	 

upselling <- read.table('orange_small_train_upselling.labels',
                        header=F,sep='\t')
d$upselling <- upselling$V1 	

appetency <- read.table('orange_small_train_appetency.labels',
                        header=F,sep='\t')
d$appetency <- appetency$V1 


###   CREATING TRAIN, CALIBRATION, AND TEST DATA SETS
# get the index for training/testing data
set.seed(123)
smp_size <- floor(0.75 * nrow(d))
train_ind <- sample(seq_len(nrow(d)), size = smp_size)
# making a "tiny" data set so I cn quickly test r markdown and graphical paramters
# this will be removed in the submitted version
tiny_ind <- sample(seq_len(nrow(d)), size = floor(0.10 * nrow(d)))
# split the data
dTrain <- d[train_ind, ]
dTest <- d[-train_ind, ]
dCal <- d[tiny_ind, ]


###  SETTING OTHER VARIABLES:
outcomes=c('churn','appetency','upselling')

vars <- setdiff(colnames(dTrain), c(outcomes,'rgroup'))

catVars <- vars[sapply(dTrain[,vars],class) %in% c('factor','character')] 

numericVars <- vars[sapply(dTrain[,vars],class) %in% c('numeric','integer')] 

rm(list=c('d','churn','appetency','upselling'))  

outcome <- 'upselling' 	
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



# Title: Running k-nearest neighbors 
# example 6.19
library('class')

nK <- 200
knnTrain <- dTrain[,selVars]  
knnCl <- dTrain[,outcome]==pos 


knnPred <- function(df) { 	
  knnDecision <- knn(knnTrain,df,knnCl,k=nK,prob=T)
  ifelse(knnDecision==TRUE, 
         attributes(knnDecision)$prob,
         1-(attributes(knnDecision)$prob))
}

dTrain.AUC <- calcAUC(knnPred(dTrain[,selVars]),dTrain[,outcome])
dTrain.AUC

dCal.AUC <- calcAUC(knnPred(dCal[,selVars]),dCal[,outcome])
dCal.AUC

dTest.AUC <- calcAUC(knnPred(dTest[,selVars]),dTest[,outcome])
dTest.AUC


# Title: Plotting 200-nearest neighbor performance 
# example 6.20

###  TRAIN KNN PREDICTIONS:
dTrain$kpred <- knnPred(dTrain[,selVars])

#  Create a vector of the predictions to be exporeted to a file:
upsell.knnTrainPred <- dTrain$kpred 
# save the output
setwd("C:/Users/Sandra/Dropbox/pred_454_team/models/upsell")
save(list = c('upsell.knnTrainPred'), file = 'upsell.knnTrainPred.csv')

# plot the predictions
plotROC(dTrain$kpred,dTrain[,outcome])

install.packages("ggplot2")
require("ggplot2")
ggplot(data=dTrain) +
  geom_density(aes(x=kpred,
                   color=as.factor(upselling),linetype=as.factor(upselling)))


###  CALIBRATION KNN PREDICTIONS:
dCal$kpred <- knnPred(dCal[,selVars])

#  Create a vector of the predictions to be exporeted to a file:
upsell.knnCalPred <- dCal$kpred
# save the output
setwd("C:/Users/Sandra/Dropbox/pred_454_team/models/upsell")
save(list = c('upsell.knnCalPred'), file = 'upsell.knnCalPred.csv')

# plot the predictions
plotROC(dCal$kpred,dCal[,outcome])

ggplot(data=dCal) +
  geom_density(aes(x=kpred,
                   color=as.factor(upselling),linetype=as.factor(upselling)))


###  TEST KNN PREDICTIONS:
dTest$kpred <- knnPred(dTest[,selVars])

#  Create a vector of the predictions to be exporeted to a file:
upsell.knnTestPred <- dTest$kpred
# save the output
setwd("C:/Users/Sandra/Dropbox/pred_454_team/models/upsell")
save(list = c('upsell.knnTestPred'), file = 'upsell.knnTestPred.csv')

# plot the predictions
plotROC(dTest$kpred,dTest[,outcome])

ggplot(data=dTest) +
  geom_density(aes(x=kpred,
                   color=as.factor(upselling),linetype=as.factor(upselling)))




