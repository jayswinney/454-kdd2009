# this script is intended to perform EDA on variables 161:200

# read in the data to R
# I'm using na.stings = '' to replace blanks with na
# this also helps R read the numerical varaibles as numerical
fp <- 'C:/Users/Jay/Documents/Northwestern/predict_454/KDD_Cup_2009/'

df <- read.csv(paste(fp,'orange_small_train.data', sep = ''),
               header = TRUE, sep = '\t', na.strings = '')
# read the target variables
churn <- read.csv(paste(fp,'orange_small_train_churn.labels', sep = ''),
                 header = FALSE)
appetency <- read.csv(paste(fp,'orange_small_train_appetency.labels', sep = ''),
                  header = FALSE)
upsell <- read.csv(paste(fp,'orange_small_train_upselling.labels', sep = ''),
                  header = FALSE)

# add the target variables to the data frame
df$churn <- churn$V1
df$appetency <- appetency$V1
df$upsell <- upsell$V1

rm(churn)
rm(appetency)
rm(upsell)
# replace -1's in the target variables with 0's
df[df$churn < 0,'churn'] <- 0
df[df$appetency < 0,'appetency'] <- 0
df[df$upsell < 0,'upsell'] <- 0

library(lattice)
library(plyr)
library(dplyr)
library(tidyr)
library(grid)
library(gridExtra)

for(i in 161:190){
  if(sum(is.na(df[,paste('Var',i,sep='')])) == dim(df)[1]) next
  #print(paste('Var',i,sep=''))
 p1 <- densityplot(df[,paste('Var',i,sep='')],
                   groups = df$churn, plot.points = FALSE,
                   main = 'churn',
                   scales=list(y=list(at=NULL)),
                   ylab = NULL,
                   xlab = paste('Var',i,sep=''))

 p2 <- densityplot(df[,paste('Var',i,sep='')],
                   groups = df$appetency, plot.points = FALSE,
                   main = 'appetency',
                   scales=list(y=list(at=NULL)),
                   ylab = NULL,
                   xlab = paste('Var',i,sep=''))

 p3 <- densityplot(df[,paste('Var',i,sep='')],
                   groups = df$upsell, plot.points = FALSE,
                   main = 'upsell',
                   scales=list(y=list(at=NULL)),
                   ylab = NULL,
                   xlab = paste('Var',i,sep=''))
 grid.arrange(p1, p2, p3, ncol=3)
}

# get the number of classes in each qualitative variable
for(i in 191:200){
  print(paste('Var',i,sep=''))
  print(length(unique(df[,paste('Var',i,sep='')])))
  # output some stastics about the classes
  # this is commented out right now because it's too long to read
  # print('churn')
  # print(aggregate(df$churn ~ df[,paste('Var',i,sep='')], FUN = mean))
  # print('---------------------')
  # print('appetency')
  # print(aggregate(df$appetency ~ df[,paste('Var',i,sep='')], FUN = mean))
  # print('---------------------')
  # print('upsell')
  # print(aggregate(df$upsell ~ df[,paste('Var',i,sep='')], FUN = mean))
  # print('_____________________________')
}

# use random forest to get variable importance
library(randomForest)
# notice I'm usinge na.roughfix,
# this imputes missing vlaues with the median/mode
rf <- randomForest(factor(churn) ~ .,
                   data = df[,c('Var173', 'Var181', 'Var193',
                                'Var195', 'Var196', 'churn')],
                   na.action=na.roughfix,  importance=TRUE)

varImpPlot(rf)
