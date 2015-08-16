# Neural Network to predict upsell

library(neuralnet)
library(dplyr)

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

# using the impute 0 method
source('data_transformations/impute_0.r')
source('kdd_tools.R')

# upsell
train_mat <- make_mat(train)
train_df <- data.frame(train_mat)
train_df$upsell <- train$upsell

# I will need an additional validation set for this
set.seed(123)
smp_size <- floor(0.80 * nrow(train))
train_ind <- sample(seq_len(nrow(train_df)), size = smp_size)
train <- train_df[train_ind, ]
val <- train_df[-train_ind, ]

f <- 'upsell ~'

for (n in colnames(train)){
  f <- paste(f, sprintf('%s + ',n), sep = '')
}
f <- formula(substr(f,1, nchar(f)-12))


upsell_nn_jay = neuralnet(f, data = train, stepmax = 20, rep = 1, algorithm =  'backprop',
                          lifesign.step = 2, hidden = 4, lifesign ='full',
                          learningrate = 0.02)
#
compute(upsell_nn_jay, val)
prediction(upsell_nn_jay)

upsell_nn_jay <- nnet(f, train, size = 50 , MaxNWts=100000)

sum(predict(upsell_nn_jay, val, type = 'raw')[,1])

neuralnet::compute(upsell_nn_jay, val)

save(upsell_nn_jay, file = 'models/upsell/upsell_nn_jay.RData')
