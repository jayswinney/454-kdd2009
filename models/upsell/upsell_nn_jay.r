# Neural Network to predict upsell

library(neuralnet)
library(dplyr)

setwd('c:/Users/Jay/Dropbox/pred_454_team')

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
f <- formula( substr(f,1, nchar(f)-12))


upsell_nn_jay = neuralnet(f, data = train_df, stepmax = 1000,
                          lifesign.step = 10, hidden = 384, lifesign ='full')



                                   