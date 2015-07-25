
setwd('c:/Users/Jay/Dropbox/pred_454_team')

df <- read.csv('data/orange_small_train.data', header = TRUE,
               sep = '\t', na.strings = '')

# get the response variables
churn_ <- read.csv('data/orange_small_train_churn.labels', header = FALSE)
appetency_ <- read.csv('data/orange_small_train_appetency.labels',
                       header = FALSE)
upsell_ <- read.csv('data/orange_small_train_upselling.labels',
                    header = FALSE)

# change -1 to 0
churn_[churn_$V1 < 0,] <- 0
appetency_[appetency_$V1 < 0,] <- 0
upsell_[upsell_$V1 < 0,] <- 0

# add response variables to the data
df$churn <- churn_$V1
df$appetency <- appetency_$V1
df$upsell <- upsell_$V1

set.seed(123)
smp_size <- floor(0.75 * nrow(df))
train_ind <- sample(seq_len(nrow(df)), size = smp_size)

# split the data
train <- df[train_ind, ]
test <- df[-train_ind, ]
# create response dataframe
test_response <- test[,c('upsell', 'churn', 'appetency')]

# save the test response vectors
save("test_response", file = 'data_transformations/response.RData')
