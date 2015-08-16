# This script loads the data and cleans it
# missing variables are replaces with
#   - 0 if numeric, "missing" binary variable is created
#   - "missing" level is added if factor
#
# script also creates training/testing split in data



# read in the data to R
# I'm using na.stings = '' to replace blanks with na
# this also helps R read the numerical varaibles as numerical
# setwd('c:/Users/Jay/Dropbox/pred_454_team')
df <- read.csv('data/orange_small_train.data', header = TRUE,
               sep = '\t', na.strings = '')
# read the target variables
churn_ <- read.csv('data/orange_small_train_churn.labels', header = FALSE)
appetency_ <- read.csv('data/orange_small_train_appetency.labels',
                        header = FALSE)
upsell_ <- read.csv('data/orange_small_train_upselling.labels',
                    header = FALSE)

churn_[churn_$V1 < 0,] <- 0
appetency_[appetency_$V1 < 0,] <- 0
upsell_[upsell_$V1 < 0,] <- 0

# impute mising data with zeros and "missing"o
# also creates missing variable column
for (i in names(df)){
  vclass <- class(df[,i])
  if(vclass == 'logical'){
    # some of the variables are 100% missing,
    # they are the only logical class vars
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

# add the target variables to the data frame
df$churn <- churn_$V1
df$appetency <- appetency_$V1
df$upsell <- upsell_$V1

# this portion of the code should be copied exactly
# in every data transformation script
# that way we will all be using the same training/testing data
set.seed(123)
smp_size <- floor(0.70 * nrow(df))
test_ind <- seq_len(nrow(df))
train_ind <- sample(test_ind, size = smp_size)
# remove train observations from test
test_ind <- test_ind[! test_ind %in% train_ind]
# create an ensemble test set
set.seed(123)
smp_size <- floor(0.15 * nrow(df))
ens_ind <- sample(test_ind, size = smp_size)
# remove ensemble observations from test
test_ind <- test_ind[! test_ind %in% ens_ind]
# partition the data
ensemble_test <- df[ens_ind, ]
train <- df[train_ind, ]
test <- df[test_ind, ]
