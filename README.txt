## Synopsis
My code for Predict 454 summer 2015 quarter team project: KDD cup 2009.

## To Do:
Preprocessing:
  - group together rare categories in categorical variables, this should help
    to reduce over fitting and also will make the data easier to work with.
    additionally the randomForest package can only work with categorical vars
    with <= 53 categories. ideally all categroical vars will have well below 53
    categories
  - missing values are a huge problem for this data set. need to come up with
    strategies to address this. ideas so far
      * mean/median/mode imputation
      * use regression to impute missing values (could be very labor intensive)
      * recommender system for categorical missing values
  - i don't think PCA will be very useful for this dataset, the missing values
    could make it difficult to gain anything useful from PCA.
