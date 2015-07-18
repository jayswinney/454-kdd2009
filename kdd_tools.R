library(dplyr)


make_mat <- function(df_mat){
  # this function turns a dataframe into matrix format
  # it assumes that the response varaible has not been removed
  df_mat <- select(df, -churn, -appetency, -upsell)
  
  for (i in names(df_mat)){
    if (class(df_mat[,i]) == 'factor'){
      for(level in unique(df_mat[,i])[2:length( unique(df_mat[,i]))]){
        df_mat[sprintf('%s_dummy_%s', i, level)] <- ifelse(df_mat[,i] == level, 1, 0)
      }
      df_mat[,i] <- NULL
    } else {
      # scale numeric variables
      # this is important for regularized logistic regression and KNN
      df_mat[,i] <- scale(df_mat[,i])
    }
  }
  
  return(data.matrix(df_mat))
  
}