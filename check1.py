# import os
import csv

import pandas as pd

fp = 'C:/Users/Jay/Documents/Northwestern/predict_454/team_project/'


target_vars = ['churn','upsell', 'appetency']


rmd = open(fp+'checkpoint1.rmd', 'wb')


read_data = '''
    ---
    title: "EDA Plots"
    author: "Udy Akpan, Joseph Dion, Sandra Duenas, Manjari Srivesta, Jay Swinney"
    date: "July 5, 2015"
    output: html_document
    ---
    ```{r, message=FALSE, echo=FALSE}
    library(lattice)
    library(plyr)
    library(dplyr)
    library(tidyr)
    library(grid)
    library(gridExtra)
    library(ROCR)
    library(fBasics)
    library(e1071)

    # set document width
    options(width = 10)
    # read in the data to R
    # I'm using na.stings = '' to replace blanks with na
    # this also helps R read the numerical varaibles as numerical
    setwd('C:/Users/Jay/Documents/Northwestern/predict_454/KDD_Cup_2009/')
    df <- read.csv('orange_small_train.data', header = TRUE,
                   sep = '\t', na.strings = '')
    # read the target variables
    churn_ <- read.csv('orange_small_train_churn.labels', header = FALSE)
    appetency_ <- read.csv('orange_small_train_appetency.labels', header = FALSE)
    upsell_ <- read.csv('orange_small_train_upselling.labels', header = FALSE)

    churn_[churn_$V1 < 0,] <- 0
    appetency_[appetency_$V1 < 0,] <- 0
    upsell_[upsell_$V1 < 0,] <- 0

    # add the target variables to the data frame
    df$churn <- churn_$V1
    df$appetency <- appetency_$V1
    df$upsell <- upsell_$V1

    signedlog <- function(x) {
     sign(x)*log(abs(x)+1)
     }
    ```
    '''.replace('    ','')


density_plot = '''
    ```{0}
    densityplot(df${1},
              groups = df$churn, plot.points = FALSE,
              main = 'churn',
              scales=list(y=list(at=NULL)),
              ylab = 'Densisty',
              xlab = '{1}')
    ```
    '''.replace('    ','')


standard_plots = '''
    ```{{r fig.width=3, fig.height=3, fig.show='hold', echo=FALSE, align='center'}}

    densityplot(~ {0}, data = df,
            groups = {1},
            plot.points = FALSE,
            auto.key = list(corner = c(1,1)),
            # key = c(1,1),
            ref = TRUE)

    qqmath(~ {0}, df,
            groups = {1},
            aspect = "fill",
            f.value = ppoints(100),
            auto.key = list(corner = c(0,1)),
            xlab = "Standard Normal Quatiles",
            ylab = "Average {0}")

    bwplot(factor({1}) ~ {0},
            groups = {1},
            data = df,
            xlab = "{0}")
    ```
    '''.replace('    ','')


standard_plots_log = '''
    ```{{r fig.width=3, fig.height=3, fig.show='hold', echo=FALSE, align='center'}}

    densityplot(~ signedlog({0}), data = df,
        groups = {1},
        plot.points = FALSE,
        auto.key = list(corner = c(1,1)),
        # key = c(1,1),
        ref = TRUE)

    qqmath(~ {0}, df,
        groups = {1},
        aspect = "fill",
        f.value = ppoints(100),
        auto.key = list(corner = c(0,1)),
        xlab = "Standard Normal Quatiles",
        ylab = "Average {0}")

    bwplot(factor({1}) ~ signedlog({0}),
        groups = {1},
        data = df,
        xlab = "{0}")

    ```
    '''.replace('    ','')


main_loop ='''
    ```{r, echo=FALSE}

    for(i in 161:190){

      if(sum(is.na(df[,paste('Var',i,sep='')])) == dim(df)[1]) next
      vname <- paste('Var',i,sep='')
      print(vname)
      print(basicStats(df[,vname]))
      p1 <- densityplot(df[,vname],
                        groups = df$churn, plot.points = FALSE,
                        main = 'churn',
                        scales=list(y=list(at=NULL)),
                        ylab = NULL,
                        xlab = vname)

      p2 <- densityplot(df[,vname],
                        groups = df$appetency, plot.points = FALSE,
                        main = 'appetency',
                        scales=list(y=list(at=NULL)),
                        ylab = NULL,
                        xlab = vname)

      p3 <- densityplot(df[,vname],
                        groups = df$upsell, plot.points = FALSE,
                        main = 'upsell',
                        scales=list(y=list(at=NULL)),
                        ylab = NULL,
                        xlab = vname)

      grid.arrange(p1, p2, p3, ncol=3)

      df[,paste(vname,'_missing',sep='')] <- is.na(df[,vname])
      #fill missing values with zeros
      df[is.na(df[,vname]),vname] <- 0

      set.seed(123)
      smp_size <- floor(0.75 * nrow(df))
      train_ind <- sample(seq_len(nrow(df)), size = smp_size)
      # split the data
      train <- df[train_ind, ]
      test <- df[-train_ind, ]

      for(j in c('churn', 'appetency', 'upsell')){
      print(sprintf('modeling for %s', j))
      logistic_reg <- eval(
        parse(
          text = sprintf(
            "glm(%s ~ poly(%s, 3) + %s_missing, data = train, family = 'binomial')",
            j, vname, vname)
        )
      )
      print(summary(logistic_reg))
      yhat <- predict(logistic_reg, test, response = 'prob')
      pred <- prediction(yhat, test[,j])
      perf <- performance(pred, measure = "tpr", x.measure = "fpr")
      # plot(perf, col=rainbow(10), main = vname)
      print(sprintf('AUC for %s:', vname))
      print(attributes(performance(pred, 'auc'))$y.values[[1]])
      print('--------------------------')

      }

      print('=====================================================================')

    }
    ```

    ```{r}
    for(i in 191:200){

    }
    ```
    '''

rmd.write(read_data)

with open(fp + 'interesting_vars.txt') as good_vars:
    vlist = csv.reader(good_vars)
    vlist = [r[0] for r in vlist]

vlist = ['Var'+str(x) for x in xrange(1,191)]

df = pd.read_csv(
        fp.replace('team_project/', 'KDD_Cup_2009/orange_small_train.data'),
        sep = '\t', na_values = '')

print df.fillna(0).values.min()

for v in vlist:
    if pd.isnull(df[v]).sum() == len(df):
        continue
    rmd.write('\n \n \n')
    rmd.write('#{0} \n'.format(v))
    # rmd.write(density_plot.format('{r, echo = FALSE}',v))
    if (df[v].skew() > 10.0) & (df[v].max() >= 100.0):
        rmd.write('Applying signed-log transforamtion<br> \n')
        for t in target_vars:
            rmd.write('\n##{0} signed-log({1})'.format(t.capitalize(), v))
            rmd.write(standard_plots_log.format(v, t))
    else:
        for t in target_vars:
            rmd.write('\n##{0} {1}'.format(t.capitalize(), v))
            rmd.write(standard_plots.format(v, t))

rmd.close()
