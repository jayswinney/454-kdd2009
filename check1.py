import os
import csv

import pandas as pd

fp = 'C:/Users/Jay/Documents/Northwestern/predict_454/'


target_vars = ['churn','upsell', 'appetency']


rmd = open(fp+'team_project/checkpoint1.rmd', 'wb')


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
    library(knitr)

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

    summary_tables <- list()

    # for(i in seq(length(names(df)))){
    #   summary_tables[[i]] <- basicStats(df[, names(df)[i]])
    #   names(summary_tables[[i]]) <- names(df)[i]
    # }
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
        ref = TRUE,
        xlab = "signedlog {0}")

    qqmath(~ signedlog({0}), df,
        groups = {1},
        aspect = "fill",
        f.value = ppoints(100),
        auto.key = list(corner = c(0,1)),
        xlab = "Standard Normal Quatiles",
        ylab = "Average signedlog {0}")

    bwplot(factor({1}) ~ signedlog({0}),
        groups = {1},
        data = df,
        xlab = "signedlog {0}")

    ```
    '''.replace('    ','')


rmd.write(read_data)

with open(fp + 'team_project/interesting_vars.txt') as good_vars:
    vlist = csv.reader(good_vars)
    vlist = [r[0] for r in vlist]



df = pd.read_csv(
        fp + 'KDD_Cup_2009/orange_small_train.data',
        sep = '\t', na_values = '')

comment_files = os.listdir(fp +  'comments/')

comments = [pd.read_csv(fp + 'comments/' + x) for x in comment_files]

comments_df = pd.concat(comments)
comments_df.loc[:,'Response'] = comments_df['Response'].str.lower()
comments_df = comments_df.set_index(['Variable', 'Response'])


print df.fillna(0).values.min()

vlist = ['Var'+str(x) for x in xrange(120,130)]

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
            if v in comments_df.index:
                if t in comments_df.ix[v].index:
                    rmd.write(comments_df.ix[v].ix[t]['Comments'])
                    rmd.write('\n \n')
    else:
        for t in target_vars:
            rmd.write('\n##{0} {1}'.format(t.capitalize(), v))
            rmd.write(standard_plots.format(v, t))
            if v in comments_df.index:
                if t in comments_df.ix[v].index:
                    rmd.write(comments_df.ix[v].ix[t]['Comments'])
                    rmd.write('\n \n')

rmd.close()
