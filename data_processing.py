
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd


# In[2]:

fp = 'C:/Users/Jay/Documents/Northwestern/predict_454/KDD_Cup_2009/'


# In[3]:

df = pd.read_csv(fp + 'orange_small_train.data', sep = '\t')


# In[19]:

def data_munging(var, vname):
    # qualitative variables
    if var.dtype == np.object:
        # this threshold will be used to decide whether or not
        # a value will be grouped into 'other'
        threshold = len(var)*0.01
        # replace NA values with 'missing'
        var = var.fillna(vname + '_missing')
        # get the count of times each value appears
        counts = var.value_counts()        
        # add values to be remapped to other if their frequency
        # is less than the threshold
        remapping = {}
        for c in counts.index:
            if counts[c] < threshold:
                remapping[c] = vname + '_other'
                
        # if some values need to be grouped into other do so
        if len(remapping) > 0:
            var = var.map(lambda x: remapping[x] if x in remapping else x)
        
        return [pd.Series(var, name = vname)]
    
    # quantitative variables            
    else:
        missing = pd.Series(pd.isnull(var),
                            name = vname + '_missing')
        # check to see if a large number of values are missing
        if missing.sum() < len(var) * 0.6:
            # replace missing values with median
            var = pd.Series(var.fillna(var.median()), name = vname)
            return [missing.astype(int), var]
        else:
            percentiles = []
            # create the qualitative versions of original
            qual = np.zeros(len(var))
            # find the percentiles 
            for i in xrange(0, 81, 20):
                percentiles.append(np.percentile(var, i))
                
            # add 1 to each observation that is greater than p
            # this creates a vector where the observations in the
            # top 20 percent have a value of 5 
            # then 4 ... 0 down to through lower values
            for p in percentiles:
                qual = qual + (var > p).astype(int)
                
            # turn the numbers into strings, add original name
            qual = vname + '_' + qual.map(str)
            # ID missing variables
            qual.loc[missing] = vname + '_missing'
            
            return [pd.Series(qual,name = vname + '_percentiles')]


# In[20]:

processed = []
for col in df:
    processed += data_munging(df[col], col)


# In[21]:

processed_df = pd.DataFrame(processed).T


# In[22]:

df.to_csv(fp + 'csv_data.csv', index = False)
processed_df.to_csv(fp + 'processed.csv', index = False)

