
# coding: utf-8

# In[2]:

import numpy as np

#sample 10 integers
sample=np.random.randint(low=1,high=100,size=10)


# In[3]:

#bootstrap re-sample 100 times by re-sampling with replacement from the original sample
resamples=[np.random.choice(sample,size=sample.shape) for i in range(100)]



# In[5]:

resample_means=np.array([resample.mean() for resample in resamples])


print 'original sample: %s' % sample
print 'sample mean: %s' % sample.mean()
print 'num of bootstrap re-samples: %s' % len(resamples)
print 'example resample: %s' % resamples[0]
print 'mean of re-samples\' means: %s' % resample_means.mean()



# In[6]:

# The number of trees in the forest is an important hyperparameter. 
#Increasing the number of trees improves the model's performance
#at the cost of computational complexity. 

#train random forest
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

X,y = make_classification(n_samples=1000,n_features=100,n_informative=20,n_clusters_per_class=2,random_state=11)
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=11)

clf=DecisionTreeClassifier(random_state=11)
clf.fit(X_train,y_train)
predictions=clf.predict(X_test)
print classification_report(y_test,predictions)


# In[ ]:



