import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn import datasets
from sklearn.metrics import mean_squared_error,explained_variance_score
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

housing_data=datasets.load_boston()

#random_state controls how we shuffle the data
X,y=shuffle(housing_data.data,housing_data.target,random_state=7)

#80% training 20% testing
num_training=int(0.8*len(X))
X_train,y_train=X[:num_training],y[:num_training]
X_test,y_test=X[num_training:],y[num_training:]

#pick a tree with 4 maximum depth
dt_regressor=DecisionTreeRegressor(max_depth=4)
dt_regressor.fit(X_train,y_train)


#Fit adaboost
ab_regressor=AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),n_estimators=400, random_state=7)
ab_regressor.fit(X_train,y_train)


y_pred_dt = dt_regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred_dt)
evs = explained_variance_score(y_test, y_pred_dt)
print ("\n#### Decision Tree performance ####")
print ("Mean squared error =", round(mse, 2))
print ("Explained variance score =", round(evs, 2))


y_pred_ab = ab_regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred_ab)
evs = explained_variance_score(y_test, y_pred_ab)
print ("\n#### AdaBoost performance ####")
print ("Mean squared error =", round(mse, 2))
print ("Explained variance score =", round(evs, 2))

"""
all the features don't contribute
equally to the output.
In case we want to discard some of them later,
we need to know which features are less important.
"""
def plot_feature_importances(feature_importances, title, feature_names):
    # Normalize the importance values
    feature_importances = 100.0 * (feature_importances / max(feature_importances))

    # Sort the index values and flip them so that they are arranged in decreasing order of importance
    index_sorted = np.flipud(np.argsort(feature_importances))

    # Center the location of the labels on the X-axis (for display purposes only)
    pos = np.arange(index_sorted.shape[0]) + 0.5

    # Plot the bar graph
    plt.figure()
    plt.bar(pos, feature_importances[index_sorted], align='center')
    plt.xticks(pos, feature_names[index_sorted])
    plt.ylabel('Relative Importance')
    plt.title(title)
    plt.show()

plot_feature_importances(dt_regressor.feature_importances_,
        'Decision Tree regressor', housing_data.feature_names)
plot_feature_importances(ab_regressor.feature_importances_,
        'AdaBoost regressor', housing_data.feature_names)
