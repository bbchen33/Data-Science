# Data from Kaggle https://www.kaggle.com/mlg-ulb/creditcardfraud
# Using oversampling or undersampling to work with imbalanced data
# Principal component analysis to visualize data distributions
# Seaborn to see the correlation between variables
# Logistic regression to predict whether a transaction is fraud or not
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import seaborn as sns
from sklearn.decomposition import PCA
df = pd.read_csv('F:\creditcardfraud\creditcard.csv')
# Count non-fraud and fraud percentage
not_fraud = df.Class.value_counts(normalize=True)[0]
fraud = df.Class.value_counts(normalize=True)[1]
plt.bar('not fraud', not_fraud)
plt.bar('fraud', fraud)
plt.ylabel('percentage')
plt.title('Percentage of non fraud and fraud')
plt.show()
# As the data suggest, the structure is highly unbalanced


shuffled_df = df.sample(frac = 1)
not_fraud_df = df[df.Class == 0][0:492]
fraud_df = df[df.Class == 1]


new_df = pd.concat([not_fraud_df, fraud_df])
not_fraud = new_df.Class.value_counts(normalize=True)[0]
fraud = new_df.Class.value_counts(normalize=True)[1]
plt.bar('not fraud', not_fraud)
plt.bar('fraud', fraud)
plt.ylabel('percentage')
plt.title('Percentage of non fraud and fraud')

# Use a for loop to get 5 randomly shuffled undersampled not_fraud data
score = []
for i in range(5):
    shuffled_df = df.sample(frac = 1)
    not_fraud_df = df[df.Class == 0][0:492]
    fraud_df = df[df.Class == 1]
    new_df = pd.concat([not_fraud_df, fraud_df])
    not_fraud = new_df.Class.value_counts(normalize=True)[0]
    fraud = new_df.Class.value_counts(normalize=True)[1]
    features = new_df.drop('Class', axis = 1)
    targets = new_df.Class
    X_train,X_test,Y_train,Y_test = train_test_split(features, targets, test_size = 0.6)
    clf = LogisticRegression().fit(X_train, Y_train)
    score.append(clf.score(X_test,Y_test))
# The scores are in the 80s range. Not super high. Could it be due to lack of normalization?
# Normalize 'Time' and 'Amount' by subtracting mean from it and dividing it to the standard deviation
features = new_df.drop('Class', axis = 1)
features.Time = (features.Time - features.Time.mean())/features.Time.std()
features.Amount = (features.Amount - features.Amount.mean())/features.Amount.std()
targets = new_df.Class

# Perform PCA to see the distance between the samples
pca = PCA(n_components = 2)
principalComponents = pca.fit_transform(features)
principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])
targets.index = [x for x in range(984)]

finalDf = pd.concat([principalDf, targets], axis = 1)
plt.scatter(finalDf['principal component 1'][finalDf.Class == 0], finalDf['principal component 2'][finalDf.Class == 0], label = 'not fraud')
plt.scatter(finalDf['principal component 1'][finalDf.Class == 1], finalDf['principal component 2'][finalDf.Class == 1], label = 'fraud')
plt.legend()
plt.xlabel('principal component 1')
plt.ylabel('principal component 2')
plt.show()
# One can see that the distribution of not fraud is much closer to each other than the fraud samples. Perhaps fraud data tend to be highly variable.

X_train,X_test,Y_train,Y_test = train_test_split(features, targets, test_size = 0.6)
clf = LogisticRegression().fit(X_train, Y_train)
score_after_normalization = clf.score(X_test,Y_test)
# I got 0.976 for score after normalization
# Try oversampling with SMOTE
features = df.drop('Class', axis =1)
targets = df.Class
X_train, X_test, Y_train, Y_test = train_test_split(features, targets, test_size=0.4, random_state=0)
sm = SMOTE(random_state = 2)
X_train_over, Y_train_over = sm.fit_sample(X_train, Y_train.ravel())
X_test_over, Y_test_over = sm.fit_sample(X_test, Y_test.ravel())
sum(Y_train_over == 0)
sum(Y_train_over == 1)
# Turned out samples fraud and non-fraud now have equal sample size from oversampling fraud
clf = LogisticRegression().fit(X_train_over, Y_train_over)
score_after_SMOTE = clf.score(X_test,Y_test)
# The score I got was 0.956. Pretty close to the previous undersampling method with normalization.

correlation = df.corr()

sns.heatmap(correlation, cmap = 'Spectral')
plt.show()
# From the heatmap, we can see there is little correlation between the variables v2, v3...... when we used the raw data (no oversampling or undersampling)
correlation2 = new_df.corr()
sns.heatmap(correlation2, cmap = 'Spectral')
# However, when I used the undersampling with non-fraud samples, there seems to be more correlation between the variables (shown in blue or green)


