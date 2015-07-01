import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.cross_validation import train_test_split

data = pd.read_csv('dataset.csv')

feature_cols = ['uuid_count','threads_created','comments_created','total_searches','total_SFVs','total_leads','saved_vendors','booked_vendors']
X = data[feature_cols]
y = data.reviews_user

knn = KNeighborsClassifier(n_neighbors=50)
knn.fit(X,y)
y_pred = knn.predict(X)

print metrics.accuracy_score(y, y_pred)

k_range = range(1,50)
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X,y)
    y_pred = knn.predict(X)
    scores.append(metrics.accuracy_score(y, y_pred))
    
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print metrics.accuracy_score(y_test, y_pred)

k_range = range(1, 51)
testing_error = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    testing_error.append(1 - metrics.accuracy_score(y_test, y_pred))
    