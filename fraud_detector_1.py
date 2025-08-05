import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv('upi_transactions_7000.csv')

#KNN Classifiers need all feature values in digits

df['Transaction Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Transaction Time'])
df['DayOfWeek'] = df['Transaction Datetime'].dt.dayofweek
df['Hour'] = df['Transaction Datetime'].dt.hour
df = pd.get_dummies(df, columns=['Platform'], drop_first=True)
df = df.drop(columns=['Date', 'Transaction Time', 'Transaction Datetime','Transaction Number'])  #Main redundant columns are gone
#print(df.shape)
#print(df.sample(5))
X = df.drop(columns=['Fraud Possibility'])

Y = df['Fraud Possibility']

X_train = X[:-1400]
X_test = X[-1340:]
#Standardizing the features(X) to increase accuracy
scaled = StandardScaler()
X_train_scaled = scaled.fit_transform(X_train)
X_test_scaled = scaled.fit_transform(X_test)

Y_train = Y[:-1400]
Y_test = Y[-1340:]

fraud_model = KNeighborsClassifier(n_neighbors=1113)
fraud_model.fit(X_train_scaled, Y_train)
Y_pred = fraud_model.predict(X_test_scaled)

print(f"Accuracy: {accuracy_score(Y_test,Y_pred)*100:.2f} %")

accuracy_scores = []
neighbors_range = range(1, 600)

for k in neighbors_range:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train_scaled, Y_train)
    Y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(Y_test, Y_pred)
    accuracy_scores.append(accuracy)

max_accuracy = max(accuracy_scores) * 100
print(f"The maximum accuracy is: {max_accuracy:.2f} when neighbors: {k}%")






#best_acc = 0
#best_nei = 1

#neighbour_size <= no. of training samples fit in the model
#for k in range(1,99):
#    fraud_model_1 = KNeighborsClassifier(n_neighbors=k)
#    fraud_model_1.fit(X_train_scaled,Y_train)
#    Y_pred_fin = fraud_model_1.predict(X_test_scaled)
#    acc_final = accuracy_score(Y_test,Y_pred_fin)
#    if acc_final > best_acc:
#        best_acc = acc_final
#        best_nei = k
#print(f"The best accuracy is {best_acc * 100:.2f}% with neighbors = {best_nei}")