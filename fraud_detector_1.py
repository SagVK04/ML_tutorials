import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('upi_transactions.csv')

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

X_train = X[:-150]
X_test = X[-140:]
#Standardizing the features(X) to increase accuracy
scaled = StandardScaler()
X_train_scaled = scaled.fit_transform(X_train)
X_test_scaled = scaled.fit_transform(X_test)

Y_train = Y[:-150]
Y_test = Y[-140:]

fraud_model = KNeighborsClassifier(n_neighbors=95)
fraud_model.fit(X_train_scaled,Y_train)
Y_pred = fraud_model.predict(X_test_scaled)
acc = accuracy_score(Y_test,Y_pred)*100

if(acc <=50):
    print("Transaction may be fraud!")
else:
    print("Transaction is safe")


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