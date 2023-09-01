import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV


df = pd.read_csv("diabetes.csv")
df=df.drop_duplicates()
df.isnull().sum()

x = df
df_new= pd.DataFrame(x)
df_new.columns =['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin','BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
df_new.head()
target_name='Outcome'
y= df_new[target_name]
X=df_new.drop(target_name,axis=1)
X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.2,random_state=0)
knn= KNeighborsClassifier()
n_neighbors = list(range(15,25))
p=[1,2]
weights = ['uniform', 'distance']
metric = ['euclidean', 'manhattan', 'minkowski']
hyperparameters = dict(n_neighbors=n_neighbors, p=p,weights=weights,metric=metric)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid_search = GridSearchCV(estimator=knn, param_grid=hyperparameters, n_jobs=-1, cv=cv, scoring='f1',error_score=0)
best_model = grid_search.fit(X_train,y_train)
knn_pred = best_model.predict(X_test)
print(knn_pred)
import pickle

# Mô hình

# Lưu trọng số của mô hình tốt nhất
best_model = grid_search.best_estimator_
best_model.fit(X, y)  # Huấn luyện lại trên toàn bộ dữ liệu
weights_path = "weights.pkl"
with open(weights_path, "wb") as f:
    pickle.dump(best_model, f)