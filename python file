# load the iris dataset as an example 
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import pandas as pd
from sklearn.datasets import load_iris 
iris = load_iris() 

# store the feature matrix (X) and response vector (y) 
X = iris.data 
y = iris.target 

# store the feature and target names 
feature_names = iris.feature_names 
target_names = iris.target_names 

# printing features and target names of our dataset 
print("Feature names:", feature_names) 
print("Target names:", target_names) 

# X and y are numpy arrays 
print("\nType of X is:", type(X)) 

# printing first 5 input rows 
print("\nFirst 5 rows of X:\n", X[:5])


# reading csv file
data = pd.read_csv("C:/Users/YASH/Downloads/weather.csv")

# shape of dataset
print("Shape:", data.shape)

# column names
print("\nFeatures:", data.columns)

# storing the feature matrix (X) and response vector (y)
X = data[data.columns[:-1]]
y = data[data.columns[-1]]

# printing first 5 rows of feature matrix
print("\nFeature matrix:\n", X.head())

# printing first 5 values of response vector
print("\nResponse vector:\n", y.head())

# load the iris datasets as an example
iris = load_iris()

# store the feature matrix (X) and response vector (y)
X = iris.data
y = iris.target

# spitting X and Y into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=1)
# printing the shapes of the new X objects
print(X_train.shape)
print(X_test.shape)

# printing the shapes of the new y objects
print(y_train.shape)
print(y_test.shape)

# load the iris dataset as an example
from sklearn.datasets import load_iris
iris = load_iris()

# store the feature matrix (X) and response vector (y)
X = iris.data
y = iris.target

# spitting X and y into training and testing sets 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)

# training the mdel on training set 
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# making predictions on the testing set
y_pred = knn.predict(X_test)

# comparing actual response values (y_test) with predicted response values (y_pred)
from sklearn import metrics
print("kNN model accuracy:", metrics.accuracy_score)(y_test, y_pred)

# making prediction for out of sample data
sample =  [[3,5,4,2],[2,3,5,4]]
preds = knn.predict(sample)
pred_species = [iris.target_names[p] for p in preds]
print("Predictions:", pred_species)

# saving the model
from sklearn.externals import joblib
joblib.dump(knn, 'iris_knn.pkl')
