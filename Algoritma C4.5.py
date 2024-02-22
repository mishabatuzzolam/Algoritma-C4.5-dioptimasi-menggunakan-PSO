# Mount drive 
from google.colab import drive
drive.mount('/content/drive')

# Memuat library
import pandas as pd
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Memuat data dari drive
df = pd.read_csv('/content/drive/MyDrive/janin.csv', sep=';') #ubah sesuai lokasi berkas data

# Membagi data menjadi data uji dan data latih 
X = df.drop('fetal_health', axis=1)  
y = df['fetal_health']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

# Model decision tree  C4.5
model = tree.DecisionTreeClassifier()  
model.fit(X_train, y_train)

# Evaluasi model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Akurasi model C4.5: ", accuracy)
