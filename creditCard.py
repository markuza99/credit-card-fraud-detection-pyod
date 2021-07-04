import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pyod.models.knn import KNN
from pyod.models.auto_encoder import AutoEncoder
from pyod.models.iforest import IForest
import sklearn
from pyod.utils import evaluate_print
from pyod.utils.example import  visualize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import scikitplot as skplt
import seaborn as sns


#citamo podatke
data  = pd.read_csv("creditcard.csv")

#prevarne transakcije
positive = data[data["Class"]== 1]
#validne transakcije
negative = data[data["Class"]== 0]


#print("positive:{}".format(len(positive)))
#print("negative:{}".format(len(negative)))

#zbog perfomansi, smanjujemo prvobitni dataset
new_data = pd.concat([positive,negative[:10000]])

#hist za sve atribute
new_data.hist(bins = 50, figsize = (20,20))
plt.tight_layout()
plt.show()

#saflujemo nas novi dataset
new_data = new_data.sample(frac=1,random_state=42)
print(new_data.shape)

#prikaz korelacija izmedju varijabli
corr = data.corr()
fig = plt.figure(figsize=(30,20))
sns.heatmap(corr, vmax=.8, square=True, annot=True)
plt.show()

#standardizujemo polje kolicina
new_data['Amount'] = StandardScaler().fit_transform(new_data['Amount'].values.reshape(-1,1))



#plt.plot(new_data["Time"], new_data["Amount"])
#plt.show()

X = new_data.drop(['Time','Class'], axis=1)
y = new_data['Class']
#razbijamo na trening i test skup
X_train, X_test, y_train,y_test = train_test_split(X,y, test_size = 0.2, stratify=y, random_state=42 )



clf_knn = KNN(contamination=0.172, n_neighbors = 5,n_jobs=-1)

clf_knn.fit(X_train)

y_train_pred = clf_knn.labels_ # binary labels (0: inliers, 1: outliers)
# Outlier scores
y_train_scores = clf_knn.decision_scores_
print("Evaluacija KNN trening:")
evaluate_print('KNN', y_train, y_train_scores)
skplt.metrics.plot_confusion_matrix(y_train,y_train_pred, normalize=False,title="Matrica konfuzije za KNN nad trening skupom")
plt.show()

y_test_pred = clf_knn.predict(X_test)
y_test_scores = clf_knn.decision_function(X_test)  # outlier scores

# Evaluate on the training data
print("Evaluacija KNN test:")
evaluate_print('KNN', y_test,y_test_scores)
skplt.metrics.plot_confusion_matrix(y_test,y_test_pred, normalize=False,title="Matrica konfuzije za KNN nad test skupom")
plt.show()

clf_ae = AutoEncoder(contamination=0.172, hidden_neurons=[58, 29, 29, 58], epochs=30)
clf_ae.fit(X_train)
y_train_pred = clf_ae.labels_
y_train_scores = clf_ae.decision_scores_
print("Evaluacija AutoEncoder trening:")
evaluate_print("AutoEncoder", y_train, y_train_scores)
skplt.metrics.plot_confusion_matrix(y_train,y_train_pred, normalize=False,title="Matrica konfuzije za AutoEncoder nad trening skupom")
plt.show()

y_test_pred = clf_ae.predict(X_test)
y_test_scores = clf_ae.decision_function(X_test)
print("Evaluacija AutoEncoder test:")
evaluate_print("AutoEncoder", y_test, y_test_scores)
skplt.metrics.plot_confusion_matrix(y_test,y_test_pred, normalize=False,title="Matrica konfuzije za AutoEncoder nad test skupom")
plt.show()

clf_if = IForest(contamination=0.172)
clf_if.fit(X_train)
y_train_pred = clf_if.labels_
y_train_scores = clf_if.decision_scores_
print("Evaluacija Isolation Forest trening:")
evaluate_print("IForest", y_train, y_train_scores)
skplt.metrics.plot_confusion_matrix(y_train,y_train_pred, normalize=False,title="Matrica konfuzije za Isolation Forest nad trening skupom")
plt.show()

y_test_pred = clf_if.predict(X_test)
y_test_scores = clf_if.decision_function(X_test)
print("Evaluacija AutoEncoder test:")
evaluate_print("IForest", y_test, y_test_scores)
skplt.metrics.plot_confusion_matrix(y_test,y_test_pred, normalize=False,title="Matrica konfuzije za Isolation Forest nad test skupom")
plt.show()