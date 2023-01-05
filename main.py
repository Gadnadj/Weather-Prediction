import warnings

warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

pd.options.display.max_columns = None
pd.options.display.max_rows = None

np.set_printoptions(suppress=True)

df = pd.read_csv("/Users/nadjar/PycharmProjects/pythonProject7/venv/bin/seattle-weather.csv")

df = df.drop('date', axis=1)

x = df.drop('weather', axis=1)
y = df['weather']

# transform the dataset
oversample = SMOTE()
x, y = oversample.fit_resample(x, y)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)




def get_test_report(model):
    return (classification_report(y_test, y_pred))


def plot_confusion_matrix(model):
    cm = confusion_matrix(y_test, y_pred)

    conf_matrix = pd.DataFrame(data=cm,
                               columns=['Drizzle', 'Fog', 'Rain', 'Snow', 'Sun'],
                               index=['Drizzle', 'Fog', 'Rain', 'Snow', 'Sun'])
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap=ListedColormap(['lightskyblue']),
                cbar=False, linewidths=0.1, annot_kws={'size': 25})

    plt.xticks(fontsize=15, fontweight='bold')
    plt.yticks(fontsize=15, fontweight='bold')

    # Add x-axis and y-axis labels
    plt.xlabel('Predicted Weather ', fontsize=20, fontweight='bold')
    plt.ylabel('Actual Weather ', fontsize=20, fontweight='bold')

    plt.show()


score_card = pd.DataFrame(columns=['Model', 'Precision Score', 'Recall Score',
                                   'Accuracy Score', 'Kappa Score', 'f1-score'])

classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
Decision_tree = classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)


print(get_test_report(Decision_tree))
acc5 = accuracy_score(y_test, y_pred)
plot_confusion_matrix(Decision_tree)
print(f"Accuracy score: {acc5}")


classifier = SVC(kernel = 'linear', random_state = 0)
SVC=classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)


acc6 = accuracy_score(y_test, y_pred)
plot_confusion_matrix(SVC)
print(get_test_report(SVC))
print(f"Accuracy score: {acc6}")





from sklearn.tree import export_graphviz
from sklearn.tree import plot_tree

# Enregistre l'arbre de décision dans un fichier dot
export_graphviz(Decision_tree, out_file='tree.dot', feature_names=list(x.columns), class_names=list(y.unique()))

# Affiche l'arbre de décision
fig = plt.figure(figsize=(150, 90), dpi= 210)
plot_tree(Decision_tree, feature_names=list(x.columns), class_names=list(y.unique()))
fig.savefig("decison_tree.png")

