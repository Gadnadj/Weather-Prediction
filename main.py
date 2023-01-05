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
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

pd.options.display.max_columns = None
pd.options.display.max_rows = None

np.set_printoptions(suppress=True)

df = pd.read_csv("/Users/nadjar/PycharmProjects/pythonProject7/venv/bin/seattle-weather.csv")

df = df.drop('date', axis=1)

x = df.drop('weather', axis=1)
y = df['weather']
y = LabelEncoder().fit_transform(y)
# transform the dataset
oversample = SMOTE()
x, y = oversample.fit_resample(x, y)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)




def get_test_report(model):
    return (classification_report(y_test, y_pred))


score_card = pd.DataFrame(columns=['Model', 'Precision Score', 'Recall Score',
                                   'Accuracy Score', 'Kappa Score', 'f1-score'])


def update_score_card(model_name):
    global score_card

    score_card = score_card.append({'Model': model_name,
                                    'Precision Score': metrics.precision_score(y_test, y_pred, pos_label='positive',
                                                                               average='micro'),
                                    'Recall Score': metrics.recall_score(y_test, y_pred, pos_label='positive',
                                                                         average='micro'),
                                    'Accuracy Score': metrics.accuracy_score(y_test, y_pred),
                                    'Kappa Score': cohen_kappa_score(y_test, y_pred),
                                    'f1-score': metrics.f1_score(y_test, y_pred, pos_label='positive',
                                                                 average='micro')},
                                   ignore_index=True)
    return (score_card)


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

from xgboost import XGBClassifier
classifier = XGBClassifier()
XGB= classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
acc7 = accuracy_score(y_test, y_pred)
plot_confusion_matrix(XGB)
print(get_test_report(XGB))
print(f"Accuracy score: {acc7}")


mylist=[]
mylist2=[]
mylist.append(acc6)
mylist2.append("SVM")
mylist.append(acc5)
mylist2.append("DTR")
mylist.append(acc7)
mylist2.append("XGBoost")


plt.rcParams['figure.figsize']=8,6
sns.set_style("darkgrid")
plt.figure(figsize=(22,8))
ax = sns.barplot(x=mylist2, y=mylist, palette = "mako", saturation =3)
plt.xlabel("Classification Models", fontsize = 30, fontweight='bold' )
plt.ylabel("Accuracy", fontsize = 30, fontweight='bold')
plt.title("Accuracy of different Classification Models", fontsize = 32, fontweight='bold')
plt.xticks(fontsize = 20, horizontalalignment = 'center', rotation = 0)
plt.yticks(fontsize = 13)
for p in ax.patches:
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy()
    ax.annotate(f'{height:.2%}', (x + width/2, y + height*1.02), ha='center', fontsize = 'x-large')
plt.show()



from sklearn.tree import export_graphviz
from sklearn.tree import plot_tree

export_graphviz(Decision_tree, out_file='tree.dot', feature_names=list(x.columns), class_names=list(y.unique()))

fig = plt.figure(figsize=(150, 90), dpi= 210)
plot_tree(Decision_tree, feature_names=list(x.columns), class_names=list(y.unique()))
fig.savefig("decison_tree.png")

