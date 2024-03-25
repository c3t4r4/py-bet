#pip install --upgrade scikit-learn

#pip3 install seaborn scikit-learn pandas numpy matplotlib

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from pandas.plotting import parallel_coordinates
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

path = "2024.csv"
df = pd.read_csv(path)

cblol = df[df['league']=='CBLOL']

attributes = cblol.columns

cblol.head(5)

sorted_cblol = cblol.sort_values(by=["teamname"], ascending=True)
sorted_cblol_encoded = pd.get_dummies(sorted_cblol, columns=['side'])
label_encoder = LabelEncoder()

sorted_cblol_encoded['side_blue_red'] = label_encoder.fit_transform(sorted_cblol_encoded['side_Blue'])

# print(sorted_cblol_encoded.dtypes)

# print(cblol.describe())

gamescblol = sorted_cblol_encoded[sorted_cblol_encoded['position']=='team']

# print(gamescblol.groupby('teamname').count())

label_encoder = LabelEncoder()

gamescblol['gameid'] = label_encoder.fit_transform(gamescblol['gameid'])
gamescblol['datacompleteness'] = label_encoder.fit_transform(gamescblol['datacompleteness'])
gamescblol['position'] = label_encoder.fit_transform(gamescblol['position'])
gamescblol['year'] = label_encoder.fit_transform(gamescblol['year'])
gamescblol['split'] = label_encoder.fit_transform(gamescblol['split'])
gamescblol['league'] = label_encoder.fit_transform(gamescblol['league'])
gamescblol['date'] = label_encoder.fit_transform(gamescblol['date'])
gamescblol['position'] = label_encoder.fit_transform(gamescblol['position'])
gamescblol['teamname'] = label_encoder.fit_transform(gamescblol['teamname'])
gamescblol['patch'] = label_encoder.fit_transform(gamescblol['patch'])

gamescblol['playerid'] = label_encoder.fit_transform(gamescblol['playerid'])
gamescblol['teamid'] = label_encoder.fit_transform(gamescblol['teamid'])

gamescblol['ban1'] = label_encoder.fit_transform(gamescblol['ban1'])
gamescblol['ban2'] = label_encoder.fit_transform(gamescblol['ban2'])
gamescblol['ban3'] = label_encoder.fit_transform(gamescblol['ban3'])
gamescblol['ban4'] = label_encoder.fit_transform(gamescblol['ban4'])
gamescblol['ban5'] = label_encoder.fit_transform(gamescblol['ban5'])
gamescblol['pick1'] = label_encoder.fit_transform(gamescblol['pick1'])
gamescblol['pick2'] = label_encoder.fit_transform(gamescblol['pick2'])
gamescblol['pick3'] = label_encoder.fit_transform(gamescblol['pick3'])
gamescblol['pick4'] = label_encoder.fit_transform(gamescblol['pick4'])
gamescblol['pick5'] = label_encoder.fit_transform(gamescblol['pick5'])


gamescblolData = gamescblol.drop(columns=['url'])
gamescblolData = gamescblolData.fillna(0)

gamescblolData.head(5)

# # Normalizar os dados usando Min-Max Scaling
scaler = MinMaxScaler()
df_normalized = scaler.fit_transform(gamescblolData)

X = df_normalized

columns = gamescblolData.columns

X_train, X_test = train_test_split(X, test_size=0.2,  random_state=42)

df_train = pd.DataFrame(data=X_train,columns=columns)

df_teste = pd.DataFrame(data=X_test,columns=columns)


print(df_train.head(5))

sns.pairplot(df_train, hue="result", height = 2, palette = 'colorblind')

# correlation matrix
corrmat = df_train.corr()
sns.heatmap(corrmat, annot = True, square = True)

# # Model development
# X_train = df_train[['sepal_length','sepal_width','petal_length','petal_width']]
# y_train = df_train['class']
# X_test = df_teste[['sepal_length','sepal_width','petal_length','petal_width']]
# y_test = df_teste['class']

# # first try decision tree
# mod_dt = DecisionTreeClassifier(max_depth = 3, random_state = 1)
# mod_dt.fit(X_train,y_train)
# prediction=mod_dt.predict(X_test)
# print('The accuracy of the Decision Tree is',"{:.3f}".format(metrics.accuracy_score(prediction,y_test)))
#
# # set figure size
# plt.figure(figsize = (10,8))
# plot_tree(mod_dt, feature_names = fn, class_names = cn, filled = True);

# from sklearn.metrics import ConfusionMatrixDisplay
#
#
#
# cm = metrics.confusion_matrix(y_test, mod_dt.predict(X_test))
# disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=cn)
# disp.plot()
# print(cm)

# # Guassian Naive Bayes Classifier
# mod_gnb_all = GaussianNB()
# y_pred = mod_gnb_all.fit(X_train, y_train).predict(X_test)
# print('The accuracy of the Guassian Naive Bayes Classifier on test data is',"{:.3f}".format(metrics.accuracy_score(y_pred,y_test)))
#
# # LDA Classifier
# mod_lda_all = LinearDiscriminantAnalysis()
# y_pred = mod_lda_all.fit(X_train, y_train).predict(X_test)
# print('The accuracy of the LDA Classifier on test data is',"{:.3f}".format(metrics.accuracy_score(y_pred,y_test)))
#
# # try different k
# acc_s = pd.Series(dtype = 'float')
# for i in list(range(1,11)):
#     mod_knn=KNeighborsClassifier(n_neighbors=i)
#     mod_knn.fit(X_train,y_train)
#     prediction=mod_knn.predict(X_test)
#     acc_s = acc_s.append(pd.Series(metrics.accuracy_score(prediction,y_test)))
#
# plt.plot(list(range(1,11)), acc_s)
# plt.suptitle("Test Accuracy vs K")
# plt.xticks(list(range(1,11)))
# plt.ylim(0.9,1.02);