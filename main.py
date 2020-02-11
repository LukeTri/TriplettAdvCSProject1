import pandas as pd
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot as plt, pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import numpy as np
import csv


def main():

    with open('2012-18_standings.csv', newline='') as f:
        new_rows = []
        reader = csv.reader(f, delimiter=',', quotechar='|')  # pass the file to our csv reader
        date = "2012-10-30"
        c = []
        dates = []
        teams = []
        dates.append(date)
        t = 0
        for row in reader:  # iterate over the rows in the file
            if t < 30:
                teams.append(row[1])
                t += 1
            if row[0] != date:
                new_rows.append(c)
                c = []
                date = row[0]
                dates.append(date)
            c.append(row)
        print(teams)

    with open('2012-18_teamBoxScore.csv', newline='') as fi:
        reader = csv.reader(fi, delimiter=',', quotechar='|')
        games = []
        t = 0
        c = []
        for row in reader:
            t += 1
            c.append(row[9])
            if t == 2:
                c.append(row[0])
                if row[13] == "Win":
                    c.append(0)
                else:
                    c.append(1)
                games.append(c)
                c = []
                t = 0
    new_games = []
    for i in games:
        c = []
        dstandings = new_rows[dates.index(i[2])-1]
        t1 = dstandings[teams.index(i[0])]
        t2 = dstandings[teams.index(i[1])]
        c.append((int(t1[4])-int(t1[5])-int(t2[4])+int(t2[5])))
        c.append((int(t1[10])-int(t1[11])-int(t2[10])+int(t2[11]))/(int(t1[4])+int(t1[5])+1))
        c.append(int(t1[19])-int(t2[19]))
        c.append(i[3])
        new_games.append(c)
    dataset = new_games
    dataset = np.asarray(dataset)
    df = pd.DataFrame(dataset, columns = ['Win Diff' , 'MOV Diff', 'Last 10 W-L Diff', "Result"])
    print(df.shape)
    print(df.head(20))
    df.plot(kind='box', subplots=True, layout=(2, 2), sharex=False, sharey=False)
    pyplot.show()
    array = df.values
    X = array[:, 0:3]
    y = array[:, 3]
    X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)

    models = []
    models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    results = []
    names = []
    print("***Using all types of data***")
    for name, model in models:
        kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
        cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
        results.append(cv_results)
        names.append(name)
        print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

    for i in range(3):
        if i == 0:
            X = array[:, 0:1]
            y = array[:, 3]
            print("***Using only W-L***")
        elif i == 1:
            X = array[:, 1:2]
            y = array[:, 3]
            print("***Using only point differential***")
        elif i == 2:
            X = array[:, 2:3]
            y = array[:, 3]
            print("***Using only last 10 games***")
        X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)
        models = []
        models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
        models.append(('LDA', LinearDiscriminantAnalysis()))
        models.append(('KNN', KNeighborsClassifier()))
        models.append(('CART', DecisionTreeClassifier()))
        models.append(('NB', GaussianNB()))
        results = []
        names = []
        for name, model in models:
            kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
            cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
            results.append(cv_results)
            names.append(name)
            print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))


if __name__ == '__main__':
    main()
