from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
from Preprocess import Preprocess


class LogisticReg:
    def __init__(self):
        self.datapath = None  # final_dataset
        self.dataset = None
        self.feature = None
        self.label = None
        self.train = None
        self.y_train = None
        self.test = None
        self.y_test = None

    def pre(self, datapath):
        self.datapath = datapath
        p = Preprocess()
        a, b, c = p.preprocess(path=datapath)
        self.dataset = a
        self.feature = b
        self.label = c

    def grid_search(self):
        # Grid Search (Hyperparameter tuning)
        logreg = LogisticRegression(class_weight="balanced")
        param = {'C': [0.001, 0.003, 0.005, 0.01, 0.03, 0.05, 0.1, 0.3, 0.5, 1, 2, 3, 3, 4, 5, 10, 20]}

        clf = GridSearchCV(logreg, param, scoring="roc_auc", refit=True, cv=10)
        clf.fit(self.feature, self.label)
        print('Best roc_auc: {:.4}, with best C: {}'.format(clf.best_score_, clf.best_params_))

        return clf

    def model(self):
        self.train, self.test, self.y_train, self.y_test = train_test_split(self.feature, self.label, test_size=0.1,
                                                                  shuffle=True, stratify=self.label, random_state=123)

        seed = 123
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        pred_test_full = 0
        cv_score = []
        i = 1

        for train_index, val_index in kf.split(self.train, self.y_train):
            print('{} of KFold {}'.format(i, kf.n_splits))
            xtr, xvl = self.feature[train_index], self.feature[val_index]
            ytr, yvl = self.label[train_index], self.label[val_index]

            # model
            lr = LogisticRegression(C=2)
            lr.fit(xtr, ytr)
            score = roc_auc_score(yvl, lr.predict(xvl))
            print('ROC AUC score:', score)
            cv_score.append(score)
            pred_test = lr.predict_proba(self.test)[:, 1]
            pred_test_full += pred_test
            i += 1

        print("=================== Evaluation ===================")
        self.eval(lr, xvl, yvl, cv_score, pred_test_full)

    def eval(self, lr, x_val, y_val, cv_score, pred_test_full):
        print('Confusion matrix\n', confusion_matrix(y_val, lr.predict(x_val)))
        print('Cv', cv_score, '\nMean cv Score', np.mean(cv_score))

        print("Linear Regression Coefficient: ", lr.coef_)
        print("Linear Score: ", lr.score(x_val, y_val))

        proba = lr.predict_proba(x_val)[:, 1]
        frp, trp, threshold = roc_curve(y_val, proba)
        roc_auc_ = auc(frp, trp)

        plt.figure(figsize=(14, 8))
        plt.title('Reciever Operating Characteristics')
        plt.plot(frp, trp, 'r', label='AUC = %0.2f' % roc_auc_)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'b--')

        plt.ylabel('True positive rate')
        plt.xlabel('False positive rate')
        plt.show()

        y_pred = pred_test_full / 5
        submit = pd.DataFrame({'test_no.': range(len(self.test)), 'Label': y_pred})
        submit['Label'] = submit['Label'].apply(lambda x: 1 if x > 0.5 else 0)
        correct = []
        for i in range(len(self.y_test)):
            if submit['Label'][i] == self.y_test[i]:
                correct.append(True)
            else:
                correct.append(False)
        submit['is_correct'] = correct
        # submit.to_csv('lr_titanic.csv.gz',index=False,compression='gzip')
        print(submit)
        print("# of corret : ", (submit['is_correct'] == True).sum())
        # submit.to_csv('lr_titanic.csv', index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Argument')
    parser.add_argument('--data', type=str, help='need .csv path')
    args = parser.parse_args()

    logreg = LogisticReg()
    logreg.pre(args.data)
    clf = logreg.grid_search()
    logreg.model()
