import pandas as pd
import numpy as np

from sklearn.model_selection import cross_val_score

class BFS2Thres():

    def __init__(self, estimator, scoring, cv = 5, thresh=0.005, direction='b', feat_ar = np.array([]),
                 use_dummy = False, min_metric=-np.finfo(np.float32).max):

        self.estimator = estimator
        self.thresh = thresh
        self.cv = list(cv) if not isinstance(cv, int) else cv
        # optimization metric, the higher the best
        self.scoring = scoring
        self.direction = direction # b or f or s
        self.feat_ar = feat_ar
        self.use_dummy = use_dummy
        self.min_metric = min_metric

    def update_estimator(self, X):
        return self.estimator

    def calc_dummy_metric(self, X, y):

        if self.use_dummy:
            if self.estimator._estimator_type=='classifier':
                from sklearn.dummy import DummyClassifier
                dummy_model = DummyClassifier(strategy='most_frequent')
            elif self.estimator._estimator_type=='regressor':
                from sklearn.dummy import DummyRegressor
                dummy_model = DummyRegressor(strategy='mean')
            m = cross_val_score(dummy_model, X, y, cv=self.cv, scoring = self.scoring).mean()

        else:
            m = self.min_metric

        return m

    def one_iter(self, X, y, step, verbose, step_root, reverse=False):

        if reverse and verbose:
            print(f'''Идем в обратном порядке на итерации {self.feat_ar[step_root]}''')

        if reverse:
            if self.mask[step]==False:
                if verbose:
                    print(f'Признак - {self.feat_ar[step]} и так не входит в модель, не рассматриваем на дополнительное исключение \n')
                return

            self.mask[step]=False

        else:
            self.mask[step]=False if self.direction == 'b' else True


        if verbose:
            print(f"На рассмотрении признак {self.feat_ar[step]}:")

        if len(self.feat_ar[self.mask])==0:
            sc_cur = self.calc_dummy_metric(X, y)
        else:
            sc_cur = cross_val_score(self.update_estimator(X), X[self.nfeat_l+self.feat_ar[self.mask].tolist()], y, cv=self.cv, scoring = self.scoring).mean()

        qual_ratio = (sc_cur-self.sc_pr)/abs(self.sc_pr)

        if verbose:
            print(f"Текущее качество - {sc_cur}, прошлое: {self.sc_pr}, улучшение: {qual_ratio:.2%}")

        cond = qual_ratio>=-self.thresh if self.direction == 'b' else qual_ratio>=self.thresh
        if cond:
            self.sc_pr = sc_cur
        else:
            self.mask[step]=True if self.direction == 'b' else False

        # случай для s, убрали признак и качество ухудшилось
        if reverse and qual_ratio<0:
            self.mask[step]=True

        if verbose:
            print(f"Новый набор признаков: {self.feat_ar[self.mask]}")

            print(f"Не включенные признаки f{self.feat_ar[[not it for it in self.mask]]} \n")


    def fit(self, X, y, verbose=False):

        if len(self.feat_ar)==0:
            self.feat_ar = X.columns
            self.nfeat_l = []
        else:
            self.nfeat_l = [it for it in X.columns if not it in self.feat_ar]

        feat_num = len(self.feat_ar)
        self.mask = [True]*feat_num if self.direction == 'b' else [False]*feat_num


        self.sc_pr = cross_val_score(self.estimator, X, y, cv=self.cv, scoring = self.scoring).mean() if self.direction == 'b' else self.calc_dummy_metric(X, y)
        first_qual = self.sc_pr

        if verbose:
            print(f"Начальное качество на признаках: {self.feat_ar[self.mask]} - {first_qual}")

        for i in range(feat_num-1, -1, -1):

            if self.direction=='s':
                self.one_iter(X, y, step=i, verbose=verbose, step_root=i, reverse=False)

                # если включили новый, идем в обратном порядке
                if self.mask[i] and i < feat_num-1:
                    # feat_num не включается
                    for j in range(i+1, feat_num):
                        self.one_iter(X, y, step=j, verbose=verbose, step_root=i, reverse=True)
            else:
                self.one_iter(X, y, step=i, verbose=verbose, step_root=i, reverse=False)

        if verbose:
                print(f"Начальное качество было - {first_qual}, итоговое на признаках: {self.feat_ar[self.mask]} - {self.sc_pr}")



