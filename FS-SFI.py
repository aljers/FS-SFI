import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors._base import NeighborsBase
from sklearn.model_selection import LeaveOneOut
from sklearn.base import ClassifierMixin
from sklearn.metrics import accuracy_score

class FS_SSv2(ClassifierMixin, NeighborsBase):
    def __init__(
            self,
            subset_size=None,
            random_state=None):
        self.subset_size = subset_size
        self.num_of_subset = None
        self.random_state = random_state
        self.feature_list = []
        self.score = []
        self.A = None
        self.B = None
        self.Nc = None

    def process_1(self, X, y, ratio, iter=10):
        self.feature_list = X.columns.tolist()

        if self.random_state == None:
            self.random_state = np.random.randint(10000000)
        np.random.seed(self.random_state)
        self.initial_ss(X, y)
        model = KNeighborsClassifier(n_neighbors=1)
        acc_set = []
        # print(f'Round 0, size = {X.shape[1]}, accuracy = {acc_set[0]}')
        result_set = []
        run = 0
        while True:
            #
            acc = self.calculate_acc(X, y, model)
            acc_set.append(acc)
            print(f'Round {run}, size = {X.shape[1]}, accuracy = {acc}')

            R = []
            for i in range(iter):
                subset_list, score = self.feature_sampling(X, y, seed=self.random_state + i)
                # print(len(R)+1)
                R_temp = self.R_calculate(subset_list, score, X.columns.tolist())
                R.append(R_temp)
            res = pd.DataFrame({'R': np.average(R, axis=0),
                                'feature': X.columns.tolist()}).sort_values(by='R', ascending=False)
            result_set.append(res)
            #
            if int(len(res) * ratio) <= max(10, self.subset_size):
                break
            top_feature = res.feature.iloc[:int(len(res) * ratio)].tolist()
            run += 1
            X = X[top_feature]
        max_value = max(acc_set)
        last_max_index = len(acc_set) - 1 - acc_set[::-1].index(max_value)
        return result_set[max(last_max_index - 1, 0)], acc_set  # , result_set[np.argmax(acc_set)]

    def process_middle(self, X, y, ratio, iter=10, common_difference=None):

        if self.random_state == None:
            self.random_state = np.random.randint(10000000)
        if len(self.feature_list) != X.shape[1]:
            p = X.shape[1] * ratio * ratio
        else:
            p = X.shape[1] * ratio
        model = KNeighborsClassifier(n_neighbors=1)
        acc_set = []
        # print(f'Round 0, size = {X.shape[1]}, accuracy = {acc_set[0]}')
        result_set = []
        if not common_difference:
            common_difference = int((X.shape[1] - p) / 10)
        print(f'Common difference = {common_difference}')
        #
        for run in range(10):
            acc = self.calculate_acc(X, y, model)
            acc_set.append(acc)
            print(f'Round {run}, size = {X.shape[1]}, accuracy = {acc}')
            R = []
            for i in range(iter):
                subset_list, score = self.feature_sampling(X, y, seed=self.random_state + i)
                # print(len(R) + 1)
                R_temp = self.R_calculate(subset_list, score, X.columns.tolist())
                R.append(R_temp)
            res = pd.DataFrame({'R': np.average(R, axis=0), 'feature': X.columns.tolist()}).sort_values(by='R',
                                                                                                        ascending=False)
            result_set.append(res)
            #
            top_feature = res.feature.iloc[:int(len(res) - common_difference)]
            print(f'Round {run}, size = {int(X.shape[1] * ratio)}, accuracy = {acc}')
            run += 1
            X = X[top_feature]
        return result_set[np.argmax(acc_set) - 1], common_difference, acc_set

    def process_2(self, X, y, ratio, dif, iter=10):

        if self.random_state == None:
            self.random_state = np.random.randint(10000000)
        if len(self.feature_list) != X.shape[1]:
            p = X.shape[1] * ratio * ratio
        else:
            p = X.shape[1] * ratio

        model = KNeighborsClassifier(n_neighbors=1)
        acc_set = []
        # print(f'Phase2 Round 0, size = {X.shape[1]}, accuracy = {acc_set[0]}')
        result_set = []
        run = 1
        while True:
            acc = self.calculate_acc(X, y, model)
            acc_set.append(acc)
            print(f'Phase 2 Round {run}, size = {X.shape[1]}, accuracy = {acc}')
            #
            R = []
            for i in range(iter):
                subset_list, score = self.feature_sampling(X, y, seed=self.random_state + i)
                R_temp = self.R_calculate(subset_list, score, X.columns.tolist())
                R.append(R_temp)
            res = pd.DataFrame({'R': np.average(R, axis=0),
                                'feature': X.columns.tolist()}).sort_values(by='R', ascending=False)
            result_set.append(res)
            #
            if len(res) - dif <= p:
                break
            top_feature = res.feature.iloc[:-dif].tolist()
            run += 1
            #
            X = X[top_feature]
        max_value = max(acc_set)
        last_max_index = len(acc_set) - 1 - acc_set[::-1].index(max_value)
        return result_set[last_max_index], acc_set

    def process(self, X, y, ratio, dif, iter=10):
        stage1_set, acc_set1 = self.process_1(X, y, ratio, iter)
        stage2_set, acc_set2 = self.process_2(X[stage1_set.feature], y, ratio, dif, iter)

        return stage2_set, acc_set1, acc_set2

    def process_L(self, X, y, ratio, dif, iter=10):
        stage1_set, acc_set1, best_stage1 = self.process_1(X, y, ratio, iter)
        if int(len(stage1_set) * (1 / ratio - ratio)) <= dif * 10:
            return stage1_set, acc_set1, []
        i = 1
        print(f'Middle stage {i} for large feature size:')
        stage_middle, difference, acc_middle = self.process_middle(X[stage1_set.feature], y,
                                                                     ratio, iter, common_difference=None)
        while max(dif, 10) < difference:
            i += 1
            print(f'Middle stage {i} for large feature size:')
            stage_middle, difference, acc_middle = self.process_middle(X[stage_middle.feature], y,
                                                                         ratio, iter,
                                                                         common_difference=int(difference / 10))
        if max(acc_set1) > max(acc_middle):
            return best_stage1, acc_set1, acc_middle
        return stage_middle, acc_set1, acc_middle

        i = 1
        print(f'Middle stage {i} for large feature size:')
        stage_middle, difference, acc_middle = self.process3_middle(X[stage1_set.feature], y,
                                                                    ratio, iter, common_difference=None)
        while max(dif, 10) < difference:
            i += 1
            print(f'Middle stage {i} for large feature size:')
            stage_middle, difference, acc_middle = self.process3_middle(X[stage_middle.feature], y,
                                                                        ratio, iter,
                                                                        common_difference=int(difference / 10))

        return stage_middle, acc_set1, acc_middle

    def calculate_acc(self, X, y, model):

        score = []
        loo = LeaveOneOut()
        for i, (train_index, test_index) in enumerate(loo.split(X)):
            train_X, test_X = X.iloc[train_index], X.iloc[test_index]
            train_y, test_y = y[train_index], y[test_index]
            model.fit(train_X, train_y)
            y_pred = model.predict(test_X)
            score.append(accuracy_score(test_y, y_pred))

        return np.average(score)

    def feature_sampling(self, X, y, seed=None):
        p = X.shape[1]
        if p % self.subset_size == 0:
            num_of_subset = int(p / self.subset_size)
        else:
            num_of_subset = int(np.floor(p / self.subset_size) + 1)
        subset_list = []
        score = []
        i = 0
        X_copy = X.copy()
        while len(subset_list) < num_of_subset - 1:
            X_sub = X_copy.sample(n=self.subset_size, replace=False, axis=1, random_state=seed + i)
            X_copy.drop(columns=X_sub.columns, inplace=True)
            subset_list.append(X_sub)
            score.append(self.seperation_score(X_sub, y))
            i += 1
        if X_copy.shape[1] > 0:
            subset_list.append(X_copy)
            score.append(self.seperation_score(X_copy, y))
        return subset_list, score

    def seperation_score(self, X_sub, y):
        if isinstance(y, pd.DataFrame):
            y = y.iloc[:, 0]

        # direct use A and B to conduct ss
        a = np.array([1 if i in X_sub.columns else 0 for i in self.feature_list])
        aA = np.sqrt(np.array(self.A).dot(a)).sum()
        aB = np.sqrt(np.array(self.B).dot(a) * self.Nc).sum()
        return aA / aB

    def R_calculate(self, subsets, ss, features):

        ss = np.array(ss)
        R = np.zeros(len(features))
        for i in range(len(features)):
            var = features[i]
            index = np.array([(var in j.columns.tolist()) for j in subsets])
            if sum(index) > 0:
                R[i] = sum(ss[index]) / sum(index)
            else:
                R[i] = 0
        return R

    def initial_ss(self, X, y):
        # this function is only used in the beginning of the first stage to confirm matrix B and W

        centers = [np.mean(X[y == i], axis=0) for i in set(y)]

        if (not self.A) and (not self.B):
            A = []
            for i in centers:
                A.append([(px - qx) ** 2.0 for px, qx in zip(i, np.mean(X, axis=0))])
            self.A = A
            B = []
            Nc = np.array([])
            for i in range(len(centers)):
                Nc = np.append(Nc, [1 / X[y == i].shape[0]] * X[y == i].shape[0])
                for _, j in X[y == i].iterrows():
                    B.append([(px - qx) ** 2.0 for px, qx in zip(centers[i], j)])
            self.B = B
            self.Nc = Nc
        return