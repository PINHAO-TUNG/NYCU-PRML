import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.metrics import accuracy_score

classtype = [0, 1]
train_num = 201
test_num = 100

feature_num = 15
feature_name = None


def entropy(sequence):
    entropy = 0
    for i in range(len(classtype)):
        pj = np.sum(sequence == classtype[i]) / len(sequence)
        if pj != 0:
            entropy -= pj * np.log2(pj)
    return entropy


def gini(sequence):
    sigama_pj = 0
    for i in range(len(classtype)):
        sigama_pj = sigama_pj + \
            (np.sum(sequence == classtype[i]) / len(sequence)) ** 2
    gini = 1 - sigama_pj
    return gini


def criterion_type(sequence, model):
    if model == 'entropy':
        return entropy(sequence)
    elif model == 'gini':
        return gini(sequence)
    else:
        return print("please chose the criterion entropy or gini")


class Tree():
    def __init__(self, x, depth, Leaf=False):
        self.x = x
        self.depth = depth
        self.Leaf = Leaf
        self.info = 0
        self.left = None
        self.right = None
        self.attribute_index = -1
        self.threshold = -1

    def setting(self, attribute_index, threshold, left, right):
        self.left = left
        self.right = right
        self.attribute_index = attribute_index
        self.threshold = threshold


class DecisionTree():
    def __init__(self, criterion='gini', max_depth=None):
        self.criterion = criterion
        self.max_depth = max_depth
        self.tree = None
        self.importance = dict.fromkeys(feature_name, 0)

    def split_Attribute(self, x):
        feature_index, threshold = 0, 0.0
        info_o = criterion_type(x.values[0:, -1], self.criterion)
        info_gain = 0
        # choice feature to split Attribute
        for col in range(len(x.columns)-1):  # minus the target column
            tmp_x = x.sort_values([x.columns[col]])  # sort every features
            # determine every feature the cutting point
            for i in range(len(x)):
                left = criterion_type(tmp_x.values[0:i+1, -1], self.criterion)
                right = criterion_type(
                    tmp_x.values[i:train_num, -1], self.criterion)
                info_m = (i + 1) * left + (train_num - i + 1) * right
                info_m = info_m / train_num
                if info_o - info_m > info_gain:
                    info_gain = info_o - info_m
                    feature_index, threshold = col, tmp_x.values[i, col]
        return info_gain, feature_index, threshold

    def divideTree(self, tree, feature_index, threshold):
        left_df = pd.DataFrame(columns=tree.x.columns)
        right_df = pd.DataFrame(columns=tree.x.columns)
        # create left and right branches
        for i in range(len(tree.x)):
            if tree.x.values[i, feature_index] <= threshold:
                left_df = left_df.append(tree.x.loc[tree.x.index[i]])
            else:
                right_df = right_df.append(tree.x.loc[tree.x.index[i]])
        left = Tree(left_df, tree.depth + 1)
        right = Tree(right_df, tree.depth + 1)
        if len(left.x) <= 0:
            left.Leaf = True
        if len(right.x) <= 0:
            right.Leaf = True
        return left, right

    def genTree(self, tree):
        info_now = criterion_type(tree.x['target'], self.criterion)
        if tree.depth >= self.max_depth or info_now == 0:
            tree.Leaf = True
            return tree
        info_gain, feature_index, threshold = self.split_Attribute(tree.x)
        if info_gain == 0:
            tree.Leaf = True
            return tree
        # divide tree
        left, right = self.divideTree(tree, feature_index, threshold)
        # select best split attribute and threshold
        tree.setting(feature_index, threshold, left, right)
        # move to children tree
        if not left.Leaf:
            self.genTree(left)
        if not right.Leaf:
            self.genTree(right)

    def fit(self, x):
        self.tree = Tree(x, 1)
        self.genTree(self.tree)

    # predict class function
    def predict_function(self, test):
        tmp = self.tree
        while True:
            if test[tmp.attribute_index] <= tmp.threshold:
                if tmp.left is None:
                    break
                tmp = tmp.left
            else:
                if tmp.right is None:
                    break
                tmp = tmp.right
        pre = np.zeros((len(classtype)), dtype=np.uint16)
        for c in range(len(classtype)):
            pre[c] = np.sum(tmp.x['target'] == classtype[c])
        y_class = classtype[np.argmax(pre)]
        return y_class

    # predict class
    def predict(self, X):
        predict_y_class = np.zeros((test_num), dtype=np.uint8)
        for i in range(test_num):
            predict_y_class[i] = self.predict_function(X.values[i])
        return predict_y_class

    # feature importance
    def count_Importance(self, tree):
        if tree.attribute_index != -1:
            self.importance[feature_name[tree.attribute_index]] += 1
        if tree.left is not None:
            self.count_Importance(tree.left)
        if tree.right is not None:
            self.count_Importance(tree.right)


def feature_importance(importance):
    importance = dict(
        sorted(importance.items(), key=lambda student: student[1]))
    importance = {k: v for k, v in importance.items() if v != 0}
    plt.style.use('seaborn-muted')
    plt.title('Feature Importance')
    plt.barh(list(importance.keys()), importance.values(), alpha=0.6)
    plt.show()

# adaboost


class ADADecisionTree():
    def __init__(self, n_estimators, train_df, criterion='gini', max_depth=3, boostrap=True, index_cor=[], weighr_u=0, weight_d=0):
        self.n_estimators = n_estimators
        self.train_df = train_df
        self.criterion = criterion
        self.max_depth = max_depth
        self.tree = None
        self.boostrap = boostrap

        self.index_correct = index_cor
        self.weight_up = weighr_u
        self.weight_down = weight_d

    def split_Attribute(self, x):
        feature_index, threshold = 0, 0.0
        info_o = criterion_type(x.values[0:, -1], self.criterion)
        info_gain = 0
        left = 0
        right = 0
        # choice feature to split Attribute
        for col in range(len(x.columns)-1):  # minus the target column
            tmp_x = x.sort_values([x.columns[col]])  # sort every features
            # determine every feature the cutting point
            for i in range(len(x)):
                left_twu_class = np.zeros(len(x), dtype=np.int64)
                left_tdu_class = np.zeros(len(x), dtype=np.int64)
                right_twu_class = np.zeros(len(x), dtype=np.int64)
                right_tdu_class = np.zeros(len(x), dtype=np.int64)
                for j in range(len(x)):
                    if j <= i:
                        if tmp_x.index[j] in self.index_correct:
                            left_twu_class[j] = self.train_df.values[j:j+1, -1]
                        else:
                            left_tdu_class[j] = self.train_df.values[j:j+1, -1]
                    else:
                        if tmp_x.index[j] in self.index_correct:
                            right_twu_class[j] = self.train_df.values[j:j+1, -1]
                        else:
                            right_tdu_class[j] = self.train_df.values[j:j+1, -1]
                left_uw = self.weight_up*criterion_type(
                    left_twu_class, self.criterion)
                left_dw = self.weight_down*criterion_type(
                    left_tdu_class, self.criterion)
                left = left_uw + left_dw
                right_uw = self.weight_up*criterion_type(
                    right_twu_class, self.criterion)
                right_dw = self.weight_down*criterion_type(
                    right_tdu_class, self.criterion)
                right = right_uw + right_dw
                info_m = (i + 1) * left + (train_num - i + 1) * right
                info_m = info_m / train_num
                if info_o - info_m > info_gain:
                    info_gain = info_o - info_m
                    feature_index, threshold = col, tmp_x.values[i, col]
        return info_gain, feature_index, threshold

    def divideTree(self, tree, feature_index, threshold):
        left_df = pd.DataFrame(columns=tree.x.columns)
        right_df = pd.DataFrame(columns=tree.x.columns)
        # create left and right branches
        for i in range(len(tree.x)):
            if tree.x.values[i, feature_index] <= threshold:
                left_df = left_df.append(tree.x.loc[tree.x.index[i]])
            else:
                right_df = right_df.append(tree.x.loc[tree.x.index[i]])
        left = Tree(left_df, tree.depth + 1)
        right = Tree(right_df, tree.depth + 1)
        if len(left.x) <= 0:
            left.Leaf = True
        if len(right.x) <= 0:
            right.Leaf = True
        return left, right

    def genTree(self, tree):
        info_now = criterion_type(tree.x['target'], self.criterion)
        if tree.depth >= self.max_depth or info_now == 0:
            tree.Leaf = True
            return tree
        info_gain, feature_index, threshold = self.split_Attribute(
            tree.x)
        if info_gain == 0:
            tree.Leaf = True
            return tree
        # divide tree
        left, right = self.divideTree(tree, feature_index, threshold)
        # select best split attribute and threshold
        tree.setting(feature_index, threshold, left, right)
        # move to children tree
        if not left.Leaf:
            self.genTree(left)
        if not right.Leaf:
            self.genTree(right)

    def fit(self, x):
        self.tree = Tree(x, 1)
        self.genTree(self.tree)

    def predict_function(self, test):
        tmp = self.tree
        while True:
            if test[tmp.attribute_index] <= tmp.threshold:
                if tmp.left is None:
                    break
                tmp = tmp.left
            else:
                if tmp.right is None:
                    break
                tmp = tmp.right
        pre = np.zeros((len(classtype)), dtype=np.uint16)
        for c in range(len(classtype)):
            pre[c] = np.sum(tmp.x['target'] == classtype[c])
        y_class = classtype[np.argmax(pre)]
        return y_class

    # predict class
    def predict(self, X):
        predict_y_class = np.zeros((test_num), dtype=np.uint8)
        for i in range(test_num):
            predict_y_class[i] = self.predict_function(X.values[i])
        return predict_y_class

    # check train accuracy
    def train_accuracy(self, X):
        y = np.zeros((train_num), dtype=np.uint8)
        for i in range(train_num):
            y[i] = self.predict_function(X.values[i])
        return y


class AdaBoost():
    def __init__(self, n_estimators, train_df, test_df, criterion='gini', max_depth=1):
        self.n_estimators = n_estimators
        self.train_df = train_df
        self.test_df = test_df
        self.criterion = criterion
        self.max_depth = max_depth
        self.tree = None
        self.forest = []
        self.total_acc = 0
        self.test_class_predict = []
        self.weight_alpha = []

    def alpha_k(self, acc):
        deta_t = 1 - acc
        alpha = 0.5*np.log((1-deta_t)/deta_t)
        return alpha

    def weight_k(self, acc, weight_up, weight_down, index_correct):
        Weight_up_new = weight_up*math.pow(math.e, -1*self.alpha_k(acc))
        Weight_down_new = weight_down*math.pow(math.e, self.alpha_k(acc))
        z_sumofweight = Weight_up_new * \
            len(index_correct) + Weight_down_new * \
            (train_num - len(index_correct))
        Weight_up_new = Weight_up_new/z_sumofweight
        Weight_down_new = Weight_down_new/z_sumofweight
        return Weight_up_new, Weight_down_new

    def Initialization(self, y_clas):
        tree_y_class = y_clas
        index_correct_new = []
        for i in range(train_num):
            if tree_y_class[i] == self.train_df.values[i:i+1, -1]:
                index_correct_new.append(self.train_df.index[i])
        return index_correct_new

    def buildAdaBoost(self):
        index_correct = self.train_df.index
        weight_up = 1/train_num
        weight_down = 1/train_num
        for n in range(self.n_estimators):
            # print(n)
            clf_gini = ADADecisionTree(
                n_estimators=self.n_estimators, criterion=self.criterion, train_df=self.train_df, max_depth=self.max_depth, index_cor=index_correct, weighr_u=weight_up, weight_d=weight_down)
            clf_gini.fit(self.train_df)
            y_predict = clf_gini.train_accuracy(self.train_df[feature_name])
            acc = accuracy_score(self.train_df['target'], y_predict)
            weight_up, weight_down = self.weight_k(
                acc, weight_up, weight_down, index_correct)
            index_correct = self.Initialization(y_predict)
            #
            y_test_class_predict = clf_gini.predict(self.test_df[feature_name])
            alpha_now = self.alpha_k(acc)
            self.forest.append(clf_gini)
            self.test_class_predict.append(y_test_class_predict)
            self.weight_alpha.append(alpha_now)

    def AdaBoost_predict(self):
        # print(self.test_class_predict)
        # print(self.weight_alpha)
        weight_alpha_Sum = sum(self.weight_alpha)
        pred_y = np.zeros((test_num), dtype=np.uint8)
        for i in range(test_num):
            temp_0 = 0
            temp_1 = 1
            for j in range(self.n_estimators):
                if self.test_class_predict[j][i] == 0:
                    temp_0 = temp_0 + self.weight_alpha[j]/weight_alpha_Sum
                else:
                    temp_1 = temp_1 + self.weight_alpha[j]/weight_alpha_Sum
            if temp_0 >= temp_1:
                pred_y[i] = 0
            else:
                pred_y[i] = 1
        return pred_y
# end of adaboost


class RandomForest():
    def __init__(self, n_estimators, max_features, boostrap=True, criterion='gini', max_depth=None):
        self.n_estimators = n_estimators
        self.max_features = int(max_features)
        self.boostrap = boostrap
        self.criterion = criterion
        self.max_depth = 15
        self.forest = []

    def split_Attribute(self, x):
        attribute_index, threshold = 0, 0.0
        info_o = criterion_type(x.values[0:, -1], self.criterion)
        info_gain = 0
        for col in range(len(x.columns)-1):  # exclude label column
            tmp_x = x.sort_values(by=[x.columns[col]])
            for i in range(len(x)):
                left = criterion_type(tmp_x.values[0:i+1, -1], self.criterion)
                right = criterion_type(
                    tmp_x.values[i:train_num, -1], self.criterion)
                info_m = ((i + 1) * left + (train_num - i + 1) * right)
                info_m = info_m / train_num
                if info_o - info_m > info_gain:
                    info_gain = info_o - info_m
                    attribute_index, threshold = col, tmp_x.values[i, col]
        attribute_index = feature_name.index(x.columns[attribute_index])
        return info_gain, attribute_index, threshold

    def partiTree(self, tree, attribute_index, threshold):
        left_df = pd.DataFrame(columns=tree.x.columns)
        right_df = pd.DataFrame(columns=tree.x.columns)
        for i in range(len(tree.x)):
            if tree.x.values[i, attribute_index] <= threshold:
                left_df = left_df.append(tree.x.iloc[i])
            else:
                right_df = right_df.append(tree.x.iloc[i])
        left = Tree(left_df, tree.depth + 1)
        right = Tree(right_df, tree.depth + 1)
        if len(left.x) <= 0:
            left.Leaf = True
        if len(right.x) <= 0:
            right.Leaf = True
        return left, right

    def featureDelete(self):
        selected = np.random.choice(feature_num,
                                    self.max_features, replace=False)
        return np.delete(np.arange(feature_num), selected)

    def growTree(self, tree):
        info_now = criterion_type(tree.x['target'], self.criterion)
        if tree.depth >= self.max_depth or info_now == 0:
            tree.Leaf = True
            return tree
        # decide split
        for _iter in range(3):
            # Random select split attribute
            tree_copy_df = tree.x.copy()
            delete_feature = self.featureDelete()
            tree_copy_df = tree_copy_df.drop(
                columns=[feature_name[i] for i in delete_feature])
            # Select best split attribute and threshold
            info_gain, attribute_index, threshold = self.split_Attribute(
                tree_copy_df)
            if info_gain == 0:
                tree.Leaf = True
                return tree
            #  partitree
            left, right = self.partiTree(tree, attribute_index, threshold)
            if len(left.x) != len(tree.x) and len(right.x) != len(tree.x):
                break
        tree.setting(attribute_index, threshold, left, right)

        # Move to tree's children
        if not left.Leaf:
            self.growTree(left)
        if not right.Leaf:
            self.growTree(right)
        return tree

    def Bagging(self, X):
        bagging_df = pd.DataFrame(columns=X.columns)
        for n in range(len(X)):
            index = np.random.randint(0, len(X))
            bagging_df = bagging_df.append(X.iloc[index])
        return bagging_df

    def build_Forest(self, X):
        for n in range(self.n_estimators):
            data = self.Bagging(X) if self.boostrap else X
            root = Tree(data, 1)
            self.forest.append(self.growTree(root))

    def predict_function(self, test):
        vote = np.zeros((len(classtype)), dtype=np.uint32)
        for n in range(self.n_estimators):
            tmp = self.forest[n]
            while True:
                if test[tmp.attribute_index] <= tmp.threshold:
                    if tmp.left is None:
                        break
                    tmp = tmp.left
                else:
                    if tmp.right is None:
                        break
                    tmp = tmp.right
            pre = np.zeros((len(classtype)), dtype=np.uint32)
            for c in range(len(classtype)):
                pre[c] = np.sum(tmp.x['target'] == classtype[c])
            vote[np.argmax(pre)] += 1
        y_class = classtype[np.argmax(vote)]
        return y_class

    def predict(self, X):
        predict_y_class = np.zeros((test_num), dtype=np.uint8)
        for i in range(test_num):
            predict_y_class[i] = self.predict_function(X.values[i])
        return predict_y_class


if __name__ == "__main__":
    file_url = "http://storage.googleapis.com/download.tensorflow.org/data/heart.csv"
    df = pd.read_csv(file_url)

    train_idx = np.load('train_idx.npy')
    test_idx = np.load('test_idx.npy')
    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]

    reversible = []
    fixed = []
    normal = []
    for i in range(201):
        if train_df.iloc[i, 12] == "reversible":
            reversible.append(1)
            fixed.append(0)
            normal.append(0)
        elif train_df.iloc[i, 12] == "fixed":
            reversible.append(0)
            fixed.append(1)
            normal.append(0)
        elif train_df.iloc[i, 12] == "normal":
            reversible.append(0)
            fixed.append(0)
            normal.append(1)
        else:
            reversible.append(0)
            fixed.append(0)
            normal.append(0)
    train_df.insert(13, column="reversible", value=reversible)
    train_df.insert(14, column="fixed", value=fixed)
    train_df.insert(15, column="normal", value=normal)
    train = train_df.drop(columns='thal')
    test_reversible = []
    test_fixed = []
    test_normal = []
    for i in range(100):
        if test_df.iloc[i, 12] == "reversible":
            test_reversible.append(1)
            test_fixed.append(0)
            test_normal.append(0)
        elif test_df.iloc[i, 12] == "fixed":
            test_reversible.append(0)
            test_fixed.append(1)
            test_normal.append(0)
        elif test_df.iloc[i, 12] == "normal":
            test_reversible.append(0)
            test_fixed.append(0)
            test_normal.append(1)
        else:
            test_reversible.append(0)
            test_fixed.append(0)
            test_normal.append(0)
    test_df.insert(13, column="reversible", value=test_reversible)
    test_df.insert(14, column="fixed", value=test_fixed)
    test_df.insert(15, column="normal", value=test_normal)
    test = test_df.drop(columns='thal')

    feature_name = [s for s in train.columns]
    del feature_name[15]

    # Question 1
    print("-------------------------------------------------------------------")
    print("Question 1")
    data = np.array([1, 2, 1, 1, 1, 1, 2, 2, 1, 1, 2])
    data[data == 2] = 0
    # Gini of data
    print("Gini of data is ", gini(data))
    # Entropy of data
    print("Entropy of data is ", entropy(data))

    # Question 2
    # Question 2.1
    print("-------------------------------------------------------------------")
    print("Question 2")
    print("Question 2.1: Showing The Accuracy Score Of Test Data by max_depth=3 and max_depth=10")
    # max_depth=3
    clf_depth3 = DecisionTree(criterion='gini', max_depth=3)
    clf_depth3.fit(train)
    y_predict = clf_depth3.predict(test[feature_name])
    acc = accuracy_score(test['target'], y_predict)
    print("criterion='gini', max_depth=3, Acc:", acc)
    # max_depth=10
    clf_depth10 = DecisionTree(criterion='gini', max_depth=10)
    clf_depth10.fit(train)
    y_predict = clf_depth10.predict(test[feature_name])
    acc = accuracy_score(test['target'], y_predict)
    print("criterion='gini', max_depth=10, Acc:", acc)

    # Question 2.2
    print("Question 2.2:  Showing The Accuracy Score Of Test Data by criterion=gini and criterion=entropy")
    # criterion=gini
    clf_gini = DecisionTree(criterion='gini', max_depth=3)
    clf_gini.fit(train)
    y_predict = clf_gini.predict(test[feature_name])
    acc = accuracy_score(test['target'], y_predict)
    print("criterion='gini', max_depth=3 Acc:", acc)
    # criterion=entropy
    clf_entropy = DecisionTree(criterion='entropy', max_depth=3)
    clf_entropy.fit(train)
    y_predict = clf_entropy.predict(test[feature_name])
    acc = accuracy_score(test['target'], y_predict)
    print("criterion ='entropy', max_depth=3, Acc:", acc)

    # Question 3
    print("-------------------------------------------------------------------")
    print("Question 3: Plot the feature importance of your Decision Tree model")
    clf_depth10.count_Importance(clf_depth10.tree)
    feature_importance(clf_depth10.importance)

    # Question 4
    print("-------------------------------------------------------------------")
    print("Question 4: Implement The AdaBooest Algorithm by using the CART ")
    print("Question 4.1: Show the accuracy score of test data by n_estimators=10 and n_estimators=100")
    # n_estimators=10
    clf_AdaBooest = AdaBoost(
        n_estimators=10, train_df=train, test_df=test)
    clf_AdaBooest.buildAdaBoost()
    y_train_predict = clf_AdaBooest.AdaBoost_predict()
    acc = accuracy_score(test['target'], y_train_predict)
    print("accuracy score of test data by n_estimators=10:", acc)
    # n_estimators=10
    clf_AdaBooest = AdaBoost(
        n_estimators=100, train_df=train, test_df=test)
    clf_AdaBooest.buildAdaBoost()
    y_train_predict = clf_AdaBooest.AdaBoost_predict()
    acc = accuracy_score(test['target'], y_train_predict)
    print("accuracy score of test data by n_estimators=100:", acc)
    # Question 5 implement the Random Forest algorithm by using the CART you just implemented from question 2.
    # You should implement three arguments for the Random Forest. n_estimators, max_features, bootstrap
    print("-------------------------------------------------------------------")
    print("Question 5")
    print("Question 5.1: Showing The Accuracy Score Of Test Data by n_estimators=10 and n_estimators=100")
    # Question 5.1
    # n_estimators=10
    clf_estimoators_10 = RandomForest(n_estimators=10,
                                      max_features=np.sqrt(test[feature_name].shape[1]))
    clf_estimoators_10.build_Forest(train)
    y_predict = clf_estimoators_10.predict(test[feature_name])
    acc = accuracy_score(test['target'], y_predict)
    print(
        f"criterion=gini, max_depth=None, max_features=sqrt(n_features), estimoators=10, Acc: {acc}")
    # n_estimators=100
    clf_estimoators100 = RandomForest(n_estimators=100,
                                      max_features=np.sqrt(test[feature_name].shape[1]))
    clf_estimoators100.build_Forest(train)
    y_predict = clf_estimoators100.predict(test[feature_name])
    acc = accuracy_score(test['target'], y_predict)
    print(
        f"criterion=gini, max_depth=None, max_features=sqrt(n_features), estimoators=100, Acc: {acc}")

    # Question 5.2
    # max_features=sqrt(n_features)
    print("Question 5.2: Showing The Accuracy Score Of Test Data by max_features=sqrt(n_features) and max_features=n_features")
    clf_sqrt_features = RandomForest(n_estimators=10,
                                     max_features=np.sqrt(test[feature_name].shape[1]))
    clf_sqrt_features.build_Forest(train)
    y_predict = clf_sqrt_features.predict(test[feature_name])
    acc = accuracy_score(test['target'], y_predict)
    print(
        f"criterion=gini, max_depth=None, n_estimators=10, max_features=sqrt(n_features), Acc: {acc}")
    # max_features=n_features
    clf_max_features = RandomForest(n_estimators=10,
                                    max_features=test[feature_name].shape[1])
    clf_max_features.build_Forest(train)
    y_predict = clf_max_features.predict(test[feature_name])
    acc = accuracy_score(test['target'], y_predict)
    print(
        f"fcriterion=gini, max_depth=None, n_estimators=10, max_features=n_features, Acc: {acc}")
