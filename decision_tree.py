import numpy as np

class DecisionTreeClassifier:
    def __init__(self, max_depth=5, min_samples_split=2, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

    def fit(self, X, y):
        self.tree = self._grow_tree(X, y, depth=0)
    
    def _gini(self, y):
        classes = np.unique(y)
        impurity = 1.0
        for c in classes:
            p = np.sum(y == c) / len(y)
            impurity -= p ** 2
        return impurity
    
    def _best_split(self, X, y):
        best_gini = float("inf")
        best_idx, best_threshold = None, None

        n_samples, n_features = X.shape

        for idx in range(n_features):
            thresholds = np.unique(X[:, idx])

            for t in thresholds:
                left_mask = X[:, idx] <= t
                right_mask = X[:, idx] > t

                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                left_y = y[left_mask]
                right_y = y[right_mask]

                gini = (len(left_y)/len(y))*self._gini(left_y) + \
                          (len(right_y)/len(y))*self._gini(right_y)
                
                if gini < best_gini:
                    best_gini = gini
                    best_idx = idx
                    best_thresh = t

        return best_idx, best_thresh
    
    def _grow_tree(self, X, y, depth):
        num_samples = len(y)
        num_classes = len(set(y))

        # dieu kien dung
        if (
            depth >= self.max_depth or 
            num_samples < self.min_samples_split or 
            num_classes == 1 or 
            num_samples < 2 * self.min_samples_leaf 
        ):
            return self._majority_class(y)
        
        # tìm split tốt nhất
        idx, thresh = self._best_split(X, y)

        # nếu không split được
        if idx is None:
            return self._majority_class(y)

        # chia dữ liệu
        left_mask = X[:, idx] <= thresh
        right_mask = X[:, idx] > thresh

        left_subtree = self._grow_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self._grow_tree(X[right_mask], y[right_mask], depth + 1)

        return (idx, thresh, left_subtree, right_subtree)

    def _majority_class(self, y):
        return np.bincount(y).argmax()

    def _predict_one(self, x, node):
        if not isinstance(node, tuple):
            return node

        idx, thresh, left, right = node

        if x[idx] <= thresh:
            return self._predict_one(x, left)
        else:
            return self._predict_one(x, right)

    def predict(self, X):
        return np.array([self._predict_one(x, self.tree) for x in X])
        


