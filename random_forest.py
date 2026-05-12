import numpy as np
from decision_tree import DecisionTreeClassifier


class RandomForestClassifier:
    def __init__(self, n_estimators=100, max_depth=5, min_samples_split=2,
                 min_samples_leaf=1, random_state=None):
        self.n_estimators      = n_estimators
        self.max_depth         = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf  = min_samples_leaf
        self.random_state      = random_state
        self.trees             = []

    def _bootstrap(self, X, y, rng):
        n = X.shape[0]
        idx = rng.choice(n, size=n, replace=True)
        return X[idx], y[idx]

    def fit(self, X, y):
        rng = np.random.default_rng(self.random_state)
        self.trees = []
        for _ in range(self.n_estimators):
            X_boot, y_boot = self._bootstrap(X, y, rng)
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
            )
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)

    def predict(self, X):
        all_preds = np.array([tree.predict(X) for tree in self.trees])
        return np.array([
            np.bincount(all_preds[:, i]).argmax()
            for i in range(X.shape[0])
        ])