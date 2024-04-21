from collections import defaultdict
from typing import List, Tuple, Dict, Any

import numpy as np
from decision_tree import DecisionTree


class RandomForest:
    def __init__(self, params: Dict[str, Any]):
        self.forest: List[DecisionTree] = []
        self.params = defaultdict(lambda: None, params)

    def train(self, X: np.ndarray, y: np.ndarray):
        for _ in range(self.params["ntrees"]):
            X_bagging, y_bagging = self.bagging(X, y)
            tree = DecisionTree(self.params)
            tree.train(X_bagging, y_bagging)
            self.forest.append(tree)

    def evaluate(self, X: np.ndarray, y: np.ndarray):
        predicted = self.predict(X)
        predicted = [round(p) for p in predicted]
        print(f"Accuracy: {round(np.mean(predicted == y), 2)}")

    def predict(self, X: np.ndarray) -> List[float]:
        tree_predictions = []
        for tree in self.forest:
            tree_predictions.append(tree.predict(X))
        forest_predictions = list(map(lambda x: sum(x) / len(x), zip(*tree_predictions)))
        return forest_predictions

    def bagging(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n_samples = X.shape[0]

        bootstrap_indices = np.random.choice(np.arange(n_samples), size=n_samples, replace=True)
        X_selected, y_selected = X[bootstrap_indices], y[bootstrap_indices]

        return X_selected, y_selected
