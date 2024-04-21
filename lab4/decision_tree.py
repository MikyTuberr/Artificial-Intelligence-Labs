from collections import defaultdict
from typing import Dict, Any, List

import numpy as np
from node import Node


class DecisionTree:
    def __init__(self, params: Dict[str, Any]):
        self.root_node: Node = Node()
        self.params = defaultdict(lambda: None, params)

    def train(self, X: np.ndarray, y: np.ndarray):
        self.root_node.train(X, y, self.params)

    def evaluate(self, X: np.ndarray, y: np.ndarray):
        predicted = self.predict(X)
        predicted = [round(p) for p in predicted]
        print(f"Accuracy: {round(np.mean(predicted == y), 2)}")

    def predict(self, X: np.ndarray) -> List[float]:
        prediction = []
        for x in X:
            prediction.append(self.root_node.predict(x))
        return prediction
