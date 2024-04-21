from typing import Tuple, List, Union
import copy
import numpy as np


class Node:
    left_child: Union["Node", None]
    right_child: Union["Node", None]
    feature_idx: Union[int, None]
    feature_value: Union[float, None]
    node_prediction: Union[float, None]

    def __init__(self) -> None:
        self.left_child = None
        self.right_child = None
        self.feature_idx = None
        self.feature_value = None
        self.node_prediction = None

    def gini_best_score(self, y: np.ndarray, possible_splits: List[int]) -> Tuple[int, float]:
        best_gain = -np.inf
        best_idx = 0

        total_samples = len(y)
        if total_samples == 0:
            return best_idx, best_gain

        total_pos = np.sum(y)
        total_neg = total_samples - total_pos

        for idx in possible_splits:
            left_pos = np.sum(y[:idx])
            left_neg = idx - left_pos
            right_pos = total_pos - left_pos
            right_neg = total_neg - left_neg

            if (left_pos + left_neg) == 0 or (right_pos + right_neg) == 0:
                continue

            left = left_pos + left_neg
            right = right_pos + right_neg
            total = left + right

            gini_left = 1 - ((left_pos / left) ** 2 + (left_neg / left) ** 2)
            gini_right = 1 - ((right_pos / right) ** 2 + (right_neg / right) ** 2)
            gini_gain = 1 - ((left / total) * gini_left + (right / total) * gini_right)

            if gini_gain > best_gain:
                best_gain = gini_gain
                best_idx = idx

        return best_idx, best_gain

    def split_data(self, X: np.ndarray, y: np.ndarray, idx: int, val: float) -> Tuple[
        Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        left_mask = X[:, idx] < val
        return (X[left_mask], y[left_mask]), (X[~left_mask], y[~left_mask])

    def find_possible_splits(self, data: np.ndarray) -> List[int]:
        possible_split_points = []
        for idx in range(data.shape[0] - 1):
            if data[idx] != data[idx + 1]:
                possible_split_points.append(idx)
        return possible_split_points

    def random_subset_selection(self, num_features: int, subset_size: int) -> List[int]:
        return np.random.choice(num_features, subset_size, replace=True).tolist()

    def find_best_split(self, X: np.ndarray, y: np.ndarray, feature_subset: Union[List[int], int, None]) -> Tuple[
        Union[int, None], Union[float, None]]:
        best_gain = -np.inf
        best_split = None

        if feature_subset is None:
            feature_subset = self.random_subset_selection(X.shape[1], X.shape[1])
        elif isinstance(feature_subset, int):
            feature_subset = self.random_subset_selection(X.shape[1], feature_subset)

        for feature_idx in feature_subset:
            order = np.argsort(X[:, feature_idx])
            y_sorted = y[order]
            possible_splits = self.find_possible_splits(X[order, feature_idx])
            idx, value = self.gini_best_score(y_sorted, possible_splits)
            if value > best_gain:
                best_gain = value
                best_split = (feature_idx, [idx, idx + 1])

        if best_split is None:
            return None, None

        best_value = np.mean(X[best_split[1], best_split[0]])

        return best_split[0], best_value

    def predict(self, x: np.ndarray) -> Union[float, None]:
        if self.feature_idx is None:
            return self.node_prediction
        if x[self.feature_idx] < self.feature_value:
            return self.left_child.predict(x)
        else:
            return self.right_child.predict(x)

    def train(self, X: np.ndarray, y: np.ndarray, params: dict) -> bool:
        self.node_prediction = np.mean(y)
        if X.shape[0] == 1 or self.node_prediction == 0 or self.node_prediction == 1:
            return True

        self.feature_idx, self.feature_value = self.find_best_split(X, y, params["feature_subset"])
        if self.feature_idx is None:
            return True

        (X_left, y_left), (X_right, y_right) = self.split_data(X, y, self.feature_idx, self.feature_value)

        if X_left.shape[0] == 0 or X_right.shape[0] == 0:
            self.feature_idx = None
            return True

        # max tree depth
        if params["depth"] is not None:
            params["depth"] -= 1
        if params["depth"] == 0:
            self.feature_idx = None
            return True

        # create new nodes
        self.left_child, self.right_child = Node(), Node()
        self.left_child.train(X_left, y_left, copy.deepcopy(params))
        self.right_child.train(X_right, y_right, copy.deepcopy(params))
