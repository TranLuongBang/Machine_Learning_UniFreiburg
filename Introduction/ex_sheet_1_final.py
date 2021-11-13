import time

import numpy as np
import openml

def euclidean_distance(
        example_1: np.ndarray,
        example_2: np.ndarray,
) -> np.float:

    sum_squared_distance = 0
    nr_features = example_1.shape[0]
    for feature_index in range(0, nr_features):
        sum_squared_distance += np.power(example_1[feature_index] - example_2[feature_index], 2)

    return np.sqrt(sum_squared_distance)


def knn(
    query_example: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    k: int = 5,
) -> np.int:

    euc_distances = [] # array containing euclidean distances
    for ind in range(X_train.shape[0]):
        euc_distances.append(euclidean_distance(X_train[ind], query_example))
    euc_y = np.column_stack((euc_distances, y_train))
    # partial_ordered array, get k first elements which are smaller than the rest
    k_smallest = np.argpartition(euc_y[:, 0], k - 1)[:k]

    counts = np.bincount(y_train[k_smallest])

    return np.argmax(counts)

# Exercise 1
# For exercise 1, we are going to use the diabetes dataset
task = openml.tasks.get_task(267)
train_indices, test_indices = task.get_train_test_split_indices()
dataset = task.get_dataset()
X, y, categorical_indicator, _ = dataset.get_data(
    dataset_format='array',
    target=dataset.default_target_attribute,
)

X_train = X[train_indices]
y_train = y[train_indices] 
X_test = X[test_indices]
y_test = y[test_indices]

NR_NEIGHBORS = 5

start = time.time()

nr_correct_pred = 0
for example, label in zip(X_test, y_test):
    y_pred = knn(example, X_train, y_train, k=NR_NEIGHBORS)
    if y_pred == label:
        nr_correct_pred += 1

end = time.time()
accuracy = nr_correct_pred / X_test.shape[0]
print(f'The accuracy of the k-nearest neighbor algorithm with k = {NR_NEIGHBORS} is {accuracy}')
print(end-start)