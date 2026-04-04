#!/usr/bin/env python3

import numpy as np

X_full = [
    [0.2, 1.1],
    [0.3, 1.0],
    [0.4, 0.9],
    [0.5, 1.2],
    [0.6, 1.0],
    [1.2, 0.1],
    [1.0, 0.2],
    [1.3, 0.3],
    [1.5, 0.4],
    [1.4, 0.2],
    [2.0, 1.5],
    [2.2, 1.7],
    [2.1, 1.6],
    [1.9, 1.4],
    [2.3, 1.8],
]

Y_full = [
    [0],
    [0],
    [0],
    [0],
    [0],
    [1],
    [1],
    [1],
    [1],
    [1],
    [1],
    [1],
    [1],
    [1],
    [1],
]

X_test = [
    [0.25, 1.05],
    [0.35, 0.95],
    [0.45, 1.10],
    [0.55, 0.85],
    [0.65, 1.15],
    [1.10, 0.15],
    [1.25, 0.25],
    [1.35, 0.35],
    [1.45, 0.30],
    [1.55, 0.45],
    [1.95, 1.45],
    [2.05, 1.55],
    [2.15, 1.65],
    [2.25, 1.75],
    [2.35, 1.85],
]

Y_test = [
    [0],
    [0],
    [0],
    [0],
    [0],
    [1],
    [1],
    [1],
    [1],
    [1],
    [1],
    [1],
    [1],
    [1],
    [1],
]

X_train = np.asarray(X_full, dtype=np.float32)
Y_train = np.asarray(Y_full, dtype=np.float32)
X_valid = np.asarray(X_full[0:5], dtype=np.float32)
Y_valid = np.asarray(Y_full[0:5], dtype=np.float32)
X_test = np.asarray(X_test, dtype=np.float32)
Y_test = np.asarray(Y_test, dtype=np.float32)

np.savez(
    "dataset.npz",
    X_train=X_train,
    Y_train=Y_train,
    X_valid=X_valid,
    Y_valid=Y_valid,
    X_test=X_test,
    Y_test=Y_test,
)

# NeuralNetwork.save_to_csv(X_train, Y_train, "train.csv")
# NeuralNetwork.save_to_csv(X_valid, Y_valid, "valid.csv")
# NeuralNetwork.save_to_csv(X_test, Y_test, "test.csv")

# X_train, Y_train = NeuralNetwork.load_from_csv("data/csv/train.csv")

# dataset = NeuralNetwork.load_from_npz("data/npz/MNIST.npz")
# NeuralNetwork.save_to_json(dataset, "mnist.json")

# dataset = load_npz_mnist("data/MNIST.npz")
# save_dataset_to_pickle(dataset, "data_pickle/mnist_dataset.pkl")

# Y_train_one_hot = NeuralNetwork.one_hot_encode(Y_train, 10)
# Y_valid_one_hot = NeuralNetwork.one_hot_encode(Y_valid, 10)
# Y_test_one_hot = NeuralNetwork.one_hot_encode(Y_test, 10)


# X_train_small = X_train[:limit]
# Y_train_small = Y_train[:limit]

# perm = model.xp.random.permutation(X_train.shape[0])[:limit]
# X_train_small = X_train[perm]
# Y_train_small = Y_train[perm]

# pred_probs, _ = model.predict(X_train_small)

# xp = model.xp
# true_classes = xp.argmax(Y_train_small, axis=1)
# pred_classes = xp.argmax(pred_probs, axis=1)

# print("True label counts:", model.tp_cpu(xp.bincount(true_classes, minlength=10)))
# print("Pred label counts:", model.tp_cpu(xp.bincount(pred_classes, minlength=10)))

# X_test = loaded_model.to_device(X_test, dtype=loaded_model.xp.float32)
# Y_test = loaded_model.to_device(Y_test, dtype=loaded_model.xp.float32)

# if loaded_model is not None:
#     results = loaded_model.evaluate_dataset(X_test, Y_test)
#     print(results[1])
