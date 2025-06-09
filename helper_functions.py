import numpy as np
from collections import Counter



def create_segment_label(value, N):
    ones = np.ones(N)
    labels = ones*value
    return labels

def generate_datasets(X_train_C, X_test_C, y_train_C, y_test_C):
  y_train = []
  y_test = []
  for idx, y in enumerate(y_train_C):
    N = X_train_C[idx].shape[0]
    y_train.append(create_segment_label(y, N))

  for idx, y in enumerate(y_test_C):
    N = X_test_C[idx].shape[0]
    y_test.append(create_segment_label(y, N))

  y_train = np.concatenate(y_train, axis=0)
  y_test = np.concatenate(y_test, axis=0)
  #print(y_train.shape, y_test.shape)

  X_train = np.vstack(X_train_C)
  X_test = np.vstack(X_test_C)
  #print(X_train.shape, X_test.shape)

  return X_train, X_test, y_train, y_test

def count_labels(y_tensor):
  # Cuenta cuántas veces aparece cada etiqueta
  labels = y_tensor.tolist()
  label_counts = Counter(labels)
  print(label_counts)
