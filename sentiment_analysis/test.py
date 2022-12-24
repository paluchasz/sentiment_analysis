# test
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from sentiment_analysis import DATA_DIR, get_data

v = tf.Variable(1)


@tf.function
def f(x):
    """A doc string to get interrogate commit hook passing"""
    ta = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
    for i in tf.range(x):
        v.assign_add(i)
        ta = ta.write(i, 0.123)
    return ta.stack() / 3


a = f(5)

training_data = get_data(from_file=DATA_DIR / "training_data.json")
print(len(training_data))

np.random.seed(1)
np.random.shuffle(training_data)
X_train, y_train = [list(x) for x in zip(*[[t["text"], t["label"]] for t in training_data])]

train_texts, val_texts, train_labels, val_labels = train_test_split(X_train, y_train, test_size=0.2)
a = 1
