import numpy as np
import json
import gzip
import os
import time

FILE_IMAGES = "train-images-idx3-ubyte.gz"
FILE_LABELS = "train-labels-idx1-ubyte.gz"
HIDDEN_SIZE = 128 

def load_mnist_gz():
    if not os.path.exists(FILE_IMAGES):
        print("Data files not found!")
        exit()
    with gzip.open(FILE_IMAGES, 'rb') as f:
        f.read(16)
        images = np.frombuffer(f.read(), dtype=np.uint8).astype(np.float32) / 255.0
        images = images.reshape(-1, 784)
    with gzip.open(FILE_LABELS, 'rb') as f:
        f.read(8)
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return images, labels

def relu(z): return np.maximum(0, z)
def softmax(z):
    e = np.exp(z - np.max(z, axis=1, keepdims=True))
    return e / np.sum(e, axis=1, keepdims=True)

X, y = load_mnist_gz()
y_onehot = np.zeros((y.size, 10))
y_onehot[np.arange(y.size), y] = 1

np.random.seed(42)
W1 = np.random.randn(784, HIDDEN_SIZE) * np.sqrt(2/784)
B1 = np.zeros(HIDDEN_SIZE)
W2 = np.random.randn(HIDDEN_SIZE, 10) * np.sqrt(2/HIDDEN_SIZE)
B2 = np.zeros(10)

lr = 0.05
epochs = 5
batch_size = 64

for epoch in range(epochs):
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X = X[indices]
    y_onehot = y_onehot[indices]
    y = y[indices]

    for i in range(0, len(X), batch_size):
        X_b = X[i:i+batch_size]
        y_b = y_onehot[i:i+batch_size]

        Z1 = np.dot(X_b, W1) + B1
        A1 = relu(Z1)
        Z2 = np.dot(A1, W2) + B2
        probs = softmax(Z2)

        dZ2 = probs - y_b
        dW2 = np.dot(A1.T, dZ2) / batch_size
        dB2 = np.sum(dZ2, axis=0) / batch_size
        
        dZ1 = np.dot(dZ2, W2.T) * (Z1 > 0)
        dW1 = np.dot(X_b.T, dZ1) / batch_size
        dB1 = np.sum(dZ1, axis=0) / batch_size

        W1 -= lr * dW1; B1 -= lr * dB1
        W2 -= lr * dW2; B2 -= lr * dB2

    Z1 = np.dot(X, W1) + B1
    A1 = relu(Z1)
    scores = np.dot(A1, W2) + B2
    preds = np.argmax(scores, axis=1)
    acc = np.mean(preds == y)
    print(f"Epoch {epoch+1}: {acc*100:.2f}%")

output = {
    "W1": W1.flatten().tolist(), "B1": B1.tolist(),
    "W2": W2.flatten().tolist(), "B2": B2.tolist()
}
with open("Weights.json", "w") as f: json.dump(output, f)
