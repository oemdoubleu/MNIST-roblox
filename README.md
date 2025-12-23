# MNIST-roblox
**MNIST-roblox** is a proof-of-concept implementation of a fully connected neural network running entirely within the Luau engine. It demonstrates how to bridge the gap between traditional Machine Learning workflows (Python/NumPy) and real-time Roblox experiences.

## Overview

The system is divided into two distinct pipelines: **Training** and **Inference**.

### 1. Training
Python for the compute-intensive task of training the network.
*   **Dataset:** MNIST
*   **Framework:** NumPy
*   **Topology:** A 2-layer Perceptron (MLP):
    *   **Input Layer:** 784 neurons (28x28 pixels).
    *   **Hidden Layer:** 128 neurons (ReLU activation).
    *   **Output Layer:** 10 neurons (Softmax activation).
*   **Output:** The trained weights (W1, W2) and biases (B1, B2) are serialized into a JSON format optimized for Lua tables.

### 2. Inference
Roblox implementation is a pure math engine that reconstructs the network using the pre-trained weights.
*   **Matrix Operations:** Since Luau lacks native matrix multiplication, we implement optimized vector-matrix dot products using 1D table representations.
*   **Activation Functions:** Custom implementations of **ReLU** (Rectified Linear Unit) for non-linearity and **Softmax** for probability distribution.
*   **Zero-Dependency:** The neural network module (`NeuralNet`) has no dependencies other than the weights file.

## Technical Deep Dive

### 1. Data Preprocessing & "Center of Mass" Logic
A raw drawing on a canvas is rarely compatible with a neural network trained on standardized data. MNIST images are centered by their **center of mass**, not their bounding box.

To achieve high accuracy the `CanvasController` performs complex preprocessing before the network ever sees the data:
1.  **Grid Extraction:** Reads the `EditableImage` buffer to extract a raw pixel grid.
2.  **Bounding Box Detection:** Scans the grid to find the tightest rectangle containing the drawing.
3.  **Mass Centering:** Calculates the center of mass $(C_x, C_y)$ based on pixel density:
    $$C_x = \frac{\sum x_i \cdot intensity_i}{\sum intensity_i}$$
4.  **Translation & Scaling:** The drawing is scaled to fit a 20x20 box within the 28x28 field, then shifted so that $(C_x, C_y)$ aligns with the center of the image.

### 2. The Forward Pass
The Lua `NeuralNet:Identify()` function performs a forward propagation pass. Mathematically, it executes:

1.  **Hidden Layer ($Z_1$):**
    $$Z_1 = W_1 \cdot X + B_1$$
    $$A_1 = \text{ReLU}(Z_1) = \max(0, Z_1)$$
    *This projects the 784 input pixels into 128 abstract features.*

2.  **Output Layer ($Z_2$):**
    $$Z_2 = W_2 \cdot A_1 + B_2$$
    *This maps the features to the 10 class scores.*

3.  **Probability Distribution (Softmax):**
    $$P_i = \frac{e^{Z_{2,i}}}{\sum_{j} e^{Z_{2,j}}}$$
    *This normalizes scores into percentages (0.0 to 1.0).*
