# GhostFaces 👻

An algebraic facial recognition system that extracts eigenfaces using **Thin Singular Value Decomposition (SVD)** and classifies subjects using a **Support Vector Machine (SVM)**.

---

## 🛠 Prerequisites

To run this project, ensure your environment meets the following requirements:

### 1. Environment
* **Python 3.9+**

### 2. Required Libraries
Install the core scientific computing and computer vision dependencies:

```bash
pip install numpy scipy opencv-python scikit-learn matplotlib
```

* `numpy`: Handles vectorization, matrix manipulations, and numerical array operations.
* `scipy`: Provides optimized linear algebra routines (`scipy.linalg.svd`) for Thin SVD decomposition.
* `opencv-python` (`cv2`): Loads images from disk in grayscale format and resizes them to uniform dimensions.
* `scikit-learn`: Handles dataset splitting (`train_test_split`), classification (`SVC`), and model evaluation metrics.
* `matplotlib`: Renders and plots the average face and principal eigenfaces.

### 3. Data Requirement
* A folder named `grayfaces` in the project root containing grayscale images with standardized dimensions ($112 \times 92$).
* Naming convention: `<gender>.<person_name>.<image_number>.jpg`

---

## 📖 Detailed Mathematical & Algorithmic Explanation

The `GhostFaces` pipeline relies on linear subspace projection:

### 1. Vectorization
Each $112 \times 92$ 2D grayscale face image is flattened into a 1D vector of length $N^2 = 10{,}304$. An image $\Gamma_i$ is treated as a single point in $\mathbb{R}^{10304}$.

### 2. Mean Centering
To focus purely on variance rather than baseline illumination, the global average face $\Psi$ across all $M$ training images is calculated:

$$\Psi = \frac{1}{M} \sum_{i=1}^M \Gamma_i$$

Each training image is centered by subtracting the mean:

$$\Phi_i = \Gamma_i - \Psi$$

These centered vectors form the columns of matrix $A \in \mathbb{R}^{N^2 \times M}$. Test faces are also centered using this exact training mean $\Psi$ to prevent data leakage.

### 3. Thin Singular Value Decomposition (SVD)
Instead of constructing a computationally prohibitive covariance matrix $C = \frac{1}{M} A A^T$ (size $10{,}304 \times 10{,}304$), Thin SVD is applied directly to $A$:

$$A = U \Sigma V^T$$

* **$U \in \mathbb{R}^{N^2 \times M}$**: The left singular vectors. The columns of $U$ are the orthonormal eigenvectors of $A A^T$, representing the **Eigenfaces**.
* **$\Sigma \in \mathbb{R}^{M \times M}$**: Diagonal matrix containing singular values $\sigma_i$, which represent the standard deviation captured along each principal component.
* **$V^T \in \mathbb{R}^{M \times M}$**: Right singular vectors.

The top $M' = 20$ columns of $U$ capture over 90% of the dataset's total variance while discarding high-frequency noise.

### 4. Subspace Projection
Each centered image vector $\Phi$ is projected onto the low-dimensional eigenface subspace spanned by $U_{M'}$:

$$\Omega = \Phi^T U_{M'}$$

This transforms every face from a $10{,}304$-dimensional pixel space down to a compact $20$-dimensional coordinate weight vector $\Omega \in \mathbb{R}^{20}$.

### 5. SVM Classification
A Support Vector Classifier (SVC) with a linear kernel is trained directly on these 20-dimensional feature weights:
* The SVM determines optimal maximum-margin hyperplanes separating the identity classes in the reduced subspace.
* During inference, unknown faces are flattened, centered with $\Psi$, projected onto $U_{M'}$, and classified by the SVM.
