## GhostFaces 👻
GhostFaces is a large-scale facial recognition pipeline designed to address the curse of dimensionality in raw image processing. By utilizing Principal Component Analysis (PCA) via Thin Singular Value Decomposition (SVD), the system compresses high-dimensional faces into compact feature vectors and classifies them using a Support Vector Machine (SVM).   

## 📁 Repository Structure
1) ToyModel.ipynb: The core Jupyter Notebook containing data ingestion, Thin SVD mathematical extraction, and SVM classification.
2) main.ipynb: The updated Jupyter Notebook in which the model and pipeline has been modified to handle thousands of messy, real world data.
3) grayfaces/: The directory storing the labelled images (already in grayface) used to train and test the protype model.
4) faceset/: The directory storing unconstrained facial images organized by individual identities (e.g., Aaron_Peirsol, Abdullah_Gul, Angelina_Jolie).   

## ⚙️ Methodology & Pipeline
1) In-Memory Streaming: Images are streamed directly from GitHub into RAM using requests and zipfile to avoid large local downloads.   
2) Grayscale Standardization: Images are converted to standard grayscale, preserving natural facial contours and avoiding destructive pseudo-infrared mappings.
3) Threshold Filtering: To prevent open-set classification failures, the dataset is filtered to retain only individuals with at least 5 images. This reduces the full dataset to 3,332 images across 355 unique people.   
4) Mean Centering: The global mean face is calculated and subtracted from every image to ensure the model targets distinct facial variations rather than shared background lighting.   
5) Thin SVD Dimensionality Reduction: Thin SVD ($A = U \Sigma V^T$) is applied directly to the centered data matrix to avoid computing a massive covariance matrix. The model retains as many components needed to capture 90% of the dataset's total variance.   
6) Classification: Coordinates are standardized and passed to a Polynomial SVM (degree 2, coef0=1.0) to classify identities.   

## 📊 Performance Results
1) Baseline Model: A controlled test on 50 images across 10 identities retained 7 components and achieved 100% accuracy using a linear SVM.   
2) Full Dataset: The final Polynomial SVM model evaluated on 355 classes achieved 28.04% accuracy.   

## ⚠️ Known Limitations
The 28.04% performance ceiling reveals fundamental constraints of the traditional Eigenface method on real-world, unconstrained datasets:   
1) Sensitivity to Alignment: PCA assumes perfect pixel alignment; slight head tilts shift pixels, which the algorithm processes as high-frequency noise rather than translated facial features.
2) Lighting vs. Identity: The top Eigenfaces capture variations in lighting (e.g., directional flash) rather than structural identity differences, heavily confounding the algorithm.
3) Linear Boundaries: Facial representations change in curved, non-linear ways as heads rotate, causing linear PCA representations to overlap and struggle with clean separation.

## 🛠 Prerequisites
The pipeline is built in Python and relies on the following core scientific libraries: numpy, scipy, cv2 (opencv-python), scikit-learn, and matplotlib. 
