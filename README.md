<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GhostFaces: Facial Recognition Project Report</title>
    <style>
        body {
            font-family: "Georgia", serif;
            line-height: 1.6;
            color: #333;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            background-color: #fcfcfc;
        }
        h1, h2, h3, h4 {
            font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
            color: #1a467a;
        }
        h1 {
            text-align: center;
            margin-bottom: 5px;
            font-size: 2.2em;
        }
        .authors {
            text-align: center;
            font-size: 1.1em;
            margin-bottom: 10px;
            font-weight: bold;
        }
        .repo-link {
            text-align: center;
            margin-bottom: 30px;
        }
        .repo-link a {
            color: #1a467a;
            text-decoration: none;
            font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
        }
        .repo-link a:hover {
            text-decoration: underline;
        }
        .abstract {
            background-color: #f4f7f6;
            padding: 20px;
            border-left: 4px solid #1a467a;
            margin-bottom: 30px;
            font-style: italic;
        }
        .callout {
            background-color: #e9f2fa;
            border: 1px solid #1a467a;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
        }
        .callout h4 {
            margin-top: 0;
            color: #1a467a;
        }
        pre {
            background-color: #f8f8f8;
            border: 1px solid #ddd;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
            font-family: "Courier New", Courier, monospace;
            font-size: 0.9em;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        table, th, td {
            border: 1px solid #ccc;
        }
        th, td {
            padding: 10px;
            text-align: center;
        }
        th {
            background-color: #f0f0f0;
        }
    </style>
</head>
<body>

    <h1>GhostFaces: A Linear Subspace Approach to Facial Recognition via Thin SVD and Polynomial Support Vector Machines</h1>
    
    <div class="authors">
        Neerav Bhuyan &amp; Saptarshi Datta<br>
        <span style="font-weight: normal; font-size: 0.9em;">Department of Data Science and Machine Learning</span>
    </div>
    
    <div class="repo-link">
        Project Repository: <a href="https://github.com/neeravbhuyan/ghostfaces" target="_blank">https://github.com/neeravbhuyan/ghostfaces</a>[cite: 9]
    </div>

    <div class="abstract">
        <strong>Abstract:</strong> This report outlines the implementation and analysis of <strong>GhostFaces</strong>, a practical facial recognition pipeline built using linear algebra and machine learning. Raw face images of 112 &times; 92 pixels naturally contain 10,304 pieces of information per photo[cite: 4, 9]. To compress this massive amount of data efficiently, we use a mathematical shortcut called Thin Singular Value Decomposition (Thin SVD) directly on the images[cite: 5, 9]. This extracts the core visual patterns, known as Eigenfaces, saving computer memory and significantly reducing execution time. Using our unconstrained dataset, we filtered out subjects with fewer than 5 images, resulting in 3,332 processed samples across 355 unique identities[cite: 6, 9]. Finally, we balanced the features using a Standard Scaler and deployed a Polynomial Support Vector Machine (SVM) to classify the faces, achieving a final accuracy of 28.04%[cite: 6, 9]. We conclude by reviewing the structural limits of this method under real-world pose, expression, and lighting changes.
    </div>

    <h2>1. Introduction</h2>
    <p>Face recognition from raw pixels presents a classic machine learning challenge known as the <em>curse of dimensionality</em>[cite: 4]. When a standard portrait is captured at a resolution of 112 &times; 92 pixels, its flattened vector representation becomes a long list of 10,304 numbers[cite: 4, 9]. Operating directly within this large space is computationally expensive and highly sensitive to noise. Minor changes in room lighting or head posture often create larger mathematical differences between two photos of the same person than photos of two entirely different people.</p>

    <p>The classical countermeasure is the <strong>Eigenface method</strong>. It uses Principal Component Analysis (PCA) to discover a smaller, simpler coordinate system that captures the most significant visual differences across the dataset[cite: 4]. The core visual patterns spanning this new space act as fundamental facial templates (Eigenfaces)[cite: 4]. This allows standard classifiers, such as Support Vector Machines (SVM), to operate on a highly compressed and efficient set of features[cite: 4].</p>

    <h2>2. Data Collection and Preprocessing Pipeline</h2>
    
    <h3>2.1 In-Memory Streaming Architecture</h3>
    <p>To eliminate local storage requirements and ensure the pipeline can be run easily on any machine, data ingestion streams the dataset directly from the project's GitHub repository[cite: 3, 9]. Utilizing Python's <code>requests</code> and <code>io.BytesIO</code> libraries, the archive is loaded into RAM, and image byte streams are extracted and decoded via OpenCV (<code>cv2.imdecode</code>)[cite: 9, 12].</p>

    <h3>2.2 Grayscale Conversion vs. Pseudo-Infrared Mapping</h3>
    <p>An earlier iteration of this project experimented with pseudo-infrared thermal mapping, which applied artificial colors to the images before converting them back to grayscale[cite: 12, 15]. The current pipeline safely decodes the image as standard color and directly converts it to single-channel grayscale (<code>cv2.COLOR_BGR2GRAY</code>)[cite: 9]. This perfectly preserves the natural lighting and structural shadows required for the algorithm to work properly[cite: 9].</p>

    <h3>2.3 The Singleton Penalty and Class Filtering</h3>
    <p>The raw dataset contains 1,434 distinct identities. However, attempting to perform an 80/20 train-test split fails when classes have only one photo[cite: 6]. If a person with only one photo is placed in the test set, the SVM will be forced to guess their identity without ever having seen them during training—a guaranteed failure[cite: 6].</p>

    <p>To fix this, the pipeline automatically filters the dataset before splitting, keeping only identities that have at least 5 images (<code>MIN_SAMPLES = 5</code>)[cite: 6, 9]. This filtering process reduced the dataset from 6,053 raw images to 3,332 processed samples[cite: 6, 9]. It also reduced the total unique identities from 1,434 down to 355[cite: 6, 9]. This ensures every person tested has a corresponding baseline image in the training data, enabling a mathematically valid train-test split[cite: 6, 9].</p>

    <h2>3. Mathematical Subspace Model</h2>

    <h3>3.1 Image Vectorization and Centering</h3>
    <p>Let a preprocessed grayscale image be flattened into a 1D column vector of length 10,304[cite: 4, 9]. Given the training images, the global mean (average) face is calculated[cite: 4]. To remove basic background lighting bias, each image is centered by subtracting this average[cite: 4].</p>

    <h3>3.2 Thin Singular Value Decomposition (SVD)</h3>
    <p>The classical PCA method requires constructing a massive covariance matrix, calculating its eigenvectors, and multiplying them back against the images[cite: 5]. To make this significantly faster, our pipeline applies a direct mathematical shortcut called <strong>Thin SVD</strong> directly on the transposed data matrix[cite: 5, 9].</p>

    <div class="callout">
        <h4>Computational Efficiency</h4>
        <p>By using direct Thin SVD, we extract the core facial patterns (the Eigenfaces) in a single step[cite: 5, 9]. This avoids calculating massive intermediate matrices, cutting processing time in half and preventing computer rounding errors[cite: 5].</p>
    </div>

    <h3>3.3 Shrinking the Data</h3>
    <p>We select the top Eigenfaces that retain 90% of the variance (the most important visual differences)[cite: 5, 9]. In our final model, this resulted in retaining exactly 93 Eigenfaces[cite: 6, 9].</p>

    <p>Every mean-centered image is mapped into a small 93-dimensional list of coordinates by comparing it against this basis[cite: 5, 9]. Instead of processing 10,304 pixels per face, the classifier only needs to evaluate 93 numbers[cite: 5, 9].</p>

    <h2>4. Implementation Architecture in Python</h2>
    <p>The entire pipeline is self-contained and implemented using standard data science libraries (NumPy, SciPy, scikit-learn)[cite: 8, 9].</p>

<pre><code class="language-python"># Step 1: In-Memory Data Streaming and Grayscale Conversion
import io, zipfile, cv2, requests
import numpy as np
from scipy.linalg import svd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Download repository into RAM
REPO_ZIP_URL = "https://github.com/neeravbhuyan/ghostfaces/archive/refs/heads/main.zip"
response = requests.get(REPO_ZIP_URL)

IMG_HEIGHT, IMG_WIDTH = 112, 92
image_vectors, labels = [], []

with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
    for file_path in archive.namelist():
        if "faceset/" in file_path and file_path.lower().endswith(('.jpg', '.jpeg', '.png')):
            parts = file_path.split('/')
            if len(parts) >= 3 and parts[-2] != "faceset":
                person_name = parts[-2]
                
                img_bytes = archive.read(file_path)
                img_array = np.frombuffer(img_bytes, dtype=np.uint8)
                img_color = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                
                if img_color is not None:
                    # Convert directly to Grayscale
                    img_gray = cv2.cvtColor(cv2.resize(img_color, (IMG_WIDTH, IMG_HEIGHT)), cv2.COLOR_BGR2GRAY)
                    image_vectors.append(img_gray.flatten())
                    labels.append(person_name)

X_raw = np.array(image_vectors, dtype=np.float32)
y_raw = np.array(labels)

# Filter out subjects with fewer than 5 photos
MIN_SAMPLES = 5
unique_labels, counts = np.unique(y_raw, return_counts=True)
valid_labels = unique_labels[counts >= MIN_SAMPLES]
valid_mask = np.isin(y_raw, valid_labels)

X = X_raw[valid_mask]
y = y_raw[valid_mask]

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Step 2: Global Mean Centering
global_mean_face = np.mean(X_train, axis=0)
X_train_centered = X_train - global_mean_face
X_test_centered = X_test - global_mean_face

# Step 3: Thin SVD
A = X_train_centered.T  
U, S, Vt = svd(A, full_matrices=False)

# Retain 90% variance
explained_variance_ratio = (S ** 2) / np.sum(S ** 2)
cumulative_variance = np.cumsum(explained_variance_ratio)
M_prime = np.searchsorted(cumulative_variance, 0.90) + 1
eigenfaces = U[:, :M_prime]

# Step 4: Map images into the 93-dimensional weight subspace
W_train = X_train_centered @ eigenfaces
W_test = X_test_centered @ eigenfaces

# Balance the features
scaler = StandardScaler()
omega_train = scaler.fit_transform(W_train)
omega_test = scaler.transform(W_test)

# Step 5: Polynomial Support Vector Machine (SVC) Classification
svm_classifier = SVC(kernel="poly", degree=2, coef0=1.0, random_state=42)
svm_classifier.fit(omega_train, y_train)

# Evaluate
y_pred = svm_classifier.predict(omega_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"SVM Classification Accuracy: {accuracy * 100:.2f}%\n")
</code></pre>

    <h2>5. Final Results</h2>
    <p>By cleaning the data and using the 93 balanced features, our Support Vector Machine successfully mapped the identities[cite: 9]. We utilized a <strong>Polynomial Kernel</strong> (degree 2)[cite: 9]. While linear classifiers try to draw flat, rigid lines to separate people, the polynomial kernel is able to draw slightly curved boundaries, helping the model understand complex interactions between facial features.</p>

    <p>The final GhostFaces pipeline achieved an accuracy of <strong>28.04%</strong> on the testing set across 355 unique identities[cite: 6, 9].</p>

    <table>
        <thead>
            <tr>
                <th>Metric Type</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>F1-Score</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>Macro Average</td>
                <td>0.1753</td>
                <td>0.1822</td>
                <td>0.1687</td>
            </tr>
            <tr>
                <td>Weighted Average</td>
                <td>0.2382</td>
                <td>0.2804</td>
                <td>0.2395</td>
            </tr>
        </tbody>
    </table>

    <div class="callout">
        <h4>Understanding the Accuracy Metric</h4>
        <p>While 28% might sound low, the model is choosing from 355 different people[cite: 9]. Guessing randomly would yield an accuracy of just 0.28%. By shrinking 10,304 pixels into just 93 numbers, our math-based model is predicting correctly at <strong>100 times better than random guessing</strong>[cite: 9].</p>
    </div>

    <h2>6. Drawbacks and Limitations of Eigenfaces</h2>
    <p>The 28.04% performance ceiling reveals fundamental constraints of the traditional Eigenface method on real-world datasets[cite: 7, 9]:</p>
    <ul>
        <li><strong>Sensitivity to Alignment:</strong> Eigenfaces assume strict pixel alignment. Slight head tilts shift features by several pixels. SVD processes these spatial misalignments as high-frequency image noise, wasting components on alignment differences rather than identity[cite: 7, 9].</li>
        <li><strong>Lighting vs. Identity:</strong> PCA captures overall variance. In unconstrained photography, sunlight or flash direction often causes more variation than actual facial shapes, leading top Eigenfaces to capture lighting artifacts rather than structural identity[cite: 7, 9].</li>
        <li><strong>Linear Boundaries:</strong> Facial representations change in curved, non-linear ways as heads rotate. Linear PCA representations overlap and struggle with clean separation[cite: 7, 9].</li>
    </ul>

    <h2>7. Conclusion</h2>
    <p>This project successfully implemented an in-memory automated facial recognition pipeline. We verified that calculating patterns directly via Thin SVD operates faster and avoids memory issues compared to older covariance methods[cite: 5].</p>

    <p>While the pipeline successfully reduced dimensionality from 10,304 pixels to just 93 weights[cite: 5], it exposed the challenges of simple mathematical models in unconstrained datasets. For future improvements, moving toward non-linear solutions like Fisherfaces (Linear Discriminant Analysis) or Deep Convolutional Neural Networks (CNNs) is recommended to push accuracy beyond this ceiling[cite: 7].</p>

</body>
</html>
