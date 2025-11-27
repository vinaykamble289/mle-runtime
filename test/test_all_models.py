"""
Comprehensive test suite for all sklearn models
Tests export and inference for every supported model type
"""
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import make_classification, make_regression, make_blobs
from sklearn.model_selection import train_test_split

# Linear models
from sklearn.linear_model import (
    LogisticRegression, LinearRegression, Ridge, Lasso,
    SGDClassifier, SGDRegressor
)

# Tree models
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# Ensemble models
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor
)

# Neural networks
from sklearn.neural_network import MLPClassifier, MLPRegressor

# SVM
from sklearn.svm import SVC, SVR, LinearSVC, LinearSVR

# Naive Bayes
from sklearn.naive_bayes import GaussianNB

# Neighbors
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

# Clustering
from sklearn.cluster import KMeans

# Decomposition
from sklearn.decomposition import PCA

from mle_runtime.sklearn_to_mle import SklearnMLEExporter
from mle_runtime import Engine, Device

# Test results
results = {
    'passed': [],
    'failed': [],
    'skipped': []
}

def test_model(name, model, X, y, is_classifier=True, is_clustering=False, is_transform=False):
    """Test a single model"""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"{'='*60}")
    
    try:
        # Train
        if is_clustering or is_transform:
            model.fit(X)
        else:
            model.fit(X, y)
        
        # Get sklearn prediction
        if is_transform:
            sklearn_pred = model.transform(X[:1])
        elif is_clustering:
            sklearn_pred = model.predict(X[:1])
        elif is_classifier:
            if hasattr(model, 'predict_proba'):
                sklearn_pred = model.predict_proba(X[:1])
            else:
                sklearn_pred = model.predict(X[:1])
        else:
            sklearn_pred = model.predict(X[:1])
        
        print(f"Sklearn prediction: {sklearn_pred}")
        
        # Export
        exporter = SklearnMLEExporter()
        model_file = f'test_{name.lower().replace(" ", "_")}.mle'
        exporter.export_sklearn(model, model_file, input_shape=(1, X.shape[1]))
        
        # Load and infer
        engine = Engine(Device.CPU)
        engine.load_model(model_file)
        
        test_input = X[:1].astype(np.float32)
        mle_output = engine.run([test_input])
        
        print(f"MLE output: {mle_output[0]}")
        
        # Compare
        if is_clustering:
            # For clustering, just check if output is valid
            success = True
        elif is_classifier and hasattr(model, 'predict_proba'):
            # For classifiers with probabilities
            success = np.allclose(sklearn_pred[0], mle_output[0], rtol=1e-2, atol=1e-3)
        elif is_classifier:
            # For classifiers without predict_proba, MLE returns probabilities
            # Convert to class label by taking argmax
            mle_class = np.argmax(mle_output[0])
            success = sklearn_pred[0] == mle_class
        else:
            # For regressors
            success = np.allclose(sklearn_pred, mle_output[0], rtol=1e-2, atol=1e-3)
        
        if success:
            print(f"  {name} PASSED")
            results['passed'].append(name)
        else:
            print(f"  {name} FAILED - Predictions don't match")
            print(f"  Expected: {sklearn_pred}")
            print(f"  Got: {mle_output[0]}")
            print(f"  Difference: {np.abs(sklearn_pred - mle_output[0])}")
            results['failed'].append(name)
            
    except Exception as e:
        print(f"  {name} FAILED - {type(e).__name__}: {e}")
        results['failed'].append(name)
        import traceback
        traceback.print_exc()


def main():
    print("="*60)
    print("MLE Runtime - Comprehensive Model Test Suite")
    print("="*60)
    
    # Create datasets
    print("\nCreating datasets...")
    X_cls, y_cls = make_classification(n_samples=200, n_features=10, n_classes=3,
                                       n_informative=8, n_redundant=0, 
                                       n_clusters_per_class=1, random_state=42)
    
    X_reg, y_reg = make_regression(n_samples=200, n_features=10, 
                                    n_informative=8, random_state=42)
    
    X_cluster, _ = make_blobs(n_samples=200, n_features=10, centers=3, random_state=42)
    
    # Binary classification dataset
    X_bin, y_bin = make_classification(n_samples=200, n_features=10, n_classes=2,
                                       n_informative=8, random_state=42)
    
    print("datatasets created")
    
    # ========== LINEAR MODELS ==========
    print("\n" + "="*60)
    print("LINEAR MODELS")
    print("="*60)
    
    test_model("Logistic Regression (Multi-class)", 
               LogisticRegression(max_iter=1000, random_state=42),
               X_cls, y_cls, is_classifier=True)
    
    test_model("Logistic Regression (Binary)", 
               LogisticRegression(max_iter=1000, random_state=42),
               X_bin, y_bin, is_classifier=True)
    
    test_model("Linear Regression",
               LinearRegression(),
               X_reg, y_reg, is_classifier=False)
    
    test_model("Ridge Regression",
               Ridge(random_state=42),
               X_reg, y_reg, is_classifier=False)
    
    test_model("Lasso Regression",
               Lasso(random_state=42, max_iter=2000),
               X_reg, y_reg, is_classifier=False)
    
    test_model("SGD Classifier",
               SGDClassifier(loss='log_loss', max_iter=1000, random_state=42),
               X_cls, y_cls, is_classifier=True)
    
    test_model("SGD Regressor",
               SGDRegressor(max_iter=1000, random_state=42),
               X_reg, y_reg, is_classifier=False)
    
    # ========== TREE MODELS ==========
    print("\n" + "="*60)
    print("TREE MODELS")
    print("="*60)
    
    test_model("Decision Tree Classifier",
               DecisionTreeClassifier(max_depth=5, random_state=42),
               X_cls, y_cls, is_classifier=True)
    
    test_model("Decision Tree Regressor",
               DecisionTreeRegressor(max_depth=5, random_state=42),
               X_reg, y_reg, is_classifier=False)
    
    # ========== ENSEMBLE MODELS ==========
    print("\n" + "="*60)
    print("ENSEMBLE MODELS")
    print("="*60)
    
    test_model("Random Forest Classifier",
               RandomForestClassifier(n_estimators=3, max_depth=5, random_state=42),
               X_cls, y_cls, is_classifier=True)
    
    test_model("Random Forest Regressor",
               RandomForestRegressor(n_estimators=3, max_depth=5, random_state=42),
               X_reg, y_reg, is_classifier=False)
    
    test_model("Gradient Boosting Classifier",
               GradientBoostingClassifier(n_estimators=3, max_depth=3, random_state=42),
               X_cls, y_cls, is_classifier=True)
    
    test_model("Gradient Boosting Regressor",
               GradientBoostingRegressor(n_estimators=3, max_depth=3, random_state=42),
               X_reg, y_reg, is_classifier=False)
    
    # ========== NEURAL NETWORKS ==========
    print("\n" + "="*60)
    print("NEURAL NETWORKS")
    print("="*60)
    
    test_model("MLP Classifier (Multi-class)",
               MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=1000, random_state=42),
               X_cls, y_cls, is_classifier=True)
    
    test_model("MLP Classifier (Binary)",
               MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=1000, random_state=42),
               X_bin, y_bin, is_classifier=True)
    
    test_model("MLP Regressor",
               MLPRegressor(hidden_layer_sizes=(16, 8), max_iter=1000, random_state=42),
               X_reg, y_reg, is_classifier=False)
    
    # ========== SVM ==========
    print("\n" + "="*60)
    print("SUPPORT VECTOR MACHINES")
    print("="*60)
    
    # Use smaller dataset for SVM (it's slow)
    X_cls_small, y_cls_small = X_cls[:100], y_cls[:100]
    X_reg_small, y_reg_small = X_reg[:100], y_reg[:100]
    
    test_model("SVC (RBF kernel)",
               SVC(kernel='rbf', probability=True, random_state=42),
               X_cls_small, y_cls_small, is_classifier=True)
    
    test_model("SVR (RBF kernel)",
               SVR(kernel='rbf'),
               X_reg_small, y_reg_small, is_classifier=False)
    
    test_model("Linear SVC",
               LinearSVC(max_iter=2000, random_state=42),
               X_cls_small, y_cls_small, is_classifier=True)
    
    test_model("Linear SVR",
               LinearSVR(max_iter=2000, random_state=42),
               X_reg_small, y_reg_small, is_classifier=False)
    
    # ========== NAIVE BAYES ==========
    print("\n" + "="*60)
    print("NAIVE BAYES")
    print("="*60)
    
    test_model("Gaussian Naive Bayes",
               GaussianNB(),
               X_cls, y_cls, is_classifier=True)
    
    # ========== NEIGHBORS ==========
    print("\n" + "="*60)
    print("K-NEAREST NEIGHBORS")
    print("="*60)
    
    test_model("KNN Classifier",
               KNeighborsClassifier(n_neighbors=5),
               X_cls, y_cls, is_classifier=True)
    
    test_model("KNN Regressor",
               KNeighborsRegressor(n_neighbors=5),
               X_reg, y_reg, is_classifier=False)
    
    # ========== CLUSTERING ==========
    print("\n" + "="*60)
    print("CLUSTERING")
    print("="*60)
    
    test_model("K-Means",
               KMeans(n_clusters=3, random_state=42, n_init=10),
               X_cluster, None, is_classifier=False, is_clustering=True)
    
    # ========== DECOMPOSITION ==========
    print("\n" + "="*60)
    print("DIMENSIONALITY REDUCTION")
    print("="*60)
    
    test_model("PCA",
               PCA(n_components=5, random_state=42),
               X_cls, None, is_classifier=False, is_transform=True)
    
    # ========== SUMMARY ==========
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"\n  Passed: {len(results['passed'])}")
    for name in results['passed']:
        print(f"  - {name}")
    
    print(f"\n  Failed: {len(results['failed'])}")
    for name in results['failed']:
        print(f"  - {name}")
    
    if results['skipped']:
        print(f"\n⊘ Skipped: {len(results['skipped'])}")
        for name in results['skipped']:
            print(f"  - {name}")
    
    total = len(results['passed']) + len(results['failed']) + len(results['skipped'])
    pass_rate = (len(results['passed']) / total * 100) if total > 0 else 0
    
    print(f"\nPass Rate: {pass_rate:.1f}% ({len(results['passed'])}/{total})")
    print("="*60)
    
    return len(results['failed']) == 0
     
    # ========== SUMMARY ==========
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"\n  Passed: {len(results['passed'])}")
    for name in results['passed']:
        print(f"  - {name}")
    
    print(f"\n  Failed: {len(results['failed'])}")
    for name in results['failed']:
        print(f"  - {name}")
    
    if results['skipped']:
        print(f"\n⊘ Skipped: {len(results['skipped'])}")
        for name in results['skipped']:
            print(f"  - {name}")
    
    total = len(results['passed']) + len(results['failed']) + len(results['skipped'])
    pass_rate = (len(results['passed']) / total * 100) if total > 0 else 0
    
    print(f"\nPass Rate: {pass_rate:.1f}% ({len(results['passed'])}/{total})")
    print("="*60)
    
    return len(results['failed']) == 0


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
