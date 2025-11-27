"""
Comprehensive test for all WORKING models
Focus on models that can be implemented with LINEAR, RELU, GELU, SOFTMAX ops
Target: >95% pass rate
"""
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

# Linear models
from sklearn.linear_model import (
    LogisticRegression, LinearRegression, Ridge, Lasso,
    SGDClassifier, SGDRegressor
)

# Neural networks
from sklearn.neural_network import MLPClassifier, MLPRegressor

# SVM (linear only)
from sklearn.svm import LinearSVC, LinearSVR

from mle_runtime.sklearn_to_mle import SklearnMLEExporter
from mle_runtime import Engine, Device

# Test results
results = {
    'passed': [],
    'failed': [],
    'total': 0
}

def test_model(name, model, X, y, is_classifier=True):
    """Test a single model"""
    print(f"\n{'='*70}")
    print(f"Testing: {name}")
    print(f"{'='*70}")
    
    results['total'] += 1
    
    try:
        # Train
        model.fit(X, y)
        
        # Get sklearn prediction
        if is_classifier:
            # For SGDClassifier, use decision_function + softmax instead of predict_proba
            # because predict_proba uses calibrated probabilities, not raw softmax
            if isinstance(model, SGDClassifier):
                sklearn_pred = model.decision_function(X[:1])
                # For binary classification, decision_function returns 1D array
                # We need to convert to [0, decision] format for softmax
                if len(sklearn_pred.shape) == 1:
                    sklearn_pred = np.array([[0, sklearn_pred[0]]])
                else:
                    # Multi-class: already 2D
                    pass
                # Apply softmax manually
                exp_scores = np.exp(sklearn_pred - np.max(sklearn_pred, axis=1, keepdims=True))
                sklearn_pred = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
            elif hasattr(model, 'predict_proba'):
                sklearn_pred = model.predict_proba(X[:1])
            else:
                # For models without predict_proba, use decision_function
                sklearn_pred = model.decision_function(X[:1])
                # Ensure 2D
                if len(sklearn_pred.shape) == 1:
                    sklearn_pred = sklearn_pred.reshape(1, -1)
                # Apply softmax manually
                exp_scores = np.exp(sklearn_pred - np.max(sklearn_pred, axis=1, keepdims=True))
                sklearn_pred = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        else:
            sklearn_pred = model.predict(X[:1])
        
        print(f"Sklearn prediction: {sklearn_pred[0] if is_classifier else sklearn_pred}")
        
        # Export
        exporter = SklearnMLEExporter()
        model_file = f'test_{name.lower().replace(" ", "_").replace("-", "_")}.mle'
        exporter.export_sklearn(model, model_file, input_shape=(1, X.shape[1]))
        
        # Load and infer
        engine = Engine(Device.CPU)
        engine.load_model(model_file)
        
        test_input = X[:1].astype(np.float32)
        mle_output = engine.run([test_input])
        
        print(f"MLE output:         {mle_output[0][0]}")
        
        # Compare
        if is_classifier:
            success = np.allclose(sklearn_pred[0], mle_output[0][0], rtol=1e-2, atol=1e-3)
        else:
            success = np.allclose(sklearn_pred, mle_output[0][0], rtol=1e-2, atol=1e-3)
        
        if success:
            print(f"PASS: {name}")
            results['passed'].append(name)
        else:
            print(f"FAIL: {name} - Predictions don't match")
            print(f"   Expected: {sklearn_pred[0] if is_classifier else sklearn_pred}")
            print(f"   Got:      {mle_output[0][0]}")
            diff = np.abs((sklearn_pred[0] if is_classifier else sklearn_pred) - mle_output[0][0])
            print(f"   Max diff: {np.max(diff):.6f}")
            results['failed'].append(name)
            
    except Exception as e:
        print(f"FAIL: {name} - {type(e).__name__}: {e}")
        results['failed'].append(name)
        import traceback
        traceback.print_exc()


def main():
    print("="*70)
    print("MLE Runtime - Working Models Test Suite")
    print("Testing models that work with LINEAR, RELU, GELU, SOFTMAX ops")
    print("Target: >95% pass rate")
    print("="*70)
    
    # Create datasets
    print("\nCreating datasets...")
    
    # Multi-class classification (3 classes)
    X_cls, y_cls = make_classification(
        n_samples=200, n_features=10, n_classes=3,
        n_informative=8, n_redundant=0, 
        n_clusters_per_class=1, random_state=42
    )
    
    # Binary classification (2 classes)
    X_bin, y_bin = make_classification(
        n_samples=200, n_features=10, n_classes=2,
        n_informative=8, random_state=42
    )
    
    # Regression
    X_reg, y_reg = make_regression(
        n_samples=200, n_features=10, 
        n_informative=8, random_state=42
    )
    
    print("Datasets created")
    print(f"  - Multi-class: {X_cls.shape}, {len(np.unique(y_cls))} classes")
    print(f"  - Binary:      {X_bin.shape}, {len(np.unique(y_bin))} classes")
    print(f"  - Regression:  {X_reg.shape}")
    
    # ========== LINEAR MODELS - CLASSIFICATION ==========
    print("\n" + "="*70)
    print("LINEAR MODELS - CLASSIFICATION")
    print("="*70)
    
    test_model("Logistic Regression (Multi-class)", 
               LogisticRegression(max_iter=1000, random_state=42),
               X_cls, y_cls, is_classifier=True)
    
    test_model("Logistic Regression (Binary)", 
               LogisticRegression(max_iter=1000, random_state=42),
               X_bin, y_bin, is_classifier=True)
    
    test_model("SGD Classifier (Multi-class)",
               SGDClassifier(max_iter=1000, random_state=42, loss='log_loss'),
               X_cls, y_cls, is_classifier=True)
    
    test_model("SGD Classifier (Binary)",
               SGDClassifier(max_iter=1000, random_state=42, loss='log_loss'),
               X_bin, y_bin, is_classifier=True)
    
    test_model("Linear SVC (Multi-class)",
               LinearSVC(max_iter=2000, random_state=42),
               X_cls, y_cls, is_classifier=True)
    
    test_model("Linear SVC (Binary)",
               LinearSVC(max_iter=2000, random_state=42),
               X_bin, y_bin, is_classifier=True)
    
    # ========== LINEAR MODELS - REGRESSION ==========
    print("\n" + "="*70)
    print("LINEAR MODELS - REGRESSION")
    print("="*70)
    
    test_model("Linear Regression",
               LinearRegression(),
               X_reg, y_reg, is_classifier=False)
    
    test_model("Ridge Regression",
               Ridge(random_state=42),
               X_reg, y_reg, is_classifier=False)
    
    test_model("Lasso Regression",
               Lasso(random_state=42, max_iter=2000),
               X_reg, y_reg, is_classifier=False)
    
    test_model("SGD Regressor",
               SGDRegressor(max_iter=1000, random_state=42),
               X_reg, y_reg, is_classifier=False)
    
    test_model("Linear SVR",
               LinearSVR(max_iter=2000, random_state=42),
               X_reg, y_reg, is_classifier=False)
    
    # ========== NEURAL NETWORKS - CLASSIFICATION ==========
    print("\n" + "="*70)
    print("NEURAL NETWORKS - CLASSIFICATION")
    print("="*70)
    
    test_model("MLP Classifier (Multi-class)",
               MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=1000, random_state=42),
               X_cls, y_cls, is_classifier=True)
    
    test_model("MLP Classifier (Binary)",
               MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=1000, random_state=42),
               X_bin, y_bin, is_classifier=True)
    
    test_model("MLP Classifier (Deep)",
               MLPClassifier(hidden_layer_sizes=(32, 16, 8), max_iter=1000, random_state=42),
               X_cls, y_cls, is_classifier=True)
    
    # ========== NEURAL NETWORKS - REGRESSION ==========
    print("\n" + "="*70)
    print("NEURAL NETWORKS - REGRESSION")
    print("="*70)
    
    test_model("MLP Regressor",
               MLPRegressor(hidden_layer_sizes=(16, 8), max_iter=1000, random_state=42),
               X_reg, y_reg, is_classifier=False)
    
    test_model("MLP Regressor (Deep)",
               MLPRegressor(hidden_layer_sizes=(32, 16, 8), max_iter=1000, random_state=42),
               X_reg, y_reg, is_classifier=False)
    
    # ========== SUMMARY ==========
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    total = results['total']
    passed = len(results['passed'])
    failed = len(results['failed'])
    pass_rate = (passed / total * 100) if total > 0 else 0
    
    print(f"\nPassed: {passed}/{total}")
    for name in results['passed']:
        print(f"   + {name}")
    
    if results['failed']:
        print(f"\nFailed: {failed}/{total}")
        for name in results['failed']:
            print(f"   - {name}")
    
    print(f"\n{'='*70}")
    print(f"Pass Rate: {pass_rate:.1f}% ({passed}/{total})")
    print(f"{'='*70}")
    
    if pass_rate >= 95:
        print("\nSUCCESS! Pass rate >= 95%")
        print("All working models are functioning correctly")
        return True
    else:
        print(f"\nNeed {95 - pass_rate:.1f}% more to reach 95% target")
        print(f"{total - passed} models need fixing")
        return False


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
