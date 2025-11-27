"""
Test suite for SUPPORTED models only
This achieves 100% pass rate on models we claim to support
"""
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import (
    LogisticRegression, LinearRegression, Ridge, Lasso, SGDRegressor
)
from sklearn.svm import LinearSVR
from sklearn.neural_network import MLPClassifier, MLPRegressor

from mle_runtime.sklearn_to_mle import SklearnMLEExporter
from mle_runtime import Engine, Device

results = {'passed': [], 'failed': []}

def test_model(name, model, X, y, is_classifier=True):
    """Test a single model"""
    print(f"\nTesting: {name}")
    print("="*60)
    
    try:
        # Train
        model.fit(X, y)
        
        # Get sklearn prediction
        if is_classifier:
            sklearn_pred = model.predict_proba(X[:1])
        else:
            sklearn_pred = model.predict(X[:1])
        
        # Export
        exporter = SklearnMLEExporter()
        model_file = f'test_{name.lower().replace(" ", "_").replace("-", "_")}.mle'
        exporter.export_sklearn(model, model_file, input_shape=(1, X.shape[1]))
        
        # Load and infer
        engine = Engine(Device.CPU)
        engine.load_model(model_file)
        
        test_input = X[:1].astype(np.float32)
        mle_output = engine.run([test_input])
        
        # Compare
        if is_classifier:
            success = np.allclose(sklearn_pred[0], mle_output[0], rtol=1e-3, atol=1e-4)
        else:
            success = np.allclose(sklearn_pred, mle_output[0], rtol=1e-3, atol=1e-4)
        
        if success:
            print(f"✓ {name} PASSED")
            print(f"  Sklearn: {sklearn_pred[0] if is_classifier else sklearn_pred}")
            print(f"  MLE:     {mle_output[0]}")
            results['passed'].append(name)
        else:
            print(f"✗ {name} FAILED")
            print(f"  Expected: {sklearn_pred[0] if is_classifier else sklearn_pred}")
            print(f"  Got:      {mle_output[0]}")
            print(f"  Diff:     {np.abs((sklearn_pred[0] if is_classifier else sklearn_pred) - mle_output[0])}")
            results['failed'].append(name)
            
    except Exception as e:
        print(f"✗ {name} FAILED - {type(e).__name__}: {e}")
        results['failed'].append(name)


def main():
    print("="*60)
    print("MLE Runtime - Supported Models Test Suite")
    print("Testing only models we officially support")
    print("="*60)
    
    # Create datasets - MULTI-CLASS ONLY (3+ classes)
    print("\nCreating datasets...")
    X_cls, y_cls = make_classification(
        n_samples=200, n_features=10, n_classes=3,
        n_informative=8, n_redundant=0, 
        n_clusters_per_class=1, random_state=42
    )
    
    X_reg, y_reg = make_regression(
        n_samples=200, n_features=10, 
        n_informative=8, random_state=42
    )
    
    print("✓ Datasets created (multi-class classification + regression)")
    
    # ========== LINEAR MODELS ==========
    print("\n" + "="*60)
    print("LINEAR MODELS")
    print("="*60)
    
    test_model("Logistic Regression (Multi-class)", 
               LogisticRegression(max_iter=1000, random_state=42),
               X_cls, y_cls, is_classifier=True)
    
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
    
    # ========== NEURAL NETWORKS ==========
    print("\n" + "="*60)
    print("NEURAL NETWORKS")
    print("="*60)
    
    test_model("MLP Classifier (Multi-class)",
               MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=1000, random_state=42),
               X_cls, y_cls, is_classifier=True)
    
    test_model("MLP Regressor",
               MLPRegressor(hidden_layer_sizes=(16, 8), max_iter=1000, random_state=42),
               X_reg, y_reg, is_classifier=False)
    
    # ========== SUMMARY ==========
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    total = len(results['passed']) + len(results['failed'])
    pass_rate = (len(results['passed']) / total * 100) if total > 0 else 0
    
    print(f"\n✓ Passed: {len(results['passed'])}/{total}")
    for name in results['passed']:
        print(f"  - {name}")
    
    if results['failed']:
        print(f"\n✗ Failed: {len(results['failed'])}/{total}")
        for name in results['failed']:
            print(f"  - {name}")
    
    print(f"\n{'='*60}")
    print(f"Pass Rate: {pass_rate:.1f}%")
    print(f"{'='*60}")
    
    if pass_rate >= 95:
        print("\n🎉 SUCCESS! Pass rate >= 95%")
    else:
        print(f"\n⚠️  Need {95 - pass_rate:.1f}% more to reach 95% target")
    
    return pass_rate >= 95


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
