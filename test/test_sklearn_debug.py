"""
Debug sklearn export and inference issues
"""
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import mle_runtime

def test_logistic_regression():
    """Test simple logistic regression"""
    print("\n" + "="*60)
    print("Testing Logistic Regression")
    print("="*60)
    
    # Create simple dataset
    X, y = make_classification(n_samples=100, n_features=4, n_classes=2, 
                               n_informative=3, n_redundant=0, random_state=42)
    
    # Train model
    model = LogisticRegression(random_state=42)
    model.fit(X, y)
    
    # Get sklearn prediction
    sklearn_pred = model.predict_proba(X[:5])
    print(f"\nSklearn prediction shape: {sklearn_pred.shape}")
    print(f"Sklearn predictions:\n{sklearn_pred}")
    
    # Check model parameters
    print(f"\nModel coef shape: {model.coef_.shape}")  # Should be [n_classes, n_features]
    print(f"Model intercept shape: {model.intercept_.shape}")
    print(f"Model coef:\n{model.coef_}")
    print(f"Model intercept: {model.intercept_}")
    
    # Export to MLE
    from mle_runtime.sklearn_to_mle import SklearnMLEExporter
    exporter = SklearnMLEExporter()
    
    try:
        exporter.export_sklearn(model, 'test_lr.mle', input_shape=(1, 4))
        print("\n✓ Export successful")
    except Exception as e:
        print(f"\n✗ Export failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Load and test inference
    try:
        from mle_runtime import Engine, Device
        engine = Engine(Device.CPU)
        engine.load_model('test_lr.mle')
        
        # Test with single sample
        test_input = X[:1].astype(np.float32)
        print(f"\nTest input shape: {test_input.shape}")
        print(f"Test input: {test_input}")
        
        mle_output = engine.run([test_input])
        print(f"\nMLE output shape: {np.array(mle_output[0]).shape}")
        print(f"MLE output:\n{mle_output[0]}")
        
        print(f"\nExpected (sklearn): {sklearn_pred[0]}")
        print(f"Got (MLE): {mle_output[0]}")
        
        # Check if close
        if np.allclose(sklearn_pred[0], mle_output[0], rtol=1e-3):
            print("\n✓ Predictions match!")
        else:
            print("\n✗ Predictions don't match")
            print(f"Difference: {np.abs(sklearn_pred[0] - mle_output[0])}")
        
    except Exception as e:
        print(f"\n✗ Inference failed: {e}")
        import traceback
        traceback.print_exc()


def test_mlp():
    """Test MLP classifier"""
    print("\n" + "="*60)
    print("Testing MLP Classifier")
    print("="*60)
    
    # Create simple dataset
    X, y = make_classification(n_samples=100, n_features=4, n_classes=2, 
                               n_informative=3, n_redundant=0, random_state=42)
    
    # Train model
    model = MLPClassifier(hidden_layer_sizes=(8,), max_iter=1000, random_state=42)
    model.fit(X, y)
    
    # Get sklearn prediction
    sklearn_pred = model.predict_proba(X[:1])
    print(f"\nSklearn prediction: {sklearn_pred}")
    
    # Check layer shapes
    print(f"\nNumber of layers: {len(model.coefs_)}")
    for i, (coef, intercept) in enumerate(zip(model.coefs_, model.intercepts_)):
        print(f"Layer {i}: coef shape {coef.shape}, intercept shape {intercept.shape}")
    
    # Export to MLE
    from mle_runtime.sklearn_to_mle import SklearnMLEExporter
    exporter = SklearnMLEExporter()
    
    try:
        exporter.export_sklearn(model, 'test_mlp.mle', input_shape=(1, 4))
        print("\n✓ Export successful")
    except Exception as e:
        print(f"\n✗ Export failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Load and test inference
    try:
        from mle_runtime import Engine, Device
        engine = Engine(Device.CPU)
        engine.load_model('test_mlp.mle')
        
        test_input = X[:1].astype(np.float32)
        mle_output = engine.run([test_input])
        
        print(f"\nExpected (sklearn): {sklearn_pred[0]}")
        print(f"Got (MLE): {mle_output[0]}")
        
        if np.allclose(sklearn_pred[0], mle_output[0], rtol=1e-2):
            print("\n✓ Predictions match!")
        else:
            print("\n✗ Predictions don't match")
            print(f"Difference: {np.abs(sklearn_pred[0] - mle_output[0])}")
        
    except Exception as e:
        print(f"\n✗ Inference failed: {e}")
        import traceback
        traceback.print_exc()


def test_random_forest():
    """Test Random Forest - known to fail"""
    print("\n" + "="*60)
    print("Testing Random Forest (Expected to fail)")
    print("="*60)
    
    X, y = make_classification(n_samples=100, n_features=4, n_classes=2, 
                               n_informative=3, n_redundant=0, random_state=42)
    
    model = RandomForestClassifier(n_estimators=3, max_depth=3, random_state=42)
    model.fit(X, y)
    
    sklearn_pred = model.predict_proba(X[:1])
    print(f"\nSklearn prediction: {sklearn_pred}")
    
    from mle_runtime.sklearn_to_mle import SklearnMLEExporter
    exporter = SklearnMLEExporter()
    
    try:
        exporter.export_sklearn(model, 'test_rf.mle', input_shape=(1, 4))
        print("\n✓ Export successful")
    except Exception as e:
        print(f"\n✗ Export failed (expected): {e}")
        return


if __name__ == '__main__':
    test_logistic_regression()
    test_mlp()
    test_random_forest()
