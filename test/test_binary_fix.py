"""Test binary classification fixes"""
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from mle_runtime.sklearn_to_mle import SklearnMLEExporter
from mle_runtime import Engine, Device

# Binary classification dataset
X, y = make_classification(n_samples=100, n_features=10, n_classes=2,
                           n_informative=8, random_state=42)

print("="*60)
print("Testing Binary Logistic Regression")
print("="*60)

# Train model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X, y)

print(f"Model coef shape: {model.coef_.shape}")
print(f"Model classes: {model.classes_}")

# Get sklearn prediction
sklearn_pred = model.predict_proba(X[:1])
print(f"Sklearn prediction: {sklearn_pred}")

# Export
exporter = SklearnMLEExporter()
exporter.export_sklearn(model, 'test_binary_lr.mle', input_shape=(1, 10))

# Load and test
engine = Engine(Device.CPU)
engine.load_model('test_binary_lr.mle')

test_input = X[:1].astype(np.float32)
mle_output = engine.run([test_input])

print(f"MLE output: {mle_output[0]}")
print(f"Expected: {sklearn_pred[0]}")

if np.allclose(sklearn_pred[0], mle_output[0], rtol=1e-3):
    print("✓ Binary LogisticRegression PASSED!")
else:
    print("✗ Binary LogisticRegression FAILED")
    print(f"Difference: {np.abs(sklearn_pred[0] - mle_output[0])}")

print("\n" + "="*60)
print("Testing Binary MLP Classifier")
print("="*60)

# Train MLP
mlp = MLPClassifier(hidden_layer_sizes=(16,), max_iter=1000, random_state=42)
mlp.fit(X, y)

print(f"Last layer coef shape: {mlp.coefs_[-1].shape}")
print(f"MLP classes: {mlp.classes_}")

sklearn_pred_mlp = mlp.predict_proba(X[:1])
print(f"Sklearn prediction: {sklearn_pred_mlp}")

# Export
exporter2 = SklearnMLEExporter()
exporter2.export_sklearn(mlp, 'test_binary_mlp.mle', input_shape=(1, 10))

# Load and test
engine2 = Engine(Device.CPU)
engine2.load_model('test_binary_mlp.mle')

mle_output_mlp = engine2.run([test_input])

print(f"MLE output: {mle_output_mlp[0]}")
print(f"Expected: {sklearn_pred_mlp[0]}")

if np.allclose(sklearn_pred_mlp[0], mle_output_mlp[0], rtol=1e-2):
    print("✓ Binary MLPClassifier PASSED!")
else:
    print("✗ Binary MLPClassifier FAILED")
    print(f"Difference: {np.abs(sklearn_pred_mlp[0] - mle_output_mlp[0])}")
