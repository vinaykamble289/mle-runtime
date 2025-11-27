"""Debug binary classification issue"""
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from mle_runtime.sklearn_to_mle import SklearnMLEExporter
from mle_runtime import Engine, Device

# Create binary dataset
X, y = make_classification(n_samples=100, n_features=5, n_classes=2, random_state=42)

# Train model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X, y)

print("Model info:")
print(f"  Classes: {model.classes_}")
print(f"  Coef shape: {model.coef_.shape}")
print(f"  Intercept shape: {model.intercept_.shape}")
print(f"  Coef: {model.coef_[0][:3]}...")
print(f"  Intercept: {model.intercept_}")

# Test prediction
test_x = X[:1]
sklearn_pred = model.predict_proba(test_x)
print(f"\nSklearn predict_proba: {sklearn_pred[0]}")

# Manual calculation
logits = np.dot(test_x, model.coef_.T) + model.intercept_
print(f"Logits (raw): {logits[0]}")

# For binary, sklearn uses sigmoid on single logit
sigmoid = 1 / (1 + np.exp(-logits[0, 0]))
print(f"Sigmoid: {sigmoid}")
print(f"Proba: [{1-sigmoid}, {sigmoid}]")

# What we're doing: expand to [-logit, logit] then softmax
expanded_logits = np.array([[-logits[0, 0], logits[0, 0]]])
print(f"\nExpanded logits: {expanded_logits[0]}")

# Apply softmax
exp_logits = np.exp(expanded_logits - np.max(expanded_logits))
softmax = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
print(f"Softmax: {softmax[0]}")

# Export and test
exporter = SklearnMLEExporter()
exporter.export_sklearn(model, 'debug_binary.mle', input_shape=(1, 5))

engine = Engine(Device.CPU)
engine.load_model('debug_binary.mle')
mle_output = engine.run([test_x.astype(np.float32)])
print(f"\nMLE output: {mle_output[0][0]}")
print(f"Expected:   {sklearn_pred[0]}")
print(f"Difference: {np.abs(sklearn_pred[0] - mle_output[0][0])}")
