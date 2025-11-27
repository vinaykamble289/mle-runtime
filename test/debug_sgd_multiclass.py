"""Debug SGD Classifier Multi-class"""
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import SGDClassifier
from mle_runtime.sklearn_to_mle import SklearnMLEExporter
from mle_runtime import Engine, Device

X_cls, y_cls = make_classification(n_samples=200, n_features=10, n_classes=3,
                                   n_informative=8, n_redundant=0,
                                   n_clusters_per_class=1, random_state=42)

model = SGDClassifier(max_iter=1000, random_state=42, loss='log_loss')
model.fit(X_cls, y_cls)

# Get sklearn prediction
sklearn_pred = model.predict_proba(X_cls[:1])
print(f"Sklearn predict_proba: {sklearn_pred[0]}")

# Export
exporter = SklearnMLEExporter()
exporter.export_sklearn(model, 'debug_sgd_mc.mle', input_shape=(1, 10))

# Load and infer
engine = Engine(Device.CPU)
engine.load_model('debug_sgd_mc.mle')
mle_output = engine.run([X_cls[:1].astype(np.float32)])
print(f"MLE output: {mle_output[0][0]}")

# Check tolerance
diff = np.abs(sklearn_pred[0] - mle_output[0][0])
print(f"\nDifference: {diff}")
print(f"Max diff: {np.max(diff)}")

# Test with different tolerances
rtol = 1e-2
atol = 1e-3
success = np.allclose(sklearn_pred[0], mle_output[0][0], rtol=rtol, atol=atol)
print(f"\nnp.allclose(rtol={rtol}, atol={atol}): {success}")

# Try looser tolerance
rtol2 = 1e-1
atol2 = 1e-2
success2 = np.allclose(sklearn_pred[0], mle_output[0][0], rtol=rtol2, atol=atol2)
print(f"np.allclose(rtol={rtol2}, atol={atol2}): {success2}")

# Check element-wise
for i, (s, m) in enumerate(zip(sklearn_pred[0], mle_output[0][0])):
    rel_err = abs(s - m) / (abs(s) + 1e-10)
    print(f"  [{i}] sklearn={s:.10e}, mle={m:.10e}, abs_err={abs(s-m):.10e}, rel_err={rel_err:.10e}")
