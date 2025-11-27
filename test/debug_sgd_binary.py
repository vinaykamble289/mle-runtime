"""Debug SGD Binary"""
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import SGDClassifier
from mle_runtime.sklearn_to_mle import SklearnMLEExporter
from mle_runtime import Engine, Device

X_bin, y_bin = make_classification(n_samples=200, n_features=10, n_classes=2,
                                   n_informative=8, random_state=42)

model = SGDClassifier(max_iter=1000, random_state=42, loss='log_loss')
model.fit(X_bin, y_bin)

print(f"Classes: {model.classes_}")
print(f"Coef shape: {model.coef_.shape}")

# Decision function
decision = model.decision_function(X_bin[:1])
print(f"\nDecision function: {decision}")
print(f"Decision shape: {decision.shape}")

# For binary, decision_function returns 1D array
# We need to convert to 2D: [0, decision]
if len(decision.shape) == 1:
    decision_2d = np.array([[0, decision[0]]])
else:
    decision_2d = decision

print(f"Decision 2D: {decision_2d}")

# Apply softmax
exp_scores = np.exp(decision_2d - np.max(decision_2d))
softmax_pred = exp_scores / np.sum(exp_scores)
print(f"Softmax: {softmax_pred}")

# Compare with predict_proba
proba = model.predict_proba(X_bin[:1])
print(f"\nPredict_proba: {proba[0]}")

# Export and test
exporter = SklearnMLEExporter()
exporter.export_sklearn(model, 'debug_sgd_bin.mle', input_shape=(1, 10))

engine = Engine(Device.CPU)
engine.load_model('debug_sgd_bin.mle')
mle_output = engine.run([X_bin[:1].astype(np.float32)])
print(f"MLE output: {mle_output[0][0]}")
