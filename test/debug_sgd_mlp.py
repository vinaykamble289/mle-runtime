"""Debug SGD Classifier and MLP Binary"""
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from mle_runtime.sklearn_to_mle import SklearnMLEExporter
from mle_runtime import Engine, Device

print("="*70)
print("1. SGD Classifier (Multi-class)")
print("="*70)

X_cls, y_cls = make_classification(n_samples=200, n_features=10, n_classes=3,
                                   n_informative=8, random_state=42)

model = SGDClassifier(max_iter=1000, random_state=42, loss='log_loss')
model.fit(X_cls, y_cls)

print(f"Classes: {model.classes_}")
print(f"Coef shape: {model.coef_.shape}")

sklearn_pred = model.predict_proba(X_cls[:1])
print(f"Sklearn predict_proba: {sklearn_pred[0]}")

exporter = SklearnMLEExporter()
exporter.export_sklearn(model, 'debug_sgd.mle', input_shape=(1, 10))

engine = Engine(Device.CPU)
engine.load_model('debug_sgd.mle')
mle_output = engine.run([X_cls[:1].astype(np.float32)])
print(f"MLE output: {mle_output[0][0]}")
print(f"Difference: {np.abs(sklearn_pred[0] - mle_output[0][0])}")

print("\n" + "="*70)
print("2. MLP Classifier (Binary)")
print("="*70)

X_bin, y_bin = make_classification(n_samples=200, n_features=10, n_classes=2,
                                   n_informative=8, random_state=42)

model2 = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=1000, random_state=42)
model2.fit(X_bin, y_bin)

print(f"Classes: {model2.classes_}")
print(f"Output layer shape: {model2.coefs_[-1].shape}")
print(f"Out activation: {model2.out_activation_}")

sklearn_pred2 = model2.predict_proba(X_bin[:1])
print(f"Sklearn predict_proba: {sklearn_pred2[0]}")

exporter2 = SklearnMLEExporter()
exporter2.export_sklearn(model2, 'debug_mlp_bin.mle', input_shape=(1, 10))

engine2 = Engine(Device.CPU)
engine2.load_model('debug_mlp_bin.mle')
mle_output2 = engine2.run([X_bin[:1].astype(np.float32)])
print(f"MLE output: {mle_output2[0][0]}")
print(f"Difference: {np.abs(sklearn_pred2[0] - mle_output2[0][0])}")
