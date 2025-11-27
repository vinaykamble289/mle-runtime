"""Debug SGD training"""
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import SGDClassifier

X_cls, y_cls = make_classification(n_samples=200, n_features=10, n_classes=3,
                                   n_informative=8, n_redundant=0,
                                   n_clusters_per_class=1, random_state=42)

# Try different SGD configurations
configs = [
    {'loss': 'log_loss', 'max_iter': 1000, 'random_state': 42},
    {'loss': 'log_loss', 'max_iter': 5000, 'random_state': 42, 'tol': 1e-4},
    {'loss': 'log_loss', 'max_iter': 1000, 'random_state': 42, 'alpha': 0.0001},
    {'loss': 'modified_huber', 'max_iter': 1000, 'random_state': 42},
]

for i, config in enumerate(configs):
    print(f"\n{'='*70}")
    print(f"Config {i+1}: {config}")
    print(f"{'='*70}")
    
    model = SGDClassifier(**config)
    model.fit(X_cls, y_cls)
    
    # Check predictions on first sample
    pred_proba = model.predict_proba(X_cls[:1])
    pred_class = model.predict(X_cls[:1])
    true_class = y_cls[:1]
    
    print(f"True class: {true_class[0]}")
    print(f"Predicted class: {pred_class[0]}")
    print(f"Predict_proba: {pred_proba[0]}")
    print(f"Max prob: {np.max(pred_proba[0]):.6f}")
    print(f"Entropy: {-np.sum(pred_proba[0] * np.log(pred_proba[0] + 1e-10)):.6f}")
    
    # Check decision function
    decision = model.decision_function(X_cls[:1])
    print(f"Decision function: {decision[0]}")
    
    # Manual softmax
    exp_dec = np.exp(decision[0] - np.max(decision[0]))
    manual_softmax = exp_dec / np.sum(exp_dec)
    print(f"Manual softmax: {manual_softmax}")
