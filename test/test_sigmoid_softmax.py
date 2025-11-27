"""Test sigmoid to softmax conversion"""
import numpy as np

logit = -1.6948253

# Sklearn uses sigmoid
sigmoid = 1 / (1 + np.exp(-logit))
print(f"Logit: {logit}")
print(f"Sigmoid: {sigmoid}")
print(f"Proba: [{1-sigmoid}, {sigmoid}]")

# Try different softmax conversions
print("\nTrying different conversions:")

# Option 1: [-logit, logit]
expanded1 = np.array([-logit, logit])
softmax1 = np.exp(expanded1) / np.sum(np.exp(expanded1))
print(f"1. softmax([-logit, logit]): {softmax1}")

# Option 2: [0, 2*logit]
expanded2 = np.array([0, 2*logit])
softmax2 = np.exp(expanded2) / np.sum(np.exp(expanded2))
print(f"2. softmax([0, 2*logit]): {softmax2}")

# Option 3: [0, logit]
expanded3 = np.array([0, logit])
softmax3 = np.exp(expanded3) / np.sum(np.exp(expanded3))
print(f"3. softmax([0, logit]): {softmax3}")

# The correct formula: sigmoid(x) = exp(x)/(1+exp(x)) = 1/(1+exp(-x))
# softmax([a, b]) = [exp(a)/(exp(a)+exp(b)), exp(b)/(exp(a)+exp(b))]
# We want: softmax([a, b]) = [1/(1+exp(-x)), exp(-x)/(1+exp(-x))]
# This means: exp(a)/(exp(a)+exp(b)) = 1/(1+exp(-x))
# And: exp(b)/(exp(a)+exp(b)) = exp(-x)/(1+exp(-x))

# If we set a=0, then: 1/(1+exp(b)) = 1/(1+exp(-x))
# So: exp(b) = exp(-x), which means b = -x

expanded4 = np.array([0, -logit])
softmax4 = np.exp(expanded4) / np.sum(np.exp(expanded4))
print(f"4. softmax([0, -logit]): {softmax4}")

# Verify
print(f"\nTarget: [{1-sigmoid}, {sigmoid}]")
print(f"Best match: softmax([0, -logit]) = {softmax4}")
