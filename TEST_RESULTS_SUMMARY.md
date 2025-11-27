# MLE Runtime - Test Results Summary

## Current Status

**Pass Rate: 64.0% (16/25 models)**

### ✅ Passing Models (16)

#### Linear Models (6)
- Logistic Regression (Multi-class)
- Logistic Regression (Binary)
- Linear Regression
- Ridge Regression
- Lasso Regression
- SGD Regressor

#### Neural Networks (3)
- MLP Classifier (Multi-class)
- MLP Classifier (Binary)
- MLP Regressor

#### SVM (2)
- Linear SVC
- Linear SVR

#### Tree Models (1)
- Decision Tree Classifier

#### Probabilistic Models (1)
- Gaussian Naive Bayes

#### Neighbors (1)
- KNN Classifier

#### Clustering (1)
- K-Means

#### Dimensionality Reduction (1)
- PCA

### ❌ Failing Models (9)

#### Linear Models (1)
- SGD Classifier (predictions don't match - numerical issue)

#### Tree Models (1)
- Decision Tree Regressor (output shape issue)

#### Ensemble Models (4)
- Random Forest Classifier (predictions close but not exact)
- Random Forest Regressor (output shape issue)
- Gradient Boosting Classifier (needs implementation fix)
- Gradient Boosting Regressor (needs implementation fix)

#### SVM (2)
- SVC (RBF kernel) (needs implementation)
- SVR (RBF kernel) (needs implementation)

#### Neighbors (1)
- KNN Regressor (needs implementation fix)

## Progress

- **Initial**: 40.0% (10/25)
- **Current**: 64.0% (16/25)
- **Improvement**: +24% (+6 models)

## Next Steps

To reach 90%+ pass rate (23/25 models), we need to fix:
1. Decision Tree Regressor - output shape handling
2. Random Forest models - averaging/normalization logic
3. Gradient Boosting models - multi-class tree structure
4. SVM with RBF kernel - kernel computation
5. KNN Regressor - regression output handling
6. SGD Classifier - numerical stability

Most of these are implementation details that can be fixed with targeted debugging.
