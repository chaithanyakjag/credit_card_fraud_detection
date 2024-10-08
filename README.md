# Credit Card Fraud Detection
## Overview

Credit card fraud is a significant concern for financial institutions worldwide. This project focuses on building and evaluating various machine learning models to detect fraudulent credit card transactions using a dataset containing both legitimate and fraudulent transactions. The goal is to develop a model that can accurately distinguish between fraudulent and non-fraudulent transactions, minimizing both false positives and false negatives.

The project explores multiple supervised classification techniques including Logistic Regression, KNN, Decision Trees, Random Forest, Naïve Bayes, and SVM, with a particular focus on addressing class imbalance using SMOTE (Synthetic Minority Oversampling Technique). A unique extension of the project involves combining supervised and unsupervised techniques by integrating PCA with Random Forest.

## Methodology
### Data Description
The dataset used for this project is sourced from Kaggle, containing 1.8 million transactions across 1000 customers and 800 merchants. It includes 22 features such as credit card number, transaction amount, merchant location, and fraud flag, which serves as the target variable.

### Preprocessing Steps:
- Feature Engineering: Introduced additional features such as age and credit card type derived from existing data.
- Scaling: Applied both Min-Max Scaling and Standard Scaling to normalize the feature values.
- Handling Class Imbalance: Used SMOTE to oversample the minority class and improve model performance.

### Models Implemented:
1. Logistic Regression:
   - Applied both without upsampling and with SMOTE upsampling.
   - After upsampling, the model performance improved significantly despite a slight drop in accuracy.
2. K-Nearest Neighbors (KNN):
   - Tuned using GridSearchCV to determine the optimal value of K.
   - Implemented both Standard and Min-Max scaling for comparison.
3. Decision Trees:
   - Tuned using GridSearchCV to optimize hyperparameters like max_depth, min_samples_split, and max_leaf_nodes.
4. Random Forest:
   - A powerful ensemble method implemented for both scaled datasets.
5. Naïve Bayes (Gaussian Naïve Bayes):
   - Performed the best overall, especially when combined with Min-Max scaling.
6. Support Vector Machines (SVM):
   - Implemented for both scaling techniques with moderate results.
7. Model Extension:
   - Integrated PCA with Random Forest to explore if the combination of unsupervised and supervised learning could yield better results.

### Performance Metrics:
The models were evaluated based on the following metrics:
- Precision: To minimize false positives.
- Recall: To ensure that as many fraudulent transactions as possible are detected.
- F1 Score: A harmonic mean of precision and recall to balance both metrics.
- Confusion Matrix: Visualized to understand the model’s performance on both classes.

## Conclusion
In conclusion, after evaluating various models for credit card fraud detection, Gaussian Naïve Bayes with min-max scaling stands out as the optimal choice. The F-1 score proves to be a more effective performance measure, accounting for both false negatives and false positives in this context. Utilizing the up-sampling technique with SMOTE yields significantly improved results. While combining supervised and unsupervised techniques provides an extension, it doesn't enhance the overall model performance. In the future there is a scope for expanding the machine learning algorithms to address broader applications 
such as financial transactions, AI-powered fraud detection, and the implementation of robust cybersecurity protocols.
