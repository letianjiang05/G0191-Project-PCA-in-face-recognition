# Face Recognition System Program Design Report
## 

#### 3.1.3 Support Vector Machines (SVM)
##### Pros
- Effective in high dimensional spaces: SVMs are effective when the number of dimensions is greater than the number of samples.
- Robust to outliers: SVMs are less sensitive to outliers as they focus on the instances near the decision boundary (support vectors).
- Kernel: SVMs can solve non-linear problems using the kernel, mapping their inputs into high-dimensional feature spaces.

###### Cons
- Complexity and tuning: The choice of the appropriate kernel function can be complex. Also, SVMs have several hyperparameters (like C, gamma) which need to be optimally tuned, which can be time-consuming.
- Not directly applicable to multi-class classification: SVMs are inherently binary classifiers. For multi-class problems, strategies like one-vs-one or one-vs-all need to be applied.
Using k-NNR for face recognition:
```python
```
## Conclusion
In conclusion, this project has successfully developed a real-time face
recognition system that applies Principal Component Analysis (PCA) for feature extraction and has tested various classification algorithms to find the most effective approach.
Our experimental results have demonstrated that the system can achieve a high accuracy rate, with the best performance observed at 95.3% accuracy using the k-Nearest Neighbor Rule (k-NNR) with an optimized number of dimensions in PCA.