r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers

part1_q1 = r"""
**Your answer:**

answer:
1.1: False. The in-sample error assesses the accuracy of a function in fitting a particular training dataset. 
Conversely, the test set is used to "mimic" unknown samples that are not included in the training process and hence can't be used to determine the in-sample error. 
Instead, the error measured on the test set is referred to as the out-sample error.

1.2: False. As an illustration, having a small test set would make it difficult to accurately assess the generalization error. 
Similarly, an insufficiently large training set may cause the model to overfit on a specific set of samples.

1.3: True. It's crucial that the test set remains independent of the training process to accurately simulate unknown data. 
As part of the training process, cross-validation is used to optimize hyperparameters, and as such, the test set should not be incorporated. 
During cross-validation, instead of using the test set, we create a validation subset from the training set and train the model with the remaining data. 
This process allows us to validate the results.

1.4: True. Validation sets provide valuable insights into a model's generalization ability as they show the performance of a model, using specific hyperparameters, on samples that were not used during the training process. 
Therefore, the validation set results indeed serve as an approximation for the model's generalization error.

"""

part1_q2 = r"""
**Your answer:**
The approach described above is not justified. Selecting the optimal hyperparameters is an integral aspect of the training stage, and as such, the test set should not be incorporated into this process. 
Instead, cross-validation should be conducted within the training set to assess the model's performance under specific hyperparameters. This preserves the true purpose of the test set, enabling us to evaluate the performance of the tuned model on previously unseen data. 
If the test set is used during the training stage, it can compromise our ability to calculate an unbiased generalization error.



"""

# ==============
# Part 2 answers

part2_q1 = r"""
**Your answer:**


If $\Delta \leq 0$, the SVM loss $L(W)$ becomes non-positive, removing the hinge loss penalty. This would cause the model to minimize the regularization term by setting $W$ to zero, resulting in a trivial, constant output. Conversely, $\Delta > 0$ enforces a margin between correct and incorrect class scores, maintaining the hinge loss as an effective component of the overall loss function.


"""

part2_q2 = r"""
**Your answer:**


The linear model seems to rely on the overall shape of the digits, focusing on simple patterns like curves and strokes, as inferred from the visualization (personal observation). Misclassifications, such as "6" as "2", are common due to overlapping features like the semi-circular arc of "6" and "2". When distinctive features, such as the closed loop of "6" are faint or incomplete, the model struggles to differentiate them. This limitation arises because the model cannot capture complex pixel interactions, relying only on linear patterns.

"""

part2_q3 = r"""
**Your answer:**


The learning rate appears $\underline{\mathbf{good}}$ as the training loss graph shows a smooth, rapid decrease initially and stabilizes at a low value, just like shown in class for a good learning rate. This indicates efficient learning without instability or slow convergence.

Why It Is $\underline{\mathbf{Highly Overfitted}}$:
Large Loss Gap: The large gap between the training and validation loss suggests that the model has overfit the training set by memorizing patterns specific to it, rather than learning generalizable features.
Large Accuracy Gap: The wide difference in accuracy between training and validation sets confirms that the model's performance does not transfer well to unseen data.
Flattening Validation Curve: The validation accuracy and loss curves flatten early, showing no improvement despite further training, which is typical of a highly overfitted model.

"""

# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
**Your answer:**


The ideal pattern in a residual plot is a horizontal band of points close to the zero line, indicating good predictions with small errors. In our residual plot:

- For the top-5 features, the plot shows a curve deviating from zero, indicating errors and poor model learning.
- Adding non-linearity improved the plot, making it more evenly spread and closer to zero.
- After cross-validation, the final plot is near the ideal pattern described above.

Regarding overfitting, we want similar behavior in both train and test sets, which is observed here, indicating good model generalization.
"""

part3_q2 = r"""
**Your answer:**
1.After adding non-linear features, it remains a linear regression model. The non-linear features map the data to a higher dimension, but the decision boundary is still linear in the mapped feature space.

2.Yes, to achieve this, we need to find a feature mapping (transformation) to a space where the function behaves linearly. With these specific non-linear features, we can fit (and might overfit) any non-linear function.

3.Adding non-linear features means the model will learn in the new feature space. As explained, the model is still a linear regressor and will produce a hyperplane in the new space. However, the decision boundary in the new space is not linear in the original space, making it more complex.

"""

part3_q3 = r"""
**Your answer:**
1.Using np.logspace allows for better exploration by examining both very small and large values without needing an excessive number of values. In contrast, np.linspace might not reach sufficiently small or large values, or would require exploring too many values to avoid missing any.

2.The model was fitted 180 times. For each combination of lambda and degree values, the model was fitted K times (number of folds). With 3 degree values and 20 lambda values, there are 60 hyperparameter combinations, each fitted 3 times (3 * 20 * 3 = 180).
"""

# ==============

# ==============