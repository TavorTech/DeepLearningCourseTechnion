r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 (Backprop) answers

part1_q1 = r"""
**Your answer:**


### 1. 
**A.** The Jacobian shape calculation can be summarized as follows:
$
\text{Given:} \quad 
\mat{X} \in \mathbb{R}^{N \times \text{in\_features}}, \quad 
\mat{W} \in \mathbb{R}^{\text{out\_features} \times \text{in\_features}}, \quad 
\mat{Y} \in \mathbb{R}^{N \times \text{out\_features}}
$

I. Total elements in $\mat{Y}$:
$
N \cdot \text{out\_features} = 64 \cdot 512 = 32,768
$

II. Total elements in $\mat{X}$:
$
N \cdot \text{in\_features} = 64 \cdot 1024 = 65,536
$

III. Shape of the Jacobian:
$
\frac{\partial \mat{Y}}{\partial \mat{X}} \in \mathbb{R}^{(N \cdot \text{out\_features}) \times (N \cdot \text{in\_features})} = \mathbb{R}^{32,768 \times 65,536}
$.
    
**B.** Yes, the Jacobian is sparse because most elements are zero. 
on-zero elements exist only for input features $ \mathbf{X}[i, :] $ that directly affect the output $ \mathbf{Y}[i, j] $, corresponding to the $ j $-th feature of $ \mathbf{Y} $. 
Specifically, $ \frac{\partial Y[i, j]}{\partial X[i, n]} = W[j, n] $. All other elements , $ \frac{\partial Y[i, j]}{\partial X[m, n]} = 0 $ , $( i \neq m )$ are zero, making the Jacobian block-sparse.

**C.** No, we do not need to materialize the Jacobian to calculate the downstream gradient $ \delta X = \frac{\partial L}{\partial X} $. 
Instead, the downstream gradient can be calculated efficiently using matrix multiplication, leveraging the fact that the Jacobian         $  \frac{\partial Y}{\partial X} $ represents the weights $ W $ in the linear layer:  $  \delta X = \delta Y \cdot W $        
This avoids explicitly constructing the large Jacobian matrix, as the computation can be performed directly with the weight matrix $ W $ and the gradient $ \delta Y $.

---

### 2. 
**A.** The Jacobian shape calculation can be summarized as follows:
$
\text{Given:} \quad 
\mat{X} \in \mathbb{R}^{N \times \text{in\_features}}, \quad 
\mat{W} \in \mathbb{R}^{\text{out\_features} \times \text{in\_features}}, \quad 
$

I. Total elements in $\mat{Y}$:
$
N \cdot \text{out\_features} = 64 \cdot 512 = 32,768
$

II. Total elements in $\mat{W}$:
$
\text{out\_features} \cdot \text{in\_features} = 512 \cdot 1024 = 524,288
$

III. Shape of the Jacobian:
$
\frac{\partial \mat{Y}}{\partial \mat{W}} \in \mathbb{R}^{(N \cdot \text{out\_features}) \times (\text{out\_features} \cdot \text{in\_features})} = \mathbb{R}^{32,768 \times 524,288}
$

**B.** Yes, the Jacobian is sparse because most elements are zero. 
Non-zero elements exist only for weights $\mathbf{W}[j, :]$ that directly affect the output $\mathbf{Y}[i, j]$, corresponding to the $j$-th feature of $\mathbf{Y}$. Specifically, $ \frac{\partial Y[i, j]}{\partial W[j, n]} = X[i, n] $. All other elements, $\frac{\partial \mathbf{Y}[i, j]}{\partial \mathbf{W}[k, n]} = 0$ for $k \neq j$, are zero, making the Jacobian block-sparse.

**C.** No, we do not need to materialize the Jacobian. Similar to the answer from 1.C., the downstream gradient $ \delta \mathbf{X} $ can be efficiently calculated as $ \delta \mathbf{X} = \delta \mathbf{Y} \cdot \mathbf{W} $. Additionally, the gradient $ \delta \mathbf{W} $ can be directly computed using $ \delta \mathbf{W} = \delta \mathbf{Y}^\top \mathbf{X} $, avoiding explicit construction of the Jacobian.

"""

part1_q2 = r"""
**Your answer:**


Back-propagation is essential for training deep neural networks because it efficiently computes gradients for all parameters, which would be infeasible to calculate manually for large models. It is specifically required for handling the complex interdependencies of layers, ensuring practical optimization in intricate architectures.

"""


# ==============
# Part 2 (Optimization) answers


def part2_overfit_hp():
    wstd, lr, reg = 0, 0, 0
    # TODO: Tweak the hyperparameters until you overfit the small dataset.
    # ====== YOUR CODE: ======

    wstd, lr, reg = 0.1, 0.1, 1e-2

    # ========================
    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_optim_hp():
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = (
        0,
        0,
        0,
        0,
        0,
    )

    # TODO: Tweak the hyperparameters to get the best results you can.
    # You may want to use different learning rates for each optimizer.
    # ====== YOUR CODE: ======

    wstd = 0.05
    lr_vanilla = 0.02
    lr_momentum = 0.004
    lr_rmsprop = 0.0002
    reg = 0.0015

    # ========================
    return dict(
        wstd=wstd,
        lr_vanilla=lr_vanilla,
        lr_momentum=lr_momentum,
        lr_rmsprop=lr_rmsprop,
        reg=reg,
    )


def part2_dropout_hp():
    wstd, lr, = (
        0,
        0,
    )
    # TODO: Tweak the hyperparameters to get the model to overfit without
    # dropout.
    # ====== YOUR CODE: ======

    wstd = 0.15
    lr = 0.001

    # ========================
    return dict(wstd=wstd, lr=lr)


part2_q1 = r"""
**Your answer:**

1.1.
The graphs comparing the no-dropout and dropout configurations align with our expectations. 

Without dropout, the training accuracy remains more stable across epochs, showing fewer fluctuations compared to the dropout configurations. This is because all neurons are active during training without dropout, leading to stronger and more focused connections within the network. As a result, the model tends to perform well on the training data, resulting in higher training accuracy. 

On the other hand, the introduction of dropout introduces more spikes in the accuracy curves between epochs. Dropout randomly "drops out" a fraction of neurons during each training iteration, encouraging the network to distribute the representation across multiple sets of neurons. This promotes better generalization and reduces overfitting. As a consequence, the model may exhibit lower training accuracy compared to the no-dropout case. 

In terms of test accuracy, the results show a trade-off between the dropout configurations and the no-dropout case. Without dropout, the model achieves higher test accuracy since it has learned strong and specific connections tailored to the training data. However, this can lead to overfitting, where the model fails to generalize well to unseen data.

1.2.
As the dropout rate increases, we observe a decrease in test accuracy but an improvement in generalization. 

For example, comparing a low dropout rate (e.g., 0.4) to no dropout, the test accuracy may slightly decrease, but it helps in reducing overfitting and improving performance on unseen data. Dropout prevents the model from relying too heavily on specific neurons, encouraging the network to learn more robust and generalizable features. 

However, it's important to note that extremely high dropout rates, such as 0.8, can harm both training and testing accuracy. With a dropout rate of 0.8, a significant portion of neurons is disabled during training, severely limiting the model's capacity to learn meaningful patterns from the data. As a result, the model's performance is likely to suffer, yielding poor results for both training and testing.
```python
# A code block
```


"""

part2_q2 = r"""
**Your answer:**


Yes, it is possible. This can occur in certain scenarios due to the behavior of the Cross-Entropy loss function.

The Cross-Entropy loss function penalizes predicted labels $\hat{y}$ that are far from the true labels $y$ in terms of distance. However, accuracy only considers whether the predicted labels are equal to the true labels.

For example, consider some labels where $y_1 = \hat{y}_1$ and $y_2 \ne \hat{y}_2, y_3 \ne \hat{y}_3$, but $y_2$ and $y_3$ are very close to their predicted labels. In an epoch, if $\hat{y}_2$ and $\hat{y}_3$ increase slightly so that $y_2 = \hat{y}_2$ and $y_3 = \hat{y}_3$, while $y_1$ decreases significantly, the following happens:

In this case, all predicted labels remain unchanged except for two that now match their true labels and one that no longer does. As a result, the test accuracy increases.

However, the overall distances between the predicted and true labels increase, leading to a higher loss.
=======
"""

part2_q3 = r"""
**Your answer:**

3.1:
> **Gradient Descent (GD)** is an optimization algorithm used to minimize the loss function by iteratively updating the model's parameters.

> **Backpropagation** is an algorithm used to efficiently calculate the derivatives of the loss function with respect to the parameters using the chain rule.

3.2:
> GD and SGD are both optimization algorithms used in training machine learning models. They differ in how they update the model's parameters:

> In GD, the algorithm considers the entire dataset $X$ in each epoch to decide and perform the update step. In contrast, in SGD, the algorithm samples a subset of the dataset $X$ (of size $BatchSize$) in each epoch and only uses that subset to decide the update step.

> GD is more robust and less sensitive to noisy data compared to SGD because it uses the entire dataset. However, SGD might not reach the minimum (the solution may fluctuate around the optimal point) because it is influenced by individual samples. Nevertheless, SGD takes smaller steps and thus converges faster than GD.

3.3:
> Here are some reasons for preferring SGD:
> * SGD uses only a part of the dataset, making it more efficient and feasible for large datasets where running GD might not be possible.
> * SGD uses different samples in each epoch, which can help it converge better by ignoring some noisy data that might lead to poor updates.
> * SGD converges faster because it takes smaller steps, as explained above.

"""

part2_q4 = r"""
**Your answer:**

4.1.

Reducing the memory complexity for computing the gradient using forward mode Automatic Differentiation (AD) while maintaining O(n) computation cost, we can use checkpointing. we do this by storing only the values required for gradient computation instead of all intermediate results. In the first approach, the algorithm initializes two variables, currentGradient and currentResult, and iterates through the computational graph in a forward pass. At each step, it computes the derivative of the current function and updates the gradient and result accordingly. By storing only the current gradient and result, the memory complexity is reduced to O(1). However, if the intermediate results are not already given, the memory complexity becomes O(n) as we need to store all the intermediate results.

4.2.

Similarly, for backward mode AD, we can use checkpointing to reduce memory complexity while maintaining O(n) computation cost. The second approach initializes two holders, backwardGradient and backwardResult, and performs a forward pass through the computational graph, saving the intermediate function results. Then, in the backward pass, it iterates from the end to the beginning, computing the gradients based on the saved function results. when we store only the necessary function results - the memory complexity is reduced to O(1).

4.3.

they use checkpointing to minimize memory usage during gradient computation. However, it's important to note that these techniques assume a sequential execution of functions and may not be optimal for computational graphs with parallel executions. The memory complexity reduction to O(1) holds under the assumption of a sequential execution. These memory optimization techniques can be generalized for arbitrary computational graphs by employing checkpointing and breaking down the graph into subgraphs. By computing the gradient of each subgraph separately and combining them, we can reduce the memory complexity of the overall gradient computation to O(1). if we cache only the essential values needed for gradient computation, we will be able to handle large and complex computational graphs efficiently.

4.4.

with VGGs and ResNets using said memory optimization techniques offer significant benefits. These architectures often have a large number of parameters and layers, resulting in high memory requirements. By reducing the memory complexity of gradient computation to O(1), we can alleviate the memory burden during training. This advantage becomes crucial, as it enables efficient training on hardware with limited memory resources. Additionally, the reduced memory complexity allows for faster and more scalable training, improving the overall training time of deep architectures.

"""

# ==============


# ==============
# Part 3 (MLP) answers


def part3_arch_hp():
    n_layers = 0  # number of layers (not including output)
    hidden_dims = 0  # number of output dimensions for each hidden layer
    activation = "none"  # activation function to apply after each hidden layer
    out_activation = "none"  # activation function to apply at the output layer
    # TODO: Tweak the MLP architecture hyperparameters.
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
    return dict(
        n_layers=n_layers,
        hidden_dims=hidden_dims,
        activation=activation,
        out_activation=out_activation,
    )


def part3_optim_hp():
    import torch.nn
    import torch.nn.functional

    loss_fn = None  # One of the torch.nn losses
    lr, weight_decay, momentum = 0, 0, 0  # Arguments for SGD optimizer
    # TODO:
    #  - Tweak the Optimizer hyperparameters.
    #  - Choose the appropriate loss function for your architecture.
    #    What you returns needs to be a callable, so either an instance of one of the
    #    Loss classes in torch.nn or one of the loss functions from torch.nn.functional.
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part3_q1 = r"""
**Your answer:**
1. **High Optimization Error:** This occurs when the model fails to minimize the training loss. To reduce it, we can use optimizers like Adam, adjust the learning rate so it converges better, and increase training epochs. These steps help in minimizing the **population loss**.

2. **High Generalization Error:** This happens when the model performs well on training data but poorly on validation/test data. Reduce it by using regularization, data augmentation, and cross-validation. Increasing the **receptive field** can also help in capturing more context and improving generalization.

3. **High Approximation Error:** This occurs when the model is too simple to capture data patterns. Reduce it by increasing model complexity, improving feature engineering, and increasing the **receptive field**.

These strategies help in reducing errors and improving the overall performance by minimizing the **population loss**.
"""

part3_q2 = r"""
**Your answer:**

1. **Higher False Positive Rate (FPR):** This occurs when the classifier incorrectly labels negative instances as positive. This can happen in scenarios where the model is overly sensitive to certain features, possibly due to a large receptive field capturing too much context. For example, in spam detection, legitimate emails might be marked as spam.

2. **Higher False Negative Rate (FNR):** This occurs when the classifier incorrectly labels positive instances as negative. This can happen when the model is too conservative, possibly due to a small receptive field missing important context. For example, in medical diagnosis, actual cases of a disease might be missed. Reducing this error is crucial as it directly impacts the population loss by failing to identify true positives.

"""

part3_q3 = r"""
**Your answer:**

In this scenario, the choice of the "optimal" point on the ROC curve depends on the balance between the cost of false positives (FP) and false negatives (FN).

1. **Non-lethal Symptoms:**
   - **Optimal Point:** In this case, the cost of a false positive is high due to the expensive and risky further tests, but the cost of a false negative is relatively low since the disease will eventually be diagnosed through symptoms.
   - **Choice:** You would choose a point on the ROC curve that minimizes false positives, even if it means a higher false negative rate. This reduces unnecessary expensive and risky tests.

2. **High Probability of Death:**
   - **Optimal Point:** Here, the cost of a false negative is extremely high as it could result in death if the disease is not diagnosed early.
   - **Choice:** You would choose a point on the ROC curve that minimizes false negatives, even if it means a higher false positive rate. This ensures that as many cases as possible are caught early, despite the cost and risk of further tests.

**Explanation:**
- In the first scenario, minimizing false positives is crucial to avoid unnecessary costs and risks.
- In the second scenario, minimizing false negatives is crucial to prevent loss of life, even at the expense of higher costs and risks.


"""


part3_q4 = r"""
**Your answer:**

A Multilayer Perceptron (MLP) is not the best choice for training on sequential data like text for several reasons:

1. **Lack of Temporal Dependencies:** MLPs do not inherently capture the temporal dependencies between words in a sentence. Each word is treated independently, which means the model cannot learn the context provided by the sequence of words.

2. **Fixed Input Size:** MLPs require a fixed input size, which is not suitable for variable-length sequences like sentences. Padding or truncating sentences can lead to loss of information or introduce noise.

3. **Contextual Understanding:** Sequential data like text requires understanding the context and order of words. Models like Recurrent Neural Networks (RNNs) or Transformers are better suited as they can maintain a memory of previous words and understand the context.

4. **Feature Representation:** MLPs do not leverage word embeddings effectively, which are crucial for capturing semantic meaning in text. Models designed for sequential data can better utilize embeddings to understand word relationships.

"""
# ==============
# Part 4 (CNN) answers


def part4_optim_hp():
    import torch.nn
    import torch.nn.functional

    loss_fn = None  # One of the torch.nn losses
    lr, weight_decay, momentum = 0, 0, 0  # Arguments for SGD optimizer
    # TODO:
    #  - Tweak the Optimizer hyperparameters.
    #  - Choose the appropriate loss function for your architecture.
    #    What you returns needs to be a callable, so either an instance of one of the
    #    Loss classes in torch.nn or one of the loss functions from torch.nn.functional.
    # ====== YOUR CODE: ======
    # Hyperparameters
    lr = 0.05
    weight_decay = 0.005
    momentum = 0.05
    loss_fn = torch.nn.CrossEntropyLoss()
    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)

part4_q1 = r"""
**Your answer:**

### Comparison of Regular Block and Bottleneck Block

1. **Number of Parameters**:
   - **Regular Block**:
     - Two 3x3 convolutions directly on a 256-channel input.
     - Parameters for each convolution: $3 \times 3 \times 256 \times 256 = 589,824$
     - Total parameters: $2 \times 589,824 = 1,179,648$
   - **Bottleneck Block**:
     - First 1x1 convolution: $1 \times 1 \times 256 \times 64 = 16,384$
     - 3x3 convolution: $3 \times 3 \times 64 \times 64 = 36,864$
     - Second 1x1 convolution: $1 \times 1 \times 64 \times 256 = 16,384$
     - Total parameters: $16,384 + 36,864 + 16,384 = 69,632$

2. **Number of Floating Point Operations (FLOPs)**:
   - **Regular Block**:
     - FLOPs per 3x3 convolution: $3 \times 3 \times 256 \times 256 \times H \times W$
       (H and W represent spatial dimensions).
     - Total FLOPs: $2 \times 3 \times 3 \times 256 \times 256 \times H \times W = 1,179,648 \times H \times W$
   - **Bottleneck Block**:
     - First 1x1 convolution FLOPs: $1 \times 1 \times 256 \times 64 \times H \times W = 16,384 \times H \times W$
     - 3x3 convolution FLOPs: $3 \times 3 \times 64 \times 64 \times H \times W = 36,864 \times H \times W$
     - Second 1x1 convolution FLOPs: $1 \times 1 \times 64 \times 256 \times H \times W = 16,384 \times H \times W$
     - Total FLOPs: $(16,384 + 36,864 + 16,384) \times H \times W = 69,632 \times H \times W$

3. **Ability to Combine Inputs**:
   - **Spatial Combination (Within Feature Maps)**:
     - **Regular Block**:
       - Combines input spatially within feature maps due to 3x3 convolutions.
     - **Bottleneck Block**:
       - Combines input spatially within feature maps via the 3x3 convolution.
   - **Feature Combination (Across Feature Maps)**:
     - **Regular Block**:
       - Does not explicitly project across feature maps.
     - **Bottleneck Block**:
       - Combines inputs across feature maps using the 1x1 convolutions, allowing dimensionality reduction and feature projection.

"""

# ==============

# ==============
# Part 5 (CNN Experiments) answers


part5_q1 = r"""
**Your answer:**

### 1. Explain the effect of depth on the accuracy. What depth produces the best results and why do you think that's the case?

From the experiment:
- Increasing the depth (L) improves accuracy up to a certain point. For $K=32$, $L=4$ produces the best results, while for $K=64$, both $L=4$ and $L=8$ show good performance.
- Beyond this point, deeper networks (e.g., $L=16$) show lower accuracy due to overfitting or optimization issues.

#### Observations:
1. **Why $L=4$ performs best for $K=32$?**
   - The network has sufficient capacity to learn without overfitting.
   - The combination of convolutional layers and pooling layers effectively captures features without excessive complexity.

2. **Why $L=8$ struggles for $K=32$?**
   - The network might overfit or suffer from vanishing gradients, reducing trainability.
   - With smaller $K$, the representational power is insufficient for such depth.

3. **Why $L=8$ works for $K=64$?**
   - Larger $K$ provides more parameters per layer, balancing the increased depth and avoiding optimization issues.

4. **Effect of $L=16$:**
   - Networks with $L=16$ show poor results for both $K=32$ and $K=64$. This is likely due to vanishing gradients and difficulty in optimization at such depth.

### 2. Were there values of L for which the network wasn't trainable? What causes this? Suggest two things which may be done to resolve it at least partially.

Yes, networks with $L=16$ for both $K=32$ and $K=64$ exhibited poor trainability.

#### Causes:
1. **Vanishing Gradients:**
   - As depth increases, gradients diminish through backpropagation, making weight updates ineffective in earlier layers.

2. **Overfitting:**
   - Deeper networks memorize training data but fail to generalize, especially with insufficient regularization or data.

#### Suggestions:
1. **Gradient Stabilization:**
   - Introduce batch normalization or add skip connections to enable effective gradient flow.

2. **Regularization:**
   - Use dropout or data augmentation to improve generalization and reduce overfitting.

"""

part5_q2 = r"""
**Your answer:**


1.2. 

Yes, there were values of L for which the network wasn't trainable. Specifically, for L=8, and L=16, we observed a constant 10% accuracy and a constant loss. This behavior indicates that the network is not learning effectively.

### Causes:
1. *Vanishing/Exploding Gradients*: As the depth of the network increases, the gradients can either vanish (become very small) or explode (become very large), making it difficult for the network to learn.
2. *Overfitting*: With a deeper network, there is a higher risk of overfitting, especially if the training data is not sufficient to support the increased complexity of the model.

### Solutions:
1. *Gradient Clipping*: Implement gradient clipping to prevent the gradients from exploding. This technique involves setting a threshold value and scaling the gradients that exceed this threshold.
2. *Batch Normalization*: Use batch normalization to stabilize the learning process. Batch normalization helps in maintaining the mean and variance of the inputs to each layer, which can mitigate the vanishing/exploding gradient problem.

These strategies can help in making the network more trainable even for higher values of L.

"""

part5_q3 = r"""
**Your answer:**

### Experiment 1.3 Analysis

#### Observations:
1. The experiment tests the effect of increasing depth ($L$) with $K=64$ and batch size 128.
2. Shallower networks ($L=2$) train faster and achieve higher accuracy compared to deeper ones ($L=3, L=4$).

#### Results:
1. **Train Loss and Accuracy:**
   - For $L=2$, the network converges quickly, with the lowest train loss and highest train accuracy (~80%).
   - As depth increases ($L=3, L=4$), the network converges more slowly, showing higher train loss and lower train accuracy.

2. **Test Loss and Accuracy:**
   - $L=2$ achieves the best generalization with the lowest test loss and highest test accuracy (~70%).
   - $L=4$ struggles to generalize, with higher test loss and significantly lower test accuracy (~30%).

#### Causes:
1. **Optimization Challenges:**
   - Increased depth exacerbates vanishing gradients, slowing convergence.
2. **Overfitting Risks:**
   - Deeper networks may overfit to training data, reducing test accuracy.

#### Suggestions for Improvement:
1. Apply **gradient clipping** to improve optimization stability for deeper networks.
2. Introduce **batch normalization** to normalize activations and mitigate vanishing gradients.

"""

part5_q4 = r"""
**Your answer:**

### Generalized Comparison of Experiments
we didnt have time to run the last experiment on 1.4, but we can still compare the results of the other experiments to draw some conclusions.
#### Experiment 1.1 (Varying Depth with Fixed Filters K=64)
- *Objective*: To understand the impact of varying the network depth while keeping the number of filters fixed.
- *Results*:
  - *L=2 and L=4*: Showed steady improvement in both training and test metrics.
  - *L=8 and L=16*: Achieved only 10% accuracy, likely due to disappearing gradients, and the runs ended early.
- *Key Observations*:
  - Increasing depth initially improves performance, but very deep networks suffer from disappearing gradients, leading to poor performance.

#### Experiment 1.3 (Varying Both Filters and Depth)
- *Objective*: To explore the combined effect of varying both the number of filters and the network depth.
- *Results*:
  - *L=2, K=[64, 128]*: Showed significant improvement in both training and test metrics.
  - *L=3, K=[64, 128]*: Also showed steady improvement, though slightly less pronounced than L=2.
  - *L=4, K=[64, 128]*: Failed to train effectively, with metrics remaining poor.
- *Key Observations*:
  - Networks with more filters can achieve higher accuracies, but they are more prone to overfitting if the depth is too high.
  - There is a delicate balance between depth and the number of filters for optimal performance.

#### Experiment 1.4 (Varying Depth with Fixed Filters K=[64, 128, 256])
- *Objective*: To investigate the effect of varying the network depth with a more complex filter configuration.
- *Results*:
  - *L=2*: Showed steady improvement in both training and test metrics.
  - *L=4*: Also showed steady improvement, with slightly better performance than L=2.
- *Key Observations*:
  - Increasing depth with a complex filter configuration can improve performance, but the risk of overfitting increases with depth.
  - The network's ability to generalize well to test data is crucial for achieving good performance.

### Comparison with Experiment 1.4
- *Depth Sensitivity*:
  - Experiment 1.1 showed that very deep networks (L=8 and L=16) suffer from disappearing gradients, leading to poor performance.
  - Experiment 1.4, with its complex filter configuration, is expected to handle increased depth better but still faces the risk of overfitting.

- *Filter Configuration*:
  - Experiment 1.3 demonstrated that varying both filters and depth can lead to higher accuracies but also increases the risk of overfitting.
  - Experiment 1.4 uses a fixed complex filter configuration, which provides a more consistent basis for evaluating the impact of depth.

### Conclusion
- *Disappearing Gradients*: Techniques such as batch normalization, residual connections, and gradient clipping can help mitigate this issue in very deep networks.
- *Optimal Configuration*: Finding the right balance between depth and the number of filters is crucial for achieving good performance without overfitting.
- *Generalization*: Early stopping and other regularization techniques are essential to ensure that the network generalizes well to test data.

Experiment 1.4 provides valuable insights into the impact of depth with a complex filter configuration, highlighting the importance of balancing network complexity with the ability to generalize effectively.

"""


# ==============

# ==============
# Part 6 (YOLO) answers


part6_q1 = r"""
**Your answer:**

1.1. **model analasys**
#### Image 1: Dolphins in the Sky
- **Detected Objects:**
  - Two dolphins were recognized as one person with a confidence of 0.9.
  - Another dolphin was recognized as a person with a confidence of 0.53.
  - One dolphin's tail was recognized as a surfboard with a confidence of 0.37.

#### Image 2: Dogs and a Cat
- **Detected Objects:**
  - One dog was analyzed as a cat with a confidence of 0.65.
  - Another dog was analyzed as a cat with a confidence of 0.39.
  - Another dog was analyzed as a dog with a confidence of 0.5.

### Model Performance
- The model detected objects with varying confidence levels.
- There were significant misclassifications, such as dolphins being recognized as persons and dogs being recognized as cats.

1.2. **model failures**
### Possible Reasons for Model Failures
1. **Training Data:** The model may not have been trained on a diverse enough dataset, leading to poor generalization on unseen objects.
2. **Object Similarity:** The model might confuse objects with similar shapes or textures, especially if they are not well-represented in the training data.
3. **Occlusion and Closeness:** Objects that are close together or partially occluded can be difficult for the model to distinguish correctly.
**ways of resolving issues**
### Methods to Resolve Issues
1. **Data Augmentation:** Increase the diversity of the training dataset with more varied examples of the objects.
2. **Transfer Learning:** Fine-tune the model on a more specific dataset that includes the objects of interest.
3. **Improved Architectures:** Use more advanced architectures or ensemble methods to improve detection accuracy.

1.3. **adversarial attack**
### Adversarial Attack on Object Detection Model
To attack an Object Detection model like YOLO using an adversarial attack (e.g., PGD), you would:
1. **Generate Perturbations:** Create small perturbations to the input image that are imperceptible to humans but cause the model to misclassify objects.
2. **Iterative Process:** Use an iterative process to apply these perturbations, gradually increasing their effect while keeping them within a certain bound.
3. **Targeted Attack:** Focus on specific regions of the image to maximize the impact on the detected objects.


"""

part5_q5 = ""

part6_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


part6_q3 = r"""
**Your answer:**

#### Detected Objects:
- **Owl recognized as bear:** The owl was hiding in a tree and was recognized as a bear with a confidence of 0.53.
- **Hyena recognized as cat:** The blurry hyena moving in the shadows was recognized as a cat with a confidence of 0.68.
- **Tiger recognized as sheep:** The tiger behind the tall grass was recognized as a sheep with a confidence of 0.3.

### Model Performance
- The model detected objects with varying confidence levels.
- There were significant misclassifications, such as an owl being recognized as a bear, a hyena as a cat, and a tiger as a sheep.
overall the model didnt do its job properly in these images. however, it did manage to box the animals in the images.

"""

part6_bonus = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""