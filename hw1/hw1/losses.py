import abc
import torch


class ClassifierLoss(abc.ABC):
    """
    Represents a loss function of a classifier.
    """

    def __call__(self, *args, **kwargs):
        return self.loss(*args, **kwargs)

    @abc.abstractmethod
    def loss(self, *args, **kw):
        pass

    @abc.abstractmethod
    def grad(self):
        """
        :return: Gradient of the last calculated loss w.r.t. model
            parameters, as a Tensor of shape (D, C).
        """
        pass


class SVMHingeLoss(ClassifierLoss):
    def __init__(self, delta=1.0):
        self.delta = delta
        self.grad_ctx = {}

    def loss(self, x, y, x_scores, y_predicted):
        """
        Calculates the Hinge-loss for a batch of samples.

        :param x: Batch of samples in a Tensor of shape (N, D).
        :param y: Ground-truth labels for these samples: (N,)
        :param x_scores: The predicted class score for each sample: (N, C).
        :param y_predicted: The predicted class label for each sample: (N,).
        :return: The classification loss as a Tensor of shape (1,).
        """

        assert x_scores.shape[0] == y.shape[0]
        assert y.dim() == 1

        # TODO: Implement SVM loss calculation based on the hinge-loss formula.
        #  Notes:
        #  - Use only basic pytorch tensor operations, no external code.
        #  - Full credit will be given only for a fully vectorized
        #    implementation (zero explicit loops).
        #    Hint: Create a matrix M where M[i,j] is the margin-loss
        #    for sample i and class j (i.e. s_j - s_{y_i} + delta).

        loss = None
        # ====== YOUR CODE: ======
        N, C = x_scores.shape

        # Gather correct class scores: s_{y_i} for each sample
        correct_class_scores = x_scores[torch.arange(N), y].unsqueeze(1)

        # Compute the margin matrix: M[i, j] = s_j - s_{y_i} + delta
        margins = x_scores - correct_class_scores + self.delta

        # Ignore the correct class by zeroing its margin
        margins[torch.arange(N), y] = 0

        # Compute the hinge loss for all margins
        hinge_loss = torch.clamp(margins, min=0)

        # Average over all samples
        loss = hinge_loss.sum() / N
        # ========================

        # TODO: Save what you need for gradient calculation in self.grad_ctx
        # ====== YOUR CODE: ======
        self.grad_ctx = {
            "x": x,
            "y": y,
            "x_scores": x_scores,
            "margins": margins
        }
        # ========================

        return loss

    def grad(self):
        """
        Calculates the gradient of the Hinge-loss w.r.t. parameters.
        :return: The gradient, of shape (D, C).

        """
        # TODO:
        #  Implement SVM loss gradient calculation
        #  Same notes as above. Hint: Use the matrix M from above, based on
        #  it create a matrix G such that X^T * G is the gradient.

        grad = None
        # ====== YOUR CODE: ======
        x = self.grad_ctx["x"]           # Input data (N, D)
        y = self.grad_ctx["y"]           # Ground truth labels (N,)
        x_scores = self.grad_ctx["x_scores"]  # Predicted scores (N, C)
        margins = self.grad_ctx["margins"]    # Margins matrix (N, C)
    
        N, D = x.shape
        _, C = x_scores.shape
    
        # Create the indicator matrix for violated margins: (N, C)
        G = (margins > 0).float()
    
        # For correct classes, adjust G to subtract the contribution from all violating classes
        row_sums = G.sum(dim=1)  # Sum of all violating classes per sample (N,)
        G[torch.arange(N), y] -= row_sums
    
        # Compute the gradient: X^T * G (D, C)
        grad = x.T @ G
    
        # Average over the number of samples
        grad /= N
        # ========================

        return grad
