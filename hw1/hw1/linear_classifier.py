import torch
from torch import Tensor
from collections import namedtuple
from torch.utils.data import DataLoader

from .losses import ClassifierLoss


class LinearClassifier(object):
    def __init__(self, n_features, n_classes, weight_std=0.001):
        """
        Initializes the linear classifier.
        :param n_features: Number or features in each sample.
        :param n_classes: Number of classes samples can belong to.
        :param weight_std: Standard deviation of initial weights.
        """
        self.n_features = n_features
        self.n_classes = n_classes

        # TODO:
        #  Create weights tensor of appropriate dimensions
        #  Initialize it from a normal dist with zero mean and the given std.

        self.weights = None
        # ====== YOUR CODE: ======
        self.weights = torch.randn(n_features, n_classes) * weight_std
        # ========================

    def predict(self, x: Tensor):
        """
        Predict the class of a batch of samples based on the current weights.
        :param x: A tensor of shape (N,n_features) where N is the batch size.
        :return:
            y_pred: Tensor of shape (N,) where each entry is the predicted
                class of the corresponding sample. Predictions are integers in
                range [0, n_classes-1].
            class_scores: Tensor of shape (N,n_classes) with the class score
                per sample.
        """

        # TODO:
        #  Implement linear prediction.
        #  Calculate the score for each class using the weights and
        #  return the class y_pred with the highest score.

        y_pred, class_scores = None, None
        # ====== YOUR CODE: ======
        # Calculate class scores
        class_scores = x @ self.weights  # Matrix multiplication: (N, n_features) x (n_features, n_classes)

        # Predict class with highest score
        y_pred = torch.argmax(class_scores, dim=1)  # Select class index with the maximum score for each sample

        # ========================

        return y_pred, class_scores

    @staticmethod
    def evaluate_accuracy(y: Tensor, y_pred: Tensor):
        """
        Calculates the prediction accuracy based on predicted and ground-truth
        labels.
        :param y: A tensor of shape (N,) containing ground truth class labels.
        :param y_pred: A tensor of shape (N,) containing predicted labels.
        :return: The accuracy in percent.
        """

        # TODO:
        #  calculate accuracy of prediction.
        #  Do not use an explicit loop.

        acc = None
        # ====== YOUR CODE: ======
        correct_predictions = (y == y_pred).sum()  # Count matching predictions
        acc = correct_predictions.item() / y.size(0)  # Divide by total samples
        # ========================

        return acc * 100

    def train(
        self,
        dl_train: DataLoader,
        dl_valid: DataLoader,
        loss_fn: ClassifierLoss,
        learn_rate=0.1,
        weight_decay=0.001,
        max_epochs=100,
    ):

        Result = namedtuple("Result", "accuracy loss")
        train_res = Result(accuracy=[], loss=[])
        valid_res = Result(accuracy=[], loss=[])

        print("Training", end="")
        for epoch_idx in range(max_epochs):
            total_correct = 0
            average_loss = 0

            # TODO:
            #  Implement model training loop.
            #  1. At each epoch, evaluate the model on the entire training set
            #     (batch by batch) and update the weights.
            #  2. Each epoch, also evaluate on the validation set.
            #  3. Accumulate average loss and total accuracy for both sets.
            #     The train/valid_res variables should hold the average loss
            #     and accuracy per epoch.
            #  4. Don't forget to add a regularization term to the loss,
            #     using the weight_decay parameter.

            # ====== YOUR CODE: ======
            # Training loop
            for x_batch, y_batch in dl_train:
                class_scores = x_batch @ self.weights  # Compute class scores
                y_predicted = torch.argmax(class_scores, dim=1)  # Predicted class labels
                loss = loss_fn(x_batch, y_batch, class_scores, y_predicted)  # Compute loss

                # Add regularization term
                reg_loss = weight_decay * (self.weights ** 2).sum()
                loss += reg_loss

                # Compute gradient and update weights
                grad = x_batch.T @ (torch.softmax(class_scores, dim=1) - \
                                    torch.nn.functional.one_hot(y_batch, num_classes=self.weights.size(1))) / x_batch.size(0)
                self.weights -= learn_rate * (grad + weight_decay * self.weights)

                # Update total_correct and average_loss for training
                y_pred = torch.argmax(class_scores, dim=1)
                total_correct += (y_pred == y_batch).sum().item()
                average_loss += loss.item()

            # Record training accuracy and average loss
            train_res.accuracy.append(100 * total_correct / len(dl_train.dataset))
            train_res.loss.append(average_loss / len(dl_train))

            # Validation evaluation
            total_correct = 0
            average_loss = 0

            for x_batch, y_batch in dl_valid:
                class_scores = x_batch @ self.weights  # Compute class scores
                y_predicted = torch.argmax(class_scores, dim=1)  # Predicted class labels
                loss = loss_fn(x_batch, y_batch, class_scores, y_predicted)  # Compute loss

                # Update total_correct and average_loss for validation
                y_pred = torch.argmax(class_scores, dim=1)
                total_correct += (y_pred == y_batch).sum().item()
                average_loss += loss.item()

            # Record validation accuracy and average loss
            valid_res.accuracy.append(100 * total_correct / len(dl_valid.dataset))
            valid_res.loss.append(average_loss / len(dl_valid))
            # ========================
            print(".", end="")

        print("")
        return train_res, valid_res

    def weights_as_images(self, img_shape, has_bias=True):
        """
        Create tensor images from the weights, for visualization.
        :param img_shape: Shape of each tensor image to create, i.e. (C,H,W).
        :param has_bias: Whether the weights include a bias component
            (assumed to be the first feature).
        :return: Tensor of shape (n_classes, C, H, W).
        """

        # TODO:
        #  Convert the weights matrix into a tensor of images.
        #  The output shape should be (n_classes, C, H, W).

        # ====== YOUR CODE: ======
        weights = self.weights

        # Remove the bias term if it exists
        if has_bias:
            weights = weights[1:, :]  # Exclude the first row (bias)
    
        # Reshape weights into image format
        n_classes = weights.size(1)
        w_images = weights.T.reshape(n_classes, *img_shape)  # Reshape to (n_classes, C, H, W)
        # ========================

        return w_images


def hyperparams():
    hp = dict(weight_std=0.0, learn_rate=0.0, weight_decay=0.0)

    # TODO:
    #  Manually tune the hyperparameters to get the training accuracy test
    #  to pass.
    # ====== YOUR CODE: ======
    # Manually tune these values:
    hp["weight_std"] = 0.01  # Standard deviation for weight initialization
    hp["learn_rate"] = 0.01  # Learning rate
    hp["weight_decay"] = 0.01  # Regularization strength
    # ========================

    return hp
