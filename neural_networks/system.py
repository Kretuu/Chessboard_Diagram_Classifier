"""Baseline classification system.

Solution outline for the COM2004/3004 assignment.

This solution will run but the dimensionality reduction and
the classifier are not doing anything useful, so it will
produce a very poor result.

version: v1.0
"""
from typing import List

import numpy as np
import scipy.linalg
import torch
from torch.utils.data import DataLoader, TensorDataset

N_DIMENSIONS = 10
TENSOR_DEVICE = "cpu"


def classify(train: np.ndarray, train_labels: np.ndarray, test: np.ndarray) -> List[str]:
    """Classify a set of feature vectors using a training set.

    This dummy implementation simply returns the empty square label ('.')
    for every input feature vector in the test data.

    Note, this produces a surprisingly high score because most squares are empty.

    Args:
        train (np.ndarray): 2-D array storing the training feature vectors.
        train_labels (np.ndarray): 1-D array storing the training labels.
        test (np.ndarray): 2-D array storing the test feature vectors.

    Returns:
        list[str]: A list of one-character strings representing the labels for each square.
    """

    # Nearest neighbour classifier implemented "just in case" the other classifier is not working
    # This classifier is additional work done, but I wanted to practice neural networks
    x = np.dot(test, train.transpose())
    modtest = np.sqrt(np.sum(test * test, axis=1))
    modtrain = np.sqrt(np.sum(train * train, axis=1))
    dist = x / np.outer(modtest, modtrain.transpose())
    nearest = np.argmax(dist, axis=1)
    label = train_labels[nearest]

    return label


# The functions below must all be provided in your solution. Think of them
# as an API that it used by the train.py and evaluate.py programs.
# If you don't provide them, then the train.py and evaluate.py programs will not run.
#
# The contents of these functions are up to you but their signatures (i.e., their names,
# list of parameters and return types) must not be changed. The trivial implementations
# below are provided as examples aand will produce a result, but the score will be low.


def reduce_dimensions(data: np.ndarray, model: dict) -> np.ndarray:
    """Reduce the dimensionality of a set of feature vectors down to N_DIMENSIONS.

    The feature vectors are stored in the rows of 2-D array data, (i.e., a data matrix).
    The dummy implementation below simply returns the first N_DIMENSIONS columns.

    Args:
        data (np.ndarray): The feature vectors to reduce.
        model (dict): A dictionary storing the model data that may be needed.

    Returns:
        np.ndarray: The reduced feature vectors.
    """

    reduced_features = np.dot((data - model["mean_train"]), model["pca_vectors"])

    return reduced_features


def feature_selection(fvectors_train: np.ndarray, labels_train: np.ndarray, model: dict):
    # Select 10 eigenvectors which have the biggest eigenvalue
    covariance_matrix = np.cov(fvectors_train, rowvar=0)
    N = covariance_matrix.shape[0]
    _, eigen_vect = scipy.linalg.eigh(covariance_matrix, subset_by_index=(N - 10, N - 1))
    eigen_vect = np.fliplr(eigen_vect)

    fvectors_train_reduced = np.dot((fvectors_train - model["mean_train"]), eigen_vect)

    return eigen_vect


def process_training_data(fvectors_train: np.ndarray, labels_train: np.ndarray) -> dict:
    """Process the labeled training data and return model parameters stored in a dictionary.

    Note, the contents of the dictionary are up to you, and it can contain any serializable
    data types stored under any keys. This dictionary will be passed to the classifier.

    Args:
        fvectors_train (np.ndarray): training data feature vectors stored as rows.
        labels_train (np.ndarray): the labels corresponding to the feature vectors.

    Returns:
        dict: a dictionary storing the model data.
    """

    model = {}
    model["labels_train"] = labels_train.tolist()
    # Store label -> index mapping
    model["label_mapping"] = {label: idx for idx, label in enumerate(np.unique(labels_train))}
    # Store index -> label mapping
    model["reverse_mapping"] = {idx: label for label, idx in model["label_mapping"].items()}

    # Choose best 10 features and perform dimensionality reduction
    model["mean_train"] = np.mean(fvectors_train)

    selected_features = feature_selection(fvectors_train, labels_train, model)
    model["pca_vectors"] = selected_features.tolist()
    model["fvectors_train"] = reduce_dimensions(fvectors_train, model).tolist()

    # Map train data and train labels to tensors. Train labels were mapped to indexes in order to use argmax in
    # neural network
    x_train_tensor = torch.tensor(model["fvectors_train"], dtype=torch.float, device=TENSOR_DEVICE)
    y_train_numbers = np.vectorize(model["label_mapping"].get)(labels_train)
    y_train_tensor = torch.tensor(y_train_numbers, dtype=torch.float, device=TENSOR_DEVICE)

    # Train neural network
    learning_rate = 0.0001
    batch_size = 10
    num_epochs = 50

    dataset = TensorDataset(x_train_tensor, y_train_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    net = NeuralNetwork()

    net.train(loader, num_epochs, learning_rate)
    w_1, b_1, w_2, b_2 = net.return_parameters()
    model["w_1"] = w_1.tolist()
    model["b_1"] = b_1.tolist()
    model["w_2"] = w_2.tolist()
    model["b_2"] = b_2.tolist()

    return model


def images_to_feature_vectors(images: List[np.ndarray]) -> np.ndarray:
    """Takes a list of images (of squares) and returns a 2-D feature vector array.

    In the feature vector array, each row corresponds to an image in the input list.

    Args:
        images (list[np.ndarray]): A list of input images to convert to feature vectors.

    Returns:
        np.ndarray: An 2-D array in which the rows represent feature vectors.
    """
    h, w = images[0].shape
    n_features = h * w
    fvectors = np.empty((len(images), n_features))
    print("Length")
    print(len(images))
    for i, image in enumerate(images):
        fvectors[i, :] = image.reshape(1, n_features)

    return fvectors


def classify_squares(fvectors_test: np.ndarray, model: dict) -> List[str]:
    """Run classifier on a array of image feature vectors presented in an arbitrary order.

    Note, the feature vectors stored in the rows of fvectors_test represent squares
    to be classified. The ordering of the feature vectors is arbitrary, i.e., no information
    about the position of the squares within the board is available.

    Args:
        fvectors_test (np.ndarray): An array in which feature vectors are stored as rows.
        model (dict): A dictionary storing the model data.

    Returns:
        list[str]: A list of one-character strings representing the labels for each square.
    """

    # Convert testing data to tensor and perform classification by trained neural network
    x_test_tensor = torch.tensor(fvectors_test, dtype=torch.float, device=TENSOR_DEVICE)

    w_1 = torch.tensor(model["w_1"], device=TENSOR_DEVICE)
    b_1 = torch.tensor(model["b_1"], device=TENSOR_DEVICE)
    w_2 = torch.tensor(model["w_2"], device=TENSOR_DEVICE)
    b_2 = torch.tensor(model["b_2"], device=TENSOR_DEVICE)
    net = NeuralNetwork(w_1=w_1, b_1=b_1, w_2=w_2, b_2=b_2)

    predictions, _ = net.pred(x_test_tensor)

    # Convert predictions indexes to strings and map index -> label
    predictions = np.array(predictions).astype(str)
    labels = np.vectorize(model["reverse_mapping"].get)(predictions)

    return labels


def classify_boards(fvectors_test: np.ndarray, model: dict) -> List[str]:
    """Run classifier on a array of image feature vectors presented in 'board order'.

    The feature vectors for each square are guaranteed to be in 'board order', i.e.
    you can infer the position on the board from the position of the feature vector
    in the feature vector array.

    In the dummy neural_networks below, we just re-use the simple classify_squares function,
    i.e. we ignore the ordering.

    Args:
        fvectors_test (np.ndarray): An array in which feature vectors are stored as rows.
        model (dict): A dictionary storing the model data.

    Returns:
        list[str]: A list of one-character strings representing the labels for each square.
    """

    # Convert testing data to tensor and perform classification by trained neural network
    x_test_tensor = torch.tensor(fvectors_test, dtype=torch.float, device=TENSOR_DEVICE)

    w_1 = torch.tensor(model["w_1"], device=TENSOR_DEVICE)
    b_1 = torch.tensor(model["b_1"], device=TENSOR_DEVICE)
    w_2 = torch.tensor(model["w_2"], device=TENSOR_DEVICE)
    b_2 = torch.tensor(model["b_2"], device=TENSOR_DEVICE)
    net = NeuralNetwork(w_1=w_1, b_1=b_1, w_2=w_2, b_2=b_2)

    predictions, pred_values = net.pred(x_test_tensor)

    # Convert predictions indexes to strings and map index -> label
    predictions = np.array(predictions).astype(str)
    labels = np.vectorize(model["reverse_mapping"].get)(predictions)

    # Find number of different chessboard (every chessboard has 64 squares)
    N = int(labels.shape[0] / 64)
    for x in range(N):
        # Index of first element from the board
        x = 64 * x

        # Assume that black figures start on the top and whites on the bottom of the board then it is known that
        # black pawns cannot appear in the first row and white pawns cannot appear in the last row of the board
        labels[x:x + 8] = np.where(labels[x:x + 8] == "p", "P", labels[x:x + 8])
        labels[x + 56:x + 64] = np.where(labels[x + 56:x + 64] == "P", "p", labels[x + 56:x + 64])

        # Iterate through all squares of the board and find if there are repeated figures more than two times
        # There is exception for pawns and empty squares. Filter these figures and put them into the filtered map
        board_data = {}
        for y in range(64):
            index = y + x
            current_label = labels[index]
            if current_label == 'p' or current_label == 'P' or current_label == '.':
                continue

            if current_label in board_data.keys():
                current_item_data = {'idx': index, 'prob': pred_values[index]}
                board_label_data = board_data[current_label]
                board_label_data[1].append(current_item_data)
                board_data[current_label] = (board_label_data[0] + 1, board_label_data[1])
            else:
                board_data[current_label] = (1, [{'idx': index, 'prob': pred_values[index]}])

        filtered_board_data = {k: v for k, v in board_data.items() if v[0] > 2}

        # For every filtered figures exclude first two with the highest probability and replace the rest with some
        # figures. Replacement figures were deduced based on looking into the common classifier error
        for label, too_many_classified in filtered_board_data.items():
            too_many_classified_sorted = sorted(too_many_classified[1], key=lambda x: x['prob'], reverse=True)
            for wrong_classified in too_many_classified_sorted[2:]:
                new_label = label
                if label == 'B':
                    if wrong_classified['prob'] > 0.8:
                        new_label = 'P'
                    else:
                        new_label = 'p'
                elif label == 'k':
                    new_label = 'P'
                labels[wrong_classified['idx']] = new_label

    return labels

class NeuralNetwork:
    def __init__(self, w_1=None, b_1=None, w_2=None, b_2=None):
        # The problem is non-linear, so it is necessary to have at least one hidden layer
        inputs = 10
        neurons_net_1 = 7900
        outputs = 13

        if w_1 is None or b_1 is None:
            self.w_1 = torch.rand(inputs, neurons_net_1, device=TENSOR_DEVICE)
            self.b_1 = torch.rand(neurons_net_1, 1, device=TENSOR_DEVICE)
            self.w_2 = torch.rand(neurons_net_1, outputs, device=TENSOR_DEVICE)
            self.b_2 = torch.rand(outputs, 1, device=TENSOR_DEVICE)
        else:
            self.w_1 = w_1
            self.b_1 = b_1
            self.w_2 = w_2
            self.b_2 = b_2

        self.dw_1 = torch.zeros(inputs, neurons_net_1, device=TENSOR_DEVICE)
        self.db_1 = torch.zeros(neurons_net_1, 1, device=TENSOR_DEVICE)
        self.dw_2 = torch.zeros(neurons_net_1, outputs, device=TENSOR_DEVICE)
        self.db_2 = torch.zeros(outputs, 1, device=TENSOR_DEVICE)

    def forward(self, x, train=False):
        x = torch.unsqueeze(x, 2)

        net1_linear_transformation = torch.matmul(torch.transpose(self.w_1, 0, 1), x) + self.b_1
        net1_activation = self.relu(net1_linear_transformation)
        net2_linear_transformation = torch.matmul(torch.transpose(self.w_2, 0, 1), net1_activation) + self.b_2

        # Use softmax function for last layer to find probability of each output
        y_hat = self.softmax(net2_linear_transformation)
        if train:
            return y_hat, net1_activation
        else:
            return y_hat

    def train(self, data, num_epochs, learning_rate):
        for epoch in range(num_epochs):
            for item in data:
                y_labels = self.y_relu_labels(item[1])
                y_hat, net1_activation = self.forward(item[0], train=True)
                loss = torch.sum(self.cross_entropy(y_labels, torch.squeeze(y_hat)))
                print("Loss: ", loss.item())
                self.backward(y_labels, y_hat, net1_activation, item[0])
                self.update(learning_rate=learning_rate)

    def backward(self, y, y_hat, net1_activation, x):
        y_newshape = torch.unsqueeze(y, dim=2)

        x = torch.unsqueeze(x, 2)

        batch_size = x.shape[0]

        # Softmax + cross entropy derivative.
        # These two derivatives cancel each other out, so it's just y_hat - y_newshape
        dnet2 = y_hat - y_newshape
        dnetdw2 = net1_activation

        dw_2 = torch.matmul(dnetdw2, torch.transpose(dnet2, -1, -2))
        db_2 = dnet2

        dnet2da1 = self.w_2
        da1dnet1 = self.relu_derivative(net1_activation)
        dnet1dw1 = x

        da1 = torch.matmul(dnet2da1, dnet2)
        dnet1 = da1 * da1dnet1
        dw_1 = torch.matmul(dnet1dw1, torch.transpose(dnet1, -1, -2))
        db_1 = dnet1

        self.dw_1 = torch.sum(dw_1, dim=0) / batch_size
        self.dw_2 = torch.sum(dw_2, dim=0) / batch_size
        self.db_1 = torch.sum(db_1, dim=0) / batch_size
        self.db_2 = torch.sum(db_2, dim=0) / batch_size

    def update(self, learning_rate):
        self.w_1 = self.w_1 - learning_rate * self.dw_1
        self.w_2 = self.w_2 - learning_rate * self.dw_2
        self.b_1 = self.b_1 - learning_rate * self.db_1
        self.b_2 = self.b_2 - learning_rate * self.db_2

    def pred(self, data):
        pred = self.forward(data)
        probs = []
        ans = []
        for t in pred:
            max_index = t.cpu().numpy().argmax()
            # Save both label index and the corresponding probability (needed for full-board classification)
            probs.append(t[max_index])
            ans.append(max_index)

        return ans, probs

    def return_parameters(self):
        return self.w_1, self.b_1, self.w_2, self.b_2

    @staticmethod
    def softmax(z):
        z = torch.squeeze(z, dim=-1)
        max_z, _ = torch.max(z, dim=1, keepdim=True)
        e_z = torch.exp(z - max_z)
        softmax = e_z / e_z.sum(dim=1, keepdim=True)
        return torch.unsqueeze(softmax, dim=2)

    @staticmethod
    def relu(z):
        return torch.max(z, torch.tensor(0.0, device=TENSOR_DEVICE))

    @staticmethod
    def y_relu_labels(y):
        # Correct class must should be set as 1 and incorrect as 0
        labels = torch.zeros(y.shape[0], 13, device=TENSOR_DEVICE)
        for label in range(len(y)):
            zeros = torch.zeros(13, device=TENSOR_DEVICE)
            zeros[int(y[label])] = 1
            labels[label] = zeros
        return labels

    @staticmethod
    def relu_derivative(z):
        return torch.where(z > 0, torch.tensor(1.0, device=TENSOR_DEVICE), torch.tensor(0.0, device=TENSOR_DEVICE))

    @staticmethod
    def cross_entropy(y, y_hat):
        # Epsilon is used to make numbers more stable (no division by 0)
        epsilon = 1e-15
        return -(y * torch.log(y_hat + epsilon) + (1 - y) * torch.log(1 - y_hat + epsilon))
