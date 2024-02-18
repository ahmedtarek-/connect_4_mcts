import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class Connect4Model(nn.Module):
    """
    A class that represents the neural network of the connect 4 model
        - It contains a convolution layer
        - It contains a residual block (for diminishing gradient problem)
        - It contains two heads:
            - Action head
            - Value head

    Attributes
    ----------
        - size (tuple): The size of the Connect 4 board represented as (rows, columns).
        - action_size (int): The number of possible actions in the Connect 4 game.
        - conv (nn.Sequential): Convolutional layer to process input features.
        - res_blocks (nn.ModuleList): List of residual blocks for addressing the vanishing gradient problem.
        - action_head (nn.Sequential): Head for predicting action probabilities.
        - value_head (nn.Sequential): Head for predicting the value of a given board state.
    
    Methods
    -------
        - forward(self, x): Performs a forward pass through the neural network.
        - predict(self, board): Makes predictions (action probabilities and value) for a given board state.
    """

    def __init__(self, board_size, action_size):
        super(Connect4Model, self).__init__()

        self.size = board_size
        self.action_size = action_size
        n_filters = 128
        n_res_blocks = 8
        self.conv = nn.Sequential(
            nn.Conv2d(3, n_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(n_filters),
            nn.ReLU()
        )
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(n_filters) for _ in range(n_res_blocks)]
        )

        self.action_head = nn.Sequential(
            nn.Conv2d(n_filters, n_filters//4, kernel_size=3, padding=1),
            nn.BatchNorm2d(n_filters//4),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(n_filters//4 * self.size[0] * self.size[1], self.action_size)
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(n_filters, n_filters//32, kernel_size=3, padding=1),
            nn.BatchNorm2d(n_filters//32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(n_filters//32 * self.size[0] * self.size[1], 1),
            nn.Tanh()
        )

    def forward(self, x):
        """
        Performs a forward pass through the neural network.

        Parameters
        ----------
            - x (torch.Tensor): Input tensor representing the Connect 4 board state.

        Returns
        -------
            - tuple: A tuple containing action probabilities and the predicted value.
        """
        x = F.relu(self.conv(x))

        for block in self.res_blocks:
            x = block(x)
        action_logits = self.action_head(x)
        value_logit = self.value_head(x)
        value = torch.tanh(value_logit)

        return F.softmax(action_logits, dim=1), value

    def predict(self, board):
        """
        A prediction method
        """
        board = torch.FloatTensor(board.astype(np.float32)).to(self.device)
        board = board.view(1, self.size)
        self.eval()
        with torch.no_grad():
            pi, v = self.forward(board)

        return pi.data.cpu().numpy()[0], v.data.cpu().numpy()[0]

class ResidualBlock(nn.Module):
    """
    A class that represents a residual block to be used in the Connect4Model
    it inherits from nn.Module

    Attributes
    ----------
        - conv_1 (nn.Conv2d): First convolutional layer in the residual block.
        - batch_norm_1 (nn.BatchNorm2d): Batch normalization layer after the first convolution.
        - conv_2 (nn.Conv2d): Second convolutional layer in the residual block.
        - batch_norm_2 (nn.BatchNorm2d): Batch normalization layer after the second convolution.
        - relu (nn.ReLU): Rectified Linear Unit activation function.

    Methods
    -------
        - forward(self, x): Performs a forward pass through the residual block.
    """

    def __init__(self, n_filters):
        super().__init__()
        self.conv_1 = nn.Conv2d(n_filters, n_filters, kernel_size=3, padding=1)
        self.batch_norm_1 = nn.BatchNorm2d(n_filters)
        self.conv_2 = nn.Conv2d(n_filters, n_filters, kernel_size=3, padding=1)
        self.batch_norm_2 = nn.BatchNorm2d(n_filters)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Performs a forward pass through the residual block.

        Parameters
        ----------
            - x (torch.Tensor): Input tensor to the residual block.

        Returns
        -------
            - torch.Tensor: Output tensor after the forward pass through the residual block.
        """
        output = self.relu(self.batch_norm_1(self.conv_1(x)))
        output = self.batch_norm_2(self.conv_2(output))
        return self.relu(output + x)
