import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

from game_utils import BOARD_SHAPE
from .generate_data import MCTS_for_data_generation
from .network import Connect4Model

epochs = 20
batch_size = 128
num_episodes = 100

def generate_self_play_data(num_episodes: int) -> list:
    """
    """
    data = []
    for _ in range(num_episodes):
        episode_data = MCTS_for_data_generation()
        data.extend(episode_data)
    return data

def train(model: Connect4Model, examples: list) -> tuple[Connect4Model, list, list]:
    """
    Trains the neural network by following the standard method
        A. Defining optimizer and loss functions
        B. Shuffle the data
        C. Training loops (for number of epochs)
            1. Choose a sample of boards and assign them to tensors
            2. Assign the policies and values to torches
            3. Make boards, policies and values contiguous (stored in a single sequential chunk)
            4. Run the model and compare with samples (loss function)
            5. Backprop and step in optimizer
            6. Append losses to track them later
    """

    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    loss_pi_function  = nn.CrossEntropyLoss()
    loss_v_function = nn.MSELoss()
    pi_losses = []
    v_losses = []

    num_batches = int(len(examples) / batch_size)

    # Shuffle data
    all_sample_ids = np.arange(len(examples))
    np.random.shuffle(all_sample_ids)
    all_sample_ids = np.array_split(all_sample_ids, num_batches)

    for epoch in range(epochs):
        model.train()

        batch_idx = 0
        while batch_idx < num_batches:
            # 1. Choose a sample of boards and assign them to tensors
            sample_ids = all_sample_ids[batch_idx]
            boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
            boards = torch.FloatTensor(np.array(boards).astype(np.float64))

            # 2. Assign the policies and values to torches
            target_pis = torch.FloatTensor(np.array(pis))
            target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))

            # 3. Make boards, policies and values contiguous (stored in a single sequential chunk)
            boards = boards.contiguous()
            target_pis = target_pis.contiguous()
            target_vs = target_vs.contiguous()

            # 4. Run the model and compare with samples (loss function)
            out_pi, out_v = model(boards)
            l_pi = loss_pi_function(out_pi, target_pis)
            l_v = loss_v_function(out_v, target_vs)
            total_loss = l_pi + l_v

            # 5. Backprop and step in optimizer
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            batch_idx += 1

        # 6. Append losses
        pi_losses.append(float(l_pi))
        v_losses.append(float(l_v))
    return model, pi_losses, v_losses

def train_network():
    """
    The main method that
        1. Generates data from mcts boards
        2. Creates a Connect4Model
        3. Trains the model
        4. Saves the model for later use
    """
    self_play_data = generate_self_play_data(num_episodes)
    model = Connect4Model(BOARD_SHAPE, 7)
    model, pi_losses, v_losses  = train(model, self_play_data[:200])

    # Save data
    torch.save(model.state_dict(), 'data/model_weights.pth')
    model = Connect4Model(BOARD_SHAPE, 7)
    model.load_state_dict(torch.load('data/model_weights.pth'))
