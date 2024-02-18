# Connect 4 using Monte Carlo Tree Search powered by a neural network
https://github.com/ahmedtarek-/connect_4_mcts

_This project is a collaborative effort by Taisiia Tikhomirova, Myriam Hamon, and Tarek Abdalfatah, who are second-year BCCN Berlin Masters students. The project is focused on the development of a Connect-4 playing agent._

## Project Goals
The primary objectives of this project include:
- Phase 1
    - Implement a Connect4 Game, with a simple human vs human implementation as well as a human vs minimax agent implementation
- Phase 2
    - Deepening Knowledge of game agents: To enhance our understanding of network-based agents through the exploration of the Alpha Zero generic model. This involves recreating a simplified version of such a network by incorporating elements that inspired us.
    - Functional Agent: To develop a functioning agent capable of playing Connect-4 in a logical and "non-stupid" manner.

## Project Overview
Our project draws inspiration from AlphaZero, a revolutionary artificial intelligence developed by DeepMind. Our approach to developing the Connect-4 playing agent is outlined as follows:

- Monte Carlo Tree Search (MCTS): We began with the implementation of a vanilla MCTS algorithm. We used it to generate a data corpus of approximately 21,000 board states. These states are accompanied by the probability distribution of moves and the resulting value of play (-1 for a loss and 1 for a win). 
    -> agent_mcts package -> mcts.py, mcts_node.py
- Residual Neural Network with Convolutional Layers (ResNet): The generated data corpus was then utilized to train a ResNet. This step was inspired by the methodologies employed in the Alpha Zero model.
    -> agent_mcts_nn package  -> network.py , train_network.py , generate_data.py
- Enhanced MCTS: We further developed a variant of the original MCTS class, integrating model predictions to inform the value assessments and policy for selecting nodes 
    -> agent_mcts_nn package -> mcts_nn.py, mcts_node.py

## Project Structure
- `main.py` will run the game in a manner ‘human’ agent against enhanced MCTS (localized in the agents/agent_mcts_nn packages)
- All games will use the game_utils.py file that defines basic functions and variables (e.g size of the board or printing of the board)
- In order to run main.py with the enhanced MCTS agent it is necessary to run the network trainer train_mcts_network.py first, 
    which will generate data (with agent_mcts_nn/generate_data.py) and train the neural network’s weights (agent_mcts_nn/train_network.py).
- You can find step by step to run the agent in the last section
    

## Coding conventions

- For the classes we used a more elaborate docstring to describe the attributes and the
    methods of this class. Example:
    ```python
    """
    A class that represents a node of a monte carlo tree search algorithm
    a node typically has a parent and children

    Attributes
    ----------
        - action: The action taken to reach this node.

    Methods
    -------
        - get_available_actions(self): Returns an array of available actions.
    """
    ```
- For the functions we mainly relied on type hinting to infer the paramters and
    return types of the function and leaving the docstring for a small description of
    the functionality. Example:
    ```python
    def find_best_move(root: MonteCarloTreeSearchNode) -> int:
    """
    Finds the best move given a root node
    """
    ```

## To run the project

1. Install requirements.txt using pip or your favorite package manager (we used conda)

2. Run tests
```bash
python run_tests.py
```

3. Run the network trainer
```bash
python train_mcts_network.py
```

4. Run the main
```bash
python main.py
```

5. To change playing mode, change line 10 in `main.py`
    - human_user_move  -> For two players
    - minmax_user_move -> For playing against computer
