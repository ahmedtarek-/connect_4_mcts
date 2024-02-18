# Connect 4 using Monte Carlo Tree Search powered by a neural network
https://github.com/ahmedtarek-/connect_4_mcts

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

## To run:

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
    - random_user_move -> For random moves
    - minmax_user_move -> For playing against computer
