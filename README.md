# Connect 4 using Monte Carlo Tree Search powered by a neural network
https://github.com/ahmedtarek-/connect_4_mcts

#### To run:

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
