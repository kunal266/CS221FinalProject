# CS221 Final Project: Deep Reinforcement Learning for playing Agar.io

Fist install the dependencies in a conda environment:

```shell script
conda create --name agario python=3
conda activate agario
pip install -r requirements.txt
```

Next, install the Agario gym environment:

```shell script
pip install -e gym-agario
```

Then run the RL Agent:

```shell script
python main.py
```

To run our Agar.io clone:

```shell script
python gym-agario/gym_agario/envs/agario.py
```