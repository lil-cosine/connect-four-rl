from game_runner import evaluate
from agent import greedy_agent
from network import ActorCriticNet
import torch

def load(path):
    net = ActorCriticNet()
    net.load_state_dict(torch.load(path, map_location="cuda"))
    net.eval()
    return greedy_agent(net)

new = load("checkpoints/ckpt_18500.pt")
old = load("checkpoints/ckpt_0.pt")

score = evaluate(new, [old], games_per_opponent=50)
print(f"Score vs checkpoint 0: {score:.2f}")
# +1.0 = always wins, 0.0 = even, -1.0 = always loses
