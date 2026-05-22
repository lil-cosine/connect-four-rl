# train.py

import os

import torch
import torch.nn.functional as F

from tqdm import tqdm

from network import ActorCriticNet
from agent import A2CAgent, greedy_agent
from selfplay import collect_episode
from game_runner import evaluate


GAMMA = 0.99
LR = 3e-3

C_VALUE = 0.6
C_ENT = 0.02

EPISODES = 2_000_000

EVAL_EVERY = 500
SAVE_EVERY = 5000


def compute_returns(rewards):

    G = 0
    returns = []

    for r in reversed(rewards):
        G = r + GAMMA * G
        returns.insert(0, G)

    return torch.tensor(
        returns,
        dtype=torch.float32
    )


def load_oldest_checkpoint(device):

    checkpoints = sorted([
        f for f in os.listdir("./checkpoints")
        if f.startswith("ckpt_") and f.endswith(".pt")
    ])

    if not checkpoints:
        return None

    old_net = ActorCriticNet().to(device)

    old_net.load_state_dict(
        torch.load(
            "checkpoints/" + checkpoints[0],
            map_location=device
        )
    )

    old_net.eval()

    return greedy_agent(old_net, device)


def train():

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    net = ActorCriticNet().to(device)

    opt = torch.optim.Adam(
        net.parameters(),
        lr=LR,
        weight_decay=1e-4
    )

    agent = A2CAgent(
        net,
        temperature=1.0,
        device=device
    )

    for ep in tqdm(range(EPISODES)):

        transitions = collect_episode(agent)

        log_probs = torch.stack([
            t[0] for t in transitions
        ])

        values = torch.stack([
            t[1] for t in transitions
        ]).squeeze(-1)

        rewards = [
            t[2] for t in transitions
        ]

        entropies = torch.stack([
            t[3] for t in transitions
        ])

        returns = compute_returns(rewards).to(device)

        advantage = returns - values.detach()

        # normalize advantage only
        if len(advantage) > 1:
            advantage = (
                advantage - advantage.mean()
            ) / (
                advantage.std() + 1e-8
            )

        loss_policy = -(
            log_probs * advantage
        ).mean()

        loss_value = F.mse_loss(
            values,
            returns
        )

        loss_entropy = entropies.mean()

        loss = (
            loss_policy
            + C_VALUE * loss_value
            - C_ENT * loss_entropy
        )

        opt.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            net.parameters(),
            max_norm=0.5
        )

        opt.step()

        if ep % SAVE_EVERY == 0 and ep > 0:

            torch.save(
                net.state_dict(),
                f"checkpoints/ckpt_{ep}.pt"
            )

        if ep % EVAL_EVERY == 0:

            opponent = load_oldest_checkpoint(device)

            if opponent is None:
                opponent = greedy_agent(net, device)

            score = evaluate(
                greedy_agent(net, device),
                [opponent],
                20
            )

            print(
                f"ep {ep:,} "
                f"— score vs oldest ckpt: {score:.2f}"
            )


if __name__ == "__main__":
    train()
