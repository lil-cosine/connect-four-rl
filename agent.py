import torch
import torch.nn.functional as F

class A2CAgent:
    def __init__(self, network, temperature=1.0, device="cuda"):
        self.net    = network
        self.temp   = temperature
        self.device = device

    def _state_tensor(self, board, player):
        raw = board.get_state(player)
        me  = [1.0 if v == 1.0  else 0.0 for v in raw]
        opp = [1.0 if v == -1.0 else 0.0 for v in raw]
        s = torch.tensor(me + opp, dtype=torch.float32, device=self.device)
        return s.unsqueeze(0)

    def _mask(self, valid_cols):
        mask = torch.full((7,), float("-inf"), device=self.device)
        mask[valid_cols] = 0.0
        return mask

    @torch.no_grad()
    def choose_col(self, board, player):
        self.net.eval()
        logits, _ = self.net(self._state_tensor(board, player))
        logits = logits[0]

        if torch.any(torch.isnan(logits)):
            return board.get_valid_cols()[0]

        logits = logits + self._mask(board.get_valid_cols())

        if torch.all(torch.isinf(logits)):
            return board.get_valid_cols()[0]

        probs = F.softmax(logits / max(self.temp, 1e-6), dim=-1)
        probs = torch.clamp(probs, min=1e-8)
        probs = probs / probs.sum()

        if torch.any(torch.isnan(probs)) or torch.any(probs < 0):
            return board.get_valid_cols()[0]

        return torch.multinomial(probs, 1).item()

    def policy_and_value(self, board, player):
        self.net.train()
        logits, value = self.net(self._state_tensor(board, player))
        logits    = logits[0] + self._mask(board.get_valid_cols())
        log_probs = F.log_softmax(logits, dim=-1)
        probs     = F.softmax(logits, dim=-1)
        return log_probs, probs, value


def greedy_agent(network, device="cpu"):
    return A2CAgent(network, temperature=0.05, device=device)
