"""Phase 3, step 3, stage 2: the ball model and its loss. Trained by v1/ball_train.py.

Our own build from the paper, Kim et al., KDD 2023 (arxiv.org/abs/2306.08206), no code taken from
its repository. Part 1 guesses who has the ball, part 2 guesses where the ball is from that. Both
read the window forwards and backwards, so a gap is filled using both of its ends.

Where we differ from the paper, because we have one match and track slots, not known players:
  - one set encoder over everybody, with team as an input, not one per team;
  - a slot may be absent for part of a window: it is masked, and the LSTM is told it is absent;
  - the known ball goes in as a channel with a known flag, to every player and to part 2;
  - one "nobody / out" class, not four points on the lines. Small sizes, one LSTM layer each.
The paper prints the MSE and the reality loss with a leading minus. Minimising that would reward
error, against the paper's own words, so both are added here.

Our own addition: with on_line, part 2 predicts a correction to the straight line between the known
ball points, not the ball itself. The correction starts at zero, so an untrained model is baseline A.
"""

import torch
from torch import nn

# Params
PLAYER_INPUTS = 7        # see PLAYER_FEATURES in v1/ball_dataset.py
BALL_INPUTS = 3          # x, y, known
DIM = 32                 # size of a player's description, far below the paper's: we have one match
HEADS = 2
SET_BLOCKS = 2
PLAYER_LSTM = 32
BALL_LSTM = 64
DROPOUT = 0.2            # the paper's
CE_WEIGHT = 20.0         # the paper's: brings who-has-the-ball to the scale of the position error
REALITY_WEIGHT = 1.0     # the paper's
KNOWN_WEIGHT = 0.2       # steps with the ball shown count less: copying them is the easy part
ANGLE_EDGE = 1e-6        # keeps arccos away from +-1, where its slope is infinite
ABSENT_LOGIT = -1e9


class SetBlock(nn.Module):
    """Every present player looks at every other. Swapping two players swaps their outputs only."""

    def __init__(self, dim):
        super().__init__()
        self.attend = nn.MultiheadAttention(dim, HEADS, batch_first=True)
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.feed = nn.Sequential(nn.Linear(dim, dim * 2), nn.ReLU(), nn.Linear(dim * 2, dim))

    def forward(self, x, absent):
        x = self.norm1(x + self.attend(x, x, x, key_padding_mask=absent, need_weights=False)[0])
        return self.norm2(x + self.feed(x))


class BallModel(nn.Module):
    def __init__(self, on_line=True):
        super().__init__()
        self.on_line = on_line
        inputs = PLAYER_INPUTS + BALL_INPUTS + 1  # +1: is this slot present
        self.embed = nn.Linear(inputs, DIM)
        self.blocks = nn.ModuleList(SetBlock(DIM) for _ in range(SET_BLOCKS))
        self.player_lstm = nn.LSTM(inputs + 2 * DIM, PLAYER_LSTM, batch_first=True, bidirectional=True)
        self.holder = nn.Linear(2 * PLAYER_LSTM, 1)
        self.nobody = nn.Linear(2 * PLAYER_LSTM, 1)
        self.to_set = nn.Linear(inputs + 2 * PLAYER_LSTM + 1, DIM)
        self.seed = nn.Parameter(torch.zeros(1, 1, DIM))
        self.pool = nn.MultiheadAttention(DIM, HEADS, batch_first=True)
        self.ball_lstm = nn.LSTM(DIM + BALL_INPUTS + 1, BALL_LSTM, batch_first=True, bidirectional=True)
        self.out = nn.Linear(2 * BALL_LSTM, 2)
        self.drop = nn.Dropout(DROPOUT)
        if on_line:
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)

    def part1(self, x, present):
        """Who has the ball. Returns (logits over slots + nobody, per-player memory), both per step."""
        b, t, n, _ = x.shape
        absent = ~present.reshape(b * t, n)
        empty = absent.all(dim=1)
        if empty.any():
            absent = absent.clone()
            absent[empty, 0] = False  # attention needs somebody to look at; the slot still counts as absent
        e = self.embed(x).reshape(b * t, n, DIM)
        for block in self.blocks:
            e = block(e, absent)
        e = e.reshape(b, t, n, DIM) * present[..., None]
        everyone = e.sum(dim=2, keepdim=True) / present.sum(dim=2, keepdim=True).clamp(min=1)[..., None]
        seq = torch.cat([x, e, everyone.expand(-1, -1, n, -1)], dim=-1)
        h = self.player_lstm(seq.permute(0, 2, 1, 3).reshape(b * n, t, -1))[0]
        h = self.drop(h).reshape(b, n, t, -1).permute(0, 2, 1, 3)
        logits = self.holder(h).squeeze(-1).masked_fill(~present, ABSENT_LOGIT)
        mean_h = (h * present[..., None]).sum(dim=2) / present.sum(dim=2, keepdim=True).clamp(min=1)
        return torch.cat([logits, self.nobody(mean_h)], dim=-1), h

    def part2(self, x, present, ball, h, probs):
        """Where the ball is: a summary of the players per step, read both ways along the window."""
        b, t, n, _ = x.shape
        z = torch.relu(self.to_set(torch.cat([x, h, probs[..., :n, None]], dim=-1)))
        absent = ~present.reshape(b * t, n)
        empty = absent.all(dim=1)
        if empty.any():
            absent = absent.clone()
            absent[empty, 0] = False
        seed = self.seed.expand(b * t, -1, -1)
        summary = self.pool(seed, z.reshape(b * t, n, DIM), z.reshape(b * t, n, DIM),
                            key_padding_mask=absent, need_weights=False)[0].reshape(b, t, DIM)
        seq = torch.cat([summary, ball, probs[..., n:]], dim=-1)
        return self.out(self.drop(self.ball_lstm(seq)[0]))

    def forward(self, players, present, ball, line):
        """players (b, t, n, 7), present (b, t, n), ball (b, t, 3), line (b, t, 2).
        Returns logits, ball x,y scaled -1..1."""
        n = players.shape[2]
        x = torch.cat([players, ball[:, :, None].expand(-1, -1, n, -1), present[..., None].float()], dim=-1)
        x = x * present[..., None]  # an absent slot says nothing, not even where the ball is
        logits, h = self.part1(x, present)
        xy = self.part2(x, present, ball, h, torch.softmax(logits, dim=-1))
        return logits, xy + line if self.on_line else xy


def reality_loss(pred, player_xy, present):
    """The paper's Eq 17: turning costs more the further the ball is from any player. Metres."""
    v = pred[:, 1:] - pred[:, :-1]
    dot = (v[:, :-1] * v[:, 1:]).sum(dim=-1)
    norm = (v[:, :-1].square().sum(dim=-1) * v[:, 1:].square().sum(dim=-1)).add(1e-12).sqrt()
    angle = torch.arccos((dot / norm).clamp(-1 + ANGLE_EDGE, 1 - ANGLE_EDGE))
    gap = (pred[:, 1:-1, None] - player_xy[:, 1:-1]).square().sum(dim=-1).add(1e-6).sqrt()
    nearest = gap.masked_fill(~present[:, 1:-1], float("inf")).amin(dim=2)
    nearest = nearest.masked_fill(torch.isinf(nearest), 0.0)
    return (torch.tanh(angle) * nearest).mean()


def position_loss(pred, truth, hidden, shown):
    """Mean squared metres on the hidden steps, plus a smaller share on the shown ones."""
    squared = (pred - truth).square().sum(dim=-1)
    return (squared * hidden).sum() / hidden.sum().clamp(min=1) + KNOWN_WEIGHT * (squared * shown).sum() / shown.sum().clamp(min=1)
