"""
Decision Transformer backbone (Chen et al., 2021) on a GPT-2 transformer.

Sizes default to the global configuration in ``pidt.config`` so that the same
class serves the 298-dimensional EMSx state used throughout the paper.
"""

import torch
import torch.nn as nn
import transformers
from pidt.config import cfg


# ============================================================
# BASE CLASS
# ============================================================

class TrajectoryModel(nn.Module):
    """Base class for trajectory models (identical to original repo)."""

    def __init__(self, state_dim, act_dim, max_length=None):
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.max_length = max_length

    def forward(self, states, actions, rewards, masks=None, attention_mask=None):
        raise NotImplementedError

    def get_action(self, states, actions, rewards, **kwargs):
        raise NotImplementedError


# ============================================================
# DECISION TRANSFORMER
# ============================================================

class DecisionTransformer(TrajectoryModel):
    """
    Decision Transformer: Reinforcement Learning via Sequence Modeling.
    Chen et al. (2021), arXiv:2106.01345

    Input/output sizes are taken from ``pidt.config.cfg`` unless given
    explicitly.
    """

    def __init__(
        self,
        state_dim: int = None,
        act_dim: int = None,
        hidden_size: int = None,
        max_length: int = None,
        max_ep_len: int = None,
        action_tanh: bool = None,
        n_layer: int = None,
        n_head: int = None,
        n_inner: int = None,
        activation_function: str = None,
        resid_pdrop: float = None,
        attn_pdrop: float = None,
    ):
        # Defaults from config
        dt = cfg.dt
        state_dim = state_dim or cfg.state.state_dim
        act_dim = act_dim or cfg.action.dim
        hidden_size = hidden_size or dt.hidden_size
        max_length = max_length or dt.context_length
        max_ep_len = max_ep_len or dt.max_ep_len
        action_tanh = action_tanh if action_tanh is not None else dt.action_tanh
        n_layer = n_layer or dt.n_layer
        n_head = n_head or dt.n_head
        n_inner = n_inner or dt.n_inner or 4 * hidden_size
        activation_function = activation_function or dt.activation_function
        resid_pdrop = resid_pdrop if resid_pdrop is not None else dt.resid_pdrop
        attn_pdrop = attn_pdrop if attn_pdrop is not None else dt.attn_pdrop

        super().__init__(state_dim, act_dim, max_length=max_length)
        self.hidden_size = hidden_size

        # ---- GPT-2 backbone ----
        config = transformers.GPT2Config(
            vocab_size=1,
            n_embd=hidden_size,
            n_layer=n_layer,
            n_head=n_head,
            n_inner=n_inner,
            activation_function=activation_function,
            resid_pdrop=resid_pdrop,
            attn_pdrop=attn_pdrop,
            n_positions=1024,
        )
        self.transformer = transformers.GPT2Model(config)

        # ---- Modality embeddings ----
        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.embed_return = nn.Linear(1, hidden_size)
        self.embed_state = nn.Linear(self.state_dim, hidden_size)
        self.embed_action = nn.Linear(self.act_dim, hidden_size)
        self.embed_ln = nn.LayerNorm(hidden_size)

        # ---- Prediction heads ----
        self.predict_state = nn.Linear(hidden_size, self.state_dim)
        self.predict_action = nn.Sequential(
            *([nn.Linear(hidden_size, self.act_dim)]
              + ([nn.Tanh()] if action_tanh else []))
        )
        self.predict_return = nn.Linear(hidden_size, 1)

    def forward(
        self,
        states,
        actions,
        rewards,
        returns_to_go,
        timesteps,
        attention_mask=None,
    ):
        batch_size, seq_length = states.shape[0], states.shape[1]

        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length), dtype=torch.long, device=states.device
            )

        # Embed each modality
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns_to_go)
        time_embeddings = self.embed_timestep(timesteps)

        # Add timestep embeddings
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings

        # Interleave: (R̂₁, s₁, a₁, R̂₂, s₂, a₂, ...)
        stacked_inputs = (
            torch.stack(
                (returns_embeddings, state_embeddings, action_embeddings), dim=1
            )
            .permute(0, 2, 1, 3)
            .reshape(batch_size, 3 * seq_length, self.hidden_size)
        )
        stacked_inputs = self.embed_ln(stacked_inputs)

        stacked_attention_mask = (
            torch.stack(
                (attention_mask, attention_mask, attention_mask), dim=1
            )
            .permute(0, 2, 1)
            .reshape(batch_size, 3 * seq_length)
        )

        # GPT-2 forward (position_ids=0 → we use timestep embeddings instead)
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
            position_ids=torch.zeros(
                stacked_attention_mask.shape,
                device=stacked_inputs.device,
                dtype=torch.long,
            ),
        )

        x = transformer_outputs["last_hidden_state"]
        x = x.reshape(batch_size, seq_length, 3, self.hidden_size).permute(
            0, 2, 1, 3
        )

        # Predictions
        return_preds = self.predict_return(x[:, 2])   # from action token
        state_preds = self.predict_state(x[:, 2])     # from action token
        action_preds = self.predict_action(x[:, 1])   # from state token

        return state_preds, action_preds, return_preds

    def get_action(self, states, actions, rewards, returns_to_go, timesteps, **kwargs):
        """Autoregressive action prediction at inference time."""
        # Truncate to context window K
        states = states[:, -self.max_length:]
        actions = actions[:, -self.max_length:]
        returns_to_go = returns_to_go[:, -self.max_length:]
        timesteps = timesteps[:, -self.max_length:]

        # Pad to max_length
        seq_len = states.shape[1]
        pad_len = self.max_length - seq_len
        device = states.device

        attention_mask = torch.cat([
            torch.zeros(pad_len, device=device),
            torch.ones(seq_len, device=device),
        ]).reshape(1, -1).to(dtype=torch.long)

        states = torch.cat([
            torch.zeros((1, pad_len, self.state_dim), device=device),
            states,
        ], dim=1).float()

        actions = torch.cat([
            torch.zeros((1, pad_len, self.act_dim), device=device),
            actions,
        ], dim=1).float()

        returns_to_go = torch.cat([
            torch.zeros((1, pad_len, 1), device=device),
            returns_to_go,
        ], dim=1).float()

        timesteps = torch.cat([
            torch.zeros((1, pad_len), device=device, dtype=torch.long),
            timesteps,
        ], dim=1)

        state_preds, action_preds, return_preds = self.forward(
            states, actions, None, returns_to_go, timesteps,
            attention_mask=attention_mask,
        )

        return action_preds[0, -1]


def build_model(device: torch.device = None) -> DecisionTransformer:
    """Build a model from the global configuration and move it to ``device``."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DecisionTransformer().to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model created: {total_params:,} params ({trainable:,} trainable)")
    print(f"  state_dim={model.state_dim}, act_dim={model.act_dim}, "
          f"hidden={model.hidden_size}, layers={cfg.dt.n_layer}, "
          f"heads={cfg.dt.n_head}, K={model.max_length}")

    return model
