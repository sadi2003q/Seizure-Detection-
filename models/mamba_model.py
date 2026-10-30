# models/mamba_model.py
# Optional. Needs PyTorch. Uses the official mamba_ssm if available (NVIDIA GPU),
# otherwise a pure-PyTorch fallback (report it if the fallback is used).
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from tqdm.auto import tqdm
from base import Model, ModelFamily, check_params

try:
    from mamba_ssm import Mamba as _OfficialMamba
    _OFFICIAL = torch.cuda.is_available()
except Exception:
    _OFFICIAL = False

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
N_CH, N_FEAT = 22, 42


class _SimpleSSM(nn.Module):
    """Pure-PyTorch selective SSM block (fallback when mamba_ssm is unavailable)."""
    def __init__(self, d_model, d_state=16, expand=2):
        super().__init__()
        d_in = d_model * expand
        self.in_proj = nn.Linear(d_model, d_in * 2)
        self.conv = nn.Conv1d(d_in, d_in, 3, padding=2, groups=d_in)
        self.x_proj = nn.Linear(d_in, d_state * 2 + 1)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_in, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_in))
        self.out_proj = nn.Linear(d_in, d_model)
        self.d_state = d_state

    def forward(self, x):
        B, L, _ = x.shape
        x_, z = self.in_proj(x).chunk(2, dim=-1)
        x_ = F.silu(self.conv(x_.transpose(1, 2))[..., :L].transpose(1, 2))
        bcd = self.x_proj(x_)
        Bm, Cm = bcd[..., :self.d_state], bcd[..., self.d_state:2 * self.d_state]
        dt = F.softplus(bcd[..., -1:])
        A = -torch.exp(self.A_log)
        h = torch.zeros(B, x_.shape[-1], self.d_state, device=x.device)
        ys = []
        for t in range(L):
            h = torch.exp(dt[:, t].unsqueeze(-1) * A) * h + \
                dt[:, t].unsqueeze(-1) * Bm[:, t].unsqueeze(1) * x_[:, t].unsqueeze(-1)
            ys.append((h * Cm[:, t].unsqueeze(1)).sum(-1))
        return self.out_proj((torch.stack(ys, 1) + x_ * self.D) * F.silu(z))


class _Block(nn.Module):
    def __init__(self, d_model, d_state):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mix = (_OfficialMamba(d_model=d_model, d_state=d_state, d_conv=4, expand=2)
                    if _OFFICIAL else _SimpleSSM(d_model, d_state))

    def forward(self, x):
        return x + self.mix(self.norm(x))


class _Net(nn.Module):
    """Input (batch, 22 channels, 42 features): the 22 channels form the sequence."""
    def __init__(self, d_model, n_layers, d_state, dropout):
        super().__init__()
        self.embed = nn.Linear(N_FEAT, d_model)
        self.pos = nn.Parameter(torch.zeros(1, N_CH, d_model))
        self.blocks = nn.ModuleList([_Block(d_model, d_state) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(d_model * 2, 1)

    def forward(self, x):
        h = self.embed(x) + self.pos
        for b in self.blocks:
            h = b(h)
        h = self.norm(h)
        return self.head(self.drop(torch.cat([h.mean(1), h.max(1).values], -1))).squeeze(-1)


class _Mamba(Model):
    name = 'Mamba'

    def __init__(self, d_model, n_layers, d_state, dropout, lr, weight_decay,
                 epochs, batch, patience, seed, verbose):
        self.params = dict(d_model=d_model, n_layers=n_layers, d_state=d_state,
                           dropout=dropout, lr=lr, weight_decay=weight_decay,
                           epochs=epochs, batch=batch, patience=patience, seed=seed)
        self.lr, self.weight_decay, self.epochs = lr, weight_decay, epochs
        self.batch, self.patience, self.seed, self.verbose = batch, patience, seed, verbose
        torch.manual_seed(seed)
        self.net = _Net(d_model, n_layers, d_state, dropout).to(DEV)

    def _T(self, X):
        return torch.tensor(X, dtype=torch.float32).reshape(-1, N_CH, N_FEAT).to(DEV)

    def fit(self, X_train, y_train):
        net = self.net
        if self.verbose:
            backend = 'official mamba_ssm' if _OFFICIAL else 'pure-PyTorch fallback (state in paper)'
            print(f"Mamba backend: {backend} | device {DEV} | "
                  f"params {sum(p.numel() for p in net.parameters()):,}")
        # early stopping uses 10% of the TRAINING rows only (val/test are never touched)
        order = np.random.default_rng(self.seed).permutation(len(y_train))
        cut = int(0.9 * len(order))
        tr_i, es_i = order[:cut], order[cut:]
        Xt, Xe = self._T(X_train[tr_i]), self._T(X_train[es_i])
        yt = torch.tensor(y_train[tr_i], dtype=torch.float32, device=DEV)
        ye = torch.tensor(y_train[es_i], dtype=torch.float32, device=DEV)

        opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        lossf = nn.BCEWithLogitsLoss()
        best, state, bad = 1e9, None, 0
        bar = tqdm(range(self.epochs), desc='Training Mamba', disable=not self.verbose)
        for _ in bar:
            net.train()
            perm = torch.randperm(len(Xt), device=DEV)
            tl = 0.0
            for i in range(0, len(Xt), self.batch):
                idx = perm[i:i + self.batch]
                opt.zero_grad()
                loss = lossf(net(Xt[idx]), yt[idx])
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                tl += loss.item() * len(idx)
            net.eval()
            with torch.no_grad():
                vl = float(np.mean([lossf(net(Xe[i:i + 1024]), ye[i:i + 1024]).item()
                                    for i in range(0, len(Xe), 1024)]))
            bar.set_postfix(train=f'{tl / len(Xt):.3f}', early_stop_loss=f'{vl:.3f}')
            if vl < best - 1e-4:
                best, bad = vl, 0
                state = {k: v.clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        net.load_state_dict(state)
        net.eval()
        return self

    def predict(self, X):
        self.net.eval()
        Xq = self._T(X)
        with torch.no_grad():
            return np.concatenate([torch.sigmoid(self.net(Xq[i:i + 1024])).cpu().numpy()
                                   for i in range(0, len(Xq), 1024)])


class MambaModel(ModelFamily):
    name = 'Mamba'

    @classmethod
    def search_space(cls, trial):
        return dict(d_model=trial.suggest_categorical('d_model', [32, 64, 128]),
                    n_layers=trial.suggest_int('n_layers', 1, 3),
                    d_state=trial.suggest_categorical('d_state', [8, 16, 32]),
                    dropout=trial.suggest_float('dropout', 0.1, 0.5),
                    lr=trial.suggest_float('lr', 1e-4, 5e-3, log=True),
                    weight_decay=trial.suggest_float('weight_decay', 1e-4, 1e-1, log=True),
                    epochs=10)          # short runs while tuning; override in build(**best, epochs=30)

    @classmethod
    def build(cls, **params):
        check_params('Mamba', params, cls._build)      # clear error on wrong names
        return cls._build(**params)

    @classmethod
    def _build(cls, d_model=64, n_layers=2, d_state=16, dropout=0.2, lr=1e-3,
               weight_decay=1e-2, epochs=30, batch=256, patience=6, seed=42, verbose=True):
        return _Mamba(d_model, n_layers, d_state, dropout, lr, weight_decay,
                      epochs, batch, patience, seed, verbose)
