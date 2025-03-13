#!/usr/bin/env python3

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)

if not torch.cuda.is_available():
    raise RuntimeError("No GPU found! Please run on a machine with CUDA support.")
device = torch.device("cuda")

L = 16
vacancy_list = [0.0, 0.1, 0.2]
Tc_map = {0.0: 2.3, 0.1: 1.975, 0.2: 1.6}
temperature_offsets = [-0.8, -0.1, -0.05, 0.0, 0.05, 0.1, 0.5]
data_file = "data/disordered_snapshots_16.npz"

EPOCHS = 50
CD_K = 10
BATCH_SIZE = 64
HID1 = 64
HID2 = 64

os.makedirs("clamped_advanced_data", exist_ok=True)

def encode_spins_to_two_channels(spin_configs):
    """Convert spins to occupant+sign channels."""
    n_samples, Ldim, _ = spin_configs.shape
    out = np.zeros((n_samples, 2 * Ldim * Ldim), dtype=np.float32)
    idx = 0
    for i in range(Ldim):
        for j in range(Ldim):
            spin_ij = spin_configs[:, i, j]
            occ = (spin_ij != 0).astype(np.float32)
            sgn = (spin_ij > 0).astype(np.float32)
            out[:, idx] = occ
            out[:, idx+1] = sgn
            idx += 2
    return out

def occupant_clamp_2channel(prob_v, occupant_data):
    """Clamp occupant bits to occupant_data, sample sign bits from prob_v."""
    out = torch.zeros_like(prob_v)
    out[:, 0::2] = occupant_data
    sign_probs = prob_v[:, 1::2]
    out[:, 1::2] = torch.bernoulli(sign_probs)
    return out

def bce_sign_bits_only(v_pred, v_true, eps=1e-7):
    """Compute BCE only for sign bits where occupant=1."""
    occ_true = v_true[:, 0::2]
    sign_pred = v_pred[:, 1::2]
    sign_true = v_true[:, 1::2]
    sign_mask = (occ_true > 0.5).float()
    sign_loss_raw = -(
        sign_true * torch.log(sign_pred + eps) +
        (1 - sign_true) * torch.log(1 - sign_pred + eps)
    )
    sign_loss = sign_loss_raw * sign_mask
    return torch.mean(sign_loss)

def correlation_matrix_sign_bits(v, h, chunk_size=32):
    """Compute (v_sign outer h) in chunks. Return correlation for sign bits."""
    device = v.device
    B = v.size(0)
    sign_idx = torch.arange(1, v.size(1), step=2, device=device)
    v_sign = v[:, sign_idx]
    out = torch.zeros(v_sign.size(1), h.size(1), device=device, dtype=v.dtype)
    for start in range(0, B, chunk_size):
        end = start + chunk_size
        v_chunk = v_sign[start:end, :]
        h_chunk = h[start:end, :]
        out += torch.einsum('bi,bj->ij', v_chunk, h_chunk)
    out /= B
    return out

class DBMAdvanced(nn.Module):
    """2-layer DBM with occupant bits pinned and skip from h2->v."""
    def __init__(self, n_vis, n_hid1, n_hid2):
        super().__init__()
        self.n_vis = n_vis
        self.n_hid1 = n_hid1
        self.n_hid2 = n_hid2
        self.W1 = nn.Parameter(torch.zeros(n_hid1, n_vis))
        self.b_v = nn.Parameter(torch.zeros(n_vis))
        self.b_h1 = nn.Parameter(torch.zeros(n_hid1))
        self.W2 = nn.Parameter(torch.zeros(n_hid1, n_hid2))
        self.b_h2 = nn.Parameter(torch.zeros(n_hid2))
        self.U_h2_v = nn.Parameter(torch.zeros(n_hid2, n_vis))
        nn.init.xavier_uniform_(self.W1, gain=1.0)
        nn.init.xavier_uniform_(self.W2, gain=1.0)
        nn.init.xavier_uniform_(self.U_h2_v, gain=1.0)

    def occupant_clamp(self, p_v, occupant_data):
        """Fill occupant bits from occupant_data, sample sign bits from p_v."""
        return occupant_clamp_2channel(p_v, occupant_data)

    def sample_h1_given_vh2(self, v, h2):
        """p(h1=1) = sigmoid(W1 v + W2^T h2 + b_h1)."""
        act_h1 = F.linear(v, self.W1, self.b_h1) + F.linear(h2, self.W2.transpose(0,1))
        p_h1 = torch.sigmoid(act_h1)
        h1_s = torch.bernoulli(p_h1)
        return p_h1, h1_s

    def sample_h2_given_h1v(self, h1, v):
        """p(h2=1) = sigmoid(W2 h1 + U_h2_v v + b_h2)."""
        act_h2 = F.linear(h1, self.W2, self.b_h2) + F.linear(v, self.U_h2_v)
        p_h2 = torch.sigmoid(act_h2)
        h2_s = torch.bernoulli(p_h2)
        return p_h2, h2_s

    def sample_v_given_h1h2(self, h1, h2):
        """p(v=1) = sigmoid(W1^T h1 + U_h2_v^T h2 + b_v)."""
        act_v = F.linear(h1, self.W1.transpose(0,1), self.b_v) + F.linear(h2, self.U_h2_v.transpose(0,1))
        p_v = torch.sigmoid(act_v)
        v_s = torch.bernoulli(p_v)
        return p_v, v_s

    def positive_phase_inference(self, v_data, M=5):
        """Approximate h1,h2 using mean-field passes."""
        B = v_data.size(0)
        device = v_data.device
        h1 = torch.zeros(B, self.n_hid1, device=device)
        h2 = torch.zeros(B, self.n_hid2, device=device)
        for _ in range(M):
            _, h1_s = self.sample_h1_given_vh2(v_data, h2)
            _, h2_s = self.sample_h2_given_h1v(h1_s, v_data)
            h1 = h1_s
            h2 = h2_s
        return (h1, h2)

    def sample_negative_phase(self, batch_size, occupant_data, k=1):
        """Block-Gibbs sampling with occupant bits pinned."""
        device = occupant_data.device
        v_init = torch.rand(batch_size, self.n_vis, device=device)
        v_init = self.occupant_clamp(v_init, occupant_data)
        v_neg = v_init.clone()
        h2_neg = torch.zeros(batch_size, self.n_hid2, device=device)
        for _ in range(k):
            _, h1_neg = self.sample_h1_given_vh2(v_neg, h2_neg)
            _, h2_neg = self.sample_h2_given_h1v(h1_neg, v_neg)
            p_v_neg, _ = self.sample_v_given_h1h2(h1_neg, h2_neg)
            v_neg = self.occupant_clamp(p_v_neg, occupant_data)
        _, h1_final = self.sample_h1_given_vh2(v_neg, h2_neg)
        _, h2_final = self.sample_h2_given_h1v(h1_final, v_neg)
        return v_neg, h1_final, h2_final

    def contrastive_divergence(self, v_data, k=1, pos_passes=5):
        """Perform occupant-pinned sign-bit training. Returns BCE loss."""
        B = v_data.size(0)
        device = v_data.device
        occupant_data = v_data[:, 0::2]
        h1_data, h2_data = self.positive_phase_inference(v_data, M=pos_passes)
        v_neg, h1_neg, h2_neg = self.sample_negative_phase(B, occupant_data, k=k)
        pos_vh1 = correlation_matrix_sign_bits(v_data, h1_data)
        neg_vh1 = correlation_matrix_sign_bits(v_neg, h1_neg)
        dW1_sign = pos_vh1 - neg_vh1
        sign_idx = torch.arange(1, self.n_vis, step=2, device=device)
        dW1_full = torch.zeros_like(self.W1)
        dW1_full[:, sign_idx] = dW1_sign.transpose(0,1)
        pos_v_sign = v_data[:, 1::2]
        neg_v_sign = v_neg[:, 1::2]
        db_v_full = torch.zeros(self.n_vis, device=device)
        db_v_full[sign_idx] = (pos_v_sign - neg_v_sign).mean(dim=0)
        db_h1 = (h1_data - h1_neg).mean(dim=0)
        pos_h1h2 = torch.bmm(h1_data.unsqueeze(2), h2_data.unsqueeze(1)).mean(dim=0)
        neg_h1h2 = torch.bmm(h1_neg.unsqueeze(2), h2_neg.unsqueeze(1)).mean(dim=0)
        dW2 = pos_h1h2 - neg_h1h2
        db_h2 = (h2_data - h2_neg).mean(dim=0)
        pos_h2v_sign = correlation_matrix_sign_bits(v_data, h2_data)
        neg_h2v_sign = correlation_matrix_sign_bits(v_neg, h2_neg)
        dU_sign = pos_h2v_sign - neg_h2v_sign
        dU_full = torch.zeros_like(self.U_h2_v)
        dU_full[:, sign_idx] = dU_sign.transpose(0,1)
        for p in [self.W1, self.b_v, self.b_h1, self.W2, self.b_h2, self.U_h2_v]:
            if p.grad is not None:
                p.grad.zero_()
        self.W1.grad = -dW1_full
        self.b_v.grad = -db_v_full
        self.b_h1.grad = -db_h1
        self.W2.grad = -dW2
        self.b_h2.grad = -db_h2
        self.U_h2_v.grad = -dU_full
        loss = bce_sign_bits_only(v_neg, v_data)
        return loss

def train_dbm_advanced_subset(spins_subset, T_val, vac_val):
    """Train DBM on subset with occupant-pinned sign-bit training."""
    X_np = encode_spins_to_two_channels(spins_subset)
    X_t = torch.tensor(X_np, dtype=torch.float32, device=device)
    model = DBMAdvanced(n_vis=2*L*L, n_hid1=HID1, n_hid2=HID2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    dataset = TensorDataset(X_t)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    POS_PASSES = 5
    for ep in range(EPOCHS):
        for (batch_data,) in loader:
            optimizer.zero_grad()
            loss = model.contrastive_divergence(batch_data, k=CD_K, pos_passes=POS_PASSES)
            optimizer.step()
        if (ep+1) % 10 == 0:
            print(f"Epoch {ep+1}/{EPOCHS}, BCE(sign-bits) = {loss.item():.4f}")
    out_name = f"trained_dbmAdvanced_T{T_val}_vac{vac_val}_clamped.pt"
    out_path = os.path.join("clamped_advanced_data", out_name)
    torch.save(model.state_dict(), out_path)
    print(f"  [Saved] {out_path}  (#samples={len(spins_subset)})")

def main():
    """Load data and train DBM for each T,vac combination."""
    if not os.path.isfile(data_file):
        raise FileNotFoundError(f"[ERROR] Missing data file: {data_file}")
    data = np.load(data_file)
    spins_all = data["spins"]
    temps_all = data["temperature"]
    vacs_all = data["vacancy_fraction"]
    if spins_all.ndim == 3 and spins_all.shape[0] == spins_all.shape[1]:
        spins_all = np.transpose(spins_all, (2,0,1))
    all_subsets = []
    for vac in vacancy_list:
        Tc = Tc_map[vac]
        for off in temperature_offsets:
            T_val = round(Tc + off, 3)
            all_subsets.append((T_val, vac))
    for (T_val, vac_val) in all_subsets:
        mask = (
            np.isclose(temps_all, T_val, atol=1e-3)
            & np.isclose(vacs_all, vac_val, atol=1e-3)
        )
        spins_subset = spins_all[mask]
        if spins_subset.shape[0] == 0:
            print(f"  [SKIP] No data for T={T_val}, vac={vac_val}")
            continue
        print(f"Train advanced DBM at T={T_val}, vac={vac_val}, #samples={spins_subset.shape[0]} ...")
        train_dbm_advanced_subset(spins_subset, T_val, vac_val)

if __name__ == "__main__":
    main()
