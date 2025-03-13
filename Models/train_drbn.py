#!/usr/bin/env python3

"""
Trains a 2-layer DRBN on an Ising-like dataset (with possible vacancies). 

"""

import os
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

L = 16
vacancy_list = [0.0, 0.1, 0.15, 0.2, 0.25]
Tc_map = {
    0.0: 2.3,
    0.1: 1.975,
    0.15: 1.75,
    0.2: 1.6,
    0.25: 1.4
}
temperature_offsets = [-0.8, -0.1, -0.05, 0.0, 0.05, 0.1, 0.5]
data_file = "data/drbn_snaps.npz"
EPOCHS_DEFAULT = 50
CD_K_DEFAULT = 10
BATCH_SIZE_DEFAULT = 32
HID1 = 64
HID2 = 64

random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)

def encode_spins_to_two_channels(spin_configs):
    """
    Convert spins in {−1,0,+1} to occupant/sign channels (shape (N, 2*L*L)).
    occupant=1 if spin!=0, sign=1 if spin>0.
    """
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

def occupant_clamp(p_v, occupant_data):
    """
    Pin occupant bits from occupant_data, sample sign bits from p_v.
    """
    out = torch.zeros_like(p_v)
    out[:, 0::2] = occupant_data
    sign_probs = p_v[:, 1::2]
    out[:, 1::2] = torch.bernoulli(sign_probs)
    return out

def masked_bce_occupancy_sign(v_pred, v_true, eps=1e-7):
    """
    Masked BCE for occupant+sign. Occupant bits get full BCE; sign bits only if occupant=1.
    """
    eps_t = torch.tensor(eps, device=v_true.device, dtype=v_true.dtype)
    occ_pred = v_pred[:, 0::2]
    occ_true = v_true[:, 0::2]
    sign_pred = v_pred[:, 1::2]
    sign_true = v_true[:, 1::2]
    occ_loss = -(
        occ_true * torch.log(occ_pred + eps_t) +
        (1 - occ_true) * torch.log(1 - occ_pred + eps_t)
    )
    sign_mask = (occ_true > 0.5).float()
    sign_loss_raw = -(
        sign_true * torch.log(sign_pred + eps_t) +
        (1 - sign_true) * torch.log(1 - sign_pred + eps_t)
    )
    sign_loss = sign_loss_raw * sign_mask
    return torch.mean(occ_loss) + torch.mean(sign_loss)

class DRBN_mod(nn.Module):
    """
    Two-layer Deep RBM with up/down passes and a contrastive_divergence method.
    """
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
        nn.init.xavier_uniform_(self.W1, gain=1.0)
        nn.init.xavier_uniform_(self.W2, gain=1.0)

    def up_pass(self, v):
        logits_h1 = F.linear(v, self.W1, self.b_h1)
        p_h1 = torch.sigmoid(logits_h1)
        h1 = torch.bernoulli(p_h1)
        logits_h2 = F.linear(h1, self.W2, self.b_h2)
        p_h2 = torch.sigmoid(logits_h2)
        h2 = torch.bernoulli(p_h2)
        return (p_h1, h1), (p_h2, h2)

    def down_pass(self, h2):
        logits_h1 = F.linear(h2, self.W2.transpose(0,1), self.b_h1)
        p_h1 = torch.sigmoid(logits_h1)
        h1 = torch.bernoulli(p_h1)
        logits_v = F.linear(h1, self.W1.transpose(0,1), self.b_v)
        p_v = torch.sigmoid(logits_v)
        v = torch.bernoulli(p_v)
        return (p_h1, h1), (p_v, v)

    def contrastive_divergence(self, v_data, occupant_data=None, k=1, clamp=False):
        device = v_data.device
        with torch.no_grad():
            (_, h1_data), (_, h2_data) = self.up_pass(v_data)
        v_neg = v_data.clone()
        for _ in range(k):
            (_, h1_up), (_, h2_up) = self.up_pass(v_neg)
            (_, h1_down), (p_v_down, v_down) = self.down_pass(h2_up)
            if clamp and occupant_data is not None:
                v_neg = occupant_clamp(p_v_down, occupant_data)
            else:
                v_neg = v_down
        with torch.no_grad():
            (_, h1_neg), (_, h2_neg) = self.up_pass(v_neg)
        pos_vh1 = torch.bmm(v_data.unsqueeze(2), h1_data.unsqueeze(1))
        neg_vh1 = torch.bmm(v_neg.unsqueeze(2), h1_neg.unsqueeze(1))
        dW1 = (pos_vh1.mean(dim=0) - neg_vh1.mean(dim=0)).transpose(0,1)
        pos_h1h2 = torch.bmm(h1_data.unsqueeze(2), h2_data.unsqueeze(1))
        neg_h1h2 = torch.bmm(h1_neg.unsqueeze(2), h2_neg.unsqueeze(1))
        dW2 = pos_h1h2.mean(dim=0) - neg_h1h2.mean(dim=0)
        db_v = (v_data - v_neg).mean(dim=0)
        db_h1 = (h1_data - h1_neg).mean(dim=0)
        db_h2 = (h2_data - h2_neg).mean(dim=0)
        for p in [self.W1, self.b_v, self.b_h1, self.W2, self.b_h2]:
            if p.grad is not None:
                p.grad.zero_()
        self.W1.grad = -dW1
        self.b_v.grad= -db_v
        self.b_h1.grad= -db_h1
        self.W2.grad = -dW2
        self.b_h2.grad= -db_h2
        with torch.no_grad():
            (_, h1_tmp), (_, h2_tmp) = self.up_pass(v_neg)
        logits_v_final = F.linear(h1_tmp, self.W1.transpose(0,1), self.b_v)
        p_v_final = torch.sigmoid(logits_v_final)
        loss = masked_bce_occupancy_sign(p_v_final, v_data)
        return loss

def train_drbn_subset(spins_subset, T, vac, clamp=False):
    X_encoded = encode_spins_to_two_channels(spins_subset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DRBN_mod(n_vis=2*L*L, n_hid1=HID1, n_hid2=HID2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    if clamp:
        occupant_arr = X_encoded[:, 0::2]
        dataset = TensorDataset(
            torch.tensor(X_encoded, dtype=torch.float32),
            torch.tensor(occupant_arr, dtype=torch.float32)
        )
    else:
        occupant_dummy = np.zeros((X_encoded.shape[0], L*L), dtype=np.float32)
        dataset = TensorDataset(
            torch.tensor(X_encoded, dtype=torch.float32),
            torch.tensor(occupant_dummy, dtype=torch.float32)
        )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE_DEFAULT, shuffle=True)
    for ep in range(EPOCHS_DEFAULT):
        for batch_data, occupant_data in loader:
            batch_data = batch_data.to(device)
            occupant_data = occupant_data.to(device) if clamp else None
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.zero_()
            loss = model.contrastive_divergence(
                v_data=batch_data,
                occupant_data=occupant_data,
                k=CD_K_DEFAULT,
                clamp=clamp
            )
            optimizer.step()
    out_folder = "clamped" if clamp else "unclamped"
    os.makedirs(out_folder, exist_ok=True)
    clamp_str = "_clamped" if clamp else ""
    out_name = f"trained_drbn_T{T}_vac{vac}{clamp_str}.pt"
    out_path = os.path.join(out_folder, out_name)
    torch.save(model.state_dict(), out_path)
    print(f"[INFO] Saved '{out_path}'")
    del model, optimizer, dataset, loader
    torch.cuda.empty_cache()

def main():
    if not os.path.isfile(data_file):
        raise FileNotFoundError(f"Could not find data file: {data_file}")
    data = np.load(data_file)
    spins_all = data["spins"]
    temps_all = data["temperature"]
    vacs_all  = data["vacancy_fraction"]
    if spins_all.ndim == 3 and spins_all.shape[0] == spins_all.shape[1]:
        spins_all = np.transpose(spins_all, (2,0,1))
    all_subsets = []
    for vac in vacancy_list:
        Tc = Tc_map[vac]
        for offset in temperature_offsets:
            T_val = round(Tc + offset, 3)
            all_subsets.append((T_val, vac))
    for (T_val, vac) in all_subsets:
        mask = (
            np.isclose(temps_all, T_val, atol=1e-5) &
            np.isclose(vacs_all,  vac,   atol=1e-5)
        )
        subset_spins = spins_all[mask]
        n_samples = subset_spins.shape[0]
        if n_samples == 0:
            print(f"No data for T={T_val}, vac={vac}, skipping.")
            continue
        print(f"Training DRBN for T={T_val}, vac={vac}, #samples={n_samples}...")
        train_drbn_subset(subset_spins, T_val, vac, clamp=False)
        train_drbn_subset(subset_spins, T_val, vac, clamp=True)

if __name__ == "__main__":
    main()
