#!/usr/bin/env python3

"""
Trains RBM, DBM, DBN, DRBN, and an advanced DBM. Occupant bits are pinned and excluded from gradients, learning only p(sign|occupant).
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import math

random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)

L = 16
vacancy_list = [0.1, 0.2]
Tc_map = {0.0: 2.3, 0.1: 1.975, 0.2: 1.6}
temperature_offsets = [-0.8, -0.1, -0.05, 0.0, 0.05, 0.1, 0.5]
data_file = "data/disordered_snapshots_16.npz"
EPOCHS_DEFAULT = 50
CD_K_DEFAULT = 10
BATCH_SIZE_DEFAULT = 64
RBM_HIDDEN = 128
HID1 = 64
HID2 = 64
HID1_ADV = 64
HID2_ADV = 64
BATCH_SIZE_ADVANCED = 32
EPOCHS_ADVANCED = 40
MAX_DATA_ADVANCED = 3000

def encode_spins_to_two_channels(spin_configs):
    """
    Convert spins to occupant+sign bits. Output shape (N, 2*L*L).
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

def bce_sign_bits_only(v_pred, v_true, eps=1e-7):
    """
    Binary cross-entropy for sign bits only.
    """
    sign_pred = v_pred[:, 1::2]
    sign_true = v_true[:, 1::2]
    eps_t = torch.tensor(eps, device=v_true.device, dtype=v_true.dtype)
    sign_loss = -(
        sign_true * torch.log(sign_pred + eps_t) +
        (1 - sign_true) * torch.log(1 - sign_pred + eps_t)
    )
    return torch.mean(sign_loss)

def get_sign_bits(v_batch):
    """
    Extract only the sign bits from occupant+sign vector.
    """
    return v_batch[:, 1::2]

def occupant_clamp(v_prob, occupant_data):
    """
    Clamp occupant bits from data, sample sign bits from probabilities.
    """
    out = torch.zeros_like(v_prob)
    out[:, 0::2] = occupant_data
    sign_prob = v_prob[:, 1::2]
    out[:, 1::2] = torch.bernoulli(sign_prob)
    return out

class RBM(nn.Module):
    def __init__(self, n_vis, n_hid):
        super(RBM, self).__init__()
        self.n_vis = n_vis
        self.n_hid = n_hid
        self.W = nn.Parameter(torch.zeros(n_vis, n_hid))
        self.v_bias = nn.Parameter(torch.zeros(n_vis))
        self.h_bias = nn.Parameter(torch.zeros(n_hid))
        nn.init.xavier_uniform_(self.W, gain=1.0)

    def sample_h_given_v(self, v):
        p_h = torch.sigmoid(v @ self.W + self.h_bias)
        h_s = torch.bernoulli(p_h)
        return p_h, h_s

    def sample_v_given_h(self, h):
        p_v = torch.sigmoid(h @ self.W.t() + self.v_bias)
        v_s = torch.bernoulli(p_v)
        return p_v, v_s

    def contrastive_divergence(self, v_data, occupant_data, k=1):
        B = v_data.size(0)
        _, h_data = self.sample_h_given_v(v_data)
        v_neg = v_data.clone()
        h_neg = h_data.clone()
        for _ in range(k):
            p_v_neg, _ = self.sample_v_given_h(h_neg)
            v_neg = occupant_clamp(p_v_neg, occupant_data)
            _, h_neg = self.sample_h_given_v(v_neg)
        pos_v_sign = get_sign_bits(v_data)
        neg_v_sign = get_sign_bits(v_neg)
        pos_vh = torch.bmm(pos_v_sign.unsqueeze(2), h_data.unsqueeze(1))
        neg_vh = torch.bmm(neg_v_sign.unsqueeze(2), h_neg.unsqueeze(1))
        dW = (pos_vh.mean(dim=0) - neg_vh.mean(dim=0))
        dW_full = torch.zeros_like(self.W)
        sign_mask = torch.arange(self.n_vis, device=v_data.device)[1::2]
        dW_full[sign_mask, :] = dW
        db_v = torch.zeros(self.n_vis, device=v_data.device)
        db_v[sign_mask] = (pos_v_sign - neg_v_sign).mean(dim=0)
        db_h = (h_data - h_neg).mean(dim=0)
        if self.W.grad is not None:
            self.W.grad.zero_()
        if self.v_bias.grad is not None:
            self.v_bias.grad.zero_()
        if self.h_bias.grad is not None:
            self.h_bias.grad.zero_()
        self.W.grad = -dW_full
        self.v_bias.grad = -db_v
        self.h_bias.grad = -db_h
        loss = bce_sign_bits_only(p_v_neg, v_data)
        return loss

def train_rbm_subset(spins_subset, T, vac):
    X_encoded = encode_spins_to_two_channels(spins_subset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rbm = RBM(n_vis=2*L*L, n_hid=RBM_HIDDEN).to(device)
    optimizer = optim.Adam(rbm.parameters(), lr=1e-3)
    occupant_arr = X_encoded[:, 0::2]
    dataset = TensorDataset(torch.tensor(X_encoded, dtype=torch.float32),
                            torch.tensor(occupant_arr, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE_DEFAULT, shuffle=True)
    for ep in range(EPOCHS_DEFAULT):
        for batch_data, occupant_data in loader:
            batch_data = batch_data.to(device)
            occupant_data = occupant_data.to(device)
            for p in rbm.parameters():
                if p.grad is not None:
                    p.grad.zero_()
            loss = rbm.contrastive_divergence(batch_data, occupant_data, k=CD_K_DEFAULT)
            optimizer.step()
    os.makedirs("clamped_data", exist_ok=True)
    out_name = f"trained_rbm_T{T}_vac{vac}_clamped.pt"
    out_path = os.path.join("clamped_data", out_name)
    torch.save(rbm.state_dict(), out_path)
    print(f"    Saved {out_path}")
    del rbm, optimizer, dataset, loader
    torch.cuda.empty_cache()

class DBM(nn.Module):
    def __init__(self, n_vis, n_hid1, n_hid2):
        super(DBM, self).__init__()
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

    def sample_h1_given_v_h2(self, v, h2):
        act_h1 = F.linear(v, self.W1, self.b_h1) + F.linear(h2, self.W2.transpose(0,1))
        p_h1 = torch.sigmoid(act_h1)
        h1_s = torch.bernoulli(p_h1)
        return p_h1, h1_s

    def sample_h2_given_h1(self, h1):
        act_h2 = F.linear(h1, self.W2, self.b_h2)
        p_h2 = torch.sigmoid(act_h2)
        h2_s = torch.bernoulli(p_h2)
        return p_h2, h2_s

    def sample_v_given_h1(self, h1):
        act_v = F.linear(h1, self.W1.transpose(0,1), self.b_v)
        p_v = torch.sigmoid(act_v)
        v_s = torch.bernoulli(p_v)
        return p_v, v_s

    def contrastive_divergence(self, v_data, occupant_data, k=1):
        device = v_data.device
        B = v_data.size(0)
        with torch.no_grad():
            h2_zeros = torch.zeros(B, self.n_hid2, device=device)
            _, h1_data = self.sample_h1_given_v_h2(v_data, h2_zeros)
            _, h2_data = self.sample_h2_given_h1(h1_data)
        v_neg = v_data.clone()
        h2_neg = h2_data.clone()
        for _ in range(k):
            _, h1_neg = self.sample_h1_given_v_h2(v_neg, h2_neg)
            _, h2_neg = self.sample_h2_given_h1(h1_neg)
            p_v_neg, _ = self.sample_v_given_h1(h1_neg)
            v_neg = occupant_clamp(p_v_neg, occupant_data)
        with torch.no_grad():
            _, h1_neg_final = self.sample_h1_given_v_h2(v_neg, h2_neg)
        pos_v_sign = get_sign_bits(v_data)
        neg_v_sign = get_sign_bits(v_neg)
        pos_vh1 = torch.bmm(pos_v_sign.unsqueeze(2), h1_data.unsqueeze(1))
        neg_vh1 = torch.bmm(neg_v_sign.unsqueeze(2), h1_neg_final.unsqueeze(1))
        dW1_sign = (pos_vh1.mean(dim=0) - neg_vh1.mean(dim=0))
        dW1_full = torch.zeros_like(self.W1)
        sign_mask = torch.arange(self.n_vis, device=device)[1::2]
        dW1_full[:, sign_mask] = dW1_sign.transpose(0, 1)
        db_v_full = torch.zeros(self.n_vis, device=device)
        db_v_full[sign_mask] = (pos_v_sign - neg_v_sign).mean(dim=0)
        db_h1 = (h1_data - h1_neg_final).mean(dim=0)
        pos_h1h2 = torch.bmm(h1_data.unsqueeze(2), h2_data.unsqueeze(1))
        neg_h1h2 = torch.bmm(h1_neg_final.unsqueeze(2), h2_neg.unsqueeze(1))
        dW2 = (pos_h1h2.mean(dim=0) - neg_h1h2.mean(dim=0))
        db_h2 = (h2_data - h2_neg).mean(dim=0)
        for p in [self.W1, self.b_v, self.b_h1, self.W2, self.b_h2]:
            if p.grad is not None:
                p.grad.zero_()
        self.W1.grad = -dW1_full
        self.b_v.grad = -db_v_full
        self.b_h1.grad = -db_h1
        self.W2.grad = -dW2
        self.b_h2.grad = -db_h2
        loss = bce_sign_bits_only(v_neg, v_data)
        return loss

def train_dbm_subset(spins_subset, T, vac):
    X_encoded = encode_spins_to_two_channels(spins_subset)
    occupant_arr = X_encoded[:, 0::2]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DBM(n_vis=2*L*L, n_hid1=HID1, n_hid2=HID2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    dataset = TensorDataset(torch.tensor(X_encoded, dtype=torch.float32),
                            torch.tensor(occupant_arr, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE_DEFAULT, shuffle=True)
    for ep in range(EPOCHS_DEFAULT):
        for (batch_data, occupant_data) in loader:
            batch_data = batch_data.to(device)
            occupant_data = occupant_data.to(device)
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.zero_()
            loss = model.contrastive_divergence(batch_data, occupant_data, k=CD_K_DEFAULT)
            optimizer.step()
    os.makedirs("clamped_data", exist_ok=True)
    out_name = f"trained_dbm_T{T}_vac{vac}_clamped.pt"
    out_path = os.path.join("clamped_data", out_name)
    torch.save(model.state_dict(), out_path)
    print(f"    Saved {out_path}")
    del model, optimizer, dataset, loader
    torch.cuda.empty_cache()

class DBN(nn.Module):
    def __init__(self, n_vis, n_hid1, n_hid2):
        super(DBN, self).__init__()
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

    def sample_h1_given_v(self, v):
        logits_h1 = F.linear(v, self.W1, self.b_h1)
        p_h1 = torch.sigmoid(logits_h1)
        h1_s = torch.bernoulli(p_h1)
        return p_h1, h1_s

    def sample_h2_given_h1(self, h1):
        logits_h2 = F.linear(h1, self.W2, self.b_h2)
        p_h2 = torch.sigmoid(logits_h2)
        h2_s = torch.bernoulli(p_h2)
        return p_h2, h2_s

    def sample_v_given_h1(self, h1):
        logits_v = F.linear(h1, self.W1.transpose(0,1), self.b_v)
        p_v = torch.sigmoid(logits_v)
        v_s = torch.bernoulli(p_v)
        return p_v, v_s

    def contrastive_divergence(self, v_data, occupant_data, k=1):
        v_neg = v_data.clone()
        for _ in range(k):
            _, h1_neg = self.sample_h1_given_v(v_neg)
            _, h2_neg = self.sample_h2_given_h1(h1_neg)
            p_v_neg, _ = self.sample_v_given_h1(h1_neg)
            v_neg = occupant_clamp(p_v_neg, occupant_data)
        p_h1_data, h1_data = self.sample_h1_given_v(v_data)
        _, h1_neg_final = self.sample_h1_given_v(v_neg)
        pos_v_sign = get_sign_bits(v_data)
        neg_v_sign = get_sign_bits(v_neg)
        pos_vh1 = torch.bmm(pos_v_sign.unsqueeze(2), h1_data.unsqueeze(1))
        neg_vh1 = torch.bmm(neg_v_sign.unsqueeze(2), h1_neg_final.unsqueeze(1))
        dW1_sign = (pos_vh1.mean(dim=0) - neg_vh1.mean(dim=0))
        dW1_full = torch.zeros_like(self.W1)
        sign_mask = torch.arange(self.n_vis, device=v_data.device)[1::2]
        dW1_full[:, sign_mask] = dW1_sign.transpose(0, 1)
        db_v_full = torch.zeros(self.n_vis, device=v_data.device)
        db_v_full[sign_mask] = (pos_v_sign - neg_v_sign).mean(dim=0)
        db_h1 = (h1_data - h1_neg_final).mean(dim=0)
        p_h2_data, h2_data = self.sample_h2_given_h1(h1_data)
        _, h2_neg_final = self.sample_h2_given_h1(h1_neg_final)
        pos_h1h2 = torch.bmm(h1_data.unsqueeze(2), h2_data.unsqueeze(1))
        neg_h1h2 = torch.bmm(h1_neg_final.unsqueeze(2), h2_neg_final.unsqueeze(1))
        dW2 = (pos_h1h2.mean(dim=0) - neg_h1h2.mean(dim=0))
        db_h2 = (h2_data - h2_neg_final).mean(dim=0)
        for p in [self.W1, self.b_v, self.b_h1, self.W2, self.b_h2]:
            if p.grad is not None:
                p.grad.zero_()
        self.W1.grad = -dW1_full
        self.b_v.grad = -db_v_full
        self.b_h1.grad = -db_h1
        self.W2.grad = -dW2
        self.b_h2.grad = -db_h2
        loss = bce_sign_bits_only(v_neg, v_data)
        return loss

def train_dbn_subset(spins_subset, T, vac):
    X_encoded = encode_spins_to_two_channels(spins_subset)
    occupant_arr = X_encoded[:, 0::2]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DBN(n_vis=2*L*L, n_hid1=HID1, n_hid2=HID2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    dataset = TensorDataset(torch.tensor(X_encoded, dtype=torch.float32),
                            torch.tensor(occupant_arr, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE_DEFAULT, shuffle=True)
    for ep in range(EPOCHS_DEFAULT):
        for (batch_data, occupant_data) in loader:
            batch_data = batch_data.to(device)
            occupant_data = occupant_data.to(device)
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.zero_()
            loss = model.contrastive_divergence(batch_data, occupant_data, k=CD_K_DEFAULT)
            optimizer.step()
    os.makedirs("clamped_data", exist_ok=True)
    out_name = f"trained_dbn_T{T}_vac{vac}_clamped.pt"
    out_path = os.path.join("clamped_data", out_name)
    torch.save(model.state_dict(), out_path)
    print(f"    Saved {out_path}")
    del model, optimizer, dataset, loader
    torch.cuda.empty_cache()

class DRBN_mod(nn.Module):
    def __init__(self, n_vis, n_hid1, n_hid2):
        super(DRBN_mod, self).__init__()
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

    def sample_h1_given_v(self, v):
        logits = F.linear(v, self.W1, self.b_h1)
        p_h1 = torch.sigmoid(logits)
        h1_s = torch.bernoulli(p_h1)
        return p_h1, h1_s

    def sample_h2_given_h1(self, h1):
        logits = F.linear(h1, self.W2, self.b_h2)
        p_h2 = torch.sigmoid(logits)
        h2_s = torch.bernoulli(p_h2)
        return p_h2, h2_s

    def sample_h1_given_h2(self, h2):
        logits = F.linear(h2, self.W2.transpose(0,1), self.b_h1)
        p_h1 = torch.sigmoid(logits)
        h1_s = torch.bernoulli(p_h1)
        return p_h1, h1_s

    def sample_v_given_h1(self, h1):
        logits = F.linear(h1, self.W1.transpose(0,1), self.b_v)
        p_v = torch.sigmoid(logits)
        v_s = torch.bernoulli(p_v)
        return p_v, v_s

    def contrastive_divergence(self, v_data, occupant_data, k=1):
        device = v_data.device
        B = v_data.size(0)
        with torch.no_grad():
            _, h1_data = self.sample_h1_given_v(v_data)
            _, h2_data = self.sample_h2_given_h1(h1_data)
        v_neg = v_data.clone()
        for _ in range(k):
            _, h1_up = self.sample_h1_given_v(v_neg)
            _, h2_up = self.sample_h2_given_h1(h1_up)
            _, h1_down = self.sample_h1_given_h2(h2_up)
            p_v_down, _ = self.sample_v_given_h1(h1_down)
            v_neg = occupant_clamp(p_v_down, occupant_data)
        with torch.no_grad():
            _, h1_neg_final = self.sample_h1_given_v(v_neg)
            _, h2_neg_final = self.sample_h2_given_h1(h1_neg_final)
        pos_v_sign = get_sign_bits(v_data)
        neg_v_sign = get_sign_bits(v_neg)
        pos_vh1 = torch.bmm(pos_v_sign.unsqueeze(2), h1_data.unsqueeze(1))
        neg_vh1 = torch.bmm(neg_v_sign.unsqueeze(2), h1_neg_final.unsqueeze(1))
        dW1_sign = (pos_vh1.mean(dim=0) - neg_vh1.mean(dim=0))
        dW1_full = torch.zeros_like(self.W1)
        sign_mask = torch.arange(self.n_vis, device=device)[1::2]
        dW1_full[:, sign_mask] = dW1_sign.transpose(0, 1)
        db_v_full = torch.zeros(self.n_vis, device=device)
        db_v_full[sign_mask] = (pos_v_sign - neg_v_sign).mean(dim=0)
        db_h1 = (h1_data - h1_neg_final).mean(dim=0)
        pos_h1h2 = torch.bmm(h1_data.unsqueeze(2), h2_data.unsqueeze(1))
        neg_h1h2 = torch.bmm(h1_neg_final.unsqueeze(2), h2_neg_final.unsqueeze(1))
        dW2 = (pos_h1h2.mean(dim=0) - neg_h1h2.mean(dim=0))
        db_h2 = (h2_data - h2_neg_final).mean(dim=0)
        for p in [self.W1, self.b_v, self.b_h1, self.W2, self.b_h2]:
            if p.grad is not None:
                p.grad.zero_()
        self.W1.grad = -dW1_full
        self.b_v.grad = -db_v_full
        self.b_h1.grad = -db_h1
        self.W2.grad = -dW2
        self.b_h2.grad = -db_h2
        loss = bce_sign_bits_only(v_neg, v_data)
        return loss

def train_drbn_subset(spins_subset, T, vac):
    X_encoded = encode_spins_to_two_channels(spins_subset)
    occupant_arr = X_encoded[:, 0::2]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DRBN_mod(n_vis=2*L*L, n_hid1=HID1, n_hid2=HID2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    dataset = TensorDataset(torch.tensor(X_encoded, dtype=torch.float32),
                            torch.tensor(occupant_arr, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE_DEFAULT, shuffle=True)
    for ep in range(EPOCHS_DEFAULT):
        for (batch_data, occupant_data) in loader:
            batch_data = batch_data.to(device)
            occupant_data = occupant_data.to(device)
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.zero_()
            loss = model.contrastive_divergence(batch_data, occupant_data, k=CD_K_DEFAULT)
            optimizer.step()
    os.makedirs("clamped_data", exist_ok=True)
    out_name = f"trained_drbn_T{T}_vac{vac}_clamped.pt"
    out_path = os.path.join("clamped_data", out_name)
    torch.save(model.state_dict(), out_path)
    print(f"    Saved {out_path}")
    del model, optimizer, dataset, loader
    torch.cuda.empty_cache()

def correlation_matrix_in_chunks(v, h, chunk_size=16):
    """
    Computes sum over smaller chunks for large B.
    """
    B, xdim = v.shape
    ydim = h.shape[1]
    device = v.device
    corr = torch.zeros(xdim, ydim, device=device, dtype=v.dtype)
    for start in range(0, B, chunk_size):
        end = start + chunk_size
        v_chunk = v[start:end]
        h_chunk = h[start:end]
        corr += torch.einsum('bi,bj->ij', v_chunk, h_chunk)
    corr /= B
    return corr

class DBMAdvanced(nn.Module):
    """
    2-layer DBM with extra coupling U_h2_v. Occupant bits pinned, sign bits learned.
    """
    def __init__(self, n_vis, n_hid1, n_hid2):
        super().__init__()
        self.n_vis  = n_vis
        self.n_hid1 = n_hid1
        self.n_hid2 = n_hid2
        self.W1  = nn.Parameter(torch.zeros(n_hid1, n_vis))
        self.b_v = nn.Parameter(torch.zeros(n_vis))
        self.b_h1= nn.Parameter(torch.zeros(n_hid1))
        self.W2  = nn.Parameter(torch.zeros(n_hid1, n_hid2))
        self.b_h2= nn.Parameter(torch.zeros(n_hid2))
        self.U_h2_v = nn.Parameter(torch.zeros(n_hid2, n_vis))
        nn.init.xavier_uniform_(self.W1)
        nn.init.xavier_uniform_(self.W2)
        nn.init.xavier_uniform_(self.U_h2_v)

    def occupant_clamp(self, v_prob, occupant_data):
        return occupant_clamp(v_prob, occupant_data)

    def sample_h1_given_vh2(self, v, h2):
        act_h1 = F.linear(v, self.W1, self.b_h1) + F.linear(h2, self.W2.transpose(0,1))
        p_h1 = torch.sigmoid(act_h1)
        h1_s = torch.bernoulli(p_h1)
        return p_h1, h1_s

    def sample_h2_given_h1(self, v, h1):
        act_h2 = F.linear(h1, self.W2, self.b_h2) + F.linear(v, self.U_h2_v)
        p_h2 = torch.sigmoid(act_h2)
        h2_s = torch.bernoulli(p_h2)
        return p_h2, h2_s

    def sample_v_given_h1h2(self, h1, h2):
        act_v = (
            F.linear(h1, self.W1.transpose(0,1), self.b_v)
            + F.linear(h2, self.U_h2_v.transpose(0,1))
        )
        p_v = torch.sigmoid(act_v)
        v_s = torch.bernoulli(p_v)
        return p_v, v_s

    def positive_phase_inference(self, v_data, M=5):
        """
        Small mean-field loop for positive phase.
        """
        B = v_data.size(0)
        device = v_data.device
        h1 = torch.zeros(B, self.n_hid1, device=device)
        h2 = torch.zeros(B, self.n_hid2, device=device)
        for _ in range(M):
            act_h1 = F.linear(v_data, self.W1, self.b_h1) + F.linear(h2, self.W2.transpose(0,1))
            p_h1 = torch.sigmoid(act_h1)
            h1 = torch.bernoulli(p_h1)
            act_h2 = F.linear(h1, self.W2, self.b_h2) + F.linear(v_data, self.U_h2_v)
            p_h2 = torch.sigmoid(act_h2)
            h2 = torch.bernoulli(p_h2)
        return h1, h2

    def contrastive_divergence(self, v_data, occupant_data, k=1, pos_passes=5):
        B = v_data.size(0)
        device = v_data.device
        h1_data, h2_data = self.positive_phase_inference(v_data, M=pos_passes)
        v_neg = v_data.clone()
        for _ in range(k):
            _, h1_neg = self.sample_h1_given_vh2(v_neg, torch.zeros(B, self.n_hid2, device=device))
            _, h2_neg = self.sample_h2_given_h1(v_neg, h1_neg)
            p_v_neg, _ = self.sample_v_given_h1h2(h1_neg, h2_neg)
            v_neg = occupant_clamp(p_v_neg, occupant_data)
        with torch.no_grad():
            _, h1_neg_final = self.sample_h1_given_vh2(v_neg, torch.zeros(B, self.n_hid2, device=device))
            _, h2_neg_final = self.sample_h2_given_h1(v_neg, h1_neg_final)
        pos_v_sign = get_sign_bits(v_data)
        neg_v_sign = get_sign_bits(v_neg)
        pos_vh1_mat = correlation_matrix_in_chunks(pos_v_sign, h1_data)
        neg_vh1_mat = correlation_matrix_in_chunks(neg_v_sign, h1_neg_final)
        dW1_sign = pos_vh1_mat - neg_vh1_mat
        dW1_full = torch.zeros_like(self.W1)
        sign_mask = torch.arange(self.n_vis, device=device)[1::2]
        dW1_full[:, sign_mask] = dW1_sign.transpose(0, 1)
        db_v_full = torch.zeros(self.n_vis, device=device)
        db_v_full[sign_mask] = (pos_v_sign - neg_v_sign).mean(dim=0)
        db_h1 = (h1_data - h1_neg_final).mean(dim=0)
        pos_h1h2 = correlation_matrix_in_chunks(h1_data, h2_data)
        neg_h1h2 = correlation_matrix_in_chunks(h1_neg_final, h2_neg_final)
        dW2 = pos_h1h2 - neg_h1h2
        db_h2 = (h2_data - h2_neg_final).mean(dim=0)
        pos_h2v = correlation_matrix_in_chunks(h2_data, pos_v_sign)
        neg_h2v = correlation_matrix_in_chunks(h2_neg_final, neg_v_sign)
        dU = pos_h2v - neg_h2v
        dU_full = torch.zeros_like(self.U_h2_v)
        dU_full[:, sign_mask] = dU
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

class SimpleRBM(nn.Module):
    """
    Simple RBM used only for pretraining hidden layers.
    """
    def __init__(self, n_vis, n_hid):
        super().__init__()
        self.n_vis = n_vis
        self.n_hid = n_hid
        self.W = nn.Parameter(torch.zeros(n_vis, n_hid))
        self.v_bias = nn.Parameter(torch.zeros(n_vis))
        self.h_bias = nn.Parameter(torch.zeros(n_hid))
        nn.init.xavier_uniform_(self.W)

    def sample_h_given_v(self, v):
        p_h = torch.sigmoid(v @ self.W + self.h_bias)
        return p_h, torch.bernoulli(p_h)

    def sample_v_given_h(self, h):
        p_v = torch.sigmoid(h @ self.W.t() + self.v_bias)
        return p_v, torch.bernoulli(p_v)

    def contrastive_divergence(self, v_data, k=1):
        _, h_data = self.sample_h_given_v(v_data)
        v_neg = v_data.clone()
        h_neg = h_data.clone()
        for _ in range(k):
            _, v_neg = self.sample_v_given_h(h_neg)
            _, h_neg = self.sample_h_given_v(v_neg)
        loss = bce_sign_bits_only(v_neg, v_data)
        return loss

def pretrain_2_rbm(X_tensor, n_vis, n_hid1, n_hid2, epochs=30, cd_k=10, lr=1e-3):
    """
    Greedy pretraining of two stacked RBMs, then copying weights to DBMAdvanced.
    """
    device = X_tensor.device
    rbm1 = SimpleRBM(n_vis, n_hid1).to(device)
    opt1 = optim.Adam(rbm1.parameters(), lr=lr)
    ds1 = TensorDataset(X_tensor)
    ld1 = DataLoader(ds1, batch_size=64, shuffle=True)
    for ep in range(epochs):
        for (batch_data,) in ld1:
            batch_data = batch_data.to(device)
            opt1.zero_grad()
            loss = rbm1.contrastive_divergence(batch_data, k=cd_k)
            loss.backward()
            opt1.step()
    hidden_data_list = []
    for (batch_data,) in ld1:
        batch_data = batch_data.to(device)
        with torch.no_grad():
            _, h_samp = rbm1.sample_h_given_v(batch_data)
        hidden_data_list.append(h_samp.cpu())
    hidden_data = torch.cat(hidden_data_list, dim=0).to(device)
    rbm2 = SimpleRBM(n_hid1, n_hid2).to(device)
    opt2 = optim.Adam(rbm2.parameters(), lr=lr)
    ds2 = TensorDataset(hidden_data)
    ld2 = DataLoader(ds2, batch_size=64, shuffle=True)
    for ep in range(epochs):
        for (batch_h1,) in ld2:
            batch_h1 = batch_h1.to(device)
            opt2.zero_grad()
            loss2 = rbm2.contrastive_divergence(batch_h1, k=cd_k)
            loss2.backward()
            opt2.step()
    dbm = DBMAdvanced(n_vis, n_hid1, n_hid2)
    with torch.no_grad():
        dbm.W1.copy_(rbm1.W.t())
        dbm.b_v.copy_(rbm1.v_bias)
        dbm.b_h1.copy_(rbm1.h_bias)
        dbm.W2.copy_(rbm2.W)
        dbm.b_h2.copy_(rbm2.h_bias)
    del rbm1, rbm2, opt1, opt2, ds1, ds2, ld1, ld2
    torch.cuda.empty_cache()
    return dbm

def reduce_dataset_if_needed(spins_subset, max_size=3000, seed=1234):
    np.random.seed(seed)
    n_all = spins_subset.shape[0]
    if n_all > max_size:
        idxs = np.random.choice(n_all, size=max_size, replace=False)
        spins_subset = spins_subset[idxs]
    return spins_subset

def train_dbm_advanced_subset(spins_subset, T, vac, use_pt=False, use_pcd=False):
    spins_subset = reduce_dataset_if_needed(spins_subset, max_size=MAX_DATA_ADVANCED)
    X_encoded = torch.tensor(encode_spins_to_two_channels(spins_subset), dtype=torch.float32)
    occupant_arr = X_encoded[:, 0::2]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_encoded = X_encoded.to(device)
    occupant_arr = occupant_arr.to(device)
    model = DBMAdvanced(n_vis=2*L*L, n_hid1=HID1_ADV, n_hid2=HID2_ADV).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    ds = TensorDataset(X_encoded, occupant_arr)
    ld = DataLoader(ds, batch_size=BATCH_SIZE_ADVANCED, shuffle=True)
    POS_PASSES = 5
    for ep in range(EPOCHS_ADVANCED):
        for (batch_data, occupant_data) in ld:
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.zero_()
            loss = model.contrastive_divergence(
                batch_data, occupant_data, k=CD_K_DEFAULT, pos_passes=POS_PASSES
            )
            optimizer.step()
    suffix = ""
    if use_pt:
        suffix = "_pt"
    elif use_pcd:
        suffix = "_pcd"
    out_name = f"trained_dbmAdvanced_T{T}_vac{vac}_clamped{suffix}.pt"
    out_path = os.path.join("clamped_data", out_name)
    torch.save(model.state_dict(), out_path)
    print(f"    Saved {out_path}")
    del model, optimizer, ds, ld, X_encoded, occupant_arr
    torch.cuda.empty_cache()

def train_dbm_advanced_pretrain_subset(spins_subset, T, vac):
    spins_subset = reduce_dataset_if_needed(spins_subset, max_size=MAX_DATA_ADVANCED)
    X_encoded = torch.tensor(encode_spins_to_two_channels(spins_subset), dtype=torch.float32)
    occupant_arr = X_encoded[:, 0::2]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_encoded = X_encoded.to(device)
    dbm = pretrain_2_rbm(
        X_encoded, n_vis=2*L*L, n_hid1=HID1_ADV, n_hid2=HID2_ADV,
        epochs=30, cd_k=10, lr=1e-3
    ).to(device)
    occupant_arr = occupant_arr.to(device)
    ds = TensorDataset(X_encoded, occupant_arr)
    ld = DataLoader(ds, batch_size=BATCH_SIZE_ADVANCED, shuffle=True)
    optimizer = optim.Adam(dbm.parameters(), lr=1e-3)
    POS_PASSES = 5
    for ep in range(EPOCHS_ADVANCED):
        for (batch_data, occupant_data) in ld:
            for p in dbm.parameters():
                if p.grad is not None:
                    p.grad.zero_()
            loss = dbm.contrastive_divergence(
                batch_data, occupant_data, k=CD_K_DEFAULT, pos_passes=POS_PASSES
            )
            optimizer.step()
    out_name = f"dbm_pretrained_T{T}_vac{vac}_clamped_advanced.pt"
    out_path = os.path.join("clamped_data", out_name)
    torch.save(dbm.state_dict(), out_path)
    print(f"    Saved {out_path}")
    del dbm, optimizer, ds, ld, occupant_arr
    torch.cuda.empty_cache()

def train_all_rbm(spins_all, temps_all, vacs_all, all_subsets):
    for (T_val, vac) in all_subsets:
        mask = (np.isclose(temps_all, T_val, atol=1e-3) &
                np.isclose(vacs_all,  vac,   atol=1e-3))
        spins_subset = spins_all[mask]
        if spins_subset.shape[0] == 0:
            print(f"No data for T={T_val}, vac={vac}. Skipping RBM.")
            continue
        print(f"Training RBM (clamped) on T={T_val}, vac={vac}, #samples={spins_subset.shape[0]}")
        train_rbm_subset(spins_subset, T_val, vac)

def train_all_dbm(spins_all, temps_all, vacs_all, all_subsets):
    for (T_val, vac) in all_subsets:
        mask = (np.isclose(temps_all, T_val, atol=1e-3) &
                np.isclose(vacs_all,  vac,   atol=1e-3))
        spins_subset = spins_all[mask]
        if spins_subset.shape[0] == 0:
            print(f"No data for T={T_val}, vac={vac}. Skipping DBM.")
            continue
        print(f"Training DBM (clamped) on T={T_val}, vac={vac}, #samples={spins_subset.shape[0]}")
        train_dbm_subset(spins_subset, T_val, vac)

def train_all_dbn(spins_all, temps_all, vacs_all, all_subsets):
    for (T_val, vac) in all_subsets:
        mask = (np.isclose(temps_all, T_val, atol=1e-3) &
                np.isclose(vacs_all,  vac,   atol=1e-3))
        spins_subset = spins_all[mask]
        if spins_subset.shape[0] == 0:
            print(f"No data for T={T_val}, vac={vac}. Skipping DBN.")
            continue
        print(f"Training DBN (clamped) on T={T_val}, vac={vac}, #samples={spins_subset.shape[0]}")
        train_dbn_subset(spins_subset, T_val, vac)

def train_all_drbn(spins_all, temps_all, vacs_all, all_subsets):
    for (T_val, vac) in all_subsets:
        mask = (np.isclose(temps_all, T_val, atol=1e-3) &
                np.isclose(vacs_all,  vac,   atol=1e-3))
        spins_subset = spins_all[mask]
        if spins_subset.shape[0] == 0:
            print(f"No data for T={T_val}, vac={vac}. Skipping DRBN.")
            continue
        print(f"Training DRBN (clamped) on T={T_val}, vac={vac}, #samples={spins_subset.shape[0]}")
        train_drbn_subset(spins_subset, T_val, vac)

def train_all_dbm_advanced(spins_all, temps_all, vacs_all, all_subsets):
    for (T_val, vac) in all_subsets:
        mask = (np.isclose(temps_all, T_val, atol=1e-3) &
                np.isclose(vacs_all,  vac,   atol=1e-3))
        spins_subset = spins_all[mask]
        n_subset = spins_subset.shape[0]
        if n_subset == 0:
            print(f"No data for T={T_val}, vac={vac}. Skipping DBMAdvanced.")
            continue
        offset = round(T_val - Tc_map[vac], 3)
        print(f"Training DBMAdvanced (clamped) on T={T_val}, vac={vac}, #samples={n_subset}")
        print("   -> advanced DBM from scratch")
        train_dbm_advanced_subset(spins_subset, T_val, vac, use_pt=False, use_pcd=False)
        print("   -> advanced DBM with pretraining")
        train_dbm_advanced_pretrain_subset(spins_subset, T_val, vac)
        if offset in [-0.8, 0.0]:
            print("   -> advanced DBM + PT")
            train_dbm_advanced_subset(spins_subset, T_val, vac, use_pt=True, use_pcd=False)
        if offset in [-0.8, 0.0]:
            print("   -> advanced DBM + PCD")
            train_dbm_advanced_subset(spins_subset, T_val, vac, use_pt=False, use_pcd=True)

def main():
    if not os.path.isfile(data_file):
        raise FileNotFoundError(f"[ERROR] data file '{data_file}' not found.")
    data_npz = np.load(data_file)
    spins_all = data_npz["spins"]
    temps_all = data_npz["temperature"]
    vacs_all  = data_npz["vacancy_fraction"]
    if spins_all.ndim == 3 and spins_all.shape[0] == spins_all.shape[1]:
        spins_all = np.transpose(spins_all, (2, 0, 1))
    all_subsets = []
    for vac in vacancy_list:
        Tc = Tc_map[vac]
        for offset in temperature_offsets:
            T_val = round(Tc + offset, 3)
            all_subsets.append((T_val, vac))
    train_all_rbm(spins_all, temps_all, vacs_all, all_subsets)
    train_all_dbm(spins_all, temps_all, vacs_all, all_subsets)
    train_all_dbn(spins_all, temps_all, vacs_all, all_subsets)
    train_all_drbn(spins_all, temps_all, vacs_all, all_subsets)
    train_all_dbm_advanced(spins_all, temps_all, vacs_all, all_subsets)

if __name__ == "__main__":
    main()
