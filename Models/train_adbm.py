#!/usr/bin/env python3

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

if not torch.cuda.is_available():
    raise RuntimeError("No GPU found! Please run on a machine with CUDA support.")
device = torch.device("cuda")
os.makedirs("advanced_data", exist_ok=True)

L = 16
vacancy_list = [0.0, 0.1, 0.2]
Tc_map = {0.0: 2.3, 0.1: 1.975, 0.2: 1.6}
temperature_offsets = [-0.8, -0.1, -0.05, 0.0, 0.05, 0.1, 0.5]
data_file = "data/disordered_snapshots_16.npz"
EPOCHS_DEFAULT = 50
CD_K_DEFAULT = 10
BATCH_SIZE_DEFAULT = 64
RBM_HIDDEN = 128
HID1 = 64
HID2 = 64

def encode_spins_to_two_channels(spin_configs):
    """
    Convert spins to a 2-channel binary representation (occupancy, sign).
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

    def sample_v_given_h_clamp_occupant(self, h, occupant_data):
        """
        Clamps the occupancy bits using occupant_data, only sampling sign bits.
        """
        p_v_all = torch.sigmoid(h @ self.W.t() + self.v_bias)
        occ_p = p_v_all[:, 0::2]
        sign_p = p_v_all[:, 1::2]
        sign_s = torch.bernoulli(sign_p)
        v_out = torch.zeros_like(p_v_all)
        v_out[:, 0::2] = occupant_data
        v_out[:, 1::2] = sign_s
        return v_out

    def masked_bce_occupancy_sign(self, v_pred, v_true, eps=1e-7):
        """
        BCE loss, applying sign loss only to occupied sites.
        """
        occ_pred = v_pred[:, 0::2]
        occ_true = v_true[:, 0::2]
        sign_pred = v_pred[:, 1::2]
        sign_true = v_true[:, 1::2]
        occ_loss = -(
            occ_true * torch.log(occ_pred + eps) +
            (1 - occ_true) * torch.log(1 - occ_pred + eps)
        )
        sign_mask = (occ_true > 0.5).float()
        sign_loss_raw = -(
            sign_true * torch.log(sign_pred + eps) +
            (1 - sign_true) * torch.log(1 - sign_pred + eps)
        )
        sign_loss = sign_loss_raw * sign_mask
        return torch.mean(occ_loss) + torch.mean(sign_loss)

    def contrastive_divergence(self, v_data, k=1, clamp_occupancy=False):
        """
        Run k steps of Gibbs sampling; clamp occupancy if specified.
        """
        _, h_data = self.sample_h_given_v(v_data)
        occupant_data = v_data[:, 0::2] if clamp_occupancy else None
        v_neg = v_data.clone()
        h_neg = h_data.clone()
        for _ in range(k):
            if clamp_occupancy and occupant_data is not None:
                v_neg = self.sample_v_given_h_clamp_occupant(h_neg, occupant_data)
            else:
                _, v_neg = self.sample_v_given_h(h_neg)
            _, h_neg = self.sample_h_given_v(v_neg)
        p_v_final, _ = self.sample_v_given_h(h_neg)
        recon_loss = self.masked_bce_occupancy_sign(p_v_final, v_data)
        return recon_loss

class DBM(nn.Module):
    """
    Simple 2-layer DBM.
    """
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

    def masked_bce_occupancy_sign(self, v_pred, v_true, eps=1e-7):
        """
        BCE loss, applying sign loss only to occupied sites.
        """
        occ_pred = v_pred[:, 0::2]
        occ_true = v_true[:, 0::2]
        sign_pred = v_pred[:, 1::2]
        sign_true = v_true[:, 1::2]
        occ_loss = -(
            occ_true * torch.log(occ_pred + eps) +
            (1 - occ_true) * torch.log(1 - occ_pred + eps)
        )
        sign_mask = (occ_true > 0.5).float()
        sign_loss_raw = -(
            sign_true * torch.log(sign_pred + eps) +
            (1 - sign_true) * torch.log(1 - sign_pred + eps)
        )
        sign_loss = sign_loss_raw * sign_mask
        return torch.mean(occ_loss) + torch.mean(sign_loss)

    def contrastive_divergence(self, v_data, k=1, clamp_occupancy=False):
        """
        Contrastive divergence for DBM, with optional occupancy clamping.
        """
        device = v_data.device
        batch_size = v_data.size(0)
        with torch.no_grad():
            h2_zeros = torch.zeros(batch_size, self.n_hid2, device=device)
            _, h1_data = self.sample_h1_given_v_h2(v_data, h2_zeros)
            _, h2_data = self.sample_h2_given_h1(h1_data)
        occupant_data = v_data[:, 0::2] if clamp_occupancy else None
        v_model = v_data.clone()
        h2_temp = torch.zeros_like(h2_data)
        for _ in range(k):
            _, h1_neg = self.sample_h1_given_v_h2(v_model, h2_temp)
            _, h2_neg = self.sample_h2_given_h1(h1_neg)
            p_v, v_new = self.sample_v_given_h1(h1_neg)
            if clamp_occupancy and occupant_data is not None:
                reassembled = torch.zeros_like(p_v)
                reassembled[:, 0::2] = occupant_data
                sign_s = torch.bernoulli(p_v[:, 1::2])
                reassembled[:, 1::2] = sign_s
                v_model = reassembled
            else:
                v_model = v_new
            h2_temp = h2_neg
        p_v_final, _ = self.sample_v_given_h1(h1_neg)
        recon_loss = self.masked_bce_occupancy_sign(p_v_final, v_data)
        return recon_loss

class DBMAdvanced(nn.Module):
    """
    Advanced DBM with optional PT, PCD, and occupancy clamping.
    """
    def __init__(
        self, n_vis, n_hid1, n_hid2,
        use_pt=False, use_pcd=False, n_pt=10, betas=None,
        buffer_size=10000, lr=1e-4, clamp_occupancy=False
    ):
        super().__init__()
        self.n_vis = n_vis
        self.n_hid1 = n_hid1
        self.n_hid2 = n_hid2
        self.use_pt = use_pt
        self.use_pcd = use_pcd
        self.n_pt = n_pt
        if betas is None:
            self.betas = np.linspace(0.0, 1.0, n_pt)
        else:
            self.betas = betas
        self.buffer_size = buffer_size
        self.lr = lr
        self.clamp_occupancy = clamp_occupancy
        self.W1 = nn.Parameter(torch.zeros(n_hid1, n_vis))
        self.b_v = nn.Parameter(torch.zeros(n_vis))
        self.b_h1 = nn.Parameter(torch.zeros(n_hid1))
        self.W2 = nn.Parameter(torch.zeros(n_hid1, n_hid2))
        self.b_h2 = nn.Parameter(torch.zeros(n_hid2))
        self.U_h2_v = nn.Parameter(torch.zeros(n_hid2, n_vis))
        nn.init.xavier_uniform_(self.W1, gain=1.0)
        nn.init.xavier_uniform_(self.W2, gain=1.0)
        nn.init.xavier_uniform_(self.U_h2_v, gain=1.0)
        nn.init.constant_(self.b_v, 0.0)
        nn.init.constant_(self.b_h1, 0.0)
        nn.init.constant_(self.b_h2, 0.0)
        if self.use_pt:
            self.register_buffer(
                "pt_buffer", torch.bernoulli(torch.rand(self.n_pt, self.buffer_size, n_vis))
            )
        elif self.use_pcd:
            self.register_buffer(
                "pcd_buffer", torch.bernoulli(torch.rand(self.buffer_size, n_vis))
            )

    def occupant_clamp_2channel(self, p_v, occupant_data):
        """
        Clamp occupancy bits from occupant_data, sample sign bits.
        """
        out = torch.zeros_like(p_v)
        out[:, 0::2] = occupant_data
        sign_prob = p_v[:, 1::2]
        sign_samp = torch.bernoulli(sign_prob)
        out[:, 1::2] = sign_samp
        return out

    def sample_h1_given_vh2(self, v, h2, beta=1.0):
        act_h1 = beta * (
            F.linear(v, self.W1, self.b_h1) +
            F.linear(h2, self.W2.transpose(0,1))
        )
        p_h1 = torch.sigmoid(act_h1)
        h1_s = torch.bernoulli(p_h1)
        return p_h1, h1_s

    def sample_h2_given_h1(self, v, h1, beta=1.0):
        act_h2 = beta * (
            F.linear(h1, self.W2, self.b_h2) +
            F.linear(v, self.U_h2_v)
        )
        p_h2 = torch.sigmoid(act_h2)
        h2_s = torch.bernoulli(p_h2)
        return p_h2, h2_s

    def sample_v_given_h1h2(self, h1, h2, beta=1.0):
        act_v = beta * (
            F.linear(h1, self.W1.transpose(0,1), self.b_v) +
            F.linear(h2, self.U_h2_v.transpose(0,1))
        )
        p_v = torch.sigmoid(act_v)
        v_s = torch.bernoulli(p_v)
        return p_v, v_s

    def positive_phase_inference(self, v_data, M=5):
        """
        Mean-field for positive phase.
        """
        device = v_data.device
        batch_size = v_data.size(0)
        h1 = torch.zeros(batch_size, self.n_hid1, device=device)
        h2 = torch.zeros(batch_size, self.n_hid2, device=device)
        for _ in range(M):
            act_h1 = (F.linear(v_data, self.W1, self.b_h1) +
                      F.linear(h2, self.W2.transpose(0,1)))
            p_h1 = torch.sigmoid(act_h1)
            h1 = torch.bernoulli(p_h1)
            act_h2 = (F.linear(h1, self.W2, self.b_h2) +
                      F.linear(v_data, self.U_h2_v))
            p_h2 = torch.sigmoid(act_h2)
            h2 = torch.bernoulli(p_h2)
        return h1, h2

    def sample_negative_phase_batch(self, batch_size, occupant_data=None, k=1):
        device = next(self.parameters()).device
        v = torch.bernoulli(torch.rand(batch_size, self.n_vis, device=device))
        h2 = torch.zeros(batch_size, self.n_hid2, device=device)
        for _ in range(k):
            _, h1 = self.sample_h1_given_vh2(v, h2)
            _, h2 = self.sample_h2_given_h1(v, h1)
            p_v, v_new = self.sample_v_given_h1h2(h1, h2)
            if occupant_data is not None:
                v = self.occupant_clamp_2channel(p_v, occupant_data)
            else:
                v = v_new
        _, h1_final = self.sample_h1_given_vh2(v, h2)
        _, h2_final = self.sample_h2_given_h1(v, h1_final)
        return v, h1_final, h2_final

    def sample_negative_phase_pcd(self, batch_size, occupant_data=None, k=1):
        """
        Use PCD buffer for negative phase.
        """
        device = next(self.parameters()).device
        n_buf = self.pcd_buffer.size(0)
        if n_buf < batch_size:
            return self.sample_negative_phase_batch(batch_size, occupant_data, k)
        idxs = np.random.choice(n_buf, size=batch_size, replace=False)
        v = self.pcd_buffer[idxs].to(device)
        h2 = torch.zeros(batch_size, self.n_hid2, device=device)
        for _ in range(k):
            _, h1 = self.sample_h1_given_vh2(v, h2)
            _, h2 = self.sample_h2_given_h1(v, h1)
            p_v, v_new = self.sample_v_given_h1h2(h1, h2)
            if occupant_data is not None:
                v = self.occupant_clamp_2channel(p_v, occupant_data)
            else:
                v = v_new
        _, h1_final = self.sample_h1_given_vh2(v, h2)
        _, h2_final = self.sample_h2_given_h1(v, h1_final)
        self.pcd_buffer[idxs] = v.detach()
        return v, h1_final, h2_final

    def free_energy_approx(self, v, beta=1.0, mean_field_steps=2):
        """
        Approximate free energy for PT swaps.
        """
        with torch.no_grad():
            h2 = torch.zeros(v.size(0), self.n_hid2, device=v.device)
            for _ in range(mean_field_steps):
                act_h1 = beta * (
                    F.linear(v, self.W1, self.b_h1) +
                    F.linear(h2, self.W2.transpose(0,1))
                )
                p_h1 = torch.sigmoid(act_h1)
                act_h2 = beta * (
                    F.linear(p_h1, self.W2, self.b_h2) +
                    F.linear(v, self.U_h2_v)
                )
                p_h2 = torch.sigmoid(act_h2)
                h2 = p_h2
        E_v = torch.sum(self.b_v * v, dim=1)
        return -E_v

    def sample_negative_phase_pt(self, batch_size, occupant_data=None, k=1):
        """
        Parallel Tempering negative phase.
        """
        device = next(self.parameters()).device
        buf = self.pt_buffer
        n_buf = buf.size(1)
        if n_buf < batch_size:
            return self.sample_negative_phase_batch(batch_size, occupant_data, k)
        idxs = np.random.choice(n_buf, size=batch_size, replace=False)
        v_reps = buf[:, idxs].to(device)
        for _ in range(k):
            for r in range(self.n_pt):
                beta_r = self.betas[r]
                v_batch = v_reps[r]
                h2 = torch.zeros(batch_size, self.n_hid2, device=device)
                _, h1 = self.sample_h1_given_vh2(v_batch, h2, beta=beta_r)
                _, h2 = self.sample_h2_given_h1(v_batch, h1, beta=beta_r)
                p_v, v_new = self.sample_v_given_h1h2(h1, h2, beta=beta_r)
                if occupant_data is not None:
                    v_reps[r] = self.occupant_clamp_2channel(p_v, occupant_data)
                else:
                    v_reps[r] = v_new
            for r in range(self.n_pt - 1):
                beta1 = self.betas[r]
                beta2 = self.betas[r + 1]
                v1 = v_reps[r]
                v2 = v_reps[r + 1]
                E1 = self.free_energy_approx(v1, beta=beta1, mean_field_steps=2)
                E2 = self.free_energy_approx(v2, beta=beta2, mean_field_steps=2)
                E1_swapped = self.free_energy_approx(v1, beta=beta2, mean_field_steps=2)
                E2_swapped = self.free_energy_approx(v2, beta=beta1, mean_field_steps=2)
                delta = (E1_swapped + E2_swapped) - (E1 + E2)
                accept_prob = torch.exp(-delta)
                rand_vals = torch.rand_like(accept_prob)
                swap_mask = (accept_prob > rand_vals)
                swap_idx = swap_mask.nonzero(as_tuple=True)[0]
                if len(swap_idx) > 0:
                    temp1 = v1[swap_idx, :].clone()
                    temp2 = v2[swap_idx, :].clone()
                    v_reps[r][swap_idx, :] = temp2
                    v_reps[r + 1][swap_idx, :] = temp1
        v_neg = v_reps[-1]
        _, h1_neg = self.sample_h1_given_vh2(
            v_neg, torch.zeros(batch_size, self.n_hid2, device=device), beta=1.0
        )
        _, h2_neg = self.sample_h2_given_h1(v_neg, h1_neg, beta=1.0)
        for r in range(self.n_pt):
            buf[r, idxs] = v_reps[r].detach()
        return v_neg, h1_neg, h2_neg

    def masked_bce_occupancy_sign(self, v_pred, v_true, eps=1e-7):
        """
        BCE loss, applying sign loss only to occupied sites.
        """
        occ_pred = v_pred[:, 0::2]
        occ_true = v_true[:, 0::2]
        sign_pred = v_pred[:, 1::2]
        sign_true = v_true[:, 1::2]
        occ_loss = -(
            occ_true * torch.log(occ_pred + eps) +
            (1 - occ_true) * torch.log(1 - occ_pred + eps)
        )
        sign_mask = (occ_true > 0.5).float()
        sign_loss_raw = -(
            sign_true * torch.log(sign_pred + eps) +
            (1 - sign_true) * torch.log(1 - sign_pred + eps)
        )
        sign_loss = sign_loss_raw * sign_mask
        return torch.mean(occ_loss) + torch.mean(sign_loss)

    def contrastive_divergence(self, v_data, k=1, pos_passes=5, use_masked_bce=True):
        """
        Main CD training step with mean-field positive phase.
        """
        device = v_data.device
        batch_size = v_data.size(0)
        h1_data, h2_data = self.positive_phase_inference(v_data, M=pos_passes)
        occupant_data = None
        if self.clamp_occupancy:
            occupant_data = v_data[:, 0::2]
        if self.use_pt:
            v_neg, h1_neg, h2_neg = self.sample_negative_phase_pt(batch_size, occupant_data, k=k)
        elif self.use_pcd:
            v_neg, h1_neg, h2_neg = self.sample_negative_phase_pcd(batch_size, occupant_data, k=k)
        else:
            v_neg, h1_neg, h2_neg = self.sample_negative_phase_batch(batch_size, occupant_data, k=k)
        pos_vh1 = torch.bmm(v_data.unsqueeze(2), h1_data.unsqueeze(1))
        neg_vh1 = torch.bmm(v_neg.unsqueeze(2), h1_neg.unsqueeze(1))
        dW1 = pos_vh1.mean(dim=0) - neg_vh1.mean(dim=0)
        dW1_t = dW1.transpose(0, 1)
        db_v = (v_data - v_neg).mean(dim=0)
        db_h1 = (h1_data - h1_neg).mean(dim=0)
        pos_h1h2 = torch.bmm(h1_data.unsqueeze(2), h2_data.unsqueeze(1))
        neg_h1h2 = torch.bmm(h1_neg.unsqueeze(2), h2_neg.unsqueeze(1))
        dW2 = pos_h1h2.mean(dim=0) - neg_h1h2.mean(dim=0)
        db_h2 = (h2_data - h2_neg).mean(dim=0)
        pos_h2v = torch.bmm(h2_data.unsqueeze(2), v_data.unsqueeze(1))
        neg_h2v = torch.bmm(h2_neg.unsqueeze(2), v_neg.unsqueeze(1))
        dU_h2v = pos_h2v.mean(dim=0) - neg_h2v.mean(dim=0)
        self.W1.grad = -dW1_t
        self.b_v.grad = -db_v
        self.b_h1.grad = -db_h1
        self.W2.grad = -dW2
        self.b_h2.grad = -db_h2
        self.U_h2_v.grad = -dU_h2v
        if use_masked_bce:
            recon_loss = self.masked_bce_occupancy_sign(v_neg, v_data)
        else:
            recon_loss = F.mse_loss(v_neg, v_data)
        return recon_loss

def pretrain_dbm_advanced_subset(
    spins_subset, T, vac,
    n_vis=2*L*L, nh1=64, nh2=64,
    rbm_epochs=50, cd_k=10, lr=1e-3,
    do_advanced_finetune=True, finetune_epochs=30,
    advanced_lr_scale=0.1, use_pt=False, use_pcd=False, suffix_extra=""
):
    """
    Pretrain DBMAdvanced with two RBMs, then optional fine-tuning.
    """
    X_encoded = torch.tensor(encode_spins_to_two_channels(spins_subset), dtype=torch.float32, device=device)
    rbm1 = RBM(n_vis, nh1).to(device)
    opt1 = optim.Adam(rbm1.parameters(), lr=lr)
    dataset = TensorDataset(X_encoded)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE_DEFAULT, shuffle=True)
    for epoch in range(rbm_epochs):
        for (batch_data,) in loader:
            opt1.zero_grad()
            loss = rbm1.contrastive_divergence(batch_data, k=cd_k, clamp_occupancy=False)
            loss.backward()
            opt1.step()
    rbm1.eval()
    hidden_data_list = []
    for (batch_data,) in loader:
        with torch.no_grad():
            _, h_samp = rbm1.sample_h_given_v(batch_data)
        hidden_data_list.append(h_samp)
    hidden_data = torch.cat(hidden_data_list, dim=0)
    rbm2 = RBM(n_vis=nh1, n_hid=nh2).to(device)
    opt2 = optim.Adam(rbm2.parameters(), lr=lr)
    dataset2 = TensorDataset(hidden_data)
    loader2 = DataLoader(dataset2, batch_size=BATCH_SIZE_DEFAULT, shuffle=True)
    for epoch in range(rbm_epochs):
        for (batch_h1,) in loader2:
            opt2.zero_grad()
            loss = rbm2.contrastive_divergence(batch_h1, k=cd_k, clamp_occupancy=False)
            loss.backward()
            opt2.step()
    dbm = DBMAdvanced(
        n_vis=n_vis, n_hid1=nh1, n_hid2=nh2,
        use_pt=use_pt, use_pcd=use_pcd, lr=lr
    ).to(device)
    with torch.no_grad():
        dbm.W1.copy_(rbm1.W.t())
        dbm.b_v.copy_(rbm1.v_bias)
        dbm.b_h1.copy_(rbm1.h_bias)
        dbm.W2.copy_(rbm2.W)
        dbm.b_h2.copy_(rbm2.h_bias)
    if do_advanced_finetune:
        adv_lr = lr * advanced_lr_scale
        adv_optimizer = optim.Adam(dbm.parameters(), lr=adv_lr)
        dataset_ft = TensorDataset(X_encoded)
        loader_ft = DataLoader(dataset_ft, batch_size=BATCH_SIZE_DEFAULT, shuffle=True)
        POS_PASSES = 5
        for epoch in range(finetune_epochs):
            for (batch_data,) in loader_ft:
                for p in dbm.parameters():
                    if p.grad is not None:
                        p.grad.zero_()
                loss = dbm.contrastive_divergence(batch_data, k=cd_k, pos_passes=POS_PASSES)
                adv_optimizer.step()
        suffix_extra += "_advanced"
        if use_pt:
            suffix_extra += "_pt"
        elif use_pcd:
            suffix_extra += "_pcd"
    out_name = f"pretrained_dbmAdvanced_T{T}_vac{vac}{suffix_extra}.pt"
    out_path = os.path.join("advanced_data", out_name)
    torch.save(dbm.state_dict(), out_path)
    print(f"    [Pretrain] Saved {out_path}")
    del rbm1, rbm2, dbm, opt1, opt2, dataset, dataset2, loader, loader2, X_encoded, hidden_data, hidden_data_list
    torch.cuda.empty_cache()

def train_dbm_advanced_subset(spins_subset, T, vac, clamp=False, use_pt=False, use_pcd=False, suffix_extra=""):
    """
    Train DBMAdvanced without RBM pretraining, optionally with PT/PCD/clamping.
    """
    X_encoded = torch.tensor(encode_spins_to_two_channels(spins_subset), dtype=torch.float32, device=device)
    model = DBMAdvanced(
        n_vis=2*L*L, n_hid1=HID1, n_hid2=HID2,
        use_pt=use_pt, use_pcd=use_pcd, n_pt=10, lr=1e-3,
        clamp_occupancy=clamp
    ).to(device)
    dataset = TensorDataset(X_encoded)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE_DEFAULT, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    POS_PASSES = 5
    for ep in range(EPOCHS_DEFAULT):
        for (batch_data,) in loader:
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.zero_()
            loss = model.contrastive_divergence(batch_data, k=CD_K_DEFAULT, pos_passes=POS_PASSES)
            optimizer.step()
    clamp_str = "_clamped" if clamp else ""
    if use_pt:
        suffix_extra += "_pt"
    elif use_pcd:
        suffix_extra += "_pcd"
    out_name = f"trained_dbmAdvanced_T{T}_vac{vac}{clamp_str}{suffix_extra}.pt"
    out_path = os.path.join("advanced_data", out_name)
    torch.save(model.state_dict(), out_path)
    print(f"    Saved {out_path}")
    del model, optimizer, dataset, loader, X_encoded
    torch.cuda.empty_cache()

def main():
    if not os.path.isfile(data_file):
        raise FileNotFoundError(f"[ERROR] data file '{data_file}' not found.")
    data_npz = np.load(data_file)
    spins_all = data_npz["spins"]
    temps_all = data_npz["temperature"]
    vacs_all = data_npz["vacancy_fraction"]
    if spins_all.ndim == 3 and spins_all.shape[0] == spins_all.shape[1]:
        spins_all = np.transpose(spins_all, (2, 0, 1))
    all_subsets = []
    for vac in vacancy_list:
        Tc = Tc_map[vac]
        for offset in temperature_offsets:
            T_val = round(Tc + offset, 3)
            all_subsets.append((T_val, vac))
    print("\n[INFO] Beginning custom training calls for ADVANCED DBM...\n")
    for (T_val, vac) in all_subsets:
        mask = (
            np.isclose(temps_all, T_val, atol=1e-3)
            & np.isclose(vacs_all, vac, atol=1e-3)
        )
        spins_subset = spins_all[mask]
        if spins_subset.shape[0] == 0:
            print(f"    No data for T={T_val}, vac={vac}. Skipping.")
            continue
        print(f"    T={T_val}, vac={vac}, #samples={spins_subset.shape[0]}")
        train_dbm_advanced_subset(
            spins_subset, T_val, vac,
            clamp=False, use_pt=False, use_pcd=False, suffix_extra=""
        )
    for (T_val, vac) in all_subsets:
        mask = (
            np.isclose(temps_all, T_val, atol=1e-3)
            & np.isclose(vacs_all, vac, atol=1e-3)
        )
        spins_subset = spins_all[mask]
        if spins_subset.shape[0] == 0:
            print(f"    No data for T={T_val}, vac={vac}. Skipping.")
            continue
        print(f"    T={T_val}, vac={vac}, #samples={spins_subset.shape[0]}")
        train_dbm_advanced_subset(
            spins_subset, T_val, vac,
            clamp=False, use_pt=False, use_pcd=True, suffix_extra=""
        )
    for (T_val, vac) in all_subsets:
        mask = (
            np.isclose(temps_all, T_val, atol=1e-3)
            & np.isclose(vacs_all, vac, atol=1e-3)
        )
        spins_subset = spins_all[mask]
        if spins_subset.shape[0] == 0:
            print(f"    No data for T={T_val}, vac={vac}. Skipping.")
            continue
        print(f"    T={T_val}, vac={vac}, #samples={spins_subset.shape[0]}")
        pretrain_dbm_advanced_subset(
            spins_subset, T_val, vac,
            n_vis=2*L*L, nh1=HID1, nh2=HID2,
            rbm_epochs=50, cd_k=10, lr=1e-3,
            do_advanced_finetune=True, finetune_epochs=30,
            advanced_lr_scale=0.1,
            use_pt=False,
            use_pcd=False,
            suffix_extra=""
        )

if __name__ == "__main__":
    main()
