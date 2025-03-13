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
vacancy_list = [0.0, 0.1, 0.2]
Tc_map = {0.0: 2.3, 0.1: 1.975, 0.2: 1.6}
temperature_offsets = [-0.8, -0.1, -0.05, 0.0, 0.05, 0.1, 0.5]
data_file = "data/disordered_snapshots_16.npz"

EPOCHS_DEFAULT = 50
CD_K_DEFAULT = 10
BATCH_SIZE_DEFAULT = 32

RBM_HIDDEN = 128
HID1 = 64
HID2 = 64

def encode_spins_to_two_channels(spin_configs):
    """Converts spins (including vacancies) to 2-channel occupant/sign bits."""
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

def masked_bce_occupancy_sign(v_pred, v_true, eps=1e-7):
    """BCE on occupant bits, masked BCE on sign bits where occupant=1."""
    eps_tensor = torch.tensor(eps, device=v_true.device, dtype=v_true.dtype)
    occ_pred = v_pred[:, 0::2]
    occ_true = v_true[:, 0::2]
    sign_pred = v_pred[:, 1::2]
    sign_true = v_true[:, 1::2]
    occ_loss = -(
        occ_true * torch.log(occ_pred + eps_tensor) +
        (1 - occ_true) * torch.log(1 - occ_pred + eps_tensor)
    )
    sign_mask = (occ_true > 0.5).float()
    sign_loss_raw = -(
        sign_true * torch.log(sign_pred + eps_tensor) +
        (1 - sign_true) * torch.log(1 - sign_pred + eps_tensor)
    )
    sign_loss = sign_loss_raw * sign_mask
    total_loss = torch.mean(occ_loss) + torch.mean(sign_loss)
    return total_loss

class RBM(nn.Module):
    """RBM with manual CD training."""
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
        p_v_all = torch.sigmoid(h @ self.W.t() + self.v_bias)
        sign_p = p_v_all[:, 1::2]
        sign_s = torch.bernoulli(sign_p)
        v_out = torch.zeros_like(p_v_all)
        v_out[:, 0::2] = occupant_data
        v_out[:, 1::2] = sign_s
        return v_out

    def contrastive_divergence(self, v_data, k=1, clamp_occupancy=False):
        """Performs CD with optional occupancy clamping."""
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
        pos_vh = torch.bmm(v_data.unsqueeze(2), h_data.unsqueeze(1))
        neg_vh = torch.bmm(v_neg.unsqueeze(2), h_neg.unsqueeze(1))
        dW = (pos_vh.mean(dim=0) - neg_vh.mean(dim=0))
        db_v = (v_data - v_neg).mean(dim=0)
        db_h = (h_data - h_neg).mean(dim=0)
        if self.W.grad is not None:
            self.W.grad.zero_()
        if self.v_bias.grad is not None:
            self.v_bias.grad.zero_()
        if self.h_bias.grad is not None:
            self.h_bias.grad.zero_()
        self.W.grad = -dW
        self.v_bias.grad = -db_v
        self.h_bias.grad = -db_h
        p_v_final, _ = self.sample_v_given_h(h_neg)
        recon_loss = masked_bce_occupancy_sign(p_v_final, v_data)
        return recon_loss

def train_rbm_subset(spins_subset, T, vac, clamp=False):
    X_encoded = encode_spins_to_two_channels(spins_subset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rbm = RBM(n_vis=2*L*L, n_hid=RBM_HIDDEN).to(device)
    optimizer = optim.Adam(rbm.parameters(), lr=1e-3)
    dataset = TensorDataset(torch.tensor(X_encoded, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE_DEFAULT, shuffle=True)
    for ep in range(EPOCHS_DEFAULT):
        for (batch_data,) in loader:
            batch_data = batch_data.to(device)
            for p in rbm.parameters():
                if p.grad is not None:
                    p.grad.zero_()
            loss = rbm.contrastive_divergence(batch_data, k=CD_K_DEFAULT, clamp_occupancy=clamp)
            optimizer.step()
    clamp_str = "_clamped" if clamp else ""
    out_name = f"trained_rbm_T{T}_vac{vac}{clamp_str}.pt"
    os.makedirs("normal", exist_ok=True)
    out_path = os.path.join("normal", out_name)
    torch.save(rbm.state_dict(), out_path)
    print(f"    Saved {out_path}")
    del rbm, optimizer, dataset, loader
    torch.cuda.empty_cache()

class DBM(nn.Module):
    """2-layer DBM, manual CD approach."""
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

    def contrastive_divergence(self, v_data, k=1, clamp_occupancy=False):
        """CD for DBM with approximate positive phase."""
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
        with torch.no_grad():
            _, h1_neg_final = self.sample_h1_given_v_h2(v_model, h2_temp)
            _, h2_neg_final = self.sample_h2_given_h1(h1_neg_final)
        pos_vh1 = torch.bmm(v_data.unsqueeze(2), h1_data.unsqueeze(1))
        neg_vh1 = torch.bmm(v_model.unsqueeze(2), h1_neg_final.unsqueeze(1))
        dW1 = (pos_vh1.mean(dim=0) - neg_vh1.mean(dim=0)).transpose(0,1)
        db_v = (v_data - v_model).mean(dim=0)
        db_h1 = (h1_data - h1_neg_final).mean(dim=0)
        pos_h1h2 = torch.bmm(h1_data.unsqueeze(2), h2_data.unsqueeze(1))
        neg_h1h2 = torch.bmm(h1_neg_final.unsqueeze(2), h2_neg_final.unsqueeze(1))
        dW2 = pos_h1h2.mean(dim=0) - neg_h1h2.mean(dim=0)
        db_h2 = (h2_data - h2_neg_final).mean(dim=0)
        for p in [self.W1, self.b_v, self.b_h1, self.W2, self.b_h2]:
            if p.grad is not None:
                p.grad.zero_()
        self.W1.grad = -dW1
        self.b_v.grad = -db_v
        self.b_h1.grad = -db_h1
        self.W2.grad = -dW2
        self.b_h2.grad = -db_h2
        recon_loss = masked_bce_occupancy_sign(v_model, v_data)
        return recon_loss

def train_dbm_subset(spins_subset, T, vac, clamp=False):
    X_encoded = encode_spins_to_two_channels(spins_subset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DBM(n_vis=2*L*L, n_hid1=HID1, n_hid2=HID2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    dataset = TensorDataset(torch.tensor(X_encoded, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE_DEFAULT, shuffle=True)
    for ep in range(EPOCHS_DEFAULT):
        for (batch_data,) in loader:
            batch_data = batch_data.to(device)
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.zero_()
            loss = model.contrastive_divergence(batch_data, k=CD_K_DEFAULT, clamp_occupancy=clamp)
            optimizer.step()
    clamp_str = "_clamped" if clamp else ""
    out_name = f"trained_dbm_T{T}_vac{vac}{clamp_str}.pt"
    os.makedirs("normal", exist_ok=True)
    out_path = os.path.join("normal", out_name)
    torch.save(model.state_dict(), out_path)
    print(f"    Saved {out_path}")
    del model, optimizer, dataset, loader
    torch.cuda.empty_cache()

class DBN(nn.Module):
    """Simple DBN with 2 stacked layers."""
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
        act_h1 = F.linear(v, self.W1, self.b_h1)
        p_h1 = torch.sigmoid(act_h1)
        h1_s = torch.bernoulli(p_h1)
        return p_h1, h1_s

    def sample_v_given_h1(self, h1):
        act_v = F.linear(h1, self.W1.transpose(0,1), self.b_v)
        p_v = torch.sigmoid(act_v)
        v_s = torch.bernoulli(p_v)
        return p_v, v_s

    def sample_h2_given_h1(self, h1):
        act_h2 = F.linear(h1, self.W2, self.b_h2)
        p_h2 = torch.sigmoid(act_h2)
        h2_s = torch.bernoulli(p_h2)
        return p_h2, h2_s

    def sample_h1_given_h2(self, h2):
        act_h1 = F.linear(h2, self.W2.transpose(0,1), self.b_h1)
        p_h1 = torch.sigmoid(act_h1)
        h1_s = torch.bernoulli(p_h1)
        return p_h1, h1_s

    def contrastive_divergence(self, v_data, k=1, clamp_occupancy=False):
        """Manual CD for DBN."""
        occupant_data = v_data[:, 0::2] if clamp_occupancy else None
        with torch.no_grad():
            _, h1_data = self.sample_h1_given_v(v_data)
            _, h2_data = self.sample_h2_given_h1(h1_data)
        h1_model = h1_data.clone()
        h2_model = h2_data.clone()
        for _ in range(k):
            _, h1_model = self.sample_h1_given_h2(h2_model)
            _, h2_model = self.sample_h2_given_h1(h1_model)
        p_v_model, v_model_samp = self.sample_v_given_h1(h1_model)
        if clamp_occupancy and occupant_data is not None:
            reassembled = torch.zeros_like(p_v_model)
            reassembled[:, 0::2] = occupant_data
            sign_s = torch.bernoulli(p_v_model[:, 1::2])
            reassembled[:, 1::2] = sign_s
            v_model = reassembled
        else:
            v_model = v_model_samp
        pos_h1h2 = torch.bmm(h1_data.unsqueeze(2), h2_data.unsqueeze(1))
        neg_h1h2 = torch.bmm(h1_model.unsqueeze(2), h2_model.unsqueeze(1))
        dW2 = pos_h1h2.mean(dim=0) - neg_h1h2.mean(dim=0)
        db_h2 = (h2_data - h2_model).mean(dim=0)
        pos_vh1 = torch.bmm(v_data.unsqueeze(2), h1_data.unsqueeze(1))
        neg_vh1 = torch.bmm(v_model.unsqueeze(2), h1_model.unsqueeze(1))
        dW1 = (pos_vh1.mean(dim=0) - neg_vh1.mean(dim=0)).transpose(0,1)
        db_v = (v_data - v_model).mean(dim=0)
        db_h1 = (h1_data - h1_model).mean(dim=0)
        for p in [self.W1, self.b_v, self.b_h1, self.W2, self.b_h2]:
            if p.grad is not None:
                p.grad.zero_()
        self.W2.grad = -dW2
        self.b_h2.grad = -db_h2
        self.W1.grad = -dW1
        self.b_v.grad = -db_v
        self.b_h1.grad = -db_h1
        recon_loss = masked_bce_occupancy_sign(p_v_model, v_data)
        return recon_loss

def train_dbn_subset(spins_subset, T, vac, clamp=False):
    X_encoded = encode_spins_to_two_channels(spins_subset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DBN(n_vis=2*L*L, n_hid1=HID1, n_hid2=HID2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    dataset = TensorDataset(torch.tensor(X_encoded, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE_DEFAULT, shuffle=True)
    for ep in range(EPOCHS_DEFAULT):
        for (batch_data,) in loader:
            batch_data = batch_data.to(device)
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.zero_()
            loss = model.contrastive_divergence(batch_data, k=CD_K_DEFAULT, clamp_occupancy=clamp)
            optimizer.step()
    clamp_str = "_clamped" if clamp else ""
    out_name = f"trained_dbn_T{T}_vac{vac}{clamp_str}.pt"
    os.makedirs("normal", exist_ok=True)
    out_path = os.path.join("normal", out_name)
    torch.save(model.state_dict(), out_path)
    print(f"    Saved {out_path}")
    del model, optimizer, dataset, loader
    torch.cuda.empty_cache()

class DRBN_mod(nn.Module):
    """2-layer network v <-> h1 <-> h2."""
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

    def sample_v_given_h1(self, h1):
        logits = F.linear(h1, self.W1.transpose(0,1), self.b_v)
        p_v = torch.sigmoid(logits)
        v_s = torch.bernoulli(p_v)
        return p_v, v_s

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

    def contrastive_divergence(self, v_data, k=1, clamp_occupancy=False):
        """CD for DRBN with up-down passes."""
        occupant_data = v_data[:, 0::2] if clamp_occupancy else None
        with torch.no_grad():
            _, h1_data = self.sample_h1_given_v(v_data)
            _, h2_data = self.sample_h2_given_h1(h1_data)
        v_model = v_data.clone()
        for _ in range(k):
            _, h1_up = self.sample_h1_given_v(v_model)
            _, h2_up = self.sample_h2_given_h1(h1_up)
            _, h1_down = self.sample_h1_given_h2(h2_up)
            p_v_down, v_down = self.sample_v_given_h1(h1_down)
            if clamp_occupancy and occupant_data is not None:
                reassembled = torch.zeros_like(p_v_down)
                reassembled[:, 0::2] = occupant_data
                sign_samp = torch.bernoulli(p_v_down[:, 1::2])
                reassembled[:, 1::2] = sign_samp
                v_model = reassembled
            else:
                v_model = v_down
        with torch.no_grad():
            _, h1_model = self.sample_h1_given_v(v_model)
            _, h2_model = self.sample_h2_given_h1(h1_model)
        pos_vh1 = torch.bmm(v_data.unsqueeze(2), h1_data.unsqueeze(1))
        neg_vh1 = torch.bmm(v_model.unsqueeze(2), h1_model.unsqueeze(1))
        dW1 = (pos_vh1.mean(dim=0) - neg_vh1.mean(dim=0)).transpose(0,1)
        db_v = (v_data - v_model).mean(dim=0)
        db_h1 = (h1_data - h1_model).mean(dim=0)
        pos_h1h2 = torch.bmm(h1_data.unsqueeze(2), h2_data.unsqueeze(1))
        neg_h1h2 = torch.bmm(h1_model.unsqueeze(2), h2_model.unsqueeze(1))
        dW2 = pos_h1h2.mean(dim=0) - neg_h1h2.mean(dim=0)
        db_h2 = (h2_data - h2_model).mean(dim=0)
        for p in [self.W1, self.b_v, self.b_h1, self.W2, self.b_h2]:
            if p.grad is not None:
                p.grad.zero_()
        self.W1.grad = -dW1
        self.b_v.grad = -db_v
        self.b_h1.grad = -db_h1
        self.W2.grad = -dW2
        self.b_h2.grad = -db_h2
        p_v_final, _ = self.sample_v_given_h1(h1_model)
        recon_loss = masked_bce_occupancy_sign(p_v_final, v_data)
        return recon_loss

def train_drbn_subset(spins_subset, T, vac, clamp=False):
    X_encoded = encode_spins_to_two_channels(spins_subset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DRBN_mod(n_vis=2*L*L, n_hid1=HID1, n_hid2=HID2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    dataset = TensorDataset(torch.tensor(X_encoded, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE_DEFAULT, shuffle=True)
    for ep in range(EPOCHS_DEFAULT):
        for (batch_data,) in loader:
            batch_data = batch_data.to(device)
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.zero_()
            loss = model.contrastive_divergence(batch_data, k=CD_K_DEFAULT, clamp_occupancy=clamp)
            optimizer.step()
    clamp_str = "_clamped" if clamp else ""
    out_name = f"trained_drbn_T{T}_vac{vac}{clamp_str}.pt"
    os.makedirs("normal", exist_ok=True)
    out_path = os.path.join("normal", out_name)
    torch.save(model.state_dict(), out_path)
    print(f"    Saved {out_path}")
    del model, optimizer, dataset, loader
    torch.cuda.empty_cache()

class DBMAdvanced(nn.Module):
    """Advanced DBM with optional PT/PCD and occupant-channel clamping."""
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

    def masked_bce_occupancy_sign(self, v_pred, v_true, eps=1e-7):
        return masked_bce_occupancy_sign(v_pred, v_true, eps=eps)

    def free_energy_approx(self, v, beta=1.0, mean_field_steps=2):
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

    def sample_negative_phase_batch(self, batch_size, occupant_data=None, k=1):
        device = self.W1.device
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
        device = self.W1.device
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
        self.pcd_buffer[idxs] = v.detach().cpu()
        return v, h1_final, h2_final

    def sample_negative_phase_pt(self, batch_size, occupant_data=None, k=1):
        device = self.W1.device
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

    def contrastive_divergence(self, v_data, k=1, pos_passes=5, use_masked_bce=True):
        device = v_data.device
        batch_size = v_data.size(0)
        h1_data, h2_data = self.positive_phase_inference(v_data, M=pos_passes)
        occupant_data = v_data[:, 0::2] if self.clamp_occupancy else None
        if self.use_pt:
            v_neg, h1_neg, h2_neg = self.sample_negative_phase_pt(batch_size, occupant_data, k=k)
        elif self.use_pcd:
            v_neg, h1_neg, h2_neg = self.sample_negative_phase_pcd(batch_size, occupant_data, k=k)
        else:
            v_neg, h1_neg, h2_neg = self.sample_negative_phase_batch(batch_size, occupant_data, k=k)
        pos_vh1 = torch.bmm(v_data.unsqueeze(2), h1_data.unsqueeze(1))
        neg_vh1 = torch.bmm(v_neg.unsqueeze(2), h1_neg.unsqueeze(1))
        dW1 = (pos_vh1.mean(dim=0) - neg_vh1.mean(dim=0)).transpose(0,1)
        db_v = (v_data - v_neg).mean(dim=0)
        db_h1 = (h1_data - h1_neg).mean(dim=0)
        pos_h1h2 = torch.bmm(h1_data.unsqueeze(2), h2_data.unsqueeze(1))
        neg_h1h2 = torch.bmm(h1_neg.unsqueeze(2), h2_neg.unsqueeze(1))
        dW2 = pos_h1h2.mean(dim=0) - neg_h1h2.mean(dim=0)
        db_h2 = (h2_data - h2_neg).mean(dim=0)
        pos_h2v = torch.bmm(h2_data.unsqueeze(2), v_data.unsqueeze(1))
        neg_h2v = torch.bmm(h2_neg.unsqueeze(2), v_neg.unsqueeze(1))
        dU_h2v = pos_h2v.mean(dim=0) - neg_h2v.mean(dim=0)
        for p in [self.W1, self.b_v, self.b_h1, self.W2, self.b_h2, self.U_h2_v]:
            if p.grad is not None:
                p.grad.zero_()
        self.W1.grad = -dW1
        self.b_v.grad = -db_v
        self.b_h1.grad = -db_h1
        self.W2.grad = -dW2
        self.b_h2.grad = -db_h2
        self.U_h2_v.grad = -dU_h2v
        if use_masked_bce:
            recon_loss = masked_bce_occupancy_sign(v_neg, v_data)
        else:
            recon_loss = F.mse_loss(v_neg, v_data)
        return recon_loss

def pretrain_2_rbm(X_tensor, n_vis, n_hid1, n_hid2, epochs=30, cd_k=10, lr=1e-3):
    """Greedy layerwise pretraining of two RBMs, returning a DBMAdvanced."""
    device = X_tensor.device
    rbm1 = RBM(n_vis, n_hid1).to(device)
    opt1 = optim.Adam(rbm1.parameters(), lr=lr)
    dataset1 = TensorDataset(X_tensor)
    loader1 = DataLoader(dataset1, batch_size=64, shuffle=True)
    for ep in range(epochs):
        for (batch_data,) in loader1:
            batch_data = batch_data.to(device)
            for p in rbm1.parameters():
                if p.grad is not None:
                    p.grad.zero_()
            _ = rbm1.contrastive_divergence(batch_data, k=cd_k)
            opt1.step()
    hidden_data_list = []
    for (batch_data,) in loader1:
        batch_data = batch_data.to(device)
        with torch.no_grad():
            _, h_samp = rbm1.sample_h_given_v(batch_data)
        hidden_data_list.append(h_samp.cpu())
    hidden_data = torch.cat(hidden_data_list, dim=0)
    rbm2 = RBM(n_hid1, n_hid2).to(device)
    opt2 = optim.Adam(rbm2.parameters(), lr=lr)
    dataset2 = TensorDataset(hidden_data)
    loader2 = DataLoader(dataset2, batch_size=64, shuffle=True)
    for ep in range(epochs):
        for (batch_h1,) in loader2:
            batch_h1 = batch_h1.to(device)
            for p in rbm2.parameters():
                if p.grad is not None:
                    p.grad.zero_()
            _ = rbm2.contrastive_divergence(batch_h1, k=cd_k)
            opt2.step()
    dbm = DBMAdvanced(n_vis=n_vis, n_hid1=n_hid1, n_hid2=n_hid2)
    with torch.no_grad():
        dbm.W1.copy_(rbm1.W.t())
        dbm.b_v.copy_(rbm1.v_bias)
        dbm.b_h1.copy_(rbm1.h_bias)
        dbm.W2.copy_(rbm2.W)
        dbm.b_h2.copy_(rbm2.h_bias)
    del rbm1, rbm2, opt1, opt2, loader1, loader2, dataset1, dataset2
    torch.cuda.empty_cache()
    return dbm

def train_dbm_advanced_subset(spins_subset, T, vac, clamp=False, use_pt=False, use_pcd=False, suffix_extra=""):
    X_encoded = torch.tensor(encode_spins_to_two_channels(spins_subset), dtype=torch.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
            batch_data = batch_data.to(device)
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
    os.makedirs("normal", exist_ok=True)
    out_path = os.path.join("normal", out_name)
    torch.save(model.state_dict(), out_path)
    print(f"    Saved {out_path}")
    del model, optimizer, dataset, loader, X_encoded
    torch.cuda.empty_cache()

def train_dbm_advanced_pretrain_subset(spins_subset, T, vac, clamp=False):
    X_encoded = torch.tensor(encode_spins_to_two_channels(spins_subset), dtype=torch.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_encoded = X_encoded.to(device)
    dbm = pretrain_2_rbm(
        X_encoded, n_vis=2*L*L, n_hid1=HID1, n_hid2=HID2,
        epochs=30, cd_k=10, lr=1e-3
    ).to(device)
    dbm.use_pt = False
    dbm.use_pcd = False
    dbm.clamp_occupancy = clamp
    dataset = TensorDataset(X_encoded)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE_DEFAULT, shuffle=True)
    optimizer = optim.Adam(dbm.parameters(), lr=1e-3)
    POS_PASSES = 5
    for ep in range(EPOCHS_DEFAULT):
        for (batch_data,) in loader:
            for p in dbm.parameters():
                if p.grad is not None:
                    p.grad.zero_()
            loss = dbm.contrastive_divergence(batch_data, k=CD_K_DEFAULT, pos_passes=POS_PASSES)
            optimizer.step()
    clamp_str = "_clamped" if clamp else ""
    out_name = f"dbm_pretrained_T{T}_vac{vac}{clamp_str}_advanced.pt"
    os.makedirs("normal", exist_ok=True)
    out_path = os.path.join("normal", out_name)
    torch.save(dbm.state_dict(), out_path)
    print(f"    Saved {out_path}")
    del dbm, optimizer, dataset, loader, X_encoded
    torch.cuda.empty_cache()

def train_all_rbm(spins_all, temps_all, vacs_all, all_subsets):
    for (T_val, vac) in all_subsets:
        mask = (
            np.isclose(temps_all, T_val, atol=1e-3) &
            np.isclose(vacs_all, vac, atol=1e-3)
        )
        spins_subset = spins_all[mask]
        if spins_subset.shape[0] == 0:
            print(f"No data for T={T_val}, vac={vac}. Skipping RBM.")
            continue
        print(f"Training RBM on T={T_val}, vac={vac}, #samples={spins_subset.shape[0]}")
        train_rbm_subset(spins_subset, T_val, vac, clamp=False)
    torch.cuda.empty_cache()

def train_all_dbm(spins_all, temps_all, vacs_all, all_subsets):
    for (T_val, vac) in all_subsets:
        mask = (
            np.isclose(temps_all, T_val, atol=1e-3) &
            np.isclose(vacs_all, vac, atol=1e-3)
        )
        spins_subset = spins_all[mask]
        if spins_subset.shape[0] == 0:
            print(f"No data for T={T_val}, vac={vac}. Skipping DBM.")
            continue
        print(f"Training DBM on T={T_val}, vac={vac}, #samples={spins_subset.shape[0]}")
        train_dbm_subset(spins_subset, T_val, vac, clamp=False)
    torch.cuda.empty_cache()

def train_all_dbn(spins_all, temps_all, vacs_all, all_subsets):
    for (T_val, vac) in all_subsets:
        mask = (
            np.isclose(temps_all, T_val, atol=1e-3) &
            np.isclose(vacs_all, vac, atol=1e-3)
        )
        spins_subset = spins_all[mask]
        if spins_subset.shape[0] == 0:
            print(f"No data for T={T_val}, vac={vac}. Skipping DBN.")
            continue
        print(f"Training DBN on T={T_val}, vac={vac}, #samples={spins_subset.shape[0]}")
        train_dbn_subset(spins_subset, T_val, vac, clamp=False)
    torch.cuda.empty_cache()

def train_all_drbn(spins_all, temps_all, vacs_all, all_subsets):
    for (T_val, vac) in all_subsets:
        mask = (
            np.isclose(temps_all, T_val, atol=1e-3) &
            np.isclose(vacs_all, vac, atol=1e-3)
        )
        spins_subset = spins_all[mask]
        if spins_subset.shape[0] == 0:
            print(f"No data for T={T_val}, vac={vac}. Skipping DRBN.")
            continue
        print(f"Training DRBN on T={T_val}, vac={vac}, #samples={spins_subset.shape[0]}")
        train_drbn_subset(spins_subset, T_val, vac, clamp=False)
    torch.cuda.empty_cache()

def train_all_dbm_advanced(spins_all, temps_all, vacs_all, all_subsets):
    """Trains DBMAdvanced in various modes."""
    for (T_val, vac) in all_subsets:
        mask = (
            np.isclose(temps_all, T_val, atol=1e-3) &
            np.isclose(vacs_all, vac, atol=1e-3)
        )
        spins_subset = spins_all[mask]
        if spins_subset.shape[0] == 0:
            print(f"No data for T={T_val}, vac={vac}. Skipping DBMAdvanced.")
            continue
        offset = round(T_val - Tc_map[vac], 3)
        n_subset = spins_subset.shape[0]
        print(f"Training DBMAdvanced on T={T_val}, vac={vac}, #samples={n_subset}")
        print("   -> advanced DBM from scratch (unclamped)")
        train_dbm_advanced_subset(spins_subset, T_val, vac, clamp=False, use_pt=False, use_pcd=False)
        print("   -> advanced DBM with pretraining (unclamped)")
        train_dbm_advanced_pretrain_subset(spins_subset, T_val, vac, clamp=False)
        if offset in [-0.8, 0.0]:
            print("   -> advanced DBM (PT on, no clamp)")
            train_dbm_advanced_subset(spins_subset, T_val, vac, clamp=False, use_pt=True, use_pcd=False)
        if offset in [-0.8, 0.0]:
            print("   -> advanced DBM (PCD on, no clamp)")
            train_dbm_advanced_subset(spins_subset, T_val, vac, clamp=False, use_pt=False, use_pcd=True)
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
    train_all_rbm(spins_all, temps_all, vacs_all, all_subsets)
    train_all_dbm(spins_all, temps_all, vacs_all, all_subsets)
    train_all_dbn(spins_all, temps_all, vacs_all, all_subsets)
    train_all_drbn(spins_all, temps_all, vacs_all, all_subsets)
    # train_all_dbm_advanced(spins_all, temps_all, vacs_all, all_subsets)

if __name__ == "__main__":
    main()
