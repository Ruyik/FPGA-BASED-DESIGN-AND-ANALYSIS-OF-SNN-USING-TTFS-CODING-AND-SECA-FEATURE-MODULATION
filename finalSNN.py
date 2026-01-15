# finalSNN.py
# Optimized SNN with TTFS Inference, SECA Mechanism, and Tiered W1 Filters.

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# -------------------------
# CONFIGURATION
# -------------------------
IMG_SIZE = 14
P = IMG_SIZE * IMG_SIZE
H = 12  # Options: 6, 12, 24
TARGET_DIGIT = 2
TARGET_SAMPLE_INDEX = 5

OUTPUT = 3
TMAX = 31
T = TMAX + 1
THR = 1.0
DECAY = 0.98
RESET = 0.0

NUM_TRAIN = 300
NUM_TEST = 100
RANDOM_SEED = 42
EPOCHS = 60
BATCH = 64
LR = 1e-3
VERBOSE_SAMPLES = 5

EXPORT_DIR = "export_finalSNN"
os.makedirs(EXPORT_DIR, exist_ok=True)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------
# DATA & ENCODING
# -------------------------
def load_mnist_3classes(train=True, n_per_class=200, img_size=IMG_SIZE):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((img_size, img_size), antialias=True),
    ])
    dataset = datasets.MNIST("./data", train=train, download=True, transform=transform)
    X_list, y_list = [], []
    counts = {0: 0, 1: 0, 2: 0}
    for img, label in dataset:
        lbl = int(label)
        if lbl not in counts: continue
        if counts[lbl] >= n_per_class:
            if all(c >= n_per_class for c in counts.values()): break
            continue
        arr = img.numpy().squeeze()
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-9)
        X_list.append(arr.reshape(-1))
        y_list.append(lbl)
        counts[lbl] += 1
    return np.stack(X_list), np.array(y_list)


def ttfs_encode_batch(X, Tmax=TMAX):
    N, P = X.shape
    times = np.floor((1.0 - X) * Tmax).astype(int)
    times = np.clip(times, 0, Tmax)
    spikes = np.zeros((N, P, Tmax + 1), dtype=np.uint8)
    for n in range(N):
        for p in range(P):
            spikes[n, p, times[n, p]] = 1
    return times, spikes


# -------------------------
# FILTER DESIGN (W1)
# -------------------------
def make_fixed_W1(img_size=IMG_SIZE, hidden_neurons=H):
    Hlist = []
    # Tier 1: Basic Structural Filters
    v = np.zeros((img_size, img_size));
    v[:, img_size // 2] = 1;
    Hlist.append(v.reshape(-1))
    h = np.zeros((img_size, img_size));
    h[img_size // 2, :] = 1;
    Hlist.append(h.reshape(-1))
    h_c = np.zeros((img_size, img_size));
    h_c[4:img_size - 4, 4:img_size - 4] = -1.0;
    Hlist.append(h_c.reshape(-1))
    c_bl = np.zeros((img_size, img_size));
    c_bl[img_size - 4:, :4] = 1.0;
    Hlist.append(c_bl.reshape(-1))
    t = np.zeros((img_size, img_size));
    t[:img_size // 2, :] = 1;
    Hlist.append(t.reshape(-1))
    b = np.zeros((img_size, img_size));
    b[img_size // 2:, :] = 1;
    Hlist.append(b.reshape(-1))

    # Tier 2: Refined Detail Filters
    if hidden_neurons >= 12:
        v_l = np.zeros((img_size, img_size));
        v_l[:, img_size // 2 - 1] = 1;
        Hlist.append(v_l.reshape(-1))
        v_r = np.zeros((img_size, img_size));
        v_r[:, img_size // 2 + 1] = 1;
        Hlist.append(v_r.reshape(-1))
        c_tr = np.zeros((img_size, img_size));
        c_tr[:4, img_size - 4:] = 1;
        Hlist.append(c_tr.reshape(-1))
        e_bc = np.zeros((img_size, img_size));
        e_bc[img_size - 2:, 4:img_size - 4] = 1.0;
        Hlist.append(e_bc.reshape(-1))
        tl = np.zeros((img_size, img_size));
        tl[:img_size // 2, :img_size // 2] = 1;
        Hlist.append(tl.reshape(-1))
        a_tl = np.zeros((img_size, img_size));
        a_tl[1, 1:img_size // 2] = 1.0;
        Hlist.append(a_tl.reshape(-1))

    # Tier 3: High-Density Specificity (Optimized for 2 vs 0 separation)
    if hidden_neurons >= 24:
        d1 = np.zeros((img_size, img_size))
        for i in range(img_size): d1[i, img_size - 1 - i] = 1.0
        Hlist.append(d1.reshape(-1))
        d2 = np.zeros((img_size, img_size))
        for i in range(img_size): d2[i, i] = 1.0
        Hlist.append(d2.reshape(-1))
        f_base = np.zeros((img_size, img_size));
        f_base[-2:, :] = 1.0;
        Hlist.append(f_base.reshape(-1))
        v2 = np.zeros((img_size, img_size));
        v2[:, 3] = 1.0;
        v2[:, 10] = 1.0;
        Hlist.append(v2.reshape(-1))
        tl_c = np.zeros((img_size, img_size));
        tl_c[:4, :4] = 1.0;
        Hlist.append(tl_c.reshape(-1))
        m_diag = np.zeros((img_size, img_size))
        for i in range(4, 10): m_diag[i, 13 - i] = 1.0
        Hlist.append(m_diag.reshape(-1))
        for offset in [-2, 2]:
            Hlist.append(np.roll(v.reshape(img_size, img_size), offset, axis=1).reshape(-1))
            Hlist.append(np.roll(h.reshape(img_size, img_size), offset, axis=0).reshape(-1))
        while len(Hlist) < hidden_neurons:
            Hlist.append(np.random.normal(0, 0.05, img_size * img_size))

    W1 = np.array(Hlist[:hidden_neurons], float)
    W1 = W1 / (np.linalg.norm(W1, axis=1, keepdims=True) + 1e-9)
    return W1


# -------------------------
# NEURON & FEATURE LOGIC
# -------------------------
def lif_hidden_numpy(sample_spikes, W1_np, threshold=THR, decay=DECAY):
    Hn, Tn = W1_np.shape[0], sample_spikes.shape[1]
    hidden_spikes = np.zeros((Hn, Tn), dtype=np.uint8)
    for h in range(Hn):
        V = 0.0
        for t in range(Tn):
            I = float(np.dot(W1_np[h], sample_spikes[:, t]))
            V = V * decay + I
            if V >= threshold:
                hidden_spikes[h, t] = 1
                V = RESET
    return hidden_spikes


def build_features_all(spikes_np, W1_np):
    N, Hn = spikes_np.shape[0], W1_np.shape[0]
    counts_arr = np.zeros((N, Hn))
    ttfs_feat = np.zeros((N, Hn))
    for i in range(N):
        hs = lif_hidden_numpy(spikes_np[i], W1_np)
        counts_arr[i] = hs.sum(axis=1)
        first = np.array([np.nonzero(hs[h])[0][0] if np.any(hs[h]) else T for h in range(Hn)])
        ttfs_feat[i] = (TMAX - first) / float(TMAX)
    max_c = counts_arr.max(axis=0)
    max_c[max_c == 0] = 1.0
    return np.concatenate([counts_arr / max_c, ttfs_feat], axis=1), counts_arr, ttfs_feat


# -------------------------
# SECA MODEL
# -------------------------
class SECA_Feature_Classifier(nn.Module):
    def __init__(self, Hn, out):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=True)
        with torch.no_grad():
            w = torch.zeros_like(self.conv.weight)
            w[0, 0, 0] = 0.5;
            w[0, 0, 1] = 1.0;
            w[0, 0, 2] = 0.5
            self.conv.weight.copy_(w)
            if self.conv.bias is not None: self.conv.bias.zero_()
        self.lin = nn.Linear(Hn * 2, out)

    def forward(self, feats):
        Hn = feats.shape[1] // 2
        counts, ttfs = feats[:, :Hn], feats[:, Hn:]
        x = self.conv(counts.unsqueeze(1)).squeeze(1)
        s = torch.sigmoid(x)
        scale = (1.0 + s) / ((1.0 + s).max(dim=1, keepdim=True)[0] + 1e-9)
        return self.lin(torch.cat([counts * scale, ttfs], dim=1)), scale


# -------------------------
# TRAINING & EVALUATION
# -------------------------
def train_and_eval():
    print(f"\n--- SNN Configuration: Hidden Neurons H = {H} ---")

    X_train, y_train = load_mnist_3classes(train=True, n_per_class=NUM_TRAIN)
    X_test, y_test = load_mnist_3classes(train=False, n_per_class=NUM_TEST)
    _, spikes_train = ttfs_encode_batch(X_train)
    _, spikes_test = ttfs_encode_batch(X_test)
    W1 = make_fixed_W1()

    feats_train, _, _ = build_features_all(spikes_train, W1)
    feats_test, _, _ = build_features_all(spikes_test, W1)

    model = SECA_Feature_Classifier(H, OUTPUT).to(device)
    Xtr = torch.from_numpy(feats_train).float().to(device)
    ytr = torch.from_numpy(y_train).long().to(device)
    Xte = torch.from_numpy(feats_test).float().to(device)
    yte = torch.from_numpy(y_test).long().to(device)

    opt = optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    for ep in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(Xtr.shape[0], device=device)
        epoch_loss = 0.0
        for i in range(0, Xtr.shape[0], BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            logits, _ = model(Xtr[idx])
            loss = loss_fn(logits, ytr[idx])
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * idx.size(0)

        if ep % 20 == 0 or ep == EPOCHS:
            model.eval()
            with torch.no_grad():
                l_te, _ = model(Xte)
                acc = (l_te.argmax(1).cpu().numpy() == y_test).mean()
            print(f"Epoch {ep:3d}/{EPOCHS} | Loss: {epoch_loss / Xtr.shape[0]:.4f} | Val Acc: {acc * 100:.2f}%")

    # Footprint Analysis
    total_params = 4 + (H * 2 * OUTPUT + OUTPUT)
    print(f"\nTotal Trainable Parameters: {total_params}")
    print(f"Estimated Memory Usage: {(total_params * 4) / 1024:.3f} KB")

    # Spiking Inference
    conv_w = model.conv.weight.detach().cpu().numpy().squeeze()
    conv_b = model.conv.bias.detach().cpu().numpy().squeeze()
    lin_w = model.lin.weight.detach().cpu().numpy()

    def seca_counts_to_scale(counts, K, bias):
        x = np.zeros(counts.size)
        k0, k1, k2 = K[0], K[1], K[2]
        for i in range(counts.size):
            if i > 0: x[i] += k0 * counts[i - 1]
            x[i] += k1 * counts[i]
            if i < counts.size - 1: x[i] += k2 * counts[i + 1]
            x[i] += bias
        s = 1.0 / (1.0 + np.exp(-x))
        return (1.0 + s) / ((1.0 + s).max() + 1e-9)

    def spiking_inference_calculate(spikes_np, W1_np, cw, cb, lw, y_test):
        preds, lats = [], []
        t_start = time.perf_counter()

        for i in range(len(y_test)):
            hs = lif_hidden_numpy(spikes_np[i], W1_np)
            counts = hs.sum(axis=1).astype(float)
            first = np.array([np.nonzero(hs[h])[0][0] if np.any(hs[h]) else T for h in range(H)])
            scale = seca_counts_to_scale(counts, cw, cb)

            hidden_mod = hs.astype(float) * scale[:, None]
            ttfs_in = np.zeros((H, T))
            for h in range(H):
                if first[h] < T: ttfs_in[h, first[h]] = 1.0
            inputs = np.concatenate([hidden_mod, ttfs_in], axis=0)

            V_out = np.zeros(OUTPUT)
            pred, earliest = -1, T
            for t in range(T):
                V_out = V_out * DECAY + np.dot(lw, inputs[:, t])
                spiking = np.where(V_out >= THR)[0]
                if spiking.size > 0:
                    pred, earliest = spiking[np.argmax(V_out[spiking])], t
                    break
            if pred == -1: pred = np.argmax(V_out)
            preds.append(pred);
            lats.append(earliest)

            if i < VERBOSE_SAMPLES:
                print(f"Sample {i} | Label: {y_test[i]} | Pred: {pred} | Latency: {earliest}t")

        t_end = time.perf_counter()
        return np.array(preds), lats, (t_end - t_start)

    print("\nRunning Inference...")
    spk_preds, lats, dur = spiking_inference_calculate(spikes_test, W1, conv_w, conv_b, lin_w, y_test)
    print(f"\nFinal Accuracy: {(spk_preds == y_test).mean() * 100:.2f}%")
    print(f"Avg Latency: {np.mean(lats):.2f} t | Total Speed: {dur:.3f}s")

    # Visual Analysis
    idx = np.where(y_test == TARGET_DIGIT)[0][TARGET_SAMPLE_INDEX]
    plt.imshow(X_test[idx].reshape(IMG_SIZE, IMG_SIZE), cmap='gray')
    plt.title(f"Label: {y_test[idx]}, Pred: {spk_preds[idx]}")
    plt.show()


if __name__ == "__main__":
    train_and_eval()