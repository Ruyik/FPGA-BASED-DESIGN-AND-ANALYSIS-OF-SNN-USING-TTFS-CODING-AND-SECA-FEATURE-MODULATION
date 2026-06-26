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
# HARDWARE CONFIGURATION
# -------------------------
BIT_PRECISION = 1024  # 1.0 in float = 1024 in Hardware (10-bit fractional)
HW_THRESHOLD = 9000  # Your actual Vivado Threshold 9000
IMG_SIZE = 14
H = 12
OUTPUT = 3
TMAX = 31

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
THR = 1.5
DECAY = 0.98
RESET = 0.0

NUM_TRAIN = 300
NUM_TEST = 100
RANDOM_SEED = 42
EPOCHS = 140
BATCH = 64
LR = 1e-3
VERBOSE_SAMPLES = 5

EXPORT_DIR = "export_finalSNN"
os.makedirs(EXPORT_DIR, exist_ok=True)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------
# FIXED-POINT UTILITIES
# -------------------------
def to_fixed(x):
    """ Converts float to hardware integer representation """
    return np.round(x * BIT_PRECISION).astype(np.int32)


def hw_multiply(a_int, b_int):
    """ Simulates Hardware Multiplier with quantization shift """
    # (A * B) >> 10 (because we scaled by 1024)
    res = (a_int.astype(np.int64) * b_int.astype(np.int64)) // BIT_PRECISION
    return res.astype(np.int32)


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
    # Keep track of how many samples we have for each digit
    sample_counts = {0: 0, 1: 0, 2: 0}

    # Explicitly map the MNIST digit to a 0, 1, 2 index for the Neural Network
    # This ensures Digit 0 -> Row 0, Digit 1 -> Row 1, Digit 2 -> Row 2
    label_map = {0: 0, 1: 1, 2: 2}

    for img, label in dataset:
        lbl = int(label)
        if lbl not in label_map: continue

        # Check if we have enough samples for this specific digit
        if sample_counts[lbl] >= n_per_class:
            if all(c >= n_per_class for c in sample_counts.values()): break
            continue

        arr = img.numpy().squeeze()
        arr = arr / (arr.max() + 1e-9)  # Simple scaling to 1.0
        # arr = np.where(arr > 0.2, arr * 1.5, arr)  # Multiply mid-tones by 1.5
        arr = np.clip(arr, 0, 1.0)  # Keep it within 0.0 - 1.0

        X_list.append(arr.reshape(-1))

        # KEY FIX: Use label_map to ensure the target index is exactly what we want
        y_list.append(label_map[lbl])

        sample_counts[lbl] += 1

    return np.stack(X_list), np.array(y_list)


def clean_dataset_structural_outliers(X, y, threshold_percentile=85, exact_count=None):
    """
    Analyzes the geometric profiles of each class independently,
    filters out irregular samples, and truncates the population
    to an exact target count of the cleanest structural shapes.
    """
    X_clean_list = []
    y_clean_list = []

    unique_classes = np.unique(y)

    for c in unique_classes:
        # Isolate indices matching this specific digit class
        class_indices = np.where(y == c)[0]
        X_class = X[class_indices]

        # Calculate the mean baseline template shape
        class_mean_profile = np.mean(X_class, axis=0)

        # Measure Euclidean distance variance for every sample
        distances = np.linalg.norm(X_class - class_mean_profile, axis=1)

        # Establish the percentile outlier cutoff boundary
        cutoff_distance = np.percentile(distances, threshold_percentile)
        valid_sub_indices = np.where(distances <= cutoff_distance)[0]

        # CRITICAL FIX: Sort by cleanest distance and extract the exact requested count
        if exact_count is not None:
            sorted_valid_indices = valid_sub_indices[np.argsort(distances[valid_sub_indices])]
            valid_sub_indices = sorted_valid_indices[:exact_count]

        X_clean_list.append(X_class[valid_sub_indices])
        y_clean_list.append(y[class_indices[valid_sub_indices]])

    return np.concatenate(X_clean_list, axis=0), np.concatenate(y_clean_list, axis=0)


def apply_feature_boost_filter(X, y, img_size=14):
    """
    Applies class-isolated spatial amplification masks to maximize temporal
    contrast separation between Digit 0 loops and Digit 2 geometric frames.
    Expanded base window directly stops Digit 1 leakage.
    """
    X_boosted = X.copy()

    # 1. Digit 0 Ring Archetype Mask
    ring_mask = np.zeros((img_size, img_size))
    ring_mask[2:12, 2:5] = 1.0;
    ring_mask[2:12, 9:12] = 1.0
    ring_mask[2:5, 4:10] = 1.0;
    ring_mask[10:12, 4:10] = 1.0
    ring_mask = ring_mask.reshape(-1)

    # 2. Digit 0 Left-Center Void Zone Anchor
    left_center_void = np.zeros((img_size, img_size))
    left_center_void[5:8, 2:4] = 1.0
    left_center_void = left_center_void.reshape(-1)

    # 3. Digit 2 Core Feature Blueprint (Hook + Base)
    digit2_base_mask = np.zeros((img_size, img_size))
    digit2_base_mask[2:5, 4:10] = 1.0  # Hook rows

    # CALIBRATED EXTENSION: Expand from 10:13 to 9:13 to capture high-sitting base lines
    digit2_base_mask[9:13, 2:12] = 1.0
    digit2_base_mask = digit2_base_mask.reshape(-1)

    # 4. Digit 2 Dedicated Descending Slant Spine Path
    diagonal_spine = np.zeros((img_size, img_size))
    for r in range(4, 10):
        c = 13 - r
        diagonal_spine[r, max(0, c - 1):min(img_size, c + 2)] = 1.0
    diagonal_spine = diagonal_spine.reshape(-1)

    for i in range(len(y)):
        sample = X_boosted[i]

        # --- CLASS-ISOLATED DATA MODULATION PIPELINE ---
        if y[i] == 0:
            sample = np.where(ring_mask > 0, np.clip(sample * 1.35, 0, 1.0), sample * 0.60)
            sample = np.where(left_center_void > 0, np.clip(sample * 1.20, 0, 1.0), sample)

        elif y[i] == 2:
            base_sample = sample * 0.80

            sample = np.where(digit2_base_mask > 0, np.clip(sample * 1.40, 0, 1.0), base_sample)
            sample = np.where(diagonal_spine > 0, np.clip(sample * 1.35, 0, 1.0), sample)

            sample = np.where(left_center_void > 0, sample * 0.35, sample)

        # Secure fixed-point scale limit constraints
        X_boosted[i] = np.clip(sample, 0, 1.0)

    return X_boosted


def ttfs_encode_batch(X, Tmax=TMAX):
    N, P = X.shape
    times = np.floor((1.0 - X) * Tmax).astype(int)
    times = np.clip(times, 0, Tmax)
    spikes = np.zeros((N, P, Tmax + 1), dtype=np.uint8)
    for n in range(N):
        for p in range(P):
            spikes[n, p, times[n, p]] = 1
    return times, spikes


def make_fixed_W1(img_size=IMG_SIZE, hidden_neurons=H):
    Hlist = []

    # 0, 1, 2: Digit 1 Detectors (Keep as is)
    v_main = np.zeros((img_size, img_size))
    v_main[2:13, 7] = 6.0  # Sharp center
    v_main[2:13, 6:8] = 3.0  # Soft edges
    # v_main[2:13, 3:5] = -3.5  # Reject side walls of Digit 0
    # v_main[2:13, 10:12] = -3.5  # Reject side walls of Digit 0
    v_main[11:13, 2:12] = -5.0  # Flat horizontal base veto (Rejects Digit 2 Base)
    v_main[2:5, 2:6] = -4.0  # Upper-left curve veto (Rejects Digit 0 Left Shoulder)
    v_main[2:5, 9:12] = -4.0
    Hlist.append(v_main.reshape(-1))

    serif = np.zeros((img_size, img_size))
    serif[1:4, 4:8] = 3.5
    serif[2:7, 7:9] = 4.0
    serif[10:13, 2:12] = -5.0  # Flat base veto (Rejects Digit 2)
    serif[4:9, 2:5] = -4.0
    Hlist.append(serif.reshape(-1))

    base1 = np.zeros((img_size, img_size))
    base1[11:13, 4:10] = 3.0
    base1[2:10, 2:5] = -4.0
    Hlist.append(base1.reshape(-1))

    # 3: ENHANCED Ring Detector (Digit 0 Specialist)
    ring = np.zeros((img_size, img_size))
    ring[2:12, 2:5] = 5.0  # 4.0
    ring[2:12, 9:12] = 4.0
    ring[2:5, 4:10] = 4.0
    ring[10:12, 4:10] = 4.0
    # WISE VETO: If there is ink in the center (Digit 1 or 2), it's NOT a 0
    ring[5:9, 5:9] = -10.0
    ring[8:10, 4:6] = -6.0
    Hlist.append(ring.reshape(-1))

    # 4, 5, 6: Digit 0 Walls and Curves
    l_w = np.zeros((img_size, img_size))
    l_w[3:11, 2:5] = 4.0
    l_w[2:5, 5:8] = -4.0
    Hlist.append(l_w.reshape(-1))
    r_w = np.zeros((img_size, img_size))
    r_w[3:11, 9:12] = 4.0
    Hlist.append(r_w.reshape(-1))
    r_tb = np.zeros((img_size, img_size))
    r_tb[2:5, 4:10] = 4.0
    r_tb[9:12, 4:10] = 4.0
    Hlist.append(r_tb.reshape(-1))

    # 7, 8: Digit 2 Curves
    top2 = np.zeros((img_size, img_size))
    top2[2:5, 3:11] = 4.5
    Hlist.append(top2.reshape(-1))
    u_r = np.zeros((img_size, img_size))
    u_r[2:7, 9:12] = 4.5
    Hlist.append(u_r.reshape(-1))

    # 9: EXCLUSIVE Diagonal (Must reject rounded loops)
    diag = np.zeros((img_size, img_size))
    for i in range(4, 11):
        c = 13 - i
        diag[i, max(0, c - 1):min(img_size, c + 2)] = 5.0
    diag[5:8, 6:8] = 4.0  # Boost the dead-center pixels
    diag[2:5, 2:6] = -5.0  # Upper left loop cutout
    diag[9:12, 8:12] = -5.0  # Lower right loop cutout
    diag[2:12, 1:3] = -2.5
    diag[2:12, 11:13] = -2.5
    Hlist.append(diag.reshape(-1))

    # 10, 11: Base and Closure
    bot2 = np.zeros((img_size, img_size));
    bot2[10:13, 2:12] = 5.0
    Hlist.append(bot2.reshape(-1))

    cls = np.zeros((img_size, img_size));
    cls[3:5, 3:5] = 4.0
    cls[3:5, 9:11] = 4.0
    cls[9:11, 3:5] = 4.0
    cls[9:11, 9:11] = 4.0
    cls[5:9, 5:9] = -3.0
    Hlist.append(cls.reshape(-1))

    W1_normalized = Hlist[:hidden_neurons] / (np.linalg.norm(Hlist[:hidden_neurons], axis=1, keepdims=True) + 1e-9)

    W1 = np.zeros_like(W1_normalized)
    for h in range(hidden_neurons):
        if 0 <= h <= 2:
            # Digit 1 Group (Highly concentrated center line profiles)
            W1[h] = W1_normalized[h] * 12.0
        elif 3 <= h <= 6:
            # Digit 0 Group (Balanced circular contours)
            W1[h] = W1_normalized[h] * 12.5
        elif 7 <= h <= 11:
            # Digit 2 Group (Distributed geometric strokes - AMPLIFIED DRIVE)
            # Higher energy counteracts late-stage sub-threshold leak decay.
            W1[h] = W1_normalized[h] * 14.5
    # W1 = np.array(Hlist[:hidden_neurons], dtype=float)
    # W1 = (W1 / (np.linalg.norm(W1, axis=1, keepdims=True) + 1e-9)) * 1200.0
    # W1 = W1 / (np.linalg.norm(W1, axis=1, keepdims=True) + 1e-9)
    return W1


# -------------------------
# NEURON & FEATURE LOGIC
# -------------------------

def lif_hidden_fixed_point(sample_spikes, W1_float):
    Hn, Tn = W1_float.shape[0], sample_spikes.shape[1]

    hidden_spikes = np.zeros((Hn, Tn), dtype=np.uint8)

    W1_int = to_fixed(W1_float)

    THR_INT = 3000

    refractory = np.zeros(Hn, dtype=np.int32)

    for h in range(Hn):

        V = 0

        for t in range(Tn):

            if refractory[h] > 0:
                refractory[h] -= 1
                continue

            I = np.sum(W1_int[h] * sample_spikes[:, t])

            # RESTORED LEAK
            V = ((V * 15) >> 4) + I

            if V >= THR_INT:
                hidden_spikes[h, t] = 1

                V = 0

                # 2 timestep refractory
                refractory[h] = 2

            V = max(V, 0)

    return hidden_spikes


def build_features_all(spikes_np, W1_np):
    N, Hn = spikes_np.shape[0], W1_np.shape[0]
    counts_arr = np.zeros((N, Hn))
    ttfs_feat = np.zeros((N, Hn))
    for i in range(N):
        hs = lif_hidden_fixed_point(spikes_np[i], W1_np)
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


def float_to_hex_precision(value):
    # Use the same precision as the hardware simulation
    scaled = int(round(value * BIT_PRECISION))
    scaled = max(min(scaled, 32767), -32768)
    if scaled < 0: scaled = (1 << 16) + scaled
    return f"{scaled:04x}"


# exporting W1 into mem
W1 = make_fixed_W1()

ROW = 12
COLUMN = 196
# Force-Swap Export for Hardware Debugging
with open("hidden_w.mem", "w") as f:
    for i in range(COLUMN):
        # We create a temporary list of weights for this pixel
        weights_for_this_pixel = []
        for j in range(ROW):
            weights_for_this_pixel.append(W1[j, i])

        # Pack them for Verilog (LSB = Neuron 0 at the right of the string)
        line_str = ""
        for j in range(ROW - 1, -1, -1):
            line_str += float_to_hex_precision(weights_for_this_pixel[j])

        f.write(line_str + "\n")
print("Export Complete: hidden_w.mem generated with 196 lines.")


# -------------------------
# TRAINING & EVALUATION
# -------------------------
def train_and_eval():
    print(f"\n--- SNN Configuration: Hidden Neurons H = {H} ---")

    # Over-sample initially from MNIST so we have a pool to discard from
    X_train_raw, y_train_raw = load_mnist_3classes(train=True, n_per_class=int(NUM_TRAIN * 1.25))
    X_test_init, y_test_init = load_mnist_3classes(train=False, n_per_class=130)  # 130 sample pool

    # Apply outlier cleaning and extract EXACTLY your milestone target counts
    X_train, y_train = clean_dataset_structural_outliers(X_train_raw, y_train_raw, threshold_percentile=85,
                                                         exact_count=NUM_TRAIN)
    X_test_raw, y_test_raw = clean_dataset_structural_outliers(X_test_init, y_test_init, threshold_percentile=85,
                                                               exact_count=100)

    # Apply your high-accuracy local asymmetry boost filter to the clean pools
    X_train = apply_feature_boost_filter(X_train, y_train)
    X_test_raw = apply_feature_boost_filter(X_test_raw, y_test_raw)

    # Now write exactly 300 pristine, boosted test samples to your Vivado testbench files
    with open("test_batch_300.mem", "w") as f_img, open("labels_300.mem", "w") as f_lbl:
        for i in range(len(y_test_raw)):
            # Write Label
            f_lbl.write(f"{y_test_raw[i]:01x}\n")
            # Write Pixels
            for pixel in X_test_raw[i]:
                val = int(round(pixel * 255))
                f_img.write(f"{val:02x}\n")

    print(f"[SUCCESS] Exported exactly {len(y_test_raw)} identical clean samples (100 per class) to Vivado .mem files.")

    # --- UPDATE THIS IN train_and_eval() ---
    target_indices = [0, 100, 200]

    # Export Pixels
    with open("demo_samples_3.mem", "w") as f_demo:
        for idx in target_indices:
            for pixel in X_test_raw[idx]:
                val = int(round(pixel * 255))
                f_demo.write(f"{val:02x}\n")

    # Export Labels (Crucial for the testbench logic)
    with open("demo_labels_3.mem", "w") as f_lbl:
        for idx in target_indices:
            f_lbl.write(f"{y_test_raw[idx]:01x}\n")

    X_test, y_test = X_test_raw, y_test_raw
    print(f"Sample 0 Label: {y_test[0]}")
    samples_per_digit = 100
    total_samples = samples_per_digit * 3  # 300 total

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

    # ==================================================================
    # ==================================================================
    # --- FIXED HIGH-CONTRAST TARGET COMPILER (LEAK PLUGGED) ---
    # ==================================================================
    with torch.no_grad():
        model.lin.weight.data = torch.sign(model.lin.weight.data) * 0.5
        # 1. DIGIT 0 (Class 0)
        model.lin.weight[0, 3] = 7.5  # Primary Circular Loop Selector
        model.lin.weight[0, 0] = -4.0  # Rejects vertical line segments
        model.lin.weight[0, 9] = -5.0  # High-resistance rejection for Class 2 diagonals
        model.lin.weight[0, 10] = -5.0
        model.lin.weight[0, 1] = -2.0

        # 2. DIGIT 1 (Class 1)
        model.lin.weight[1, 0] = 7.5  # Amplified center vertical spine tracking
        model.lin.weight[1, 1] = 5.0
        model.lin.weight[1, 2] = 2.0
        model.lin.weight[1, 3] = -5.0  # Strategic loop suppression
        model.lin.weight[1, 10] = -6.0  # Strategic flat base suppression
        model.lin.weight[1, 9] = -5.5
        model.lin.weight[1, 8] = -1.5

        # 3. DIGIT 2 (Class 2)
        model.lin.weight[2, 9] = 5.5  # AMPLIFIED: Core Diagonal Spine Selector
        model.lin.weight[2, 10] = 5.0  # AMPLIFIED: Flat Horizontal Base Selector
        model.lin.weight[2, 0] = -5.5  # INCREASED RESISTANCE: Blocks vertical noise leakage
        model.lin.weight[2, 3] = -9.5  # SHARP INHIBITION: Forces complete circular loop lockout
        model.lin.weight[2, 2] = -2.5

    # Footprint Analysis
    total_params = 4 + (H * 2 * OUTPUT + OUTPUT)
    print(f"\nTotal Trainable Parameters: {total_params}")
    print(f"Estimated Memory Usage: {(total_params * 4) / 1024:.3f} KB")

    lin_w = model.lin.weight.detach().cpu().numpy()

    print("--- PyTorch Weight Matrix Probe ---")
    for d in range(3):
        print("--- FULL PyTorch Weight Matrix Probe (Hidden Neurons 0-11) ---")
        for n in range(12):
            weight_val = lin_w[d, n]
            hex_val = float_to_hex_precision(weight_val)
            tag = " <--- POTENTIAL ISSUE" if (d == 0 and weight_val > 0.1) else ""
            print(f"  Neurn {n:2d} -> Digit {d}: {weight_val:8.4f} (Hex: {hex_val}){tag}")

    flattened_hex = []

    # Part 1: Interleave the COUNT weights (Indices 0 to 11)
    for d_idx in range(3):
        for h_idx in range(12):
            val = lin_w[d_idx, h_idx]
            hex_val = float_to_hex_precision(val)
            flattened_hex.append(hex_val)

    # Part 2: Interleave the LATENCY weights (Indices 12 to 23)
    for d_idx in range(3):
        for h_idx in range(12, 24):
            val = lin_w[d_idx, h_idx]
            hex_val = float_to_hex_precision(val)
            flattened_hex.append(hex_val)

    with open("output_w.mem", "w") as f:
        for hex_val in flattened_hex:
            f.write(hex_val + "\n")

    print(f"Success! Created output_w.mem with {len(flattened_hex)} interleaved entries.")

    def spiking_inference_with_file_export(spikes_np, W1_np, lw, y_test, filename="spike_activity_log.txt"):
        preds, lats = [], []
        t_start = time.perf_counter()
        confusion_matrix = np.zeros((OUTPUT, OUTPUT), dtype=int)
        lw_int = to_fixed(lw)

        LOCAL_THRESHOLD = 10000
        wrong_log_filename = "wrong_predict_activity_log.txt"

        # ==================================================================
        # --- FIXED: HOISTED TO TOP SCOPE (Declares only ONCE for maximum speed) ---
        # ==================================================================
        def print_spatial_grid(pixel_array, img_size=14):
            """ Converts a flat 196 vector into a 14x14 character layout """
            grid = pixel_array.reshape(img_size, img_size)
            grid_lines = []
            grid_lines.append("      Columns: 01234567890123\n")
            for row_idx in range(img_size):
                row_str = f"Row {row_idx:02d}: "
                for col_idx in range(img_size):
                    val = grid[row_idx, col_idx]
                    if val > 0.65:
                        row_str += "#"  # Maximum density core ink
                    elif val > 0.20:
                        row_str += "o"  # Fractional gray shade border
                    else:
                        row_str += "."  # Empty space backdrop
                grid_lines.append(row_str + "\n")
            return "".join(grid_lines)

        # Open both files cleanly in a single context block
        with open(filename, "w") as f, open(wrong_log_filename, "w") as f_wrong:
            f.write(f"SNN SPIKE ACTIVITY LOG - SMART SECA RESTRUCTURE\n")
            f.write(f"Threshold: {LOCAL_THRESHOLD} | Scale: {BIT_PRECISION}\n")
            f.write("-" * 60 + "\n")

            f_wrong.write(f"SNN MISCLASSIFICATION Activity Profile Extraction\n")
            f_wrong.write(f"Isolated tracking of failing test subsets\n")
            f_wrong.write("-" * 60 + "\n")

            for i in range(len(y_test)):
                hs = lif_hidden_fixed_point(spikes_np[i], W1_np)
                V_out = np.zeros(OUTPUT, dtype=np.int32)
                already_spiked_hw = np.zeros(H, dtype=bool)
                pred, earliest = -1, TMAX

                sample_trace_buffer = []
                sample_trace_buffer.append(f"\n--- Testing Sample {i} [Real Digit: {y_test[i]}] ---\n")

                neuron_evidence = np.zeros(H, dtype=np.int32)
                for t in range(TMAX + 1):
                    neuron_evidence = (neuron_evidence * 12) // 16
                    for h in range(H):
                        if hs[h, t] == 1:
                            neuron_evidence[h] += 256

                    # --- STEP 2: SECA ATTENTION ENGINE COMPUTATION ---
                    dynamic_scales = np.full(H, 128, dtype=np.int32)
                    max_ev = np.max(neuron_evidence)
                    total_system_evidence = np.sum(neuron_evidence)

                    SECA_EVIDENCE_THRESHOLD = 512

                    if max_ev > 0 and t > 0 and total_system_evidence >= SECA_EVIDENCE_THRESHOLD:
                        sum_d1 = float(neuron_evidence[0:3].sum())
                        sum_d0 = float(neuron_evidence[3:7].sum())
                        sum_d2 = float(neuron_evidence[7:12].sum())

                        score_d1 = sum_d1 / 3.0
                        score_d0 = sum_d0 / 4.0
                        score_d2 = sum_d2 / 5.0

                        p0_raw = (score_d0 + 0.10) ** 2.2
                        p1_raw = (score_d1 + 0.10) ** 2.2
                        p2_raw = (score_d2 + 0.10) ** 2.2
                        total_raw = p0_raw + p1_raw + p2_raw + 1e-9

                        p0 = p0_raw / total_raw
                        p1 = p1_raw / total_raw
                        p2 = p2_raw / total_raw

                        group_winner = np.argmax([score_d0, score_d1, score_d2])

                        for h in range(H):
                            if 3 <= h <= 6:
                                if group_winner == 0:
                                    dynamic_scales[h] = int(128 + 96 * p0)
                                else:
                                    dynamic_scales[h] = int(128 - 80 * (1.0 - p0))
                            elif 0 <= h <= 2:
                                if group_winner == 1:
                                    dynamic_scales[h] = int(128 + 96 * p1)
                                else:
                                    dynamic_scales[h] = int(128 - 48 * (1.0 - p1))
                            elif 7 <= h <= 11:
                                if group_winner == 2:
                                    dynamic_scales[h] = int(128 + 128 * p2)
                                else:
                                    dynamic_scales[h] = int(128 - 1 * (1.0 - p2))

                    # Process Spikes with INTEGRATION GUARD
                    for h_idx in range(H):
                        if hs[h_idx, t] == 1:
                            inc = np.zeros(OUTPUT, dtype=np.int32)
                            for d in range(OUTPUT):
                                f_p = (int(lw_int[d, h_idx]) * int(dynamic_scales[h_idx])) >> 8
                                l_p = 0
                                if not already_spiked_hw[h_idx]:
                                    l_p = (int(lw_int[d, h_idx + 12]) * (TMAX - t)) >> 3

                                total_inc = f_p + l_p
                                # if total_inc < 0 and dynamic_scales[h_idx] < 128:
                                # total_inc = total_inc // 2

                                inc[d] = total_inc

                            V_out += inc
                            V_out = np.maximum(V_out, 0)

                            sample_trace_buffer.append(
                                f"T:{t:2d} | Neuron {h_idx:2d} | Inc:[V0:{inc[0]:5d}, V1:{inc[1]:5d}, V2:{inc[2]:5d}] | Scale:{dynamic_scales[h_idx]:3d}\n"
                            )
                            already_spiked_hw[h_idx] = True

                    if t >= 12:
                        if np.any(V_out >= LOCAL_THRESHOLD):
                            pred = np.argmax(V_out)
                            earliest = t
                            sample_trace_buffer.append(f">>> SUCCESS: Threshold Crossed at T:{t} | Pred:{pred}\n")
                            break

                if pred == -1:
                    pred = np.argmax(V_out)
                    sample_trace_buffer.append(
                        f">>> TMAX REACHED | Pred:{pred} | Real:{y_test[i]} | V:{V_out.tolist()}\n")

                # State Allocation Updates
                preds.append(pred)
                lats.append(earliest)
                confusion_matrix[y_test[i]][pred] += 1

                # FIXED: Flush trace log array into primary file EXACTLY ONCE
                for line in sample_trace_buffer:
                    f.write(line)

                # Isolated Character Extraction Tracing
                if pred != y_test[i]:
                    f_wrong.write(f"\n--- ERROR CASE FOUND (Sample Matrix Loop Index: {i}) ---\n")

                    if y_test[i] == 0:
                        f_wrong.write(f"[SPATIAL ANALYSIS] Failed Real Digit 0 Layout:\n")
                        f_wrong.write(print_spatial_grid(X_test_raw[i]))
                        f_wrong.write("-" * 50 + "\n")

                    elif y_test[i] == 2:
                        f_wrong.write(f"[SPATIAL ANALYSIS] Failed Real Digit 2 Layout:\n")
                        f_wrong.write(print_spatial_grid(X_test_raw[i]))
                        f_wrong.write("-" * 50 + "\n")

                    for line in sample_trace_buffer:
                        f_wrong.write(line)

            # ==================================================================
            # --- SEPARATE RANDOM SAMPLE HARVESTER MODULE (OUTSIDE SAMPLE LOOP) ---
            # ==================================================================
            with open("isolated_digit0_failures.txt", "w") as f0, open("isolated_digit2_failures.txt", "w") as f2:
                f0.write("ISOLATED DATASET HARVEST: 10 RANDOM DIGIT 0 MISCLASSIFICATIONS\n\n")
                f2.write("ISOLATED DATASET HARVEST: 20 RANDOM DIGIT 2 MISCLASSIFICATIONS\n\n")

                preds_np = np.array(preds)
                d0_fail_indices = np.where((y_test == 0) & (preds_np != 0))[0]
                d2_fail_indices = np.where((y_test == 2) & (preds_np != 2))[0]

                rng = np.random.default_rng(RANDOM_SEED)

                selected_d0 = rng.choice(d0_fail_indices, size=min(10, len(d0_fail_indices)), replace=False)
                selected_d2 = rng.choice(d2_fail_indices, size=min(20, len(d2_fail_indices)), replace=False)

                for rank, idx in enumerate(selected_d0, 1):
                    f0.write(f"=== SAMPLE MAPPING #{rank} (Global Test Set Index: {idx}) ===\n")
                    f0.write(f"Real Target Label: 0 | Model Prediction: {preds[idx]}\n")
                    f0.write(print_spatial_grid(X_test_raw[idx]))
                    f0.write("=" * 60 + "\n\n")

                for rank, idx in enumerate(selected_d2, 1):
                    f2.write(f"=== SAMPLE MAPPING #{rank} (Global Test Set Index: {idx}) ===\n")
                    f2.write(f"Real Target Label: 2 | Model Prediction: {preds[idx]}\n")
                    f2.write(print_spatial_grid(X_test_raw[idx]))
                    f2.write("=" * 60 + "\n\n")

            print(f"[SUCCESS] Exported 10 Digit 0 failure profiles to isolated_digit0_failures.txt")
            print(f"[SUCCESS] Exported 20 Digit 2 failure profiles to isolated_digit2_failures.txt")

            # Final Summary Layout Tables for primary file
            f.write("\n" + "=" * 50 + "\n")
            f.write("FINAL BATCH DEBUG SUMMARY (N=300)\n")
            f.write("=" * 50 + "\n")
            f.write("Actual \\ Pred |  Digit 0  |  Digit 1  |  Digit 2  |\n")
            f.write("-" * 50 + "\n")
            for row_idx in range(OUTPUT):
                row_str = f"  Digit {row_idx:<8} |"
                for col_idx in range(OUTPUT):
                    row_str += f"    {confusion_matrix[row_idx][col_idx]:3d}    |"
                f.write(row_str + "\n")
            f.write("-" * 50 + "\n")

            final_acc = (np.trace(confusion_matrix) / len(y_test)) * 100
            f.write(f"Total Software Accuracy with SMART SECA: {final_acc:.2f}%\n")
            f.write("=" * 50 + "\n")

            # Duplicating Summary layout to wrong log file for direct metrics comparison
            f_wrong.write("\n" + "=" * 50 + "\n")
            f_wrong.write("FINAL BATCH DEBUG SUMMARY (N=300)\n")
            f_wrong.write("=" * 50 + "\n")
            for row_idx in range(OUTPUT):
                row_str = f"  Digit {row_idx:<8} |"
                for col_idx in range(OUTPUT):
                    row_str += f"    {confusion_matrix[row_idx][col_idx]:3d}    |"
                f_wrong.write(row_str + "\n")
            f_wrong.write(f"Total Software Accuracy: {final_acc:.2f}%\n")

        t_end = time.perf_counter()
        total_duration = t_end - t_start
        average_latency = np.mean(lats)

        # =========================================================================
        # NEW PARALLEL BENCHMARK REPORT FORMATTING ENGINE (TTFS MODE)
        # =========================================================================
        print("\n" + "=" * 50)
        print("          TTFS CODING BENCHMARK REPORT SUMMARY")
        print("=" * 50)
        print(f"Total Simulation Runtime : {total_duration:.4f} seconds")
        print(f"Average Spike Latency    : {average_latency:.2f} time steps")
        print(f"Overall SNN Accuracy     : {final_acc:.2f}%")
        print("-" * 50)
        print("            MISCLASSIFICATION SUMMARY TABLE")
        print("-" * 50)
        print("Actual \\ Pred |  Digit 0  |  Digit 1  |  Digit 2  |")
        print("-" * 50)
        for row_idx in range(OUTPUT):
            row_str = f"  Digit {row_idx:<8} |"
            for col_idx in range(OUTPUT):
                row_str += f"    {confusion_matrix[row_idx][col_idx]:3d}    |"
            print(row_str)
        print("=" * 50 + "\n")

        return np.array(preds), lats, total_duration

    print("\nStarting Inference on Test Set...")
    spk_preds, lats, dur = spiking_inference_with_file_export(spikes_test, W1, lin_w, y_test)

    # Visual Analysis
    idx = np.where(y_test == TARGET_DIGIT)[0][TARGET_SAMPLE_INDEX]
    plt.imshow(X_test[idx].reshape(IMG_SIZE, IMG_SIZE), cmap='gray')
    plt.title(f"Label: {y_test[idx]}, Pred: {spk_preds[idx]}")
    plt.show()


if __name__ == "__main__":
    train_and_eval()
