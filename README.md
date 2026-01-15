# FPGA-BASED-DESIGN-AND-ANALYSIS-OF-SNN-USING-TTFS-CODING-AND-SECA-FEATURE-MODULATION

TTFS-SNN-FPGA-Optimization

This repository contains the source code for a Time-to-First-Spike (TTFS) Spiking Neural Network (SNN) enhanced with a Spiking Efficient Channel Attention (SECA) mechanism. This project was developed as part of a Final Year Project (FYP) focused on optimizing neuromorphic architectures for low-cost FPGA deployment.


🚀 Overview

Traditional Artificial Neural Networks are computationally expensive for edge devices. This project implements a lightweight, two-layer SNN that utilizes TTFS coding to achieve ultra-low latency. To bridge the accuracy gap common in temporal networks, a SECA block is integrated to dynamically scale important feature channels.

Key Features
⦁	TTFS Encoding: Converts MNIST pixel intensities into temporal spike latencies.
⦁	SECA Mechanism: A trainable 1D-convolutional attention block that modulates feature importance.
⦁	Tiered W1 Filters: Fixed structural filters (6, 12, or 24 neurons) optimized for MNIST digit differentiation (0, 1, and 2).
⦁	Early-Stop Inference: A hardware-oriented inference loop that exits as soon as a classification spike is detected.
⦁	Sub-Kilobyte Footprint: The entire model requires less than 1KB of memory, making it ideal for resource-constrained FPGAs.

🛠️ Project Structure

⦁	finalSNN.py: The main script for training the SECA/W3 layers and running the spiking inference analysis.
⦁	requirements.txt: List of necessary Python libraries.
⦁	data/: Directory where the MNIST dataset will be downloaded.
⦁	export_finalSNN/: Directory where trained weights and bias artifacts are saved for hardware reference.

📦 Prerequisites

Ensure you have Python 3.9+ installed. You will need the following libraries:
NumPy
PyTorch
TorchVision
Matplotlib
To install all dependencies, run:
pip install -r requirements.txt

💻 Usage

1. Training and Analysis
To train the model and generate the performance metrics (Accuracy, Latency, and Memory Footprint), run:
python finalSNN.py

2. Modifying Network Capacity
You can modify the number of hidden neurons to compare different hardware profiles. Open finalSNN.py and change the H variable in the CONFIG section:
H = 12  # Set to 6, 12, or 24 for comparative analysis
