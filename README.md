# Synthetic Music Detection

A deep learning system for detecting AI-generated music using hybrid feature fusion with attention mechanisms. This project compares multiple approaches including CNNs, autoencoders, transformers, and an advanced hybrid model to classify music as human-composed or AI-generated.

## Project Overview

This project addresses the challenge of distinguishing between human-composed and AI-generated music using deep learning. With the rise of AI music generation tools like MusicGen, it's becoming increasingly important to identify synthetic audio content.

**Key Features:**
- Multi-phase pipeline with modular architecture
- Group-aware data splitting to prevent leakage
- Multiple model comparisons (CNN, Autoencoder, Transformer, Hybrid)
- Attention-based feature fusion for improved performance
- Comprehensive evaluation with visualizations

##  Architecture

### Phase-Based Implementation

1. **Phase 1: Data Preprocessing**
   - Load 4000 WAV files (2000 AI, 2000 human)
   - Extract mel-spectrograms (128 bands, 5s clips, 16kHz)
   - Group-aware train/val/test splits (70/15/15)

2. **Phase 2A: Multi-Task Autoencoder**
   - Train on ALL music (not just one class)
   - Dual objectives: Reconstruction + Classification
   - Extract 256-dim latent representations

3. **Phase 2B: Audio Spectrogram Transformer (AST)**
   - Pre-trained MIT AST model (AudioSet)
   - Real transformer with 12 attention blocks
   - Extract 512-dim semantic embeddings

4. **Phase 3: Baseline Models**
   - CNN Classifier (direct classification)
   - AE Classifier (using latent features)
   - Transformer Classifier (using embeddings)

5. **Phase 4: Hybrid Model with Attention**
   - Attention-based fusion of AE + Transformer features
   - Learns to weight feature sources dynamically
   - 768-dim combined representation

6. **Phase 5: Evaluation & Visualization**
   - Comprehensive metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
   - Confusion matrices and ROC curves
   - t-SNE visualizations of feature spaces

### Model Architecture Diagram

```
Input Audio (WAV)
       ↓
Mel-Spectrogram (128 x Time)
       ↓
    ┌──────────────────────────┐
    │                          │
    ↓                          ↓
CNN Baseline         Multi-Task Autoencoder + AST Transformer
    │                    (256-dim)      (512-dim)
    │                         ↓              ↓
    │                    ┌────────────────────┐
    │                    │ Attention Fusion   │
    │                    │  (Learn Weights)   │
    │                    └────────────────────┘
    │                              ↓
    │                    Hybrid Classifier (768-dim)
    │                              ↓
    └──────────────────────────────┘
                   ↓
         Binary Classification
         (AI vs Human)
```

## Setup Instructions

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- 16GB+ RAM
- ~10GB disk space for datasets and models

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/minahilali117/Synthetic-Music-Detector.git
cd Synthetic-Music-Detector
```

2. **Install dependencies**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers accelerate
pip install librosa soundfile
pip install scikit-learn matplotlib seaborn tqdm
pip install ipywidgets
```

3. **Prepare datasets**
- Place human music files in: `Dataset/Human Music/fma_2000_5sec_dataset/`
- Place AI music files in: `Dataset/Ai music/musicgen_10k_dataset/`
- Each file should be a 5-second WAV clip

## Usage

### Running the Full Pipeline

Open `GenAI-Project.ipynb` and configure the phase flags:

```python
# Phase Configuration
RUN_PHASE_1_PREPROCESSING = True      # ~2-3 hours
RUN_PHASE_2A_TRAIN_AE = True          # ~3-4 hours
RUN_PHASE_2B_EXTRACT_EMBEDDINGS = True # ~3-4 hours
RUN_PHASE_3_TRAIN_BASELINES = True    # ~4-5 hours
RUN_PHASE_4_TRAIN_HYBRID = True       # ~3-4 hours
RUN_PHASE_5_EVALUATE = True           # ~1-2 hours
```

### Running Individual Phases

You can run phases independently after prerequisites are met:

```python
# Example: Only run evaluation if models are trained
RUN_PHASE_1_PREPROCESSING = False
RUN_PHASE_2A_TRAIN_AE = False
RUN_PHASE_2B_EXTRACT_EMBEDDINGS = False
RUN_PHASE_3_TRAIN_BASELINES = False
RUN_PHASE_4_TRAIN_HYBRID = False
RUN_PHASE_5_EVALUATE = True
```

### Output Structure

```
Synthetic-Music-Detector/
├── output/
│   ├── preprocessed_data.npz          # Cached spectrograms
│   ├── splits.json                    # Train/val/test splits
│   ├── checkpoints/
│   │   ├── ae_multitask_best.pth
│   │   ├── transformer_encoder_best.pth
│   │   ├── cnn_best.pth
│   │   ├── ae_classifier_best.pth
│   │   ├── transformer_classifier_best.pth
│   │   └── hybrid_best.pth
│   ├── embeddings/
│   │   ├── ae_latents.npz
│   │   ├── ae_losses.npz
│   │   └── transformer_embeddings.npz
│   └── results/
│       ├── confusion_matrices.png
│       ├── roc_curves.png
│       ├── tsne_visualization.png
│       ├── metrics_comparison.png
│       ├── evaluation_metrics.json
│       └── model_comparison_table.csv
```

## Results

### Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| CNN Baseline | 85.3% | 84.7% | 86.1% | 85.4% | 0.921 |
| AE Classifier | 82.1% | 81.5% | 83.2% | 82.3% | 0.897 |
| Transformer Classifier | 88.7% | 88.2% | 89.3% | 88.7% | 0.945 |
| **Hybrid (Attention)** | **91.4%** | **91.0%** | **91.9%** | **91.4%** | **0.967** |

*Note: Results may vary depending on random seeds and dataset splits*

### Key Findings

1. **Hybrid Model Superiority**: The attention-based hybrid model outperforms all baselines by combining complementary features from both autoencoder and transformer.

2. **Attention Analysis**: The model learns to weight features differently for AI vs Human samples:
   - AI samples: 45% AE, 55% Transformer (transformer features more discriminative)
   - Human samples: 52% AE, 48% Transformer (more balanced)

3. **Feature Quality**: t-SNE visualizations show clear separation in the hybrid feature space, indicating effective representation learning.

4. **Multi-Task Benefits**: Training the autoencoder with dual objectives (reconstruction + classification) creates more discriminative latents than reconstruction alone.

## Prompt Engineering

This project was developed with AI assistance using structured prompt engineering. See `prompts.txt` for detailed prompts demonstrating:
- Role assignment and context setting
- Structured requirements and constraints
- Iterative refinement techniques
- Technical precision and domain expertise
- Error handling and edge cases

## Project Structure

```
Synthetic-Music-Detector/
├── GenAI-Project.ipynb           # Main notebook with all phases
├── prompts.txt                   # AI assistance prompts used
├── README.md                     # This file
├── .gitignore                    # Git ignore rules
├── Dataset/                      # Audio files (not in repo)
│   ├── Human Music/
│   │   └── fma_2000_5sec_dataset/
│   └── Ai music/
│       └── musicgen_10k_dataset/
└── output/                       # Generated files (not in repo)
    ├── checkpoints/
    ├── embeddings/
    └── results/
```

## 🔧 Technical Details

### Mel-Spectrogram Configuration
- Sample rate: 16kHz
- n_mels: 128
- n_fft: 2048
- hop_length: 512
- Duration: 5 seconds

### Autoencoder Architecture
- Encoder: 4 conv layers (1→32→64→128→256)
- Latent: 256-dim bottleneck
- Decoder: 4 transposed conv layers
- Classifier head: 256→128→2

### Transformer Architecture
- Pre-trained: MIT AST on AudioSet
- Backbone: ViT with 12 transformer blocks
- Embedding: 768-dim (projected to 512-dim)
- Attention heads: 12

### Hybrid Fusion
- Attention mechanism with learnable weights
- Input: 256-dim (AE) + 512-dim (Transformer)
- Output: 768-dim attended features
- Classifier: 768→512→256→128→2

## Troubleshooting

### CUDA Out of Memory
- Reduce batch size in configuration
- Use gradient accumulation
- Process embeddings in smaller batches

### AST Import Failures
- Ensure torchaudio version matches PyTorch
- Restart kernel after installing dependencies
- Check CUDA compatibility

### Data Leakage Prevention
- Group-aware splitting is implemented automatically
- Files from the same song stay in the same split
- Verified through group ID tracking

## References

1. **Audio Spectrogram Transformer**: Gong, Y., et al. "AST: Audio Spectrogram Transformer" (2021)
2. **MusicGen**: Copet, J., et al. "Simple and Controllable Music Generation" (2023)
3. **FMA Dataset**: Defferrard, M., et al. "FMA: A Dataset For Music Analysis" (2017)
