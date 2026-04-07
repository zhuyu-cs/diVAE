
# diVAE: Dual-Identifiable Variational Autoencoder Enables Cross-Subject Cyclic Variables in Neural Population Dynamics

## Overview
Neural systems often exhibit **periodic activity patterns**, yet inferring intrinsic cyclic variables from high-dimensional neural observations remains fundamentally ill-posed. Existing identifiable representation learning approaches rely on extrinsic experimental covariates, overlooking the **intrinsic spatial organization** of the neural population itself.

We introduce **diVAE** (Dual-Identifiable Variational Autoencoder), a generative framework that leverages **joint spatial and temporal constraints** to extract conserved cyclic variables from neural population activity. Key contributions include:

1. **Dual Identifiability**: Jointly exploiting spatial structure and temporal dynamics of neural populations to achieve principled, identifiable latent representations — without relying on external behavioral labels.
2. **Conserved Circular Manifold Recovery**: Extracting a cross-subject circular latent manifold that faithfully encodes intrinsic cyclic variables (e.g., circadian phase) from high-dimensional observations.
3. **Attribution-Based Interpretability**: Gradient-based attribution analysis linking latent cyclic dimensions to spatially organized neuronal ensembles across the daily cycle.

Applied to large-scale **Ca²⁺ imaging data** from the **suprachiasmatic nucleus (SCN)** — the mammalian central circadian pacemaker — diVAE:
- Recovered a conserved circular manifold encoding **circadian phase** across subjects
- Enabled accurate **hourly time decoding** on held-out subjects
- Revealed **phase-dependent recruitment of spatially clustered neuronal ensembles** that shift sequentially across the 24-hour cycle


---

## Directory Structure

```
├── data/                        # neural data
├── diVAE/                       # Core model implementation
│   ├── models/                  # Model architectures
│   ├── utils/                   # Utility functions
│   ├── config.py                # Training configuration
│   ├── save_Gradient.py         # Attribution gradient extraction
│   ├── save_lv.py               # Latent variable extraction
│   └── train.py                 # Main training entry point
├── Figure2/                     # Figure 2 analysis & visualization
│   ├── fig2a_classifications/
│   │   ├── latents/             # Pre-saved latent variables (ready to use)
│   │   └── classifier_baseline.py
│   └── fig2b_c_d_3a/
│       └── vis_tsne.py
├── Figure4/                     # Figure 4 analysis
│   ├── fig4b_analysis/
│   ├── fig4d_TraceContrast/
│   └── GenerateSCN_data/
└── Figure5/                     # Figure 5 analysis
    ├── all_lv_grad.pkl          # Pre-saved gradient data (see Data section)
    ├── fig_5b.py
    ├── fig_5c.py
    ├── fig_5e.py
    ├── fig_s5ab.py
    └── fig_s5c.py
```

---

## Environment Setup

### Requirements

```bash
pip install torch mmcv==0.5.0 umap-learn scikit-learn scipy matplotlib
```
---

## Data


The large-scale Ca²⁺ imaging dataset from the suprachiasmatic nucleus (SCN) used in this project is publicly available via the original publication:

> **Cell Research** — Data Availability Section:
> 📄 [https://www.nature.com/articles/s41422-024-00956-x#data-availability](https://www.nature.com/articles/s41422-024-00956-x#data-availability)

Please follow the data access instructions described in the paper's **Data Availability** section to download the raw Ca²⁺ imaging data. Once downloaded, place the data under the `data/` directory.

- Preprocessed data is availiable at [Download preprocessed all_scn.pkl](https://drive.google.com/file/d/1luiECdVaQMY6J0igDrZ_WFtuhNTzaJDF/view?usp=sharing).

**Pre-saved analysis variables** are also provided for convenience so that figures can be reproduced without retraining:

- **Figure 2**: Latent variables are already saved in `Figure2/fig2a_classifications/latents/`.  
  → You can run Figure 2 analyses **directly**.

- **Figure 5**: The attribution gradient file (`all_lv_grad.pkl`) is hosted on Google Drive due to file size:
  - 📥 [Download all_lv_grad.pkl](https://drive.google.com/file/d/1ZcS8YFGokr7zCotOcu9rnYbTnArJxgQw/view?usp=drive_link)
  - Place the downloaded file at `Figure5/all_lv_grad.pkl`

> **Note**: All figures can be reproduced directly using the pre-saved variables above.  
> Retraining the model and re-extracting variables is **optional**.
---

## Usage

The recommended execution order is:

```
Train Model → Extract Latent Variables → Extract Attribution Gradients → Figure Analysis
```

### Step 1: Train the Model

We suggest the following hardware:
- ≥1× A100 NVIDIA GPU
- ≥64GB RAM

```bash
cd diVAE
python train.py config.py
```

Pretrained weights are included in this repository and can be found at:
```bash
./Figure4/GenerateSCN_data/pretrained_weights/weights.pth
```

Update the `load_from` parameter in `diVAE/config.py` with your model checkpoint path before proceeding.

---

### Step 2: Save Latent Variables

```bash
python save_lv.py config.py
```

This extracts and saves the inferred latent variables for all subjects/sessions.

---

### Step 3: Save Attribution Gradients

```bash
python save_Gradient.py  config.py
```

This computes and saves gradient-based attribution scores linking latent dimensions to individual neurons.

---

### Step 4: Figure Reproduction

Once the latent variables and gradients are saved (or downloaded from the links above), all figures can be generated by running the corresponding scripts:

#### Figure 2

```bash
# Fig 2a — Classification analysis
cd Figure2/fig2a_classifications
python classifier_baseline.py

# Fig 2b/c/d & Fig 3a — t-SNE visualization
cd Figure2/fig2b_c_d_3a/
python vis_tsne.py
```


#### Figure 4

```bash
# Navigate into respective subdirectories and run analysis scripts
cd Figure4/GenerateSCN_data
python generate_activity.py --step 1 --blend_mode avg --num_repeats 5 --output_dir ./generated_activity/

cd Figure4/fig4b_analysis
python prepare_sorted_data.py
python plot_temporal_correlation.py
python plot_org_activity.py
python plot_activity_heatmaps.py

cd Figure4/fig4d_TraceContrast

python batch_train.py \
    --input_dir ../GenerateSCN_data/generated_activity \
    --output_dir ./results/ \
    --task standard \
    --num_repeats 5 \
    --gpu 0 \
    --batch_size 64 \
    --lr 0.001 \
    --repr_dims 16 \
    --epochs 100
```
Note that the TraceContrast results of Figure 4 have been included in `Figure4/fig4d_TraceContrast/results`.

#### Figure 5

```bash
cd /data/Cooperation/VAE/open-source/Figure5

python Figure5/fig_5b.py
python Figure5/fig_5c.py
python Figure5/fig_5e.py
python Figure5/fig_s5ab.py
python Figure5/fig_s5c.py
```

> All output figures will be saved to the corresponding figure directories automatically.
