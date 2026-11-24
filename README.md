# COMP 257 — Final Project (Checkpoint #1)

Unsupervised feature analysis, dimensionality reduction, and clustering on the **UMIST Cropped Face Dataset** (`umist_cropped.mat`, 575 grayscale `112×92` images across 20 individuals).

## Project Structure

```text
.
├── umist_cropped.mat      # UMIST Cropped face dataset
├── checkpoint1.py         # End-to-end Checkpoint #1 script (Sections 1–4)
├── checkpoint1_report.md  # Written analysis & rubric justifications
├── plots/                 # Generated visualizations (01–05)
└── FinalProject.pdf       # Project instructions and rubric
```

## Setup & Run

Requires Python 3.10+ and the following packages:

```bash
pip install numpy pandas scipy matplotlib scikit-learn
python checkpoint1.py
```

Running `checkpoint1.py` loads `umist_cropped.mat`, performs stratified splitting and scaling, computes PCA/t-SNE and K-Means/Hierarchical clustering, prints evaluation metrics and cluster purity, verifies outputs with `assert` checks, and saves all 5 figures to `plots/`.

## Checkpoint #1 Summary

1. **Data Preparation & EDA Insights:** Loaded 575 images into a Pandas `DataFrame` (10,304 pixel features + `label`), checked quality (0 NaNs, 10 exact duplicates), and visualized sample faces, class imbalance, brightness variation, scaling effect, and head-pose rotation (`plots/01_sample_faces.png`, `plots/01b_dataset_insights.png`).
2. **Stratified Split & Normalization:** 70/15/15 split (`402 / 86 / 87`) preserving class ratios (19–48 images/person), with `MinMaxScaler` (`[0, 1]`) fit on train only (`plots/02_split_distribution.png`).
3. **Dimensionality Reduction:** Evaluated PCA by variance retention (`80%→22`, `85%→32`, `90%→52`, `95%→99` components; selected **PCA 90% = 52 components**) and 2D t-SNE (`plots/03_dim_reduction.png`).
4. **Clustering (`k = 20` on PCA-90%):** Tuned K-Means (`k`) and Hierarchical (`linkage`) (`plots/04_clustering_tuning.png`, `plots/05_clustering_2d.png`).

| Method                        | Silhouette |    ARI    |    NMI    |  Purity   |
| :---------------------------- | :--------: | :-------: | :-------: | :-------: |
| K-Means (`k=20`)              | **0.230**  |   0.360   |   0.682   | **0.560** |
| Hierarchical (`ward`, `k=20`) |   0.223    | **0.397** | **0.714** |   0.552   |

See [checkpoint1_report.md](checkpoint1_report.md) for the full analysis.
