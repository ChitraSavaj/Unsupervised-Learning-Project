"""COMP 257 - Final Project, Checkpoint #1: Exploration, Dim. Reduction & Clustering."""
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.io
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score

OUT = Path("plots")
OUT.mkdir(exist_ok=True)
SEED = 42

# --- 1. Data Preparation ---
mat = scipy.io.loadmat("umist_cropped.mat")
facedat = mat["facedat"][0]
names = [str(n[0]) for n in mat["dirnames"][0]]

images, labels = [], []
for person, imgs in zip(names, facedat):
    for i in range(imgs.shape[2]):
        images.append(imgs[:, :, i].ravel())
        labels.append(person)

df = pd.DataFrame(images)
df["label"] = labels
X_raw = df.drop(columns="label").to_numpy(dtype=np.float32)
n_dups = int(df.drop(columns="label").duplicated().sum())
print(
    f"Dataset: {len(df)} images, {X_raw.shape[1]} pixels (112x92), {df['label'].nunique()} people | "
    f"NaNs: {int(np.isnan(X_raw).sum())}, exact dups: {n_dups}, "
    f"pixels: [{int(X_raw.min())}, {int(X_raw.max())}], mean={X_raw.mean():.1f}"
)

# Sample image per person (4x5 grid)
fig, axes = plt.subplots(4, 5, figsize=(10, 8))
for ax, person in zip(axes.flat, names):
    img = df.loc[df["label"] == person].iloc[0, :-1].to_numpy(dtype=np.uint8).reshape(112, 92)
    ax.imshow(img, cmap="gray")
    ax.set_title(person)
    ax.axis("off")
fig.suptitle("UMIST Sample Face per Person (20 Individuals)")
fig.tight_layout()
fig.savefig(OUT / "01_sample_faces.png", dpi=150)
plt.close(fig)

# --- 2. Data Splitting & Normalization ---
X = X_raw
y = df["label"].to_numpy()
y_int = pd.Categorical(y, categories=names).codes

# 70% train / 15% val / 15% test (stratified so smallest class of 19 keeps ~13/3/3)
X_train, X_tmp, y_train, y_tmp = train_test_split(X, y_int, test_size=0.30, stratify=y_int, random_state=SEED)
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=SEED)

scaler = MinMaxScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)
X_test_s = scaler.transform(X_test)
print(f"Split sizes -> train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}")

# Preprocessing & dataset insights
fig, axes = plt.subplots(2, 2, figsize=(13, 8))
per_person_counts = df["label"].value_counts().reindex(names)
axes[0, 0].bar(names, per_person_counts, color="#4c72b0")
axes[0, 0].axhline(per_person_counts.mean(), color="crimson", ls="--", label=f"Mean ({per_person_counts.mean():.1f})")
axes[0, 0].set(title=f"Class Imbalance (19–48 imgs/person | {n_dups} exact dups)", xlabel="Person", ylabel="Images")
axes[0, 0].legend(fontsize=8)

per_person_bright = pd.Series(X_raw.mean(axis=1)).groupby(df["label"]).mean().reindex(names)
axes[0, 1].bar(names, per_person_bright, color="#dd8452")
axes[0, 1].axhline(X_raw.mean(), color="navy", ls="--", label=f"Global mean ({X_raw.mean():.1f})")
axes[0, 1].set(title="Mean Brightness per Person (Justifies Normalization)", xlabel="Person", ylabel="Mean Pixel (0–255)")
axes[0, 1].legend(fontsize=8)

axes[1, 0].hist(X_train.ravel(), bins=40, color="#4c72b0", alpha=0.55, label="Raw (train)")
ax_s = axes[1, 0].twiny()
ax_s.hist(X_train_s.ravel(), bins=40, color="#55a868", alpha=0.45, label="Normalized [0, 1]")
axes[1, 0].set(title="Pixel Distribution: Raw vs. MinMaxScaler", xlabel="Raw Pixel [1-239]", ylabel="Pixel Count")
ax_s.set_xlabel("Normalized Pixel [0, 1]", color="#2f7d32")
axes[1, 0].legend(loc="upper right", fontsize=8)
ax_s.legend(loc="center right", fontsize=8)

# Pose sequence for person 1a
p_imgs = facedat[0]
idx = np.linspace(0, p_imgs.shape[2] - 1, 6, dtype=int)
strip = np.hstack([p_imgs[:, :, i] for i in idx])
axes[1, 1].imshow(strip, cmap="gray")
axes[1, 1].set_title("Head-Pose Rotation Within One Person (1a: Profile → Frontal)")
axes[1, 1].axis("off")

fig.tight_layout()
fig.savefig(OUT / "01b_dataset_insights.png", dpi=150)
plt.close(fig)

counts = pd.DataFrame({
    "Train": pd.Series(y_train).value_counts().sort_index(),
    "Val": pd.Series(y_val).value_counts().sort_index(),
    "Test": pd.Series(y_test).value_counts().sort_index(),
})
counts.index = names
ax = counts.plot(kind="bar", figsize=(12, 4), title="Stratified Split: Images per Person")
ax.set_xlabel("Person")
ax.set_ylabel("Image Count")
plt.tight_layout()
plt.savefig(OUT / "02_split_distribution.png", dpi=150)
plt.close()

# --- 3. Dimensionality Reduction: PCA (by variance %) & t-SNE ---
# t-SNE from sklearn instead of a Keras Autoencoder
pca_full = PCA(random_state=SEED).fit(X_train_s)
cumvar = np.cumsum(pca_full.explained_variance_ratio_)

var_targets = (0.80, 0.85, 0.90, 0.95)
var_counts = {}
for pct in var_targets:
    n_comp = int(PCA(n_components=pct, random_state=SEED).fit(X_train_s).n_components_)
    var_counts[pct] = n_comp
    print(f"PCA({int(pct * 100)}%) -> {n_comp} components")

pca = PCA(n_components=0.90, random_state=SEED)
X_train_pca = pca.fit_transform(X_train_s)
X_val_pca = pca.transform(X_val_s)
X_test_pca = pca.transform(X_test_s)
print(f"Selected PCA(90%) -> {pca.n_components_} components")

X_train_tsne = TSNE(n_components=2, perplexity=30, random_state=SEED).fit_transform(X_train_pca)

def label_centers(ax, coords, group_ids, fmt):
    for g in np.unique(group_ids):
        cx, cy = np.median(coords[group_ids == g], axis=0)
        ax.text(cx, cy, fmt(g), fontsize=7, fontweight="bold", ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="gray", alpha=0.82, lw=0.5))

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
max_k = min(len(cumvar), var_counts[0.95] + 30)
axes[0].plot(range(1, max_k + 1), cumvar[:max_k], marker=".", markersize=2)
for pct, n_comp in var_counts.items():
    color = "crimson" if pct == 0.90 else "gray"
    axes[0].axhline(pct, color=color, ls="--", alpha=0.45)
    axes[0].axvline(n_comp, color=color, ls="--", alpha=0.45)
    axes[0].plot(n_comp, pct, "o", color=color, markersize=5)
    axes[0].text(n_comp + 2, pct - 0.04, f"{int(pct * 100)}% ({n_comp})", fontsize=8, color=color)
axes[0].set(title="PCA Variance Thresholds vs. Components", xlabel="Components", ylabel="Cumulative Variance")

axes[1].scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, cmap="tab20", s=20)
axes[1].set(title="PCA (First 2 Components)", xlabel="PC1", ylabel="PC2")

axes[2].scatter(X_train_tsne[:, 0], X_train_tsne[:, 1], c=y_train, cmap="tab20", s=20)
label_centers(axes[2], X_train_tsne, y_train, lambda i: names[i])
axes[2].set(title=f"t-SNE (2D from PCA-90%, {pca.n_components_} comps)", xlabel="Dim 1", ylabel="Dim 2")

fig.tight_layout()
fig.savefig(OUT / "03_dim_reduction.png", dpi=150)
plt.close(fig)

# --- 4. Clustering: K-Means & Hierarchical ---
def purity_score(y_true, y_pred):
    ct = pd.crosstab(y_pred, y_true)
    return ct.max(axis=1).sum() / len(y_true), ct

# Tune K-Means: sweep k
ks = [5, 10, 15, 20, 25, 30]
inertias, km_sils = [], []
for k in ks:
    km = KMeans(n_clusters=k, n_init=20, random_state=SEED).fit(X_train_pca)
    inertias.append(km.inertia_)
    km_sils.append(silhouette_score(X_train_pca, km.labels_))

# Tune Hierarchical: compare linkages at k=20
linkages = ["ward", "complete", "average"]
hc_sils = [
    silhouette_score(X_train_pca, AgglomerativeClustering(n_clusters=20, linkage=lk).fit_predict(X_train_pca))
    for lk in linkages
]
best_linkage = linkages[int(np.argmax(hc_sils))]

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].plot(ks, inertias, "o-")
axes[0].axvline(20, color="crimson", ls="--", alpha=0.6, label="Selected k=20")
axes[0].set(title="K-Means Elbow Method", xlabel="k", ylabel="Inertia")
axes[0].legend(fontsize=8)
axes[1].plot(ks, km_sils, "o-")
axes[1].axvline(20, color="crimson", ls="--", alpha=0.6, label="Selected k=20")
axes[1].set(title="K-Means Silhouette vs k", xlabel="k", ylabel="Silhouette")
axes[1].legend(fontsize=8)
bars = axes[2].bar(linkages, hc_sils, color=["#4c72b0", "#55a868", "#c44e52"])
axes[2].bar_label(bars, fmt="%.3f", padding=2, fontsize=9)
axes[2].set(title="Hierarchical Silhouette (k=20)", xlabel="Linkage", ylabel="Silhouette", ylim=(0, max(hc_sils) * 1.15))
fig.tight_layout()
fig.savefig(OUT / "04_clustering_tuning.png", dpi=150)
plt.close(fig)

# Final models at k=20 (matches 20 known individuals; best Silhouette plateau)
km_labels = KMeans(n_clusters=20, n_init=20, random_state=SEED).fit_predict(X_train_pca)
hc_labels = AgglomerativeClustering(n_clusters=20, linkage=best_linkage).fit_predict(X_train_pca)

print(f"\nBest Hierarchical linkage: {best_linkage} (sil={max(hc_sils):.3f})")
print(f"{'Method':<22} {'Sil':>6} {'ARI':>6} {'NMI':>6} {'Purity':>7}")
for name, pred in [("K-Means (k=20)", km_labels), (f"Hierarchical ({best_linkage})", hc_labels)]:
    pur, ct = purity_score(y_train, pred)
    sil = silhouette_score(X_train_pca, pred)
    ari = adjusted_rand_score(y_train, pred)
    nmi = normalized_mutual_info_score(y_train, pred)
    print(f"{name:<22} {sil:6.3f} {ari:6.3f} {nmi:6.3f} {pur:7.3f}")

# Per-cluster purity summary for the stronger method
_, ct_hc = purity_score(y_train, hc_labels)
cluster_summary = pd.DataFrame({
    "size": ct_hc.sum(axis=1),
    "dominant_person": [names[c] for c in ct_hc.idxmax(axis=1)],
    "dominant_count": ct_hc.max(axis=1),
    "purity": (ct_hc.max(axis=1) / ct_hc.sum(axis=1)).round(3),
})
print("\nHierarchical cluster composition:")
print(cluster_summary.to_string())

# 2D clustering visualizations (projected onto t-SNE with cluster/person ID labels)
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, col, title, fmt in [
    (axes[0], y_train, "True Labels (20 People: 1a–1t)", lambda i: names[i]),
    (axes[1], km_labels, "K-Means Clusters (C0–C19)", lambda i: f"C{i}"),
    (axes[2], hc_labels, f"Hierarchical ({best_linkage}, C0–C19)", lambda i: f"C{i}"),
]:
    ax.scatter(X_train_tsne[:, 0], X_train_tsne[:, 1], c=col, cmap="tab20", s=20)
    label_centers(ax, X_train_tsne, col, fmt)
    ax.set(title=title, xlabel="t-SNE 1", ylabel="t-SNE 2")
fig.tight_layout()
fig.savefig(OUT / "05_clustering_2d.png", dpi=150)
plt.close(fig)

# --- Sanity checks ---
assert len(df) == 575 and df["label"].nunique() == 20
assert len(X_train) + len(X_val) + len(X_test) == 575
assert not np.isnan(X_train_pca).any()
assert 0 <= purity_score(y_train, km_labels)[0] <= 1
assert all((OUT / f).exists() for f in [
    "01_sample_faces.png", "01b_dataset_insights.png", "02_split_distribution.png",
    "03_dim_reduction.png", "04_clustering_tuning.png", "05_clustering_2d.png",
])
print("\nAll checks passed. Plots saved to plots/")
