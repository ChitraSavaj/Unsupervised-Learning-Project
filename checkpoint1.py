"""COMP 257 - Final Project, Checkpoint #1: Exploration, Dim. Reduction & Clustering."""
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.io
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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
print(f"Dataset: {len(df)} images, {df.shape[1] - 1} pixels (112x92), {df['label'].nunique()} people")

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
X = df.drop(columns="label").to_numpy(dtype=np.float32)
y = df["label"].to_numpy()
y_int = pd.Categorical(y, categories=names).codes

# 70% train / 15% val / 15% test (stratified so smallest class of 19 keeps ~13/3/3)
X_train, X_tmp, y_train, y_tmp = train_test_split(X, y_int, test_size=0.30, stratify=y_int, random_state=SEED)
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=SEED)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)
X_test_s = scaler.transform(X_test)
print(f"Split sizes -> train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}")

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

# --- 3. Dimensionality Reduction: PCA & t-SNE ---
# t-SNE from sklearn instead of a Keras Autoencoder
pca_full = PCA(n_components=100, random_state=SEED).fit(X_train_s)
cumvar = np.cumsum(pca_full.explained_variance_ratio_)
for n in (10, 20, 50, 100):
    print(f"PCA({n:3d}) cumulative variance: {cumvar[n - 1]:.3f}")

pca = PCA(n_components=50, random_state=SEED)
X_train_pca = pca.fit_transform(X_train_s)
X_val_pca = pca.transform(X_val_s)
X_test_pca = pca.transform(X_test_s)

X_train_tsne = TSNE(n_components=2, perplexity=30, random_state=SEED).fit_transform(X_train_pca)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
axes[0].plot(range(1, 101), cumvar, marker=".", markersize=3)
for n in (10, 20, 50, 100):
    axes[0].axvline(n, color="gray", ls="--", alpha=0.5)
    axes[0].text(n + 1, cumvar[n - 1] - 0.05, f"{n}: {cumvar[n-1]:.0%}", fontsize=8)
axes[0].set(title="PCA Cumulative Explained Variance", xlabel="Components", ylabel="Variance Ratio")

axes[1].scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, cmap="tab20", s=18)
axes[1].set(title="PCA (First 2 Components)", xlabel="PC1", ylabel="PC2")

axes[2].scatter(X_train_tsne[:, 0], X_train_tsne[:, 1], c=y_train, cmap="tab20", s=18)
axes[2].set(title="t-SNE (2D from PCA-50)", xlabel="Dim 1", ylabel="Dim 2")

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
axes[0].set(title="K-Means Elbow Method", xlabel="k", ylabel="Inertia")
axes[1].plot(ks, km_sils, "o-")
axes[1].set(title="K-Means Silhouette vs k", xlabel="k", ylabel="Silhouette")
axes[2].bar(linkages, hc_sils, color=["#4c72b0", "#55a868", "#c44e52"])
axes[2].set(title="Hierarchical Silhouette (k=20)", xlabel="Linkage", ylabel="Silhouette")
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

# 2D clustering visualizations (projected onto t-SNE for visual clarity)
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, col, title in [
    (axes[0], y_train, "True Labels (20 People)"),
    (axes[1], km_labels, "K-Means Clusters (k=20)"),
    (axes[2], hc_labels, f"Hierarchical Clusters ({best_linkage}, k=20)"),
]:
    ax.scatter(X_train_tsne[:, 0], X_train_tsne[:, 1], c=col, cmap="tab20", s=18)
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
    "01_sample_faces.png", "02_split_distribution.png", "03_dim_reduction.png",
    "04_clustering_tuning.png", "05_clustering_2d.png",
])
print("\nAll checks passed. Plots saved to plots/")
