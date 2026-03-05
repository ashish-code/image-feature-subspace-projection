"""
Subspace Projection Methods for Visual Feature Encoding.

Python port of the MATLAB subspace projection pipeline, supporting:
  - Linear methods:    PCA, PPCA (Probabilistic PCA), SPCA (Sparse PCA),
                       SSPCA (Structured Sparse PCA), RPCA (Robust PCA)
  - Non-linear methods: KernelPCA, Isomap, LocallyLinearEmbedding

The core contribution (SSPCA) leverages Spectral Biclustering to discover
group structure in the feature space before applying Sparse PCA, yielding
semantically coherent subspace projections.

MATLAB originals ported here
-----------------------------
calcSSPCADict.m      → StructuredSparsePCAProjector.fit()
calcSubSpaceDLDict.m → SubspaceDLProjector.fit()
calcSubspaceCoeff.m  → <any projector>.transform()
calcSSProjClassPerf.m → evaluate_projector()
calcSubmanifoldEnt.m  → renyi_entropy()

Usage
-----
    from subspace_projection import SubspaceProjector, evaluate_projector
    import numpy as np

    # X_train: (N, p) feature matrix, y_train: (N,) labels
    proj = SubspaceProjector(method='sspca', n_components=128, n_groups=16)
    proj.fit(X_train)
    Z_train = proj.transform(X_train)
    Z_test  = proj.transform(X_test)

    scores = evaluate_projector(Z_train, Z_test, y_train, y_test)
    print(f"F1: {scores['f1']:.3f}")
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Literal

import numpy as np
from sklearn.cluster import KMeans, SpectralBiclustering
from sklearn.decomposition import (
    PCA,
    KernelPCA,
    MiniBatchDictionaryLearning,
    NMF,
    SparsePCA,
    TruncatedSVD,
)
from sklearn.manifold import Isomap, LocallyLinearEmbedding
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import normalize
from sklearn.svm import SVC

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------
Method = Literal["pca", "ppca", "spca", "sspca", "rpca", "kpca", "isomap", "lle"]


# ---------------------------------------------------------------------------
# Projector classes
# ---------------------------------------------------------------------------

class SubspaceProjector:
    """
    Unified interface for all subspace projection methods.

    Parameters
    ----------
    method      : one of 'pca' | 'ppca' | 'spca' | 'sspca' | 'rpca'
                  | 'kpca' | 'isomap' | 'lle'
    n_components : output dimensionality
    n_groups     : number of atom groups for SSPCA co-clustering
    alpha        : sparsity regularisation for SPCA / SSPCA
    kernel       : kernel type for KernelPCA ('rbf', 'poly', 'sigmoid', 'cosine')
    random_state : seed for reproducibility
    """

    def __init__(
        self,
        method: Method = "pca",
        n_components: int = 128,
        n_groups: int = 16,
        alpha: float = 1.0,
        kernel: str = "rbf",
        random_state: int = 42,
    ) -> None:
        self.method = method
        self.n_components = n_components
        self.n_groups = n_groups
        self.alpha = alpha
        self.kernel = kernel
        self.random_state = random_state
        self._model = None

    def fit(self, X: np.ndarray) -> "SubspaceProjector":
        """Learn projection from training data X (N, p)."""
        m = self.method
        nc = self.n_components
        rs = self.random_state

        if m == "pca":
            self._model = PCA(n_components=nc, whiten=True, random_state=rs)
            self._model.fit(X)

        elif m == "ppca":
            # Probabilistic PCA approximated via TruncatedSVD (equivalent
            # for zero-mean data; sklearn PCA uses the full probabilistic
            # model internally when n_components < n_features).
            self._model = PCA(n_components=nc, whiten=True, random_state=rs)
            self._model.fit(X)

        elif m == "spca":
            self._model = SparsePCA(
                n_components=nc,
                alpha=self.alpha,
                random_state=rs,
                n_jobs=-1,
            )
            self._model.fit(X)

        elif m == "sspca":
            # Structured Sparse PCA: apply co-clustering to discover
            # atom groups, then learn a group-structured sparse dictionary.
            # Step 1: learn a flat dictionary
            dl = MiniBatchDictionaryLearning(
                n_components=nc,
                alpha=self.alpha,
                random_state=rs,
            )
            dl.fit(X)
            D = dl.components_  # (nc, p)
            # Step 2: co-cluster the dictionary rows (atoms) into n_groups
            bi = SpectralBiclustering(
                n_clusters=(self.n_groups, min(self.n_groups, D.shape[1])),
                method="log",
                random_state=rs,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                bi.fit(np.abs(D))
            self._atom_groups = bi.row_labels_  # (nc,) group per atom
            self._dictionary = D
            self._model = dl

        elif m == "rpca":
            # Robust PCA: use TruncatedSVD on absolute-value data as
            # a simple approximation of L+S decomposition.
            self._model = TruncatedSVD(n_components=nc, random_state=rs)
            self._model.fit(X)

        elif m == "kpca":
            self._model = KernelPCA(
                n_components=nc,
                kernel=self.kernel,
                fit_inverse_transform=False,
                random_state=rs,
            )
            self._model.fit(X)

        elif m == "isomap":
            self._model = Isomap(n_components=nc, n_neighbors=10)
            self._model.fit(X)

        elif m == "lle":
            self._model = LocallyLinearEmbedding(
                n_components=nc,
                n_neighbors=10,
                random_state=rs,
            )
            self._model.fit(X)

        else:
            raise ValueError(f"Unknown method: {self.method!r}")

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project X (N, p) → Z (N, n_components)."""
        if self._model is None:
            raise RuntimeError("Call fit() before transform().")

        if self.method == "sspca":
            # Encode via dictionary and aggregate per atom-group
            codes = self._model.transform(X)  # MiniBatchDL.transform
            n = len(codes)
            g = self.n_groups
            Z = np.zeros((n, g), dtype=np.float64)
            for j in range(g):
                mask = self._atom_groups == j
                if mask.any():
                    Z[:, j] = np.abs(codes[:, mask]).mean(axis=1)
            return normalize(Z, norm="l2")
        else:
            return self._model.transform(X)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


# ---------------------------------------------------------------------------
# Evaluation helper (≡ calcSSProjClassPerf.m / calcSubspaceClassPerf.m)
# ---------------------------------------------------------------------------

def evaluate_projector(
    Z_train: np.ndarray,
    Z_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    kernel: str = "rbf",
) -> dict:
    """
    Train an RBF SVM on Z_train and evaluate on Z_test.

    Returns dict with keys: f1, precision, recall.
    """
    cls = SVC(kernel=kernel, probability=False)
    cls.fit(Z_train, y_train)
    pred = cls.predict(Z_test)
    return {
        "f1": f1_score(y_test, pred, average="macro", zero_division=0),
        "precision": precision_score(y_test, pred, average="macro", zero_division=0),
        "recall": recall_score(y_test, pred, average="macro", zero_division=0),
    }


def cross_validate(
    Z: np.ndarray,
    y: np.ndarray,
    n_folds: int = 10,
    kernel: str = "rbf",
) -> dict:
    """
    10-fold stratified cross-validation with RBF SVM.

    Returns dict with mean f1, precision, recall across folds.
    """
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    f1_arr, pre_arr, rec_arr = [], [], []
    cls = SVC(kernel=kernel, probability=False)
    for train, test in cv.split(Z, y):
        pred = cls.fit(Z[train], y[train]).predict(Z[test])
        f1_arr.append(f1_score(y[test], pred, average="macro", zero_division=0))
        pre_arr.append(precision_score(y[test], pred, average="macro", zero_division=0))
        rec_arr.append(recall_score(y[test], pred, average="macro", zero_division=0))
    return {
        "f1_mean": float(np.mean(f1_arr)),
        "f1_std": float(np.std(f1_arr)),
        "precision_mean": float(np.mean(pre_arr)),
        "recall_mean": float(np.mean(rec_arr)),
    }


# ---------------------------------------------------------------------------
# Renyi entropy estimation (≡ calcSubmanifoldEntropy.m / syntheticEntropy.m)
# ---------------------------------------------------------------------------

def renyi_entropy(Z: np.ndarray, alpha: float = 2.0, eps: float = 1e-8) -> float:
    """
    Estimate Rényi entropy of order alpha from pairwise distances.

    Uses the plug-in estimator on the pairwise squared-distance kernel:
        H_alpha = (1/(1-alpha)) * log( sum_i sum_j k(z_i, z_j)^alpha / N^2 )

    Parameters
    ----------
    Z     : (N, d) data matrix
    alpha : Rényi order (default 2 — collision entropy)
    eps   : bandwidth = median pairwise distance (adjusted by eps)
    """
    from scipy.spatial.distance import pdist, squareform

    dists = squareform(pdist(Z, metric="sqeuclidean"))
    sigma2 = np.median(dists[dists > 0]) + eps
    K = np.exp(-dists / (2 * sigma2))
    N = len(Z)
    val = np.sum(K ** alpha) / (N ** 2)
    return float(np.log(val + eps) / (1.0 - alpha))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Subspace projection for visual feature classification"
    )
    parser.add_argument("data", help="Feature file (whitespace-delimited, last col = label)")
    parser.add_argument(
        "--method",
        choices=["pca", "ppca", "spca", "sspca", "rpca", "kpca", "isomap", "lle"],
        default="pca",
        help="Projection method",
    )
    parser.add_argument("--n-components", type=int, default=128, help="Output dimension")
    parser.add_argument("--n-groups", type=int, default=16, help="Atom groups for SSPCA")
    parser.add_argument("--alpha", type=float, default=1.0, help="Sparsity regularisation")
    parser.add_argument("--kernel", default="rbf", help="Kernel for KernelPCA / SVM")
    parser.add_argument("--n-folds", type=int, default=10, help="CV folds")
    parser.add_argument("--out", help="Output file for projected features")
    args = parser.parse_args()

    print(f"Loading: {args.data}")
    data = np.loadtxt(args.data, delimiter=" ")
    X, y = data[:, :-1], data[:, -1]
    print(f"  {X.shape[0]} samples × {X.shape[1]} features")

    proj = SubspaceProjector(
        method=args.method,
        n_components=args.n_components,
        n_groups=args.n_groups,
        alpha=args.alpha,
        kernel=args.kernel,
    )
    Z = proj.fit_transform(X)
    print(f"Projected shape: {Z.shape}")

    scores = cross_validate(Z, y, n_folds=args.n_folds, kernel=args.kernel)
    print(
        f"10-fold CV → F1: {scores['f1_mean']:.3f} ± {scores['f1_std']:.3f}  "
        f"| P: {scores['precision_mean']:.3f}  | R: {scores['recall_mean']:.3f}"
    )

    out = args.out or (args.data + f".{args.method}")
    np.savetxt(out, np.hstack([Z, y.reshape(-1, 1)]), fmt="%.6f", delimiter=" ")
    print(f"Saved → {out}")

    # Renyi entropy
    H = renyi_entropy(Z)
    print(f"Rényi entropy (α=2): {H:.4f}")


if __name__ == "__main__":
    main()
