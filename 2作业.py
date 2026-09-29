
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


np.set_printoptions(precision=4, suppress=True)

# Save output files in a subfolder beside this script.
OUTPUT_DIR = Path(__file__).resolve().parent / "svd_compression_output"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


def save_figure(fig, filename):
    """Save a figure and close it."""
    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


output_lines = []


def report(text=""):
    """Print a report line and keep it for the summary file."""
    text = str(text)
    print(text)
    output_lines.append(text)


# ---------------------------------------------------------------------
# Data and model preparation
# ---------------------------------------------------------------------
report("=" * 80)
report("SVD Low-Rank Compression of a Logistic Regression Weight Matrix")
report("=" * 80)

digits = load_digits()
X, y = digits.data, digits.target

report("Dataset: scikit-learn Digits")
report(f"Total samples: {X.shape[0]}")
report(f"Feature dimension: {X.shape[1]}")
report(f"Number of classes: {len(np.unique(y))}")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=0,
    stratify=y,
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# random_state makes the experiment reproducible.
clf = LogisticRegression(max_iter=2000, random_state=0)

with warnings.catch_warnings(record=True) as caught_warnings:
    warnings.simplefilter("always", ConvergenceWarning)
    clf.fit(X_train, y_train)

for warning in caught_warnings:
    if issubclass(warning.category, ConvergenceWarning):
        report(f"Training warning: {warning.message}")

W = clf.coef_.copy()
b = clf.intercept_.copy()
baseline_acc = clf.score(X_test, y_test)

m, n = W.shape
number_of_classes = len(clf.classes_)
maximum_possible_rank = min(m, n)

report(f"Training samples: {len(X_train)}")
report(f"Test samples: {len(X_test)}")
report(f"Weight matrix W shape: {W.shape}")
report(f"Bias vector b shape: {b.shape}")
report(f"Baseline test accuracy: {baseline_acc:.4%}")
report(
    f"Theoretical rank bound: rank(W) <= min({m}, {n}) "
    f"= {maximum_possible_rank}. Each row contains one class's weights, "
    f"so the rank cannot exceed the number of classes "
    f"({number_of_classes})."
)


# ---------------------------------------------------------------------
# Part A: SVD and singular value spectrum
# ---------------------------------------------------------------------
report("\n" + "=" * 80)
report("Part A: SVD and Singular Value Spectrum")
report("=" * 80)

U, singular_values, Vt = np.linalg.svd(W, full_matrices=False)
numerical_rank = np.linalg.matrix_rank(W)

report(f"U shape: {U.shape}")
report(f"Singular value vector shape: {singular_values.shape}")
report(f"Vt shape: {Vt.shape}")
report(f"Singular values (descending):\n{singular_values}")
report(f"Smallest singular value: {singular_values[-1]:.8e}")
report(f"Numerical rank of W: {numerical_rank}")
report(f"Theoretical rank bound: {maximum_possible_rank}")

indices = np.arange(1, len(singular_values) + 1)
fig_a, axes_a = plt.subplots(1, 2, figsize=(12, 4.5))

axes_a[0].plot(indices, singular_values, "bo-", linewidth=1.5, markersize=4)
axes_a[0].axvline(
    numerical_rank,
    color="gray",
    linestyle=":",
    linewidth=2,
    label=f"Numerical rank = {numerical_rank}",
)
axes_a[0].set_xlabel("Singular value index")
axes_a[0].set_ylabel("Singular value")
axes_a[0].set_title("Singular Value Spectrum")
axes_a[0].set_xticks(indices)
axes_a[0].grid(True, alpha=0.3)
axes_a[0].legend()

positive = singular_values > 0
axes_a[1].semilogy(
    indices[positive],
    singular_values[positive],
    "ro-",
    linewidth=1.5,
    markersize=4,
)
axes_a[1].axvline(
    numerical_rank,
    color="gray",
    linestyle=":",
    linewidth=2,
    label=f"Numerical rank = {numerical_rank}",
)
axes_a[1].set_xlabel("Singular value index")
axes_a[1].set_ylabel("Singular value (log scale)")
axes_a[1].set_title("Singular Value Spectrum (Log Scale)")
axes_a[1].set_xticks(indices)
axes_a[1].grid(True, alpha=0.3)
axes_a[1].legend()

fig_a.tight_layout()
part_a_path = save_figure(fig_a, "part_a_singular_values.png")
report(f"Singular value plots saved to: {part_a_path}")


# ---------------------------------------------------------------------
# Part B: SVD and positive semidefinite matrix
# ---------------------------------------------------------------------
report("\n" + "=" * 80)
report("Part B: SVD and Positive Semidefinite Matrix")
report("=" * 80)

G = W @ W.T
eigenvalues = np.linalg.eigvalsh(G)
squared_singular_values = singular_values**2

# Scale-aware tolerance for identifying numerical zero values.
scale = max(
    float(np.max(np.abs(eigenvalues))),
    float(np.max(squared_singular_values)),
    np.finfo(W.dtype).tiny,
)
tol = 10 * max(W.shape) * np.finfo(W.dtype).eps * scale

nonzero_eigenvalues = np.sort(eigenvalues[eigenvalues > tol])
nonzero_squared_singular_values = np.sort(
    squared_singular_values[squared_singular_values > tol]
)

nonzero_match = (
    nonzero_eigenvalues.shape == nonzero_squared_singular_values.shape
    and np.allclose(
        nonzero_eigenvalues,
        nonzero_squared_singular_values,
        rtol=1e-5,
        atol=tol,
    )
)

# eigvalsh returns ascending values; singular values are descending.
max_difference = np.max(
    np.abs(eigenvalues - squared_singular_values[::-1])
)

report(f"G = W W^T shape: {G.shape}")
report(f"Is G symmetric? {np.allclose(G, G.T)}")
report(f"Zero-detection tolerance: {tol:.8e}")
report(f"Nonzero eigenvalues of G:\n{nonzero_eigenvalues}")
report(f"Nonzero squared singular values:\n{nonzero_squared_singular_values}")
report(f"Do the nonzero values match? {nonzero_match}")
report(f"Maximum absolute difference after sorting: {max_difference:.8e}")
report(
    "Explanation: G = W W^T is symmetric because (W W^T)^T = W W^T. "
    "For any real vector x, x^T G x = ||W^T x||_2^2 >= 0, so G is "
    "positive semidefinite. Its nonzero eigenvalues equal the squared "
    "nonzero singular values of W."
)


# ---------------------------------------------------------------------
# Part C: Truncated SVD compression
# ---------------------------------------------------------------------
report("\n" + "=" * 80)
report("Part C: Truncated SVD Compression")
report("=" * 80)

ranks = [1, 2, 4, 6, 8, 10]
original_params = m * n
frobenius_norm_W = np.linalg.norm(W, ord="fro")
results = []

for requested_k in ranks:
    if requested_k > numerical_rank:
        report(
            f"Warning: requested rank {requested_k} exceeds the numerical "
            f"rank of W ({numerical_rank}); it cannot add further "
            f"approximation capacity."
        )

    # SVD supplies at most min(m, n) components.
    k = min(requested_k, len(singular_values))

    Uk = U[:, :k]
    sk = singular_values[:k]
    Vtk = Vt[:k, :]

    # Reconstruct only for calculating the matrix approximation error.
    Wk = (Uk * sk) @ Vtk
    relative_error = (
        np.linalg.norm(W - Wk, ord="fro") / frobenius_norm_W
    )

    # Evaluate directly with the low-rank factors, without using Wk.
    logits = (X_test @ Vtk.T) @ (Uk * sk).T + b
    y_pred = clf.classes_[np.argmax(logits, axis=1)]
    accuracy = np.mean(y_pred == y_test)

    # Count weight parameters only; the unchanged bias is excluded.
    compressed_params = k * (m + n)
    compression_ratio = original_params / compressed_params
    parameter_reduction = 1 - compressed_params / original_params

    results.append(
        {
            "k": requested_k,
            "effective_k": k,
            "relative_error": relative_error,
            "accuracy": accuracy,
            "compressed_params": compressed_params,
            "compression_ratio": compression_ratio,
            "parameter_reduction": parameter_reduction,
        }
    )

report(f"Original weight parameter count: {original_params}")
report(f"Baseline test accuracy: {baseline_acc:.4%}")
report(
    "k | Relative Frobenius Error | Test Accuracy | Compressed Parameters "
    "| Compression Ratio | Parameter Reduction"
)
report("-" * 105)

for row in results:
    report(
        f"{row['k']:2d} | "
        f"{row['relative_error']:.6f}                 | "
        f"{row['accuracy']:.4%}     | "
        f"{row['compressed_params']:4d}                 | "
        f"{row['compression_ratio']:.3f}x             | "
        f"{row['parameter_reduction']:.2%}"
    )

report(
    "Parameter counts cover the weight matrix only; the unchanged bias "
    "vector is excluded. A compression ratio below 1 means the factors "
    "use more parameters than the original matrix."
)
report(
    "For inference, the low-rank factors can be applied directly as "
    "(X_test @ V_k.T) @ (U_k * s_k).T + b, avoiding storage of the full "
    "reconstructed W_k. This can reduce storage and matrix multiplication "
    "work, although actual speed depends on hardware and implementation."
)

ks = [row["k"] for row in results]
errors = [row["relative_error"] for row in results]
accuracies = [row["accuracy"] * 100 for row in results]
parameter_counts = [row["compressed_params"] for row in results]

fig_c, axes_c = plt.subplots(1, 3, figsize=(18, 4.5))

axes_c[0].plot(ks, errors, "bo-", linewidth=1.5, markersize=5)
axes_c[0].set_xlabel("Rank k")
axes_c[0].set_ylabel("Relative Frobenius Error")
axes_c[0].set_title("Reconstruction Error vs Rank")

axes_c[1].plot(
    ks,
    accuracies,
    "go-",
    linewidth=1.5,
    markersize=5,
    label="Compressed model",
)
axes_c[1].axhline(
    baseline_acc * 100,
    color="r",
    linestyle="--",
    linewidth=2,
    label=f"Baseline: {baseline_acc:.2%}",
)

result_6 = next(row for row in results if row["k"] == 6)
axes_c[1].scatter(
    [6],
    [result_6["accuracy"] * 100],
    color="black",
    marker="*",
    s=180,
    zorder=5,
    label="Selected k=6",
)
axes_c[1].set_xlabel("Rank k")
axes_c[1].set_ylabel("Test Accuracy (%)")
axes_c[1].set_title("Test Accuracy vs Rank")
axes_c[1].legend()

axes_c[2].plot(ks, parameter_counts, "mo-", linewidth=1.5, markersize=5)
axes_c[2].axhline(
    original_params,
    color="r",
    linestyle="--",
    linewidth=2,
    label=f"Original: {original_params}",
)
axes_c[2].set_xlabel("Rank k")
axes_c[2].set_ylabel("Compressed Weight Parameters")
axes_c[2].set_title("Parameter Count vs Rank")
axes_c[2].legend()

for ax in axes_c:
    ax.set_xticks(ranks)
    ax.grid(True, alpha=0.3)

fig_c.tight_layout()
part_c_path = save_figure(fig_c, "part_c_compression_results.png")
report(f"Compression comparison plots saved to: {part_c_path}")


# ---------------------------------------------------------------------
# Part D: Cholesky decomposition
# ---------------------------------------------------------------------
report("\n" + "=" * 80)
report("Part D: Cholesky Decomposition")
report("=" * 80)

eps = 1e-3
G_eps = G + eps * np.eye(G.shape[0])

try:
    L = np.linalg.cholesky(G_eps)
    cholesky_error = np.linalg.norm(L @ L.T - G_eps, ord="fro")

    report(f"Regularization epsilon: {eps}")
    report(f"Lower-triangular matrix L shape: {L.shape}")
    report(f"Reconstruction error ||L L^T - G_eps||_F: {cholesky_error:.8e}")
    report(f"Is L L^T approximately equal to G_eps? {np.allclose(L @ L.T, G_eps)}")
    report(
        "Explanation: Adding epsilon times the identity shifts all "
        "eigenvalues of G upward by epsilon, making G_eps positive "
        "definite and suitable for Cholesky decomposition. Cholesky is "
        "used here to factor G_eps; the weight compression itself uses "
        "truncated SVD."
    )
except np.linalg.LinAlgError as error:
    report(f"Cholesky decomposition failed: {error}")


# ---------------------------------------------------------------------
# Part E: Results interpretation (120-180 words)
# ---------------------------------------------------------------------
report("\n" + "=" * 80)
report("Part E: Results Interpretation")
report("=" * 80)

result_8 = next(row for row in results if row["k"] == 8)
acc_6 = result_6["accuracy"]
accuracy_change_6_pp = (acc_6 - baseline_acc) * 100

report(f"Baseline test accuracy: {baseline_acc:.4%}")
report(f"Test accuracy at k=6: {acc_6:.4%}")
report(
    "Accuracy change at k=6 relative to baseline: "
    f"{accuracy_change_6_pp:+.4f} percentage points"
)
report(
    f"Weight parameters at k=6: {result_6['compressed_params']}; "
    f"parameter reduction: {result_6['parameter_reduction']:.2%}"
)
report(f"Test accuracy at k=8: {result_8['accuracy']:.4%}")

essay = (
    f"I would choose k=6 as a practical compromise. The original "
    f"classifier reaches {baseline_acc:.2%} test accuracy, while the "
    f"rank-6 model reaches {acc_6:.2%}, a change of "
    f"{accuracy_change_6_pp:+.2f} percentage points. Its weight factors "
    f"use {result_6['compressed_params']} parameters instead of "
    f"{original_params}, reducing weight storage by "
    f"{result_6['parameter_reduction']:.1%}. Rank 8 is an alternative "
    f"when accuracy matters more. Truncating smaller singular values "
    f"removes directions that contribute less to the weight matrix's "
    f"Frobenius norm, but these directions may still contain useful "
    f"class-discriminative information. Thus, small reconstruction error "
    f"does not guarantee unchanged predictions. This resembles SVD "
    f"compression in large models: smaller factors approximate dense "
    f"weights to reduce storage, with possible quality loss. LLM layers "
    f"differ in shape, spectrum, activations, and sensitivity to "
    f"approximation error, which can accumulate across layers. Assigning "
    f"the same rank everywhere may over-compress sensitive layers and "
    f"waste parameters on less sensitive ones. Rank allocation should "
    f"reflect layer sensitivity and the overall storage or quality budget."
)

essay_word_count = len(essay.split())
report(
    f"\nResults discussion ({essay_word_count} words; assignment target: "
    "120-180 words):"
)
report(essay)

if not 120 <= essay_word_count <= 180:
    report("Warning: revise the discussion to meet the required word count.")


# Save the text report.
summary_path = OUTPUT_DIR / "svd_compression_summary.txt"
summary_path.write_text("\n".join(output_lines), encoding="utf-8")

print("\n" + "=" * 80)
print("Experiment files saved:")
print(f"1. {part_a_path}")
print(f"2. {part_c_path}")
print(f"3. {summary_path}")
print("=" * 80)