import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

np.set_printoptions(precision=4, suppress=True)

# ============================================================
# 起始代码：训练分类器并提取权重矩阵
# ============================================================
digits = load_digits()
X, y = digits.data, digits.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

clf = LogisticRegression(max_iter=2000, random_state=0)
clf.fit(X_train, y_train)

W = clf.coef_.copy()        # shape: (10, 64)
b = clf.intercept_.copy()   # shape: (10,)
baseline_acc = clf.score(X_test, y_test)

print("W shape:", W.shape)
print("Baseline accuracy:", baseline_acc)


# ============================================================
# Part A — SVD 与奇异值谱
# ============================================================
print("\n" + "=" * 50)
print("Part A — SVD & Singular Value Spectrum")
print("=" * 50)

U, s, Vt = np.linalg.svd(W, full_matrices=False)  # U:(10,10), s:(10,), Vt:(10,64)

print(f"U shape: {U.shape}")
print(f"s shape: {s.shape}")
print(f"Vt shape: {Vt.shape}")
print(f"Singular values: {s}")

# 数值秩：大于阈值的奇异值个数
threshold = 1e-10
numerical_rank = np.sum(s > threshold)
print(f"Numerical rank (threshold={threshold}): {numerical_rank}")

# 绘制奇异值谱
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.plot(s, 'bo-', markersize=6)
plt.xlabel('Index i')
plt.ylabel('Singular value σᵢ')
plt.title('Singular Value Spectrum')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.semilogy(s, 'ro-', markersize=6)
plt.xlabel('Index i')
plt.ylabel('Singular value σᵢ (log scale)')
plt.title('Singular Value Spectrum (log)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('part_a_singular_values.png', dpi=150, bbox_inches='tight')
print("Saved: part_a_singular_values.png")
plt.show()


# ============================================================
# Part B — 将 SVD 与 PSD 矩阵联系起来
# ============================================================
print("\n" + "=" * 50)
print("Part B — SVD & PSD Matrix Connection")
print("=" * 50)

# 构造 G = W @ W.T，形状 (10, 10)，对称半正定
G = W @ W.T
eigvals = np.linalg.eigvalsh(G)  # 返回升序排列的特征值
eigvals_sorted_desc = eigvals[::-1]  # 降序以便与 s**2 比较

print(f"G = W @ W.T shape: {G.shape}")
print(f"Eigenvalues of G (descending): {np.round(eigvals_sorted_desc, 4)}")
print(f"s**2 (singular values squared): {np.round(s**2, 4)}")

# 验证：特征值应等于奇异值的平方（允许浮点误差）
max_diff = np.max(np.abs(np.sort(eigvals)[::-1] - s**2))
print(f"\nMax |λᵢ - σᵢ²|: {max_diff:.2e}")
print(f"Match (tol=1e-6): {max_diff < 1e-6}")

# 简短解释（打印输出）
explanation_b = """
Explanation:
G = W·Wᵀ 是半正定(PSD)的，因为对于任意非零向量 x：
xᵀGx = xᵀ(WWᵀ)x = (Wᵀx)ᵀ(Wᵀx) = ||Wᵀx||² ≥ 0
即二次型非负，故 G 为 PSD。其特征值为 σᵢ² ≥ 0，
与 SVD 的奇异值平方一一对应。
"""
print(explanation_b)


# ============================================================
# Part C — 截断 SVD 压缩
# ============================================================
print("\n" + "=" * 50)
print("Part C — Truncated SVD Compression")
print("=" * 50)

ranks = [1, 2, 4, 6, 8, 10]
results = []

m, n = W.shape  # m=10, n=64
frobenius_norm_W = np.linalg.norm(W, 'fro')

for k in ranks:
    # 秩-k 近似: W_k = U_k · Σ_k · V_kᵀ
    Wk = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]

    # 相对 Frobenius 误差
    rel_error = np.linalg.norm(W - Wk, 'fro') / frobenius_norm_W

    # 用压缩后的权重评估准确率
    logits = X_test @ Wk.T + b
    y_pred = np.argmax(logits, axis=1)
    acc = np.mean(y_pred == y_test)

    # 分解后参数量: k*(m+n)
    factor_params = k * (m + n)
    original_params = m * n
    compression_ratio = original_params / factor_params

    results.append((k, rel_error, acc, factor_params, compression_ratio))

    print(f"k={k:2d} | rel_err={rel_error:.4f} | acc={acc:.4f} | "
          f"params={factor_params:4d} | compress_ratio={compression_ratio:.2f}x")

print(f"\nOriginal params: {original_params}")
print("k  | Rel Error | Accuracy | Params | Compression")
for row in results:
    print(f"{row[0]:2d} |   {row[1]:.4f}   |   {row[2]:.4f}  | {row[3]:5d} |     {row[4]:.2f}x")

# 绘制压缩结果图
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

ks = [r[0] for r in results]
errors = [r[1] for r in results]
accs = [r[2] for r in results]
params = [r[3] for r in results]

axes[0].plot(ks, errors, 'bo-', markersize=8)
axes[0].set_xlabel('Rank k')
axes[0].set_ylabel('Relative Frobenius Error')
axes[0].set_title('Compression Error vs Rank')
axes[0].grid(True, alpha=0.3)

axes[1].plot(ks, accs, 'go-', markersize=8)
axes[1].axhline(y=baseline_acc, color='r', linestyle='--', label=f'Baseline ({baseline_acc:.4f})')
axes[1].set_xlabel('Rank k')
axes[1].set_ylabel('Test Accuracy')
axes[1].set_title('Accuracy vs Rank')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

axes[2].plot(ks, params, 'mo-', markersize=8)
axes[2].axhline(y=original_params, color='r', linestyle='--', label=f'Original ({original_params})')
axes[2].set_xlabel('Rank k')
axes[2].set_ylabel('Parameter Count')
axes[2].set_title('Params vs Rank')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('part_c_compression_results.png', dpi=150, bbox_inches='tight')
print("\nSaved: part_c_compression_results.png")
plt.show()


# ============================================================
# Part D — Cholesky 回顾
# ============================================================
print("\n" + "=" * 50)
print("Part D — Cholesky Decomposition")
print("=" * 50)

eps = 1e-3
G_eps = W @ W.T + eps * np.eye(W.shape[0])  # 加入正则项确保严格正定
L = np.linalg.cholesky(G_eps)  # 下三角矩阵

reconstruction_error = np.linalg.norm(L @ L.T - G_eps)
print(f"G_eps shape: {G_eps.shape}")
print(f"L shape: {L.shape}")
print(f"||L·Lᵀ - G_ε|| = {reconstruction_error:.2e}")
print(f"Cholesky successful: {reconstruction_error < 1e-6}")

explanation_d = """
Explanation:
加入 εI (ε > 0) 使 G_ε = WWᵀ + εI 成为严格正定(PD)矩阵。
因为 WWᵀ 可能是半正定但非严格正定（存在零特征值），
而 εI 将所有特征值至少提升 ε > 0，保证所有特征值 > 0，
从而满足 Cholesky 分解的正定性要求。
"""
print(explanation_d)


# ============================================================
# Part E — 结果解释
# ============================================================
print("\n" + "=" * 50)
print("Part E — Results Interpretation")
print("=" * 50)

best_k_idx = np.argmax([r[2] for r in results])  # 选准确率最高的
best_k = results[best_k_idx][0]

explanation_e = f"""
Results Interpretation (~150 words):

1. Choice of rank k:
   I would choose k=6 or k=8 as the optimal trade-off point.
   At k=6, accuracy remains {results[3][2]:.4f} (near baseline {baseline_acc:.4f})
   while achieving ~{results[3][4]:.1f}x compression. Going from k=6 to k=8
   yields diminishing returns — accuracy improves only slightly while
   parameter count increases by 33%.

2. Information loss from truncation:
   Truncating smaller singular values discards the "less important"
   directions in weight space. These directions typically capture
   fine-grained class distinctions or noise-like patterns that
   contribute minimally to overall classification performance.
   The singular value spectrum shows rapid decay, confirming that
   most information is concentrated in the top few components.

3. Connection to LLM compression:
   This exercise mirrors LLM weight compression: large linear layers
   have low effective rank, so truncated SVD can dramatically reduce
   parameters with minimal quality loss. The same trade-off between
   compression ratio and task performance applies at scale.

4. Why per-layer rank matters:
   Real LLMs need layer-specific rank allocation because different
   layers capture different types of information (attention vs FFN,
   early vs late layers). Uniform k across all layers is suboptimal —
   some layers may need higher rank to preserve critical knowledge
   while others can be aggressively compressed. Adaptive methods like
   importance scoring or sensitivity analysis can determine optimal
   per-layer ranks automatically.
"""
print(explanation_e)

# 保存完整结果摘要
summary = f"""
=== SVD Model Compression Summary ===
Weight matrix W shape: {W.shape}
Baseline accuracy: {baseline_acc:.4f}
Numerical rank of W: {numerical_rank}

Part A - Top 5 singular values: {s[:5]}
Part B - Max |λ - σ²|: {max_diff:.2e}

Part C - Compression Results:
"""
for r in results:
    summary += f"  k={r[0]:2d}: err={r[1]:.4f}, acc={r[2]:.4f}, params={r[3]}, ratio={r[4]:.2f}x\n"

summary += f"\nPart D - Cholesky reconstruction error: {reconstruction_error:.2e}\n"

with open('svd_compression_summary.txt', 'w', encoding='utf-8') as f:
    f.write(summary)
print("\nSaved: svd_compression_summary.txt")
print("\nAll done!")
