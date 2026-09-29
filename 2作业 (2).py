import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

np.set_printoptions(precision=4, suppress=True)


# 起始代码：训练分类器并提取权重矩阵
digits = load_digits()
X, y = digits.data, digits.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=0,
    stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

clf = LogisticRegression(max_iter=2000, random_state=0)
clf.fit(X_train, y_train)

W = clf.coef_.copy()
b = clf.intercept_.copy()
baseline_acc = clf.score(X_test, y_test)

# 同时记录控制台输出，最后保存为文本
output_lines = []


def report(text=""):
    text = str(text)
    print(text)
    output_lines.append(text)


report(f"W 的形状：{W.shape}")
report(f"基线准确率：{baseline_acc:.2%}")


# Part A：SVD 与奇异值谱

report("\nPart A：SVD 与奇异值谱")

U, s, Vt = np.linalg.svd(W, full_matrices=False)
numerical_rank = np.linalg.matrix_rank(W)

report(f"U 的形状：{U.shape}")
report(f"s 的形状：{s.shape}")
report(f"Vt 的形状：{Vt.shape}")
report(f"奇异值：{s}")
report(f"最小奇异值：{s[-1]:.6e}")
report(f"W 的数值秩：{numerical_rank}")

indices = np.arange(1, len(s) + 1)

fig_a, axes_a = plt.subplots(1, 2, figsize=(11, 4))

axes_a[0].plot(indices, s, "bo-")
axes_a[0].set_xlabel("Singular value index")
axes_a[0].set_ylabel("Singular value")
axes_a[0].set_title("Singular Value Spectrum")
axes_a[0].set_xticks(indices)
axes_a[0].grid(True, alpha=0.3)

# 对数坐标无法显示零，只绘制严格大于零的值
positive = s > 0
axes_a[1].semilogy(indices[positive], s[positive], "ro-")
axes_a[1].set_xlabel("Singular value index")
axes_a[1].set_ylabel("Singular value (log scale)")
axes_a[1].set_title("Singular Value Spectrum (Log)")
axes_a[1].set_xticks(indices)
axes_a[1].grid(True, alpha=0.3)

fig_a.tight_layout()
fig_a.savefig(
    "part_a_singular_values.png",
    dpi=150,
    bbox_inches="tight"
)



# Part B：将 SVD 与 PSD 矩阵联系起来

report("\nPart B：SVD 与 PSD 矩阵的关系")

G = W @ W.T
eigvals = np.linalg.eigvalsh(G)
squared_s = s ** 2

# 根据矩阵尺度和浮点精度设置判零容差
scale = max(np.max(np.abs(eigvals)), np.max(squared_s))
tol = 10 * max(W.shape) * np.finfo(G.dtype).eps * scale

nonzero_eigvals = np.sort(eigvals[eigvals > tol])
nonzero_squared_s = np.sort(squared_s[squared_s > tol])

matched = (
    nonzero_eigvals.shape == nonzero_squared_s.shape
    and np.allclose(
        nonzero_eigvals,
        nonzero_squared_s,
        rtol=1e-5,
        atol=tol
    )
)

max_diff = np.max(np.abs(eigvals[::-1] - squared_s))

report(f"G 是否对称：{np.allclose(G, G.T)}")
report(f"G 的特征值：{eigvals}")
report(f"判零容差：{tol:.6e}")
report(f"G 的非零特征值：{nonzero_eigvals}")
report(f"非零奇异值的平方：{nonzero_squared_s}")
report(f"非零特征值与奇异值平方是否匹配：{matched}")
report(f"全部特征值与奇异值平方的最大差异：{max_diff:.6e}")

report(
    "解释：G = WWᵀ 满足 Gᵀ = G，因此它是对称矩阵。"
    "对任意实向量 x，都有 xᵀGx = (Wᵀx)ᵀ(Wᵀx)"
    " = ||Wᵀx||² ≥ 0，因此 G 是半正定矩阵。"
)



# Part C：截断 SVD 压缩

report("\nPart C：截断 SVD 压缩")

m, n = W.shape
original_params = m * n
frobenius_norm_W = np.linalg.norm(W, ord="fro")

ranks = [1, 2, 4, 6, 8, 10]
results = []

for k in ranks:
    Uk = U[:, :k]
    Sigma_k = np.diag(s[:k])
    Vtk = Vt[:k, :]

    # 构造秩不超过 k 的近似矩阵
    Wk = Uk @ Sigma_k @ Vtk

    rel_error = (
        np.linalg.norm(W - Wk, ord="fro")
        / frobenius_norm_W
    )

    logits = X_test @ Wk.T + b
    y_pred = clf.classes_[np.argmax(logits, axis=1)]
    acc = np.mean(y_pred == y_test)

    # 将 Sigma_k 吸收到 Uk 中，保存两个因子
    # 参数量不包含保持不变的偏置 b
    factor_params = k * (m + n)

    compression_ratio = original_params / factor_params
    reduction = 1 - factor_params / original_params

    results.append(
        (
            k,
            rel_error,
            acc,
            factor_params,
            compression_ratio,
            reduction
        )
    )

report(f"原始权重参数量：{original_params}")
report(f"基线准确率：{baseline_acc:.2%}")
report("k | 相对误差 | 测试准确率 | 分解后参数量 | 压缩倍数 | 参数减少比例")

for k, rel_error, acc, factor_params, ratio, reduction in results:
    report(
        f"{k:2d} | {rel_error:.6f} | {acc:.2%} | "
        f"{factor_params:4d} | {ratio:.2f}x | {reduction:.2%}"
    )

report(
    "说明：压缩倍数 = 原始权重参数量 / 分解后权重参数量。"
    "压缩倍数小于 1 或参数减少比例为负，表示参数量反而增加。"
)
report(
    "实际压缩需要保存 Uk @ Sigma_k 和 Vtk 两个因子；"
    "本代码重构完整 Wk 是为了方便评估。"
)

ks = [row[0] for row in results]
errors = [row[1] for row in results]
accuracies = [row[2] * 100 for row in results]
params = [row[3] for row in results]

fig_c, axes_c = plt.subplots(1, 3, figsize=(16, 4))

axes_c[0].plot(ks, errors, "bo-")
axes_c[0].set_xlabel("Rank k")
axes_c[0].set_ylabel("Relative Frobenius Error")
axes_c[0].set_title("Error vs Rank")

axes_c[1].plot(ks, accuracies, "go-")
axes_c[1].axhline(
    baseline_acc * 100,
    color="r",
    linestyle="--",
    label=f"Baseline: {baseline_acc:.2%}"
)
axes_c[1].set_xlabel("Rank k")
axes_c[1].set_ylabel("Test Accuracy (%)")
axes_c[1].set_title("Accuracy vs Rank")
axes_c[1].legend()

axes_c[2].plot(ks, params, "mo-")
axes_c[2].axhline(
    original_params,
    color="r",
    linestyle="--",
    label=f"Original: {original_params}"
)
axes_c[2].set_xlabel("Rank k")
axes_c[2].set_ylabel("Weight Parameter Count")
axes_c[2].set_title("Parameters vs Rank")
axes_c[2].legend()

for ax in axes_c:
    ax.set_xticks(ranks)
    ax.grid(True, alpha=0.3)

fig_c.tight_layout()
fig_c.savefig(
    "part_c_compression_results.png",
    dpi=150,
    bbox_inches="tight"
)



# Part D：Cholesky 回顾

report("\nPart D：Cholesky 分解")

eps = 1e-3
G_eps = G + eps * np.eye(G.shape[0])

L = np.linalg.cholesky(G_eps)

reconstruction_error = np.linalg.norm(
    L @ L.T - G_eps,
    ord="fro"
)

report(f"L 的形状：{L.shape}")
report(f"Cholesky 重构误差：{reconstruction_error:.6e}")
report(f"LLᵀ 是否约等于 G_eps：{np.allclose(L @ L.T, G_eps)}")

report(
    "解释：加入 εI（ε > 0）会使 WWᵀ 的所有特征值增加 ε，"
    "从而将半正定矩阵变成正定矩阵，满足 Cholesky 分解的条件。"
)



# Part E：结果解释

report("\nPart E：结果解释")

# 根据本次实验结果，选择偏重参数节省的折中方案 k=6
result_6 = next(row for row in results if row[0] == 6)
result_8 = next(row for row in results if row[0] == 8)

acc_6 = result_6[2]
params_6 = result_6[3]
reduction_6 = result_6[5] * 100
drop_6 = (baseline_acc - acc_6) * 100

acc_8 = result_8[2]
reduction_8 = result_8[5] * 100
drop_8 = (baseline_acc - acc_8) * 100

explanation_e = (
    "我选择 k=6，作为压缩程度与准确率之间的折中。"
    f"此时权重参数量由{original_params}减少到{params_6}，"
    f"减少约{reduction_6:.2f}%；"
    f"测试准确率从{baseline_acc:.2%}下降到{acc_6:.2%}，"
    f"下降{drop_6:.2f}个百分点。"
    f"如果更看重准确率，可以选择 k=8，其准确率为{acc_8:.2%}，"
    f"仅下降{drop_8:.2f}个百分点，但参数量只减少{reduction_8:.2f}%。"
    "截断较小的奇异值会丢弃对应奇异方向上的权重信息，"
    "这些部分对矩阵整体的贡献较小，但仍可能包含有用的分类信息，"
    "因此不能直接将其视为噪声。"
    "本实验与大型神经网络及LLM的SVD压缩原理相似："
    "用两个较小的矩阵近似原权重，在存储开销与模型性能之间权衡。"
    "真实LLM各层的矩阵尺寸、奇异值分布、输入特征及误差敏感度不同，"
    "而且误差可能逐层传播，因此需要结合各层特点和整体压缩预算分配秩，"
    "而不是所有层统一使用相同的 k。"
)

report(explanation_e)



with open(
    "svd_compression_summary.txt",
    "w",
    encoding="utf-8"
) as f:
    f.write("\n".join(output_lines))

print("\n已保存：")
print("part_a_singular_values.png")
print("part_c_compression_results.png")
print("svd_compression_summary.txt")


plt.show()