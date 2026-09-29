import numpy as np


np.set_printoptions(precision=4, suppress=True)

X = np.array([
    [1.0, 0.2, 0.0, 0.4],
    [0.8, 0.1, 0.3, 0.5],
    [-0.2, 0.9, 0.4, 0.1],
    [0.0, 0.7, 0.8, -0.1],
    [0.3, 0.2, -0.6, 0.9],
])
a = np.array([1.0, 1.0, 0.0, 0.0])
b = np.array([0.0, 1.0, 1.0, 0.0])
a_hat = a / np.linalg.norm(a)
delta_W = np.outer(b, a_hat)

print("X shape:", X.shape)
print("delta_W shape:", delta_W.shape)


# Part A - 向量空间与内积
norm_a = np.linalg.norm(a)
norm_b = np.linalg.norm(b)
dot_ab = a @ b
alignments = X @ a_hat

print("\nPart A")
print("||a|| =", norm_a)
print("||b|| =", norm_b)
print("a^T b =", dot_ab)
print("alignments with a_hat =", alignments)

# 简答（Part A）：a^T b = 1，不等于 0，所以 a 与 b 不正交。
# a、b 都是非零向量，因此 span{a} 和 span{b} 的维数都为 1。
# a_hat^T x 很大且为正，表示 x 与 a_hat 方向高度同向；很大且为负，
# 表示 x 与 a_hat 方向高度反向。绝对值越大，沿适配器输入方向的分量越大。


# Part B - 正交投影
def project_onto_unit_vector(x, u):
    """返回 x 到单位向量 u 所张成直线上的正交投影。"""
    return (u @ x) * u


projections = []
residuals = []
orthogonality_checks = []

for x in X:
    p = project_onto_unit_vector(x, a_hat)
    r = x - p
    projections.append(p)
    residuals.append(r)
    orthogonality_checks.append(a_hat @ r)

projections = np.array(projections)
residuals = np.array(residuals)
orthogonality_checks = np.array(orthogonality_checks)

print("\nPart B")
print("projections:\n", projections)
print("a_hat^T residuals:", orthogonality_checks)
print("residuals orthogonal to a_hat:",
      np.allclose(orthogonality_checks, 0.0, atol=1e-10))

# 简答（Part B）：若残差 r 与 a 正交，则 a_hat^T r = 0，因而
# delta_W r = b(a_hat^T r) = 0，所以适配器会完全忽略该残差分量。


# Part C - 低秩适配器的行为
# X 的每一行是一个输入 x，因此右乘 delta_W.T 等价于逐行计算 delta_W @ x。
Y = X @ delta_W.T
Y_from_projection = projections @ delta_W.T
rank_delta = np.linalg.matrix_rank(delta_W)

print("\nPart C")
print("adapter outputs Y:\n", Y)
print("rank(delta_W) =", rank_delta)
print("max |DeltaW x - DeltaW proj(x)| =",
      np.max(np.abs(Y - Y_from_projection)))

b_hat = b / np.linalg.norm(b)
direction_alignments = []
for i, y in enumerate(Y):
    if np.linalg.norm(y) > 1e-12:
        direction_alignment = abs((y / np.linalg.norm(y)) @ b_hat)
        direction_alignments.append(direction_alignment)
        print(i, direction_alignment)

print("all nonzero outputs parallel to b:",
      np.allclose(direction_alignments, 1.0, atol=1e-10))
print("outputs unchanged after projection:",
      np.allclose(Y, Y_from_projection, atol=1e-10))

# delta_W 是两个非零向量的外积；它的所有列都是 b 的倍数，故列空间只有
# 一个独立方向，矩阵秩为 1。


# Part D - 适配器的特征分析
H = delta_W.T @ delta_W
vals, vecs = np.linalg.eigh(H)
idx = np.argsort(vals)[::-1]
vals = vals[idx]
vecs = vecs[:, idx]

tolerance = 1e-10
nonzero_count = np.count_nonzero(vals > tolerance)
leading_alignment = abs(vecs[:, 0] @ a_hat)

print("\nPart D")
print("H:\n", H)
print("eigenvalues:", vals)
print("number above tolerance:", nonzero_count)
print("|v1^T a_hat| =", leading_alignment)

# 最大特征值对应的特征向量与 a_hat 的绝对内积为 1，说明二者平行或反平行；
# 因此适配器只对输入方向 a_hat 敏感。其余三个正交方向的特征值均为 0。


# Part E - 最终解释（共 5 句）
# 1. 秩一更新只读取输入在 a_hat 方向上的分量，并把所有非零输出限制在
#    span{b} 这一条直线上。
# 2. 内积 a_hat^T x 给出输入与适配器输入方向的带符号对齐程度。
# 3. 正交投影 (a_hat^T x)a_hat 保留这一有效分量，而正交残差被更新完全忽略。
# 4. H 只有一个非零特征值，且其主特征向量与 a_hat 对齐，从而揭示了唯一
#    敏感方向和该方向上的作用强度，这比逐项观察矩阵更清楚。
# 5. 在大模型中，低秩更新只需训练和存储少量参数，因而可显著降低显存、
#    计算与存储成本，同时保留原模型权重。


# 可选拓展 - 秩 r = 2 的适配器
# A 把 4 维输入压缩成 2 维，B 再把 2 维结果映射回 4 维。
A = np.array([
    [1.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 1.0],
])
B = np.array([
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [0.0, 1.0],
])

delta_W_rank2 = B @ A
H_rank2 = delta_W_rank2.T @ delta_W_rank2
vals_rank2 = np.linalg.eigvalsh(H_rank2)[::-1]

rank_rank2 = np.linalg.matrix_rank(delta_W_rank2)
nonzero_eigenvalues_rank2 = np.count_nonzero(vals_rank2 > tolerance)
lora_parameter_count = A.size + B.size
full_parameter_count = delta_W_rank2.size

print("\nOptional extension: rank r = 2")
print("A shape:", A.shape)
print("B shape:", B.shape)
print("delta_W = B @ A:\n", delta_W_rank2)
print("rank(delta_W) =", rank_rank2)
print("eigenvalues of delta_W.T @ delta_W:", vals_rank2)
print("number of nonzero eigenvalues:", nonzero_eigenvalues_rank2)
print("trainable parameters in A and B:", lora_parameter_count)
print("parameters in a full 4x4 update:", full_parameter_count)

# 这里 rank(delta_W)=2，且 delta_W.T @ delta_W 恰好有 2 个非零特征值。
# A 有 2*4=8 个参数，B 有 4*2=8 个参数，合计 16 个，与完整 4*4
# 矩阵的 16 个参数相同。因此在这个很小的 4 维示例中，r=2 并不节省参数；
# LoRA 的参数优势主要出现在矩阵维度很大、而 r 远小于输入和输出维度时。