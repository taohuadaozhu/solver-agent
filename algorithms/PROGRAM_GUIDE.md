# python_multialg 程序详细介绍

本文档对应当前 `python_multialg` 版本，说明程序主入口、执行流程、关键算法、局部搜索、meme 超启发、archive 存储机制、变种群机制，并给出当前包内函数/方法的功能、接口和用法说明。

## 1. 总体定位

`python_multialg` 是从原 MultiAlg/C++ 代码迁移并扩展的 Python 连续优化算法库。核心设计保持原项目结构：

- `Population` 是基础种群类，负责种群矩阵、随机数、适应度统计、精英保留。
- `MultiMet` 是多算法求解器类，通过多重继承组合所有算法 mixin。
- `algorithms/*.py` 中每个文件对应一个算法族或机制。
- 算法函数原则上只生成新解，写入 `solver.newpop`；统一由外层流程调用 `Evaluation()`、`pop_update()`、`worst_and_best()`、`Elist()` 完成评价和更新。
- `archive.py` 放外部 archive 和候选存储机制。
- `surrogate.py` 放 PCA、RBF、Kriging/EI、二次近似和线性方程求解等代理模型通用算子。
- `popvary.py` 放变种群、选择、子空间、协同进化等可复用机制。

当前 `algorithm_operators()` 已注册 70 个全局/进化算法，`local_operators()` 注册 20 个局部搜索算子，`meme_hyper_operators()` 注册 9 个 meme 超启发策略。

## 2. 主程序和入口

### 2.1 `python_multialg/main.py`

`main.py` 是最薄的一层入口：

```python
from .multimethod import main

if __name__ == "__main__":
    main()
```

运行：

```bash
python3 -m python_multialg.main
```

它实际调用 `multimethod.main()`，默认运行 CSA 路径。

### 2.2 `multimethod.main()`

功能：

- 解析 CLI 参数。
- 调用 `run_csa()`。
- 输出最终 best、代数、耗时。

典型用法：

```bash
python3 -m python_multialg.main --nvar 1000 --popsize 20 --maxgen 100 --seed 1
```

### 2.3 `evaluate_methods.main()`

这是当前最常用的算法评测入口。

功能：

- 按问题函数、算法组、重复次数批量运行算法。
- 支持 `algorithms`、`local_search`、`meme` 三组。
- 使用每个问题的默认搜索域，也允许 `--lb/--ub` 覆盖。

典型用法：

```bash
python3 -m python_multialg.evaluate_methods \
  --groups algorithms local_search \
  --problems rosenbrock griewank ackley rastrigin schwefel \
  --nvar 100 --popsize 20 --generations 20 --local-generations 20
```

### 2.4 其他测试/加速入口

- `benchmark_operators.py`：带 timeout 的算法/局部搜索批量测试。
- `benchmark_archive_popvary.py`：测试 archive 和变种群机制组合。
- `abca_meme.py`：ABCA + meme 超启发局部搜索选择。
- `island_meme.py`：岛模型、多种群协同、meme 局部强化。
- `fast.py`：NumPy 版 CSA 快速路径。
- `fast_topology.py`：NumPy 向量化 pool 拓扑、岛模型和批量候选搜索。
- `generated_eo_accel.py`：纯 Python 的 ADE/新增局部算子 benchmark 入口。

## 3. 标准执行流程

标准求解循环如下：

```python
solver = MultiMet(popsize, nvar, lb, ub, evaluate)
solver.Initial()
for gen in range(maxgen):
    solver.DE(0.8, 1, 0.5, 0, solver.Popsize)  # 或任意算法
    solver.Evaluation(True, 0, solver.Popsize)
    solver.pop_update(0, solver.Popsize)
    solver.worst_and_best()
    solver.Elist()
```

流程含义：

1. `Initial()` 初始化 `pop/newpop/velocity/ibest/gbest`。
2. 算法函数生成新种群，写入 `newpop`。
3. `Evaluation(True, ...)` 评价 `newpop`，写入 `newpop_fit`。
4. `pop_update()` 用 `newpop` 覆盖 `pop`，并更新个体历史最优 `ibest`。
5. `worst_and_best()` 找当前种群最好/最差个体。
6. `Elist()` 做全局精英保留，维护 `gbest`。

局部搜索算子通常直接修改某个 `newpop[i]`，局部函数内部可能临时调用目标函数比较候选，但最终仍由外层统一评估。

## 4. 核心数据结构

### 4.1 `Population`

定义位置：`population.py`

`Population[T]` 是底层种群容器，不持有目标函数，只负责种群矩阵、适应度数组、全局最优、排序和随机数工具。Python 版本使用 `list[list[float]]` 保存矩阵，不需要 C++ 版本的手工释放逻辑。

主要字段：

- `Popsize`：种群规模。
- `Nvar`：变量维度。
- `Lbound/Ubound`：搜索上下界。
- `pop`：当前种群。
- `newpop`：新生成种群。
- `pop_fit/newpop_fit`：适应度。
- `gbest/gbest_fit`：全局最优。
- `cur_best/cur_worst`：当前种群最好/最差下标。
- `CRold/CRnew`：群体适应度集中度/收敛率指标。
- `stored/hold`：`gauss()` 的 Box-Muller 正态随机缓存。
- `seed/aktseed/aktrand/rgrand`：兼容 C++ CMA-ES 随机流的内部随机数状态。

主要方法：

| 方法 | 输入 | 输出/副作用 |
|---|---|---|
| `__init__(psize, nn, lb, ub)` | 种群规模、维度、边界 | 创建 `pop/newpop/pop_fit/newpop_fit/gbest` 和随机状态 |
| `CreateMatrix(nRow, nCol)` | 行列数 | 返回零矩阵 |
| `randval(low, high)` | 下界、上界 | 返回千分粒度均匀随机数 |
| `heap_sort(num, length, cbit)` | 矩阵、长度、排序列 | 按指定列原地排序前 `length` 行 |
| `swap(x, y)` | 两个值 | 返回 `(y, x)` |
| `worst_and_best()` | 无 | 更新 `cur_best/cur_worst` |
| `Elist()` | 无 | 更新 `gbest` 或把 `gbest` 回填到当前最差个体 |
| `average_fit()` | 无 | 返回当前种群平均适应度 |
| `CRfit()` | 无 | 更新 `CRold/CRnew` |
| `square(d)` | 数值 | 返回平方 |
| `uniform()` | 无 | 返回内部 LCG 均匀随机数 |
| `gauss()` | 无 | 返回标准正态随机数 |
| `myhypot(a, b)` | 两个数 | 返回 `math.hypot(a, b)` |

### 4.2 `MultiMet`

定义位置：`multimethod.py`

`MultiMet` 是统一求解器，通过多重继承组合所有算法 mixin：

```python
class MultiMet(
    ADEMixin,
    AESSPSOMixin,
    AdamMixin,
    ...
    WOAMixin,
    LocalSearchMixin,
    Population[float],
):
    ...
```

构造接口：

```python
MultiMet(
    psize: int = 10,
    nn: int = 1000,
    lb: float = 0.0,
    ub: float = 10.0,
    evaluate: FitnessFunc = Sphere,
)
```

其中 `FitnessFunc = Callable[[list[float], int], float]`，目标函数返回越小越好的适应度。

在 `Population` 基础上增加：

- `velocity`：PSO/CSO/SAMSO 等速度。
- `ibest/ibest_fit`：个体历史最优。
- `AC1/AC2/AW`：APSO 自适应参数。
- `OA/OArow`：OLPSO 正交学习矩阵。
- `ant_tao`：ACO 信息素/候选记忆。
- `trial/pr/neigh/ngh`：ABCA/POA/局部搜索等辅助状态。
- `ade_*`：ADE/SHADE 类自适应差分进化状态。
- `meme_*`：meme 超启发策略奖励和 Q 表。
- `sade_amss_archive/sade_amss_archive_fit`：SADE-AMSS 外部存档。
- `PB_center/PB_sigma`：PBILc 概率模型中心和尺度。
- `BAT_r/BAT_A/BAT_v`：BATA 脉冲率、响度、速度。
- `I/Index`：FA 亮度和排序下标。
- `Ind_meme/SubDecBase`：meme 编号和子问题分解缓存。

主要方法：

| 方法 | 输入 | 输出/副作用 |
|---|---|---|
| `Initial()` | 无 | 初始化 `pop/newpop/velocity/ibest/gbest` 和所有算法状态 |
| `Evaluation(s, p_start, p_end)` | `s=True` 评 `newpop`，否则评 `pop` | 写入 `newpop_fit` 或 `pop_fit` |
| `pop_update(p_start, p_end)` | 个体区间 | 无条件接受 `newpop`，并更新 `ibest` |
| `pop_better_update(p_start, p_end)` | 个体区间 | 只接受更优 `newpop`，并更新 `ibest` |
| `CreateOA()` | 无 | 创建或懒初始化 OLPSO 正交表 |
| `OAValue(row, col)` | 正交表行列 | 返回 `0/1` 编码 |
| `randnorm(miu, score)` | 均值、标准差 | 返回正态扰动 |
| `_clip(value)` | 数值 | 截断到 `[Lbound, Ubound]` |
| `_wrap(value)` | 数值 | 越界后环绕到另一侧边界 |
| `_fitness_row(row)` | 解向量 | 返回 `row[:Nvar] + [fitness]` |

### 4.3 mixin 模块和公开算法方法

`algorithms/__init__.py` 统一导出 mixin，`MultiMet` 继承后直接暴露同名算法方法。

| Mixin | 文件 | 公开算法/主要方法 |
|---|---|---|
| `GAMixin` | `algorithms/ga.py` | `GA`、`NGA`、`LGA`、`VAGA`、`EAGA`、`select`、`crossover`、`mutate` |
| `PSOMixin` | `algorithms/pso.py` | `PSO`、`CPSO`、`SPSO`、`CMPSO`、`APSO_1~5`、`OLPSO`、`Subgradient`、`SparseSubgradient` |
| `GPSOMixin` / `AESSPSOMixin` | `gpso.py`、`aesspso.py` | `GPSO`、`AESSPSO` |
| `DEMixin` | `algorithms/de.py` | `DE`、`DE_ARCHIVE_SURROGATE`、`differential_mutate`、`differential_crossover` |
| `ADEMixin` / `SHADEMixin` / `IMODEMixin` | `ade.py`、`shade.py`、`imode.py` | `ADE`、`SHADE`、`IMODE` 和策略/记忆更新子函数 |
| `SAMixin` / `SADE*Mixin` | `sa.py`、`sade_*.py` | `SA`、`SADE_AMSS`、`SADE_AMSS_ORIG`、`SADE_ATDSC`、`SADE_Sammon` |
| `CMAESMixin` | `algorithms/cmaes.py` | `CMAES`、`CMAES_parametersetting`、`CMAES_initial`、`CMAES_updateDistribution` |
| `HSMixin` / `ACOMixin` / `COAMixin` / `CSAMixin` | `hs.py`、`aco.py`、`coa.py`、`csa.py` | `HS`、`ACO`、`COA`、`CSA` 和对应子函数 |
| `ABCAMixin` / `POAMixin` / `ILSMixin` / `VNSMixin` / `GRASPMixin` | 对应算法文件 | `ABCA`、`POA`、`ILS`、`VNS`、`GRASP` |
| `PBILCMixin` / `BATAMixin` / `FAMixin` / `BAMixin` | 对应算法文件 | `PBILC`、`BATA`、`FA`、`BA` |
| `AdamMixin` / `BFGSMixin` / `FRCGMixin` / `RMSPropMixin` / `SDMixin` / `SQPMixin` | 对应算法文件 | `Adam`、`BFGS`、`FRCG`、`RMSProp`、`SD`、`SQP` |
| PlatEMO 风格 mixins | `autov.py`、`cso.py`、`doa.py` 等 | `AutoV`、`CSO`、`DOA`、`ECPO`、`EGO`、`FEP`、`FROFI`、`GWO`、`KMA`、`L2SMEA`、`MFEA`、`MFEA_II`、`MGO`、`MVPA`、`MiSACO`、`Nelder_Mead`、`OFA`、`SACC_EAM_II`、`SACOSO`、`SAMSO`、`SAPO`、`SSIO_RL`、`WOA` |
| `LocalSearchMixin` | `algorithms/local_search.py` | 20 个 `newpop_*` 局部搜索算子和 `meme_*` 超启发策略 |

### 4.4 公共辅助模块

| 模块 | 主要函数/类 | 输入输出 | 用途 |
|---|---|---|---|
| `algorithms/common.py` | `squared_distance(left, right, nvar)` | 两个向量和维度，返回平方距离 | archive、popvary 多样性计算 |
| `algorithms/archive.py` | `build_archive`、`best_archive`、`diverse_archive`、`truncation_archive`、`radius_density_archive`、`dual_archive`、`archive_sample` | 候选、适应度、容量、策略，返回 archive 和适应度 | SHADE、IMODE、SADE、MiSACO、ECPO 等 |
| `algorithms/popvary.py` | `select_population`、`fill_population`、`linear_population_size`、`random_subspace_groups`、`elite_indices`、`adaptive_rmp` | 候选、适应度、规模、维度等，返回选择结果或参数 | 变种群、子空间、协同进化 |
| `algorithms/surrogate.py` | `KrigingModel`、`QuadModel`、`kriging_fit/predict`、`quad_fit/predict`、`pca_fit/transform/inverse`、`expected_improvement` | 样本和值，返回模型、预测或填充准则 | EGO、DE archive surrogate、代理辅助算法 |

## 5. 关键进化算法原理

### 5.1 GA/NGA/LGA/VAGA/EAGA

基于选择、交叉、变异生成新种群。`NGA` 加入 niche 排挤，`LGA` 加入局部步长搜索，`VAGA/EAGA` 是变异/精英风格 GA 变体。

### 5.2 PSO 系列

- `PSO`：速度由惯性项、个体最优、全局最优共同驱动。
- `CPSO`：加入收缩系数。
- `SPSO`：PSO 后叠加稀疏子梯度修正，当前版本为轻量化实现。
- `APSO_1~5`：按代数、群体状态或梯度信息自适应调整 PSO 参数。
- `OLPSO`：正交学习选择个人最优和全局最优的组合方向，高维时使用轻量正交采样。
- `GPSO/AESSPSO`：PlatEMO 风格 PSO 变体。

### 5.3 DE/ADE/SHADE/IMODE/SADE

- `DE`：差分变异 + 交叉。
- `ADE`：多策略参数串 `seri`、成功历史、自适应活动种群。
- `SHADE`：current-to-pbest/1、历史记忆 `F/CR`、外部 archive。
- `IMODE`：基于 archive 的多策略差分进化轻量版。
- `SADE-ATDSC`、`SADE-Sammon`：自适应 DE 和降维/子空间风格变体。

### 5.4 CMAES

维护均值、协方差、步长和特征系统，通过多元正态采样生成候选，再根据优秀样本更新分布。当前保留 CMA-ES 主要内部函数。

### 5.5 ABC/ACO/HS/CSA/FA/BA/BATA/POA/COA

这些是群智能/启发式算法：

- `ABCA`：雇佣蜂、观察蜂、侦察蜂。
- `ACO`：路径搜索和信息素更新。
- `HS`：和声记忆、音调微调。
- `CSA`：Levy 飞行和巢发现。
- `FA`：萤火虫吸引和随机扰动。
- `BA/BATA`：蜜蜂/蝙蝠风格邻域搜索。
- `POA/COA`：植物生长和混沌优化。

### 5.6 PlatEMO 新增连续算法

当前新增：

`EGO, FEP, FRCG, FROFI, GPSO, GWO, IMODE, KMA, L2SMEA, MFEA-II, MFEA, MGO, MVPA, MiSACO, Nelder-Mead, OFA, RMSProp, SACC-EAM-II, SACOSO, SA, SADE-AMSS, SADE-ATDSC, SADE-Sammon, SAMSO, SAPO, SD, SHADE, SQP, SSIO-RL, WOA`

实现原则：

- 每个算法有独立 `.py` 文件。
- 只负责生成新解。
- archive 和变种群辅助机制调用 `archive.py/popvary.py`。
- 大多数实现是连续优化轻量版，不是逐行复刻 PlatEMO MATLAB 源码。

## 6. 关键局部搜索算法

`local_search.py` 当前提供 20 个局部搜索算子：

- `bit_climbing`：随机维度爬山。
- `simplex`：轻量 Nelder-Mead 单纯形。
- `box_complex`：Box complex 约束搜索。
- `powell`：方向集线搜索。
- `newton`：稀疏有限差分牛顿/拟牛顿。
- `mcts`：轻量蒙特卡罗树式 rollout。
- `simulated_annealing`：模拟退火。
- `tabu_search`：禁忌搜索。
- `pattern_search`：模式搜索。
- `threshold_accepting`：阈值接受。
- `elite_contraction`：向精英收缩。
- `coordinate_descent`：坐标下降。
- `mcts_local_search`：局部版 MCTS。
- `elite_line_search`：沿精英方向线搜索。
- `multi_scale_gaussian`：多尺度高斯扰动。
- `random_subspace_pattern`：随机子空间模式搜索。
- `newton_subspace`：子空间牛顿。
- `gradient_backtracking`：梯度回溯。
- `cauchy_basin_hop`：Cauchy 盆地跳跃。
- `opposition_elite_blend`：反向学习和精英混合。

使用方式：

```python
for i in range(solver.Popsize):
    solver.newpop_pattern_search(i, 5, 5, 0.1)
```

或通过 `evaluate_methods.local_operators()` 统一调用。

## 7. meme 超启发算法

`meme_hyper_operators()` 只包含超启发策略，不再混入 `local_operators()`。

当前策略：

- `meme_random_walk`：随机局部算子游走。
- `meme_simple_random`：均匀随机选择局部算子。
- `meme_randperm`：随机排列算子应用。
- `meme_inheritance`：继承上一轮表现较好的局部算子选择。
- `meme_subprob_decomposition`：按子问题分解选择局部算子。
- `meme_biasd_roulette`：基于奖励的偏置轮盘赌。
- `meme_q_learning`：Q-learning 选择局部算子。
- `meme_biasd_roulette_generated_eo`：包含 Generated_EO 新算子的轮盘赌。
- `meme_q_learning_generated_eo`：包含 Generated_EO 新算子的 Q-learning。

这些策略本质上不直接定义搜索方向，而是在 `LocalSearchMixin.apply_meme_action()` 上选择局部搜索动作。

## 8. archive 存储机制

文件：[algorithms/archive.py](algorithms/archive.py)

核心策略：

- `best`：保存适应度最好的个体。
- `worst`：保存最差个体，主要用于扰动/对照。
- `random`：随机保存，保留多样性。
- `diverse`：优先最大化与已选样本距离。
- `mixed`：一部分 best，一部分 diverse。

SHADE/IMODE/MiSACO/ECPO 使用 archive 的方式：

- `update_external_archive()`：合并旧 archive 和候选，再裁剪到容量。
- `archive_sample()`：从 archive 抽样作为差分项来源。
- `archive_value()`：按维度抽取 archive 值，用于 ECPO 的维度注入。

注意：当前 SHADE archive 是轻量通用版本，不是严格“只保存被成功替换父代”的原论文完整实现。

## 9. 变种群机制

文件：[algorithms/popvary.py](algorithms/popvary.py)

核心机制：

- `select_population()`：统一候选选择。
  - `best`：选择最优。
  - `tournament`：锦标赛选择。
  - `roulette`：适应度轮盘赌。
  - `rank_diverse`：排名 + 多样性。
- `fill_population()`：候选不足时补齐。
  - `random`：随机生成。
  - `cycle`：循环复制已有候选。
  - `mutate_best`：围绕最优候选变异。
- `linear_population_size()`：线性缩减种群规模。
- `random_subspace_groups()`：随机维度分组。
- `best_worst_pairs()`：竞争学习用 winner/loser 配对。
- `elite_indices()`：返回精英下标。
- `adaptive_rmp()`：多任务进化中的随机交配概率自适应。

## 10. 函数接口和用法参考

约定：

- 算法 mixin 的 public 方法大多输入 `p_start/p_end`，表示作用的种群区间。
- 算法方法输出通常为副作用：写入 `self.newpop`，部分方法同时写入 `self.newpop_fit` 或维护内部状态。
- 辅助函数返回列表、下标、archive、统计值或评测结果。
- CLI `main()` 函数无显式返回，输出到 stdout 或写 CSV。

### 10.1 核心类和主流程

| 函数/方法 | 功能 | 输入 | 输出/副作用 | 用法 |
|---|---|---|---|---|
| `Population.__init__(psize, nn, lb, ub)` | 初始化基础种群状态 | 种群规模、维度、边界 | 创建 `pop/newpop/fit/gbest` | `Population(20, 100, -5, 10)` |
| `Population.CreateMatrix(nRow, nCol)` | 创建零矩阵 | 行列数 | `list[list[float]]` | 内部初始化使用 |
| `Population.randval(low, high)` | 千分粒度均匀随机数 | 下界、上界 | `float` | 算法采样 |
| `Population.heap_sort(num, length, cbit)` | 按指定列排序矩阵前段 | 矩阵、长度、列 | 原地排序 | CMAES/排序辅助 |
| `Population.swap(x, y)` | 交换两个值 | 任意两个值 | `(y, x)` | 兼容原 C++ 风格 |
| `Population.worst_and_best()` | 找当前最好/最差个体 | 无 | 更新 `cur_best/cur_worst` | 每代评估后调用 |
| `Population.Elist()` | 全局精英保留 | 无 | 更新/注入 `gbest` | 每代末调用 |
| `Population.average_fit()` | 平均适应度 | 无 | `float` | 收敛统计 |
| `Population.CRfit()` | 计算收敛率指标 | 无 | 更新 `CRnew` | 初始化/统计 |
| `Population.square(d)` | 平方 | 标量 | 标量 | 数学辅助 |
| `Population.uniform()` | 原 C++ LCG 随机数 | 无 | `[0,1)` 浮点 | 兼容随机流 |
| `Population.gauss()` | Box-Muller 高斯随机数 | 无 | `float` | 兼容随机流 |
| `Population.myhypot(a, b)` | 欧氏范数辅助 | 两个数 | `math.hypot` | 数学辅助 |
| `MultiMet.__init__(psize, nn, lb, ub, evaluate)` | 创建多算法求解器 | 种群、维度、边界、目标函数 | 初始化算法状态 | `MultiMet(20,100,-5,10,Ackley)` |
| `MultiMet.randnorm(miu, score)` | 高斯随机数 | 均值、尺度 | `float` | 算法扰动 |
| `MultiMet.pop_update(p_start, p_end)` | 接受 `newpop` | 区间 | 更新 `pop/ibest` | 每代评估后 |
| `MultiMet.pop_better_update(p_start, p_end)` | 只接受更优解 | 区间 | 条件更新 `pop/ibest` | 贪婪选择 |
| `MultiMet.CreateOA()` | 创建正交表 | 无 | 更新 `OA` | 初始化 OLPSO |
| `MultiMet.OAValue(row, col)` | 读取正交表值 | 行列 | `0/1` | OLPSO |
| `MultiMet.Initial()` | 初始化完整求解状态 | 无 | 设置 `pop/fit/ibest/gbest` | 求解前调用 |
| `MultiMet.Evaluation(s, p_start, p_end)` | 评价种群 | `s=True` 评 `newpop`，否则评 `pop` | 写 fit 数组 | 每代调用 |
| `MultiMet._clip(value)` | 截断到边界 | 标量 | 边界内标量 | 算法内部 |
| `MultiMet._wrap(value)` | 越界环绕 | 标量 | 环绕后标量 | PSO 等 |
| `MultiMet._fitness_row(row)` | 返回带适应度行 | 决策向量 | `x + [fit]` | 候选比较 |
| `run_csa(...)` | 默认 CSA 求解 Sphere | 维度、种群、代数等 | `RunResult` | `run_csa(seed=1)` |
| `multimethod.main()` | CLI 主入口 | 命令行参数 | stdout | `python3 -m python_multialg.main` |

### 10.2 目标函数

所有目标函数接口近似：

```python
Function(var: Sequence[float], num: Optional[int] = None) -> float
```

`var` 是决策向量，`num` 指定前多少维参与计算，默认使用全部。

| 函数 | 功能 | 用法 |
|---|---|---|
| `randnorm(miu, score)` | 生成正态扰动 | `randnorm(0, 1)` |
| `Sphere` | 球函数，单峰可分 | `Sphere(x, n)` |
| `Rosenbrock` | Rosenbrock 谷函数，强变量耦合 | `Rosenbrock(x, n)` |
| `Dixon_Price` | Dixon-Price 函数 | `Dixon_Price(x, n)` |
| `Griewank` | 多峰弱耦合函数 | `Griewank(x, n)` |
| `Ackley` | 多峰平台函数 | `Ackley(x, n)` |
| `Michalewicz` | 多峰非线性函数，最小值为负 | `Michalewicz(x, n)` |
| `Rastrigin` | 高频多峰可分函数 | `Rastrigin(x, n)` |
| `Schwefel` | 多峰大范围函数 | `Schwefel(x, n)` |
| `Sum_Squares` | 带维度权重的平方和 | `Sum_Squares(x, n)` |
| `Trid` | 含邻接耦合的函数 | `Trid(x, n)` |
| `Zakharov` | 含线性组合高次项函数 | `Zakharov(x, n)` |
| `Weierstrass` | 分形多峰函数 | `Weierstrass(x, n)` |
| `High_Conditioned_Elliptic` | 高条件数椭球函数 | `High_Conditioned_Elliptic(x, n)` |
| `Schwefel_P222` | Schwefel 2.22 | `Schwefel_P222(x, n)` |
| `Schwefel_P221` | Schwefel 2.21 | `Schwefel_P221(x, n)` |
| `Tablet` | Tablet 病态函数 | `Tablet(x, n)` |
| `Salomon` | Salomon 多峰函数 | `Salomon(x, n)` |

### 10.3 全局/进化算法函数

下面 public 算法函数统一为 `MultiMet` 方法。输入一般为算法参数和 `[p_start, p_end)` 区间；输出一般为写入 `self.newpop`，部分算法同步维护内部状态或 `newpop_fit`。

| 算法函数 | 功能 | 关键输入 | 输出/副作用 | 用法 |
|---|---|---|---|---|
| `GA(pc, pm, p_start, p_end)` | 标准遗传算法 | 交叉率、变异率、区间 | 新种群 | `s.GA(0.8,0.15,0,s.Popsize)` |
| `NGA(pc, pm, mdis, p_start, p_end)` | niche GA | niche 距离 | 新种群 | 批量测试 |
| `LGA(pc, pm, step, nst, p_start, p_end)` | 局部增强 GA | 局部步长/步数 | 新种群 | 批量测试 |
| `VAGA(p_start, p_end)` | 变异自适应 GA | 区间 | 新种群 | 批量测试 |
| `EAGA(p_start, p_end)` | 精英自适应 GA | 区间 | 新种群 | 批量测试 |
| `PSO(w, c1, c2, max_ve, p_start, p_end)` | 标准粒子群 | 惯性、学习因子、最大速度 | 更新速度和新种群 | `s.PSO(...)` |
| `CPSO(...)` | 收缩粒子群 | 同 PSO | 更新速度和新种群 | 批量测试 |
| `SPSO(..., Gen, MaxGen, ...)` | 子梯度轻量 PSO | 代数、PSO 参数 | 稀疏梯度修正新解 | 高维可用 |
| `CMPSO(...)` | Cauchy/混合 PSO 入口 | 代数、PSO 参数 | 新种群 | 批量测试 |
| `APSO_1~APSO_5(...)` | 自适应 PSO 系列 | 代数/速度参数 | 新种群 | `algorithm_operators()` |
| `OLPSO(w, c, max_ve, p_start, p_end)` | 正交学习 PSO | 正交学习参数 | 新种群 | 高维自动轻量采样 |
| `GPSO(...)` | PlatEMO GPSO 风格封装 | PSO 参数 | 新种群 | 新增算法测试 |
| `AESSPSO(Beta, Gamma, p_start, p_end)` | 自适应指数平滑 PSO | Beta/Gamma | 新种群 | 新增算法测试 |
| `DE(F, S, cr, p_start, p_end)` | 差分进化 | 差分权重、策略、交叉率 | 新种群 | `s.DE(0.8,1,0.5,0,N)` |
| `DE_ARCHIVE_SURROGATE(...)` | archive + 代理预筛 DE | archive 策略、代理模型 | radius-density/truncation archive + Quad/Kriging 子空间预筛 | PlatEMO 机制组合 |
| `ADE(seri_pool, p_start, p_end)` | 自适应 DE | 策略池 | 新种群和 ADE 状态 | ADE 测试 |
| `SHADE(p_start, p_end)` | 成功历史自适应 DE | 区间 | archive + 新种群 | `s.SHADE(0,N)` |
| `IMODE(Gen, MaxGen, p_start, p_end)` | 多策略 archive DE | 代数 | archive + 新种群 | PlatEMO 新增 |
| `SA(Gen, MaxGen, p_start, p_end)` | 模拟退火 | 代数/温度衰减 | 新种群和候选评价 | PlatEMO 新增 |
| `SADE_AMSS(Gen, MaxGen, p_start, p_end, K, maxd, Gm)` | 代理辅助多子空间 DE | 子空间数、最大维数、子空间迭代数 | RBF 代理筛选后生成新种群 | PlatEMO 新增 |
| `SADE_AMSS_ORIG(Gen, MaxGen, p_start, p_end, K, maxd, Gm, archive_size)` | PlatEMO 原版依赖风格 SADE-AMSS | 子空间参数、archive 容量 | 持久 archive + PCA 空间 + RBF 代理筛选 | 原版依赖算子 |
| `SADE_ATDSC(...)` | 自适应 DE 变体 | 代数 | 新种群 | PlatEMO 新增 |
| `SADE_Sammon(...)` | 子空间 DE 变体 | 采样维度 | 新种群 | PlatEMO 新增 |
| `CMAES(firstrun, p_start, p_end)` | CMA-ES | 是否首轮、区间 | 采样新种群和分布状态 | `s.CMAES(g,0,N)` |
| `Adam(...)` | Adam 梯度优化 | 学习率、beta、梯度维数 | 新种群和 Adam 状态 | 高维稀疏梯度 |
| `RMSProp(...)` | RMSProp 梯度优化 | 学习率、rho、梯度维数 | 新种群和 cache | PlatEMO 新增 |
| `BFGS(...)` | 拟牛顿 BFGS | 回溯参数、梯度维数 | 新种群和 Hessian 对角近似 | 局部强化 |
| `FRCG(...)` | Fletcher-Reeves/Polak-Ribiere 风格共轭梯度 | 梯度维数 | 新种群和方向 | PlatEMO 新增 |
| `SD(...)` | Steepest Descent | 梯度维数 | 新种群 | PlatEMO 新增 |
| `SQP(...)` | 轻量 SQP/信赖域梯度步 | 梯度维数 | 新种群 | PlatEMO 新增 |
| `ABCA(limit, p_start, p_end)` | 人工蜂群 | 侦察限制 | 新种群 | 可接 meme |
| `ACO(epsl, p_start, p_end)` | 蚁群连续搜索 | 信息素参数 | 新种群和 `ant_tao` | 批量测试 |
| `HS(srate, trate, bw, p_start, p_end)` | 和声搜索 | 记忆率、微调率、带宽 | 新种群 | 批量测试 |
| `CSA(pa, p_start, p_end)` | 布谷鸟搜索 | 发现概率 | 新种群 | 默认主程序 |
| `COA(chaos_n, p_start, p_end)` | 混沌优化 | 混沌次数 | 新种群 | 批量测试 |
| `FA(gama, alpha0, betamin, MGEN, p_start, p_end)` | 萤火虫算法 | 吸引/随机参数 | 新种群 | 批量测试 |
| `BA(p_start, p_end)` | 蜜蜂算法 | 区间 | 新种群 | 批量测试 |
| `BATA(gen, p_start, p_end)` | 蝙蝠算法 | 当前代 | 新种群 | 批量测试 |
| `POA(p_start, p_end)` | 植物生长算法 | 区间 | 新种群 | 批量测试 |
| `ILS(nst, p_start, p_end)` | 迭代局部搜索 | 局部步数 | 新种群 | 全局算法入口 |
| `VNS(nst, p_start, p_end)` | 变邻域搜索 | 邻域步数 | 新种群 | 全局算法入口 |
| `GRASP(alfa, nst, p_start, p_end)` | 贪婪随机自适应搜索 | 贪婪系数、步数 | 新种群 | 全局算法入口 |
| `PBILC(p_start, p_end, learn_rate)` | 概率模型增量学习 | 学习率 | 新种群 | 全局算法入口 |
| `CSO(phi, p_start, p_end)` | 竞争群优化 | 均值学习因子 | 新种群和速度 | PlatEMO 已接入 |
| `DOA(Gen, MaxGen, p_start, p_end)` | 动态优化算法 | 代数 | 新种群 | PlatEMO 已接入 |
| `ECPO(...)` | ECP 优化算法 | 策略、archive/popvary 策略 | archive/变种群选择后生成新种群 | 支持策略实验 |
| `AutoV(weight, p_start, p_end)` | 自动变量/算子选择 | 权重矩阵 | 新种群 | PlatEMO 已接入 |
| `EGO(p_start, p_end, infill)` | EGO 轻量代理采样 | infill 数 | 围绕精英生成候选 | PlatEMO 新增 |
| `FEP(p_start, p_end)` | 快速进化规划 | 自适应 sigma | Cauchy/Gaussian 变异 | PlatEMO 新增 |
| `FROFI(p_start, p_end)` | 鲁棒精英/中位方向搜索 | 区间 | 新种群 | PlatEMO 新增 |
| `GWO(Gen, MaxGen, p_start, p_end)` | 灰狼优化 | 代数 | alpha/beta/delta 引导新种群 | PlatEMO 新增 |
| `KMA(p_start, p_end, clusters)` | 聚类均值吸引搜索 | 簇数 | 新种群 | PlatEMO 新增 |
| `L2SMEA(p_start, p_end, group_size)` | 大规模子空间进化 | 子空间大小 | 局部维度更新 | PlatEMO 新增 |
| `MFEA(p_start, p_end, tasks)` | 多因子进化 | 任务数/RMP | 跨任务交配生成新种群 | PlatEMO 新增 |
| `MFEA_II(p_start, p_end, tasks)` | 自适应 MFEA-II | 任务数/RMP | 新种群 | PlatEMO 新增 |
| `MGO(Gen, MaxGen, p_start, p_end)` | 山地瞪羚优化轻量版 | 代数 | 领袖和角度扰动 | PlatEMO 新增 |
| `MVPA(p_start, p_end)` | 多队伍竞争优化 | 区间 | winner-loser 学习 | PlatEMO 新增 |
| `MiSACO(p_start, p_end)` | 多种群 ACO 风格连续采样 | archive | 新种群 | PlatEMO 新增 |
| `Nelder_Mead(p_start, p_end, stepn)` | Nelder-Mead 封装 | 步数 | 调用 simplex 局部搜索 | PlatEMO 新增 |
| `OFA(Gen, MaxGen, p_start, p_end)` | 光学/焦点吸引式搜索 | 代数 | 新种群 | PlatEMO 新增 |
| `SACC_EAM_II(p_start, p_end, group_size)` | 协同协作子空间进化 | 子空间大小 | 新种群 | PlatEMO 新增 |
| `SACOSO(p_start, p_end)` | 协同竞争群优化 | 区间 | winner-loser 新种群 | PlatEMO 新增 |
| `SAMSO(Gen, MaxGen, p_start, p_end)` | 自适应多策略群优化 | 代数 | 新种群和速度 | PlatEMO 新增 |
| `SAPO(Gen, MaxGen, p_start, p_end)` | 自适应群体优化 | 代数 | 新种群 | PlatEMO 新增 |
| `SSIO_RL(p_start, p_end)` | 强化学习选择搜索动作 | 内部 Q 值 | 新种群和 Q 值 | PlatEMO 新增 |
| `WOA(Gen, MaxGen, p_start, p_end)` | 鲸鱼优化 | 代数 | 包围/螺旋更新 | PlatEMO 新增 |

### 10.4 算法内部辅助函数

这些函数一般不直接从外部调用，主要服务对应算法。

| 模块 | 函数 | 功能 | 输入输出 |
|---|---|---|---|
| `abca.py` | `EmployedBee` | 雇佣蜂阶段生成邻域候选 | 输入个体编号和临时数组，写候选 |
| `abca.py` | `OnlookerBee` | 观察蜂概率选择并搜索 | 输入区间，写 `newpop` |
| `abca.py` | `ScoutBee` | 停滞后随机重启 | 输入 limit 和区间，写 `newpop` |
| `aco.py` | `path_finding` | 按信息素产生路径 | 输入 epsl/区间，写 `newpop` |
| `aco.py` | `phe_updating` | 更新信息素表 | 输入区间，更新 `ant_tao` |
| `adam.py` | `_ensure_adam_state` | 初始化 Adam 一阶/二阶矩 | 无输入，维护状态 |
| `adam.py` | `_finite_gradient` | 有限差分梯度 | 输入向量/维度/步长，返回梯度 |
| `ade.py` | `seri_diff_left_index/right_index/weight_index` | 解析 ADE 策略串下标 | 输入 term，返回下标 |
| `ade.py` | `search_state` | 判断搜索阶段 | 输入进度/停滞代数，返回状态 |
| `ade.py` | `default_seri_policy` | 初始化策略概率 | 返回策略表 |
| `ade.py` | `choose_policy_action` | epsilon/权重采样策略 | 返回动作下标 |
| `ade.py` | `sample_left_source/sample_right_source` | ADE 差分源采样 | 返回源编码 |
| `ade.py` | `normalize_seri/sample_seri/reset_seri_pool` | 生成和修正 ADE 策略串 | 返回/更新策略池 |
| `ade.py` | `_ade_source/_ade_base` | 解析源个体和基向量 | 返回向量 |
| `ade.py` | `build_ade_candidate` | 依据策略串生成候选 | 返回候选向量 |
| `ade.py` | `collect_ade_trial_stats/update_ade_memory/update_seri_policy` | 成功历史统计和更新 | 更新 ADE 内部状态 |
| `ade.py` | `share_seri_policy/update_shade_population_size/adapt_successful_seri` | 策略迁移、种群规模、自适应参数 | 更新 ADE 状态 |
| `ade.py` | `compact_active_population/allocate_algorithm_state` | 活动种群压缩和状态分配 | 更新内部状态 |
| `aesspso.py` | `_aesspso_weight` | 计算自适应权重 | 返回权重 |
| `autov.py` | `_default_autov_weight` | 默认变量操作权重 | 返回权重矩阵 |
| `autov.py` | `_autov_pick_type` | 按累计概率选类型 | 返回类型下标 |
| `autov.py` | `_autov_tournament` | 锦标赛选配 | 返回个体下标 |
| `ba.py` | `NeighborFlowerPatch` | 蜜蜂邻域花斑搜索 | 输入邻域和点，写局部候选 |
| `bfgs.py` | `_ensure_bfgs_state/_bfgs_gradient` | 初始化 BFGS 状态/计算梯度 | 更新状态/返回梯度 |
| `cmaes.py` | `pop_heap_adjust/pop_heap_sort/newpop_heap_adjust/newpop_heap_sort` | 堆排序辅助 | 原地排序 |
| `cmaes.py` | `pop_niche/newpop_localstep/pop_random_variance/pop_random_entropy` | niche、局部步长、随机统计 | 更新种群/返回统计 |
| `cmaes.py` | `CMAES_parametersetting/CMAES_initial/CMAES_sampleGenerate/CMAES_updateDistribution` | CMA-ES 参数、初始化、采样、分布更新 | 更新 CMA-ES 状态 |
| `cmaes.py` | `sortIndex/adaptC2/updateEigensystem/eigen/ql/householder` | CMA-ES 线性代数和排序 | 返回/更新矩阵 |
| `coa.py` | `chaos` | 混沌序列扰动 | 输入候选和次数，修改候选 |
| `csa.py` | `levy_cuckoo/nest_discover` | Levy 飞行和巢发现 | 写 `newpop` |
| `de.py` | `differential_mutate/differential_crossover` | DE 变异和交叉 | 写 `newpop` |
| `ecpo.py` | `_ecpo_pop_factor/_ecpo_candidates` | ECPO 候选规模和候选生成 | 返回候选集 |
| `fep.py` | `_ensure_fep_state` | 初始化 EP sigma | 更新状态 |
| `frcg.py` | `_ensure_frcg_state/_sparse_gradient` | 共轭梯度状态和稀疏梯度 | 更新/返回梯度 |
| `hs.py` | `newpop_worst_best` | 和声搜索中替换最差/保留最好 | 修改新种群 |
| `ils.py` | `localsearch` | ILS 单体局部搜索 | 修改个体向量 |
| `imode.py` | `_ensure_imode_state` | 初始化 IMODE archive | 更新状态 |
| `mfea.py` | `_ensure_mfea_state` | 初始化 MFEA 统计 | 更新状态 |
| `mfea_ii.py` | `_ensure_mfea_ii_state` | 初始化 MFEA-II 统计 | 更新状态 |
| `misaco.py` | `_ensure_misaco_state` | 初始化 MiSACO archive | 更新状态 |
| `poa.py` | `plant_growth` | 植物生长步骤 | 写 `newpop` |
| `pso.py` | `_sample_dimensions_by_magnitude` | 选速度幅值大和随机维度 | 返回维度列表 |
| `pso.py` | `SparseSubgradient/Subgradient` | 稀疏/全量子梯度 | 返回梯度向量 |
| `pso.py` | `Cauchy_mutation` | Cauchy 变异 | 修改个体 |
| `pso.py` | `Orthogonal_P/LightOrthogonal_P` | OLPSO 正交学习 | 更新学习向量 |
| `rmsprop.py` | `_ensure_rmsprop_state` | 初始化 RMSProp cache | 更新状态 |
| `shade.py` | `_ensure_shade_state` | 初始化 SHADE 记忆和 archive | 更新状态 |
| `ssio_rl.py` | `_ensure_ssio_state` | 初始化 RL Q 值 | 更新状态 |

### 10.5 局部搜索和 meme 函数

| 函数 | 功能 | 输入 | 输出/用法 |
|---|---|---|---|
| `newpop_bit_climbing(popi, L, scale)` | 随机维度爬山 | 个体、迭代、尺度 | 写 `newpop[popi]` |
| `_candidate_row(values)` | 构造带适应度候选 | 向量 | 返回 `x+[fit]` |
| `_random_dims(count)` | 随机维度 | 数量 | 返回维度列表 |
| `_local_search_count()` | 局部算子数量 | 无 | 返回整数 |
| `_line_search_row(row, direction, step, budget)` | 线搜索 | 当前行、方向、步长、预算 | 返回最优候选行 |
| `newpop_simplex(popi, L, stepn, scale)` | Nelder-Mead 单纯形 | 个体、步数、尺度 | 写 `newpop` |
| `newpop_box_complex(popi, L, stepN, scale)` | Box complex 搜索 | 个体、步数、尺度 | 写 `newpop` |
| `newpop_powell(popi, L, stepn, scale)` | Powell 方向搜索 | 个体、步数、尺度 | 写 `newpop` |
| `newpop_newton(popi, L, stepn, scale)` | 稀疏牛顿/梯度搜索 | 个体、步数、尺度 | 写 `newpop` |
| `newpop_mcts(popi, L, stepn, scale)` | 蒙特卡罗树式 rollout | 个体、步数、尺度 | 写 `newpop` |
| `newpop_simulated_annealing(popi, L, stepn, scale)` | 模拟退火 | 个体、步数、尺度 | 写 `newpop` |
| `newpop_tabu_search(popi, L, stepn, scale)` | 禁忌搜索 | 个体、步数、尺度 | 写 `newpop` |
| `newpop_pattern_search(popi, L, stepn, scale)` | 模式搜索 | 个体、步数、尺度 | 写 `newpop` |
| `newpop_threshold_accepting(popi, L, stepn, scale)` | 阈值接受搜索 | 个体、步数、尺度 | 写 `newpop` |
| `newpop_elite_contraction(popi, L, scale)` | 向精英收缩 | 个体、步数、尺度 | 写 `newpop` |
| `newpop_coordinate_descent(popi, L, scale)` | 坐标下降 | 个体、步数、尺度 | 写 `newpop` |
| `newpop_mcts_local_search(popi, L, scale)` | 局部 MCTS | 个体、步数、尺度 | 写 `newpop` |
| `newpop_elite_line_search(popi, L, scale)` | 精英方向线搜索 | 个体、步数、尺度 | 写 `newpop` |
| `newpop_multi_scale_gaussian(popi, L, scale)` | 多尺度高斯扰动 | 个体、步数、尺度 | 写 `newpop` |
| `newpop_random_subspace_pattern(popi, L, scale)` | 随机子空间模式搜索 | 个体、步数、尺度 | 写 `newpop` |
| `newpop_newton_subspace(popi, L, scale)` | 子空间牛顿 | 个体、步数、尺度 | 写 `newpop` |
| `newpop_gradient_backtracking(popi, L, scale)` | 梯度回溯 | 个体、步数、尺度 | 写 `newpop` |
| `newpop_cauchy_basin_hop(popi, L, scale)` | Cauchy 盆地跳跃 | 个体、步数、尺度 | 写 `newpop` |
| `newpop_opposition_elite_blend(popi, L, scale)` | 反向学习和精英混合 | 个体、步数、尺度 | 写 `newpop` |
| `meme_selection(popi, X, scale, Iter)` | 指定动作编号执行局部搜索 | 个体、动作、尺度、迭代 | 写 `newpop` |
| `_ensure_meme_q/_ensure_meme_stats` | 初始化 Q 表和统计 | 无 | 更新状态 |
| `_meme_range(p_start, p_end)` | meme 作用区间 | 区间 | 返回区间 |
| `_meme_reward(oldfit, newfit)` | 奖励函数 | 旧/新适应度 | 返回奖励 |
| `_ranked_meme_actions(op_count)` | 按奖励排序动作 | 动作数量 | 返回动作列表 |
| `update_local_guides(index)` | 更新局部引导统计 | 个体下标 | 更新状态 |
| `apply_meme_action(index, action, scale, iter_count)` | 执行一个局部搜索动作 | 个体、动作、尺度、迭代 | 写 `newpop` 并返回奖励 |
| `meme_q_learning(...)` | Q-learning 选择局部算子 | 代数、尺度、区间 | 更新 Q 表并应用局部搜索 |
| `meme_q_learning_range(...)` | 指定范围 Q-learning | 代数、尺度、区间 | 同上 |
| `meme_q_learning_generated_eo(...)` | 含新增算子的 Q-learning | 代数、尺度、区间 | 同上 |
| `meme_random_walk(scale)` | 随机游走式 meme | 尺度 | 应用局部搜索 |
| `meme_simple_random(scale)` | 均匀随机 meme | 尺度 | 应用局部搜索 |
| `meme_randperm(scale)` | 随机排列 meme | 尺度 | 应用局部搜索 |
| `meme_inheritance(scale)` | 继承式 meme | 尺度 | 应用局部搜索 |
| `meme_subprob_decomposition(Gen, MaxG, kk, scale)` | 子问题分解 meme | 代数、分组、尺度 | 应用局部搜索 |
| `meme_biasd_roulette(...)` | 奖励偏置轮盘赌 | 代数、尺度、区间 | 应用局部搜索 |
| `meme_biasd_roulette_range(...)` | 指定范围轮盘赌 | 代数、尺度、区间 | 应用局部搜索 |
| `meme_biasd_roulette_generated_eo(...)` | 含新增算子的轮盘赌 | 代数、尺度、区间 | 应用局部搜索 |

### 10.6 archive 函数

| 函数 | 功能 | 输入 | 输出/用法 |
|---|---|---|---|
| `_squared_distance(left, right, nvar)` | 前 `nvar` 维平方距离 | 两向量、维度 | `float` |
| `_ordered_archive(pop, fit, order, size)` | 按给定顺序抽取 archive | 种群、适应度、顺序、容量 | `(archive, archive_fit)` |
| `best_archive(pop, fit, size)` | best 策略 archive | 种群、适应度、容量 | `(archive, fit)` |
| `build_archive(pop, fit, size, strategy)` | 通用 archive 构造 | `best/worst/random/diverse/mixed/truncation/radius_density` | `(archive, fit)` |
| `dual_archive(pop, fit, convergence_size, diversity_size)` | C-TAEA 风格双 archive | 种群、适应度、两个容量 | 收敛 archive + 多样性 archive |
| `diverse_archive(pop, fit, size)` | 距离多样性 archive | 种群、适应度、容量 | `(archive, fit)` |
| `truncation_archive(pop, fit, size)` | SPEA2 风格最近邻截断 archive | 种群、适应度、容量 | `(archive, fit)` |
| `radius_density_archive(pop, fit, size)` | EM-SAEA 半径密度删除 archive | 种群、适应度、容量 | `(archive, fit)` |
| `archive_value(archive, bit)` | 从 archive 随机取某维 | archive、维度 | 值或 `None` |
| `update_external_archive(...)` | 合并旧 archive 和候选并裁剪 | archive、候选、容量、策略 | 新 archive |
| `update_sade_amss_archive(...)` | SADE-AMSS 专用 archive 合并裁剪 | archive、候选、容量 | mixed archive |
| `archive_sample_indices(archive, count)` | 从 archive 抽样下标 | archive、数量 | 下标列表 |
| `archive_sample(archive, fallback, nvar)` | 从 archive 采样行，空时回退 | archive、回退种群、维度 | 向量 |

### 10.7 popvary 函数

| 函数 | 功能 | 输入 | 输出/用法 |
|---|---|---|---|
| `_squared_distance(left, right, nvar)` | 平方距离 | 两向量、维度 | `float` |
| `select_best(candidates, fits, size)` | 选择最优候选 | 候选、适应度、数量 | `(selected, selected_fit)` |
| `select_population(candidates, fits, size, strategy, nvar)` | 通用候选选择 | 策略和候选 | `(selected, fit)` |
| `_tournament_order(fits, size)` | 锦标赛选择顺序 | 适应度、数量 | 下标列表 |
| `_roulette_order(fits, size)` | 轮盘赌选择顺序 | 适应度、数量 | 下标列表 |
| `_rank_diverse_order(candidates, fits, size, nvar)` | 排名 + 多样性顺序 | 候选、适应度、维度 | 下标列表 |
| `fill_population(candidates, size, nvar, lb, ub, randval, strategy)` | 补齐种群 | 候选、容量、边界、随机函数 | 行列表 |
| `linear_population_size(initial, minimum, generation, max_generation)` | 线性种群缩减 | 初始/最小规模、代数 | 当前规模 |
| `random_subspace_groups(nvar, group_size)` | 随机维度分组 | 维度、组大小 | 维度组列表 |
| `evaluated_injection_popvary(...)` | DRL-SAEA 风格真实评估解注入维护 | 当前种群、真实评估新解、容量 | 去重后种群 |
| `subspace_population_maintenance(...)` | AVG-SAEA 风格子空间种群维护 | 种群、适应度、维度组 | 子空间种群 |
| `merge_subspace_population(...)` | 子空间种群重组 | 子空间种群、维度数、容量 | 完整维度种群 |
| `best_worst_pairs(fits, size)` | winner/loser 配对 | 适应度、数量 | `(best,worst)` 列表 |
| `elite_indices(fits, count)` | 精英下标 | 适应度、数量 | 下标列表 |
| `adaptive_rmp(success, trials, low, high)` | 自适应随机交配概率 | 成功数、尝试数、范围 | `float` |

### 10.8 评测、并行、加速函数

| 模块 | 函数 | 功能 | 输入输出/用法 |
|---|---|---|---|
| `evaluate_methods.py` | `EvalResult.improvement` | 计算初始/最终适应度比 | 属性 |
| `evaluate_methods.py` | `_finish_generation` | 一代统一收尾 | 输入 solver 和区间，无返回 |
| `evaluate_methods.py` | `_run_operator` | 单算子单问题运行 | 返回 `EvalResult` |
| `evaluate_methods.py` | `algorithm_operators` | 注册全局算法 | 返回 `(name, callable)` 列表 |
| `evaluate_methods.py` | `local_operators` | 注册局部搜索 | 返回列表 |
| `evaluate_methods.py` | `meme_hyper_operators` | 注册 meme 超启发 | 返回列表 |
| `evaluate_methods.py` | `meme_operators` | meme 兼容入口 | 返回超启发列表 |
| `evaluate_methods.py` | `run_group` | 批量运行一组算子 | 返回 `EvalResult` 列表 |
| `evaluate_methods.py` | `main` | CLI 评测入口 | stdout |
| `benchmark_operators.py` | `_select_operator` | 按组和名称取算子 | 返回 callable |
| `benchmark_operators.py` | `_run_one` | 子进程运行单算子 | 写 Queue |
| `benchmark_operators.py` | `run_with_timeout` | 带 timeout 运行 | 返回 `EvalResult` |
| `benchmark_operators.py` | `write_csv` | 写 benchmark CSV | 文件输出 |
| `benchmark_operators.py` | `print_summary` | 打印 Top 汇总 | stdout |
| `benchmark_archive_popvary.py` | `run_ecpo_variant` | 单个 archive/popvary 组合测试 | 返回 dict |
| `benchmark_archive_popvary.py` | `write_csv/print_summary/main` | CSV、汇总和 CLI | 文件/stdout |
| `benchmark.py` | `_run_task/_run_tasks/main` | CSA 多次 benchmark | 返回/输出任务结果 |
| `abca_meme.py` | `_meme_*` | ABCA meme 策略适配器 | 输入 solver/代数/尺度，修改 solver |
| `abca_meme.py` | `run_abca_meme` | ABCA+meme 单任务 | 返回 `SolveResult` |
| `abca_meme.py` | `_run_task/_run_tasks/main` | 批量和 CLI | 返回/输出结果 |
| `island_meme.py` | `_finish_generation` | 岛内一代收尾 | 修改 solver |
| `island_meme.py` | `_selected_meme_indices` | 选择局部搜索精英 | 返回下标 |
| `island_meme.py` | `_apply_meme` | 岛内应用 meme | 修改 solver |
| `island_meme.py` | `_run_island_epoch/_run_epoch` | 岛模型 epoch | 返回岛状态 |
| `island_meme.py` | `_elite_rows/_inject_migrant/_migrate` | 迁移机制 | 读取/注入精英 |
| `island_meme.py` | `_pool_rebalance` | pool 拓扑重排 | 交换各岛个体 |
| `island_meme.py` | `_make_islands` | 初始化多岛 | 返回岛状态列表 |
| `island_meme.py` | `_persistent_worker/_recv_checked/_run_island_persistent` | 常驻子进程岛模型 | IPC/返回结果 |
| `island_meme.py` | `run_island_abca_meme/main` | 岛模型 API/CLI | 返回 `IslandRunResult` |
| `fast.py` | `NumpyMultiMet.*` | NumPy CSA 状态和流程 | 与 `MultiMet` 对应，数组化 |
| `fast.py` | `run_csa_fast` | NumPy CSA 快速求解 | 返回结果 dict/对象 |
| `fast_topology.py` | 基准函数 `sphere...zakharov` | NumPy 批量目标函数 | 输入二维数组，返回适应度数组 |
| `fast_topology.py` | `_make_evaluator/_parallel_evaluate` | 目标函数封装和线程评估 | 返回 evaluator/fit |
| `fast_topology.py` | `_structural_candidates` | 函数结构候选 | 返回候选数组 |
| `fast_topology.py` | `_pool_rebalance/_inject_candidates` | pool 拓扑重排和候选注入 | 更新数组 |
| `fast_topology.py` | `_elite_local_search` | 向量化精英局部搜索 | 更新 pop/fit |
| `fast_topology.py` | `_zakharov_projection_search/_michalewicz_coordinate_search` | 专用结构强化 | 更新 pop/fit |
| `fast_topology.py` | `run_fast_pool/main` | 快速 pool 拓扑求解/CLI | 返回 `FastResult` |
| `generated_eo_accel.py` | `_refresh_best/_call_local/_make_solver` | solver 刷新、局部算子调用、构造 solver | 内部辅助 |
| `generated_eo_accel.py` | `run_local_method/run_ade_method/run_method` | 单方法运行 | 返回结果行 |
| `generated_eo_accel.py` | `run_benchmark/write_csv/rows_to_csv/main` | 批量运行和 CSV | 输出表格 |
| `compare_serial_parallel.py` | `main` | 串行/并行公平对比 CLI | stdout |

## 11. 常用开发示例

### 11.1 增加一个全局算法

1. 在 `algorithms/new_alg.py` 新建 mixin。
2. 方法只生成 `self.newpop`。
3. 在 `algorithms/__init__.py` 导入并加入 `__all__`。
4. 在 `multimethod.py` 的 `MultiMet` 继承链加入 mixin。
5. 在 `evaluate_methods.algorithm_operators()` 注册。
6. 运行：

```bash
python3 -m compileall python_multialg
python3 -m python_multialg.evaluate_methods --groups algorithms --problems ackley --nvar 10 --popsize 6 --generations 1
```

### 11.2 增加一个局部搜索算子

1. 在 `LocalSearchMixin` 增加 `newpop_xxx(self, popi, L, scale)`。
2. 在 `evaluate_methods.local_operators()` 注册。
3. 如需 meme 超启发可选择，在 `apply_meme_action()` 动作表中加入。

### 11.3 增加 archive 或变种群机制

- archive：优先放到 `archive.py`，保持输入为候选行和适应度，输出为新 archive。
- 变种群：优先放到 `popvary.py`，保持纯函数风格，便于 ECPO/ADE/MFEA/SACC 复用。

## 12. 当前测试建议

快速 smoke test：

```bash
python3 -m python_multialg.evaluate_methods --groups algorithms --problems ackley --nvar 10 --popsize 6 --generations 1 --repeats 1
```

100 维全局算法测试：

```bash
python3 -m python_multialg.benchmark_operators \
  --problems rosenbrock griewank ackley rastrigin schwefel \
  --groups algorithms \
  --nvar 100 --popsize 20 --generations 20 --timeout 3
```

archive/popvary 测试：

```bash
python3 -m python_multialg.benchmark_archive_popvary \
  --problems rosenbrock griewank ackley rastrigin schwefel \
  --nvar 100 --popsize 20 --generations 50
```
