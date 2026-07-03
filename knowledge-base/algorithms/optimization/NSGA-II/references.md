# NSGA-II 参考资料

## 经典论文

- K. Deb, A. Pratap, S. Agarwal, T. Meyarivan, "A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II", IEEE Transactions on Evolutionary Computation, 2002.
- K. Deb, "Multi-Objective Optimization using Evolutionary Algorithms", Wiley, 2001.

## 重要概念与实现参考

- 非支配排序（Fast Non-dominated Sorting）
- 拥挤距离（Crowding Distance）
- 精英策略（Elitism）
- SBX 交叉（Simulated Binary Crossover）
- 多项式变异（Polynomial Mutation）

## 工具与框架

- PlatEMO：一个多目标进化算法平台，包含 NSGA-II、NSGA-III、MOEA/D 等实现。
- jMetal：Java 生态中的多目标优化框架，常用于算法对比实验。

## 相关扩展

- NSGA-III：针对多目标（尤其 3 个及以上目标）优化的扩展。
- r-NSGA-II：带参考点偏好的 NSGA-II 变体。
- NSGA-II with constraints：带约束处理的 NSGA-II 版本。
