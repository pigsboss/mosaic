# MOSAIC: Multidimensional Observable Sampling and Aperture Interferometric Coding
**多维观测量采样与孔径干涉编码系统**

**MOSAIC** 是一个面向下一代空间光干涉与高分辨率天文成像的前沿计算光学探索项目。本项目聚焦于**光学稀疏合成孔径 (Optical Sparse Aperture)**，旨在突破传统物理口径的衍射极限，通过前端硬件调制与后端统计算法的软硬协同，实现复杂目标特征的高维重构。

## 🌌 核心特性 (Key Features)

*   **多维观测量稀疏采样 (Sparse Sampling of Observables):** 针对深空目标（如系外行星、特征标志物）在时间、空间、光谱等多维度的天然稀疏性，建立高效的非全采样数学物理模型。
*   **主动在线孔径编码 (Online Aperture Coding):** 构建可参数化配置的光瞳面相位/振幅调制掩膜模型，主动重塑系统的点扩散函数 (PSF) 与光学传递函数 (OTF)，将高维特征映射至低维干涉图样。
*   **统计重构与目标提取 (Target Extraction & Statistical Reconstruction):** 引入压缩感知 (Compressive Sensing) 与贝叶斯推断框架，求解病态逆问题 $y = \Phi x + \epsilon$。在极低信噪比与稀疏数据下，精准剥离深空背景，提取关键物理特征。
*   **异构高性能加速 (High-Performance Computing):** 底层张量计算与迭代求解器针对异构硬件进行深度优化，全面支持基于 **SYCL** 和 **OpenCL** 的加速算子，满足海量多模态数据重构的算力需求。
