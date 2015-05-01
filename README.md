ISPH_NVIDIA_CUDA_CONTEST
========================

GPU Accelerated Incompressible Fluid Simulation for NVIDIA’s 2013/2014 CUDA Campus Programming Contest

简介

光滑粒子流体动力学（SPH）方法在物理引擎中已经有了广泛的应用，代表产品如 Nvidia PhysX。但是，SPH 方法的可压缩性严重影响了水等不可压缩流体的视觉效果。针对这一问题，本作品采用高效的预测校正不可压缩 SPH （PCISPH）算法来模拟不可压缩流体，并利用 CUDA 技术对 PCISPH 进行并行优化，提高了流体的视觉真实感，取得了较高的加速比。
