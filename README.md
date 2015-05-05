ISPH_NVIDIA_CUDA_CONTEST
========================

GPU Accelerated Incompressible Fluid Simulation for NVIDIA’s 2013/2014 CUDA Campus Programming Contest

项目简介：光滑粒子流体动力学（SPH）方法在物理引擎中已经有了广泛的应用，代表产品如 Nvidia PhysX。但是，SPH 方法的可压缩性严重影响了水等不可压缩流体的视觉效果。针对这一问题，本作品采用弱可压缩 SPH (WCSPH)方法 以及更高效的预测校正不可压缩 SPH （PCISPH）算法来模拟不可压缩流体，并利用 CUDA 技术对 WCSPH 和 PCISPH 进行并行优化，提高了流体的视觉真实感。与基于 CPU 的实现相比，基于 GPU 的流体模拟程序在性能上可以提升一个数量级。

参考文献：

(1) 标准 SPH 方法：

M. M¨uller, D. Charypar, M. Gross. Particle-based fluid simulation for interactive applications[C]//Proceedings of the 2003 ACM SIGGRAPH/Eurographics Symposium on Computer Animation. 2003. Aire-la-Ville, Geneva, Switzerland: Eurographics Association, SCA ’03.

(2) WCSPH 方法：

M. Becker, M. Teschner. Weakly compressible SPH for free surface flows[C]//Proceedings of the 2007 ACM SIGGRAPH/Eurographics Symposium on Computer Animation. 2007. Aire-la-Ville, Switzerland, Switzerland: Eurographics Association, SCA ’07.

(3) PCISPH 方法：

B. Solenthaler, R. Pajarola. Predictive-corrective incompressible SPH[J]. ACM Transactions on Graphics (Proceedings SIGGRAPH), 2009, 28(3):1–6.


优酷视频连接：
    http://v.youku.com/v_show/id_XODIwMzI1MTQw.html?from=y1.7-1.2

    
开发环境：
    Windows 7 & Visual C++ 2010 & CUDA Toolkit v7.0 CUDA Toolkit(默认安装位置为C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.0)。


关于程序参数修改：
    SPH模拟的稳定性对于物理参数以及时间步长的设置非常敏感，如需更改请设置在合理范围内，以免模拟程序运行出错（表现为粒子位置异常）。
    

关于按键操作：

    1. 空格 暂停/开始
    2. f/F 上一种模拟方法
    3. g/G 下一种模拟方法  
    4. 1 显示/隐藏均匀网格边界
    5. 2 显示/隐藏容器边界
    6. 3 显示/隐藏均匀网格
    7. c/C 改变摄像机移动模式 旋转or平移
    8. h/H 显示帮助信息
    9. l/L 切换光源位置控制模式
    10. j/J 切换流体粒子显示模式
    11. a/A + d/D + w/W + s/S + q/Q + z/Z 移动相机/（视觉效果上等同于移动物体）
    12. b/B加载/取消加载3d模型（本程序中3D模型为Stanford bunny）
    13. [ 和 ] 切换场景 
    14.鼠标左键 控制移动/旋转 + 鼠标右键控制放大缩小
    
    
    



