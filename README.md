ISPH_NVIDIA_CUDA_CONTEST
========================

GPU Accelerated Incompressible Fluid Simulation for NVIDIA’s 2013/2014 CUDA Campus Programming Contest

Description：

SPH method has a wide application in physics engines such as NVIDIA PhysX. However, it is difficult for SPH to satisfy the incompressibility condition, which leads to compressibility artifact in realistic fluid simulations. To solve this problem, the efficient PCISPH algorithm is used to simulate incompressible fluids. Furthermore, the algorithm is parallelized and optimized on the GPU with CUDA. This has improved the visual realism of the fluids as well as the simulation performance.

References：

(1) Standard SPH：

M. M¨uller, D. Charypar, M. Gross. Particle-based fluid simulation for interactive applications[C]//Proceedings of the 2003 ACM SIGGRAPH/Eurographics Symposium on Computer Animation. 2003. Aire-la-Ville, Geneva, Switzerland: Eurographics Association, SCA ’03.

(2) Weakly Compressible SPH (WCSPH)：

M. Becker, M. Teschner. Weakly compressible SPH for free surface flows[C]//Proceedings of the 2007 ACM SIGGRAPH/Eurographics Symposium on Computer Animation. 2007. Aire-la-Ville, Switzerland, Switzerland: Eurographics Association, SCA ’07.

(3) Predictive-Corrective Incompressible SPH (PCISPH) ：

B. Solenthaler, R. Pajarola. Predictive-corrective incompressible SPH[J]. ACM Transactions on Graphics (Proceedings SIGGRAPH), 2009, 28(3):1–6.


Portfolio:

YouTube Link: https://www.youtube.com/watch?v=oOEX17RnOgU&feature=youtu.be

Youku Link http://v.youku.com/v_show/id_XODIwMzI1MTQw.html?from=y1.7-1.2
    
Development Enviorenment：

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
    
    
    



