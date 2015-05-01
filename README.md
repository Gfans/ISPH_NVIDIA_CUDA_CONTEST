ISPH_NVIDIA_CUDA_CONTEST
========================

GPU Accelerated Incompressible Fluid Simulation for NVIDIA’s 2013/2014 CUDA Campus Programming Contest


简介

光滑粒子流体动力学（SPH）方法在物理引擎中已经有了广泛的应用，代表产品如 Nvidia PhysX。但是，SPH 方法的可压缩性严重影响了水等不可压缩流体的视觉效果。针对这一问题，本作品采用高效的预测校正不可压缩 SPH （PCISPH）算法来模拟不可压缩流体，并利用 CUDA 技术对 PCISPH 进行并行优化，提高了流体的视觉真实感，取得了较高的加速比。


优酷视频连接：
    http://v.youku.com/v_show/id_XODIwMzI1MTQw.html?from=y1.7-1.2

    
关于编译：
    编译代码需安装VC++2010 & CUDA Toolkit v5.5CUDA Toolkit，默认安装位置为C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v5.5。
    编译后的可执行文件名为ISPH_NVIDIA_CUDA_CONTEST.exe，位于Release目录。


关于程序参数修改：
    SPH模拟的稳定性对于物理参数以及时间步长的设置非常敏感，如需更改请设置在合理范围内，以免模拟程序运行出错（表现为粒子位置异常）。
    
    
关于可执行文件：
    可执行文件位于bin目录，请勿随意修改bin目录下相关依赖文件。本程序基于Visual C++ 2010 开发，若没有安装VC++2010，请双击bin目录下的vcredist_x86安装VC++2010运行库(X86)。请安装最新显卡驱动（包含nvcuda.dll），从而使用本程序。
    

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
    
    
    



