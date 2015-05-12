ISPH_NVIDIA_CUDA_CONTEST
========================

GPU Accelerated Incompressible Fluid Simulation for NVIDIA’s 2013/2014 CUDA Campus Programming Contest

Description：

SPH method has a wide application in physics engines such as NVIDIA PhysX. However, it is difficult for SPH to satisfy the incompressibility condition, which leads to compressibility artifact in realistic fluid simulations. To solve this problem, the efficient PCISPH algorithm is used to simulate incompressible fluids. Furthermore, the algorithm is parallelized and optimized on the GPU with CUDA. This has improved the visual realism of the fluids as well as the simulation performance.


References：

[1] Standard SPH：

M. M¨uller, D. Charypar, M. Gross. Particle-based fluid simulation for interactive applications[C]//Proceedings of the 2003 ACM SIGGRAPH/Eurographics Symposium on Computer Animation. 2003. Aire-la-Ville, Geneva, Switzerland: Eurographics Association, SCA ’03.

[2] Weakly Compressible SPH (WCSPH)：

M. Becker, M. Teschner. Weakly compressible SPH for free surface flows[C]//Proceedings of the 2007 ACM SIGGRAPH/Eurographics Symposium on Computer Animation. 2007. Aire-la-Ville, Switzerland, Switzerland: Eurographics Association, SCA ’07.

[3] Predictive-Corrective Incompressible SPH (PCISPH) ：

B. Solenthaler, R. Pajarola. Predictive-corrective incompressible SPH[J]. ACM Transactions on Graphics (Proceedings SIGGRAPH), 2009, 28(3):1–6.

[4] R.  C.  Hoetzlein,  “Fast  Fixed-Radius  Nearest  Neighbors:  InteractiveMillion-Particle Fluids,” inGPU Tech. Conf., 2014.

Portfolio:

YouTube Link: https://www.youtube.com/watch?v=oOEX17RnOgU&feature=youtu.be

Youku Link http://v.youku.com/v_show/id_XODIwMzI1MTQw.html?from=y1.7-1.2

    
Development Environment：

Windows 7 & Visual C++ 2010 & CUDA Toolkit v7.0 (The default CUDA toolkit installation location is C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.0)

Coding style:

Coding style for this project generally follows the Google C++ Style Guide 

Note:

The CUDA tookit has been upgraded from v5.5 to v7.0.


About Compilation & Parameter Setting & Keyboard Commands：

Please refer to the short documentation (Currently it's only written in Chinese)
