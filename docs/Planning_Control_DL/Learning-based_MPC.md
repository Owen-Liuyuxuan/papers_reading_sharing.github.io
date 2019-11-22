time: 20191122
pdf_source: https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/351561/08754713.pdf?sequence=1&isAllowed=y
short_title: LMPC_GP
# Learning-based Model Predictive Control for Autonomous Racing

这篇来自ETH的论文讲述了一个data-driven MPC for racing的算法系统构建。核心思路是使用GP(高斯过程)补偿动力学模型的不确定性与建模误差，然后用根据此不确定性构建非线性优化问题作为MPC的框架进行求解。文章第二章介绍了GP模型以及稀疏GP回归算法。从第三节开始按顺序说明了名义动力学模型、GP补偿问题表述、MPC损失函数表述、带有预测不确定性的MPC约束函数表述、解耦MPC与GP并简化MPC计算的方法、简化GP迭代计算的方法。

## 名字动力学模型
对于Racing Car,这里选择的动力学建模是较为精确的动力学模型，并且轮胎模型选择的是Pacejka模型(具体看公式)
$$
\dot{\mathbf{x}}=\left[\begin{array}{c}{v_{x} \cos \varphi-v_{y} \sin \varphi} \\ {v_{x} \sin \varphi+v_{y} \cos \varphi} \\ {r} \\ {\frac{1}{m}\left(F_{R, y}+F_{F, y} \cos \delta-m v_{x} r\right)} \\ {\frac{1}{I_{z}}\left(F_{F, y} l_{F} \cos \delta-F_{R, y} l_{R}+\tau_{\mathrm{TV}}\right)} \\ {\Delta \delta} \\ {\Delta T}\end{array}\right]
$$
$$
\begin{aligned} r_{\text {target }} &=\delta \frac{v_{x}}{l_{F}+l_{R}} \\ \tau_{\mathrm{TV}} &=\left(r_{\text {target }}-r\right) P_{\mathrm{TV}} \\ \alpha_{R} &=\arctan \left(\frac{v_{y}-l_{R} r}{v_{x}}\right) \\ \alpha_{F} &=\arctan \left(\frac{v_{y}+l_{F} r}{v_{x}}\right)-\delta \\ F_{R, y} &=D_{R} \sin \left(C_{R} \arctan \left(B_{R} \alpha_{R}\right)\right) \\ F_{F, y} &=D_{F} \sin \left(C_{F} \arctan \left(B_{F} \alpha_{F}\right)\right) \end{aligned}
$$

积分使用的是RK4,$T_s = 50ms$

## GP模型回归问题描述
GP模型的输入为
$$
z = [v_x;v_y; r;\delta + \frac{1}{2}\Delta\delta; T+ \frac{1}{2}\Delta T]
$$
名义运动学的预测结果与实际测量结果的差值就是GP的目标输出。
$$
\mathbf{y}_{k}=\mathbf{B}_{d}^{\dagger}\left(\mathbf{x}_{k+1}-\mathbf{f}\left(\mathbf{x}_{k}, \mathbf{u}_{k}\right)\right)=\mathbf{d}_{\mathrm{true}}\left(\mathbf{z}_{k}\right)+\mathbf{w}_{k}
$$
其中$B_d = [0_{3\times 3}; I{3\times 3}; 0_{2\times}3]$说明只有一部分值需要补偿(其实只有xy方向加速度还有角加速度需要补偿),$B_d^\dagger$为伪逆.

## 均值与方差的传递
$$
\begin{aligned} \boldsymbol{\mu}_{k+1}^{\mathrm{x}}=&\mathbf{f}\left(\boldsymbol{\mu}_{k}^{\mathrm{x}}, \mathbf{u}_{k}\right)+\mathbf{B}_{d} \boldsymbol{\mu}^{d}\left(\boldsymbol{\mu}_{k}^{\mathrm{z}}\right) \\ 
\mathbf{\Sigma}_{k+1}^{\mathrm{x}}=&\left[\nabla_{x} \mathbf{f}\left(\boldsymbol{\mu}_{k}^{\mathrm{x}}, \mathbf{u}_{k}\right) \quad \mathbf{B}_{d}\right] \\ 
&\left[\begin{array}{cc}{\mathbf{\Sigma}_{k}^{\mathbf{x}}} & {\mathbf{\Sigma}^{d}\left(\boldsymbol{\mu}_{k}^{\mathrm{z}}\right)+\mathbf{\Sigma}^{\mathrm{w}}}\end{array}\right] \\ 
&\left[\nabla_{x} \mathbf{f}\left(\boldsymbol{\mu}_{k}^{\mathbf{x}}, \mathbf{u}_{k}\right) \quad \mathbf{B}_{d}\right]^{T}
\end{aligned}
$$

## 损失函数
这里不誊抄其公式，原因是由于Racing的性质使得它只需要绕圈，损失函数与基础的非线性MPC没有本质区别。

## 带有不确定性的约束

距离轨迹点的偏离概率大于$p$对应的距离偏差值为
$$
R_{GP}(\sum^{XY}_k) = \sqrt{\chi^2_2(p) \lambda_{max}(\sum_k^{XY})}
$$
其中$\sum^{XY}$表达$XY$位移分量的covariance矩阵。$\lambda$为取特征值操作，$\chi^2_2$为表达卡方分布。根号里面的意思是，先取最大特征值，也就是$XY$偏差矩阵中取出主方向的方差值，然后卡方分布表达的是正态分布平方值的概率分布。

最后约束可以表达为
$$
\left\|\left[\begin{array}{c}{\mu_{k}^{X}} \\ {\mu_{k}^{X}}\end{array}\right]-\left[\begin{array}{c}{X_{c}\left(\theta_{k}\right)} \\ {Y_{c}\left(\theta_{k}\right)}\end{array}\right]\right\|^{2} \leq\left\|R\left(\theta_{k}\right)-R_{\mathrm{GP}}\left(\Sigma_{k}^{X Y}\right)\right\|^{2}
$$
实际运算时只对前几步有效。

其他关于力、输入、输入变化率的固定约束这里不再誊写。

## 计算考虑

由于约束中带有方差，所以会和实际采取的控制结果耦合，这里采取了一个实时运算的简化。由于上一时刻的优化结果会保存到这一时刻，因而可以近似认为优化前后的控制输入差不会太大，进而方差的传递只以本时刻第一次的结果为准，本时刻后续迭代优化不再改变方差。

## 在线学习GP补偿参数

作者开发了一套在线选择数据添加、选择数据替换、去除outlier的算法，这里不复述。