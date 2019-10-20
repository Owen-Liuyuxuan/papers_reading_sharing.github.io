# Lookahead Optimizer:ksteps forward, 1 step back

![image](res/look_ahead_optimizer.png)

本质上来说就是保存两套参数值，用一套参数值正常更新N步(每变一次需要使用新的权值计算更新方向)，然后用第二套参数往最终这N步的总更新方向走一步