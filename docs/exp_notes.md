20220511
1. mlp 与 grid 在稀疏输入上对深度的拟合表现有明显差异
   - mlp更为compact
   - grid在每个voxel参数量自由度较小，但总参数量较大，优化快但是容易过拟合
   - 需要加正则 plenoxel/regnerf 里有一定分析

2. depth guide
   - 评估深度不一致性
   - mvsnet
   - deep fusion

20220519

gaussian nll loss在以深度不一致性作为深度不确定性衡量的情况下能够使深度回归更平滑，但rgb效果与之前接近，同时由于考虑到了光线上期望深度的方差，因此对于抑制floater现象也有一定帮助，相当于同时满足了之前的视线loss与深度loss
与视线loss+深度loss比较
gaussian_nll_loss rgb http://30.2.136.23:8000/openNGP_exps/fuyinshi_1/202205111604_nll_top4_upsample/validation/38749.png
视线loss+深度loss rgb http://30.2.136.23:8000/openNGP_exps/fuyinshi_1/202204182253_srgb/validation/32499.png
gaussian_nll_loss depth http://30.2.136.23:8000/openNGP_exps/fuyinshi_1/202205111604_nll_top4_upsample/validation/38749_depth.png
视线loss+深度loss depth http://30.2.136.23:8000/openNGP_exps/fuyinshi_1/202204182253_srgb/validation/32499_depth.png
与深度loss(baseline)比较
深度loss rgb http://30.2.136.23:8000/openNGP_exps/fuyinshi_1/debug_ngp_depth/eval/0113.png
gaussian_nll_loss rgb http://30.2.136.23:8000/openNGP_exps/fuyinshi_1/202205111604_nll_top4_upsample/eval/0113.png

20220526

1. 稀疏的输入能够提升ngp拟合效果 (psnr ~27 -> ~30)
gt http://30.2.136.23:8008/fuyinshi_1/images/10.jpg
97输入 http://30.2.136.23:8000/openNGP_exps/fuyinshi_1/202205111604_nll_top4_upsample/eval_train/0001.png
21输入 http://30.2.136.23:8000/openNGP_exps/fuyinshi_1/202205111604_nll_small_dataset/validation/33749.png
但是这种策略需要控制新视角的位置，同时pose不准确带来的不一致反而不如更多数据后来的平滑
97输入 http://30.2.136.23:8000/openNGP_exps/fuyinshi_1/202205111604_nll_top4_upsample/eval/0176.png
21输入 http://30.2.136.23:8000/openNGP_exps/fuyinshi_1/202205111604_nll_small_dataset/eval/0176.png

2. freeze训练好的ngp，对pose进行优化
在lego数据上可以看到pose的确是被优化了，但收敛后依然和施加的随机扰动有一定距离
随机扰动
   trans mean perturb: 0.0946830073756591
   rots mean perturb: 5.682368967511158
收敛误差(http://30.2.136.23:8000/openNGP_exps/lego/202205062018)
   trans error 8e-3
   rots error 0.177

该训练策略的有效性是intuitive的，但是收敛结果的gap是否说明只靠pixel loss是不足以将pose优化到超过colmap的结果？
patch NCC loss instead of pixel loss

3. Neural Rays for Occlusion-aware Image-based Rendering
本文不直接训练ngp而是对训练数据的每个ray设置了一个可学习参数
对于新视角的ray通过训练数据的ray来进行fusion
在训练数据的ray的表示上引入了visibility，因此对于新视角ray的fusion可以考虑mvs下visibility的一致性来实现occlusion-aware
同时fusion公式满足volume rendering的设定(借鉴NeuS)
