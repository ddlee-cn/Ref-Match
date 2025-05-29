# [T-CSVT 2022] Reference-Guided Landmark Image Inpainting with Deep Feature Matching

[Jiacheng Li](jclee@mail.ustc.edu.cn), Zhiwei Xiong(http://staff.ustc.edu.cn/~zwxiong), and Dong Liu(https://faculty.ustc.edu.cn/dongeliu/)

Published in IEEE Transactions on Circuits and Systems for Video Technology (T-CSVT), 2022

[Paper](https://ieeexplore.ieee.org/document/9840396) | [Project](https://ddlee-cn.github.io/publication/2022-07-26-TCSVT-RefMatch.html)


## Abstract
Despite impressive progress made by recent image inpainting methods, they often fail to predict the original content when the corrupted region contains unique structures, especially for landmark images. Applying similar images as a reference is helpful but introduces a style gap of textures, resulting in color misalignment. To this end, we propose a style-robust approach for reference-guided landmark image inpainting, taking advantage of both the representation power of learned deep features and the structural prior from the reference image. By matching deep features, our approach builds style-robust nearest-neighbor mapping vector fields between the corrupted and reference images, in which the loss of information due to corruption leads to mismatched mapping vectors. To correct these mismatched mapping vectors based on the relationship between the uncorrupted and corrupted regions, we introduce mutual nearest neighbors as reliable anchors and interpolate around these anchors progressively. Finally, based on the corrected mapping vector fields, we propose a two-step warping strategy to complete the corrupted image, utilizing the reference image as a structural “blueprint”, avoiding the style misalignment problem. Extensive experiments show that our approach effectively and robustly assists image inpainting methods in restoring unique structures in the corrupted image.




## Dependencies

```
# download VGG19 weight at https://download.pytorch.org/models/vgg19-dcbb9e9d.pth

pip install -r requirments.txt

sh pyvfc/install.sh

```

## Acknowledgement

RefMatch is built based on [Neural Best-Buddies](https://github.com/kfiraberman/neural_best_buddies), [VFC](https://github.com/jiayi-ma/VFC), and [pyvfc](https://github.com/cramppet/pyvfc).


# Citation
```
@ARTICLE{9840396,  
author={Li, Jiacheng and Xiong, Zhiwei and Liu, Dong},  
journal={IEEE Transactions on Circuits and Systems for Video Technology},   
title={Reference-Guided Landmark Image Inpainting with Deep Feature Matching},   
year={2022},  
volume={},  
number={},  
pages={1-1},  
doi={10.1109/TCSVT.2022.3193893}}
```