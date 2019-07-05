# Hierarchical-Attention-Model-for-Intrusion-Detection
The idea of Hierarchical Attention Model for Intrusion Detection comes from the application of Attention in NLP.

In this paper, we separately adopt two kinds of attention mechanism. The overall view of the system is as follows：
![include](https://github.com/FlamingJay/Hierarchical-Attention-Model-for-Intrusion-Detection/blob/master/figure/model.png)


Here, we apply the location-based attention on the features. That is the feature-based attention. This is good for visualization in the next step.

![include](https://github.com/FlamingJay/Hierarchical-Attention-Model-for-Intrusion-Detection/blob/master/figure/location-attention.png)

Then, we apply the dot-product attention on different timestep aiming to improve the performance of model.

![include](https://github.com/FlamingJay/Hierarchical-Attention-Model-for-Intrusion-Detection/blob/master/figure/dot.png)

The attention used in this paper all belongs to global attention as shown below.

![include](https://github.com/FlamingJay/Hierarchical-Attention-Model-for-Intrusion-Detection/blob/master/figure/global.png)


When timestep equals to 10, the accuracy can reach more than 98.7%.

We visualize the attention map shown below.
![include](https://github.com/FlamingJay/Hierarchical-Attention-Model-for-Intrusion-Detection/blob/master/figure/normal_1.png)
![include](https://github.com/FlamingJay/Hierarchical-Attention-Model-for-Intrusion-Detection/blob/master/figure/normal_2.png)


Finally, this paper is now in the investment stage. Please contact me if u have any question or request

601275571@qq.com

Thanks a lot!
