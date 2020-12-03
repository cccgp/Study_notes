### 矩阵分解
- pytorch实现的一个库。
- 安装tensorly
- 分为三个步骤：训练完整模型，矩阵分解，微调。
- 1. 首先利用上一层的alexNet.py训练一个模型。
- 2. 再使用alex_10%_v1.py:分解alexNet_model_10%
- 3. 再使用alex_10%_v2.py:微调分解后的模型
- 同理vgg
- 这里只有alexNet和vggNet都是10%的模型代码，20%的进行相应修改即可。
- decompositions.py就是提供的接口
