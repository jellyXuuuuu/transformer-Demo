prepare.py 文件用于模型框架的建立

image_prepare.py 文件则用于图像处理

本项目仅包含模型评估部分，运用用IMGAENET数据集预训练完毕的模型(`ViT_base_patch16_384_pretrained.pdparams`)来评估

模型太大无法上传github, 获取链接: [AI Studio](https://aistudio.baidu.com/aistudio/datasetdetail/105741)

然后解压pretrained.zip文件获取

相对应的label文件为val_list.txt, 即文件路径 - 真实标签值, 用于对比用模型评估出来的预测标签值

### 实验结果
    - Ground truth lable index: 65, Pred label index:65
    - Ground truth lable index: 970, Pred label index:795
    - Ground truth lable index: 230, Pred label index:230
    - Ground truth lable index:809, Pred label index:809

分别为ILSVRC2012_val_00000001~4.JPEG图像源

其中标签970与预测值795不一致, 即ILSVRC2012_val_00000002.JPEG图的评估不准确, 另外三图预估值与真实值相同

暂时仅测试了以上四个图片, 整体评估数据集约5G