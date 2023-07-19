Set up an architecture for Vision Transformer, in order to classify three car models(搭建一个Vision Transformer模型，包含不同车辆的图像进行分类):['0':'汽车','1':'摩托车','2':'货车'].

- Explain for the different python files:
	The vit1.py is for the whole process;
	The vit0.py is for training only while vit_eval.py is for evaluation only;
	The training results are traning-loss.png & training-acc.png;
	The eval result is 0.815833330154419 as the accuracy rate.

- Changed 'Tensor.numpy()[0]' into float(Tensor) in order to avoid terminal warnings.

- result for vit1.py:


- result for vit_eval.py:
	文件已存在
	生成数据列表完成！
	模型在验证集上的准确率为： 0.815833330154419

