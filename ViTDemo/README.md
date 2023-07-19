Set up an architecture for Vision Transformer, in order to classify three car models(搭建一个Vision Transformer模型，包含不同车辆的图像进行分类):

['0':'汽车','1':'摩托车','2':'货车'].

- Explain for the different python files:

	The vit1.py is for the whole process;

	The vit0.py is for training only while vit_eval.py is for evaluation only;

	The training results are traning-loss.png & training-acc.png;

	The eval result is 0.815833330154419 as the accuracy rate.

- Changed 'Tensor.numpy()[0]' into float(Tensor) in order to avoid terminal warnings.

- result for vit1.py:

	```
	epo: 1, step: 50, loss is: 1.0213838815689087, acc is: 0.46875
	epo: 2, step: 100, loss is: 0.8762409090995789, acc is: 0.625
	epo: 3, step: 150, loss is: 0.9546507000923157, acc is: 0.53125
	epo: 4, step: 200, loss is: 0.39328888058662415, acc is: 0.875
	epo: 5, step: 250, loss is: 0.21988703310489655, acc is: 0.9375
	epo: 6, step: 300, loss is: 0.24376368522644043, acc is: 0.90625
	epo: 7, step: 350, loss is: 0.23820441961288452, acc is: 0.90625
	epo: 8, step: 400, loss is: 0.16654880344867706, acc is: 0.9375
	epo: 9, step: 450, loss is: 0.015371461398899555, acc is: 1.0
	epo: 11, step: 500, loss is: 0.05601257458329201, acc is: 0.96875
	save model to: /home/flyslice/xy/test/transformer-Demo/ViTDemo/work/checkpoints/save_dir_500.pdparams
	epo: 12, step: 550, loss is: 0.07314401119947433, acc is: 0.96875
	epo: 13, step: 600, loss is: 0.042762547731399536, acc is: 1.0
	epo: 14, step: 650, loss is: 0.024706225842237473, acc is: 1.0
	epo: 15, step: 700, loss is: 0.04962686449289322, acc is: 1.0
	epo: 16, step: 750, loss is: 0.006833759136497974, acc is: 1.0
	epo: 17, step: 800, loss is: 0.2226804494857788, acc is: 0.9375
	epo: 18, step: 850, loss is: 0.13770261406898499, acc is: 0.90625
	epo: 19, step: 900, loss is: 0.012483458034694195, acc is: 1.0
	epo: 21, step: 950, loss is: 0.006592174060642719, acc is: 1.0
	epo: 22, step: 1000, loss is: 0.016878241673111916, acc is: 1.0
	save model to: /home/flyslice/xy/test/transformer-Demo/ViTDemo/work/checkpoints/save_dir_1000.pdparams
	epo: 23, step: 1050, loss is: 0.0035787748638540506, acc is: 1.0
	epo: 24, step: 1100, loss is: 0.004507001955062151, acc is: 1.0
	epo: 25, step: 1150, loss is: 0.0021869021002203226, acc is: 1.0
	epo: 26, step: 1200, loss is: 0.002283216919749975, acc is: 1.0
	epo: 27, step: 1250, loss is: 0.0023040869273245335, acc is: 1.0
	epo: 28, step: 1300, loss is: 0.002177279442548752, acc is: 1.0
	epo: 29, step: 1350, loss is: 0.0012638989137485623, acc is: 1.0
	epo: 31, step: 1400, loss is: 0.0016838698647916317, acc is: 1.0
	epo: 32, step: 1450, loss is: 0.0017979578115046024, acc is: 1.0
	epo: 33, step: 1500, loss is: 0.0013187980512157083, acc is: 1.0
	save model to: /home/flyslice/xy/test/transformer-Demo/ViTDemo/work/checkpoints/save_dir_1500.pdparams
	epo: 34, step: 1550, loss is: 0.0014346085954457521, acc is: 1.0
	epo: 35, step: 1600, loss is: 0.0016081215580925345, acc is: 1.0
	epo: 36, step: 1650, loss is: 0.0011500888504087925, acc is: 1.0
	epo: 37, step: 1700, loss is: 0.001023828168399632, acc is: 1.0
	epo: 38, step: 1750, loss is: 0.0012713398318737745, acc is: 1.0
	epo: 39, step: 1800, loss is: 0.0010248067555949092, acc is: 1.0
	```

- result for vit_eval.py:

	```
	文件已存在
	生成数据列表完成！
	模型在验证集上的准确率为： 0.815833330154419
	```
