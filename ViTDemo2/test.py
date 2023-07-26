import os
import zipfile
import random
import paddle
from paddle import fluid
import numpy
# import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import PIL.Image as Image
from paddle.io import Dataset 
import numpy as np 
import sys
import paddle.nn as nn
from multiprocessing import cpu_count  
from paddle.nn import MaxPool2D,Conv2D,BatchNorm2D
from paddle.nn import Linear 
import random
from paddle.nn.initializer import TruncatedNormal, Constant 
from paddle.nn import TransformerEncoderLayer, TransformerEncoder
from paddle.regularizer import L2Decay

from prepare import VisionTransformer
from image_prepare import transform
import paddle.nn.functional as F

# 开启0号GPU
use_gpu = True
paddle.set_device('gpu:0') if use_gpu else paddle.set_device('cpu')

# 实例化模型
model = VisionTransformer(
        patch_size=16,
        class_dim=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        epsilon=1e-6)
# 加载模型参数
params_file_path="/home/flyslice/xy/test/transformer-Demo/ViTDemo2/ViT_base_patch16_384_pretrained.pdparams"
# DeiT_base_distilled_patch16_384_pretrained.pdparams
model_state_dict = paddle.load(params_file_path)
model.load_dict(model_state_dict)

model.eval()

with open('/home/flyslice/xy/test/transformer-Demo/ViTDemo2/val_list.txt') as flist:
    line = flist.readline()
img_path, label = line.split(' ')
print("imgpath", img_path)
print("label", label)
img_path = os.path.join('/home/flyslice/xy/test/transformer-Demo/ViTDemo2/', img_path)

# img_path = '/home/flyslice/xy/test/transformer-Demo/ViTDemo2/ILSVRC2012_test_00000001.jpg'

with open(img_path, 'rb') as f:
    img = f.read()

transformed_img = transform(img)

# true_label = int(1)
true_label = int(label)

x_data = paddle.to_tensor(transformed_img[np.newaxis,:, : ,:])
logits = model(x_data)
pred = F.softmax(logits)
pred_label = int(np.argmax(pred.numpy()))
# print("x_data", x_data)
# print("pred", pred)
# print("logits", logits)

print("Ground truth lable index: {}, Pred label index:{}".format(
        true_label, pred_label))
img = Image.open(img_path)
plt.imshow(img)
plt.axis('off')
plt.show()

# acc_set = []
# avg_loss_set = []
# for batch_id, data in enumerate(val_loader()):
#     x_data, y_data = data
#     y_data = y_data.reshape([-1, 1])
#     img = paddle.to_tensor(x_data)
#     label = paddle.to_tensor(y_data)
#     # 运行模型前向计算，得到预测值
#     logits = model(img)
#     # 多分类，使用softmax计算预测概率
#     pred = F.softmax(logits)
#     # 计算交叉熵损失函数
#     loss_func = paddle.nn.CrossEntropyLoss(reduction='none')
#     loss = loss_func(logits, label)
#     # 计算准确率
#     acc = paddle.metric.accuracy(pred, label)

#     acc_set.append(acc.numpy())
#     avg_loss_set.append(loss.numpy())
# print("[validation] accuracy/loss: {}/{}".format(np.mean(acc_set), np.mean(avg_loss_set)))