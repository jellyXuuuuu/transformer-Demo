# Transformer
用于机器翻译-translator, `Encoder-Decoder` structure

`"Attention Is All You Need"` 中提到的encoder和decoder结构都是由6个编码/解码block组成.

Seq2seq model, `T2T`(Tensor2Tensor), [https://github.com/tensorflow/tensor2tensor](https://github.com/tensorflow/tensor2tensor).

    

## Encoder:
Encoding: Input -> Positional Encoding

    PE_{pos,i} = sin(pos/10000^{2i/d_model}) 当i=偶数
    PE_{pos,i} = cos(pos/10000^{2i/d_model}) 当i=奇数

基本框架:

    Self-Attention -> Feed Forward Neural Network


## Decoder:
基本框架:

    Self-Attention -> Encoder-Decoder Attention -> Feed Forward Neural Network

Decoder 的输出的形状[句子字的个数，字向量维度]

可以把最后的输出看成`多分类任务`, 也就是预测字的概率. 经过一个nn.Linear(字向量维度, 字典中字的个数)全连接层, 再通过Softmax输出每一个字的概率.

## Algorithms
### Self-Attention
又名`Scaled Dot-Product Attention`
Q, K, V 3个矩阵.
`Q`可以理解成词的"查询"向量; `K`可以理解词的"被查"向量; `V`可以理解成词的"内容"向量.
QKV长度均为64.
它们是通过3个不同的权值矩阵由嵌入向量X乘以三个不同的权值矩阵W^Q, W^K, W^K得到(三个矩阵尺寸也相同, 512x64).

#### Self-Attention

Self-Attention步骤:

1. Input转化为Embedding(运用实例: sentence_transformers[https://github.com/UKPLab/sentence-transformers](https://github.com/UKPLab/sentence-transformers))
2. Embedding -> q, k, v
3. 每个embedding计算score = q·k
4. socre归一化, 除以 $\sqrt{d_k}$ (稳定梯度)
5. score加softmax激活函数
6. 然后对softmax点乘v(Value)
7. 最后相加所有v, z=sum(v)

`Add`部分("Add & Layer normalization") - 采用[残差网络](https://zhuanlan.zhihu.com/p/42706477)中的`short-cut`结构 (解决深度学习中的退化问题)

    残差网络是由一系列残差块组成的;
    残差块有`resnet_v1`, `resnet_v2`;
    深度残差网络有很多旁路的支线将输入直接连到后面的层，使得后面的层可以直接学习残差，这些支路就叫做shortcut.


#### Multi-head Attention
相当于h(默认h=8)个不同的self-attention的集成(ensemble).

Multi-head Attention步骤:

1. 同意input转为embeddings X(R:直接用下一个output - 除了第一个encoder以外的所有encoders情况)
2. 分到8个heads($W_0^Q$~$W_7^Q$, $W_0^K$~$W_7^K$, $W_0^V$~$W_7^V$), 用加权矩阵乘以X or R(详见上self-attention部分)
3. 计算attention(z), 用前面得到的QKV矩阵
4. concatenate Zs, 乘以权重weight

`Add & Norm`

同self-attention一样, multi-head attention也加入了`short-cut`机制

把Multi-Head Attention的输入的a矩阵直接加上Multi-Head Attention的输出b矩阵(让network训练的更深)

得到的和为$\overline{b}$矩阵

之后layer normalization([图像展示见文中](https://zhuanlan.zhihu.com/p/403433120))归一化(加快训练速度, 加速收敛)

    Layer Normalization：是在一个句上的进行归一化.
    Batch Normalization：是把每句话的同一维度上的字看成一组做归一化.

得到$\overline{b}.

- 为什么使用`multi-head attention`: Multi-head attention允许模型共同关注来自不同位置的不同表示子空间的信息

### Feed Forward Neural Network
全称 `Position-wise Feed-Forward Networks`

把layer normalization输出$\overline{b}$, 经过两个全连接层`linear`, 中间有ReLU function.

In addition to attention sub-layers, each of the layers in our encoder and decoder contains a fully connected feed-forward network, which is applied to each position separately and identically. This consists of two linear transformations with a ReLU activation in between.

输入输出长度均为512, inner-layer长度为2048.

    nn.Linear(d_model, d_ff, bias=False)
    nn.ReLU()
    nn.Linear(d_ff, d_model, bias=False)


### Encoder-Decoder Attention
在解码器中, Transformer block比编码器中多了个encoder-cecoder attention

Q来自于decoder的上一个output, K和V都来自于encoder的output

由于在机器翻译中，解码过程是一个顺序操作的过程，也就是当解码第k个特征向量时，我们只能看到第k-1及其之前的解码结果, 所以这种情况下的multi-head attention叫做`Masked multi-head attention`

#### Masked multi-head attention
Masked Multi-Head Attention的结构和Multi-Head Attention的结构是一样的, 只是输入时被掩盖的数据

掩码mask是 -1e9 (负无穷)

    if mask is not None:
    scores = scores.masked_fill(mask == 0, -1e9)

    p_attn = scores.softmax(dim=-1)

"mask的目的是为了防止网络看到不该看到的内容"

[https://blog.csdn.net/zhaohongfei_358/article/details/125858248](https://blog.csdn.net/zhaohongfei_358/article/details/125858248)

    Transformer推理时是一个一个词预测, 而训练时会把所有的结果一次性给到Transformer, 但效果等同于一个一个词给, 而之所以可以达到该效果, 就是因为对tgt进行了掩码, 防止其看到后面的信息, 也就是不要让前面的字具备后面字的上下文信息.

因为神经网络的本质就是不断的进行矩阵相乘, 例如: `XW1W2W3⋯Wn→O`, X为输入,O为输出. 在这之中，X的第二个行向量本身就不会让你的第一个行向量的结果改变.