### 创建一个 Tensor
PyTorch 里的数据类型，主要为：
- 整数型 torch.uint8、torch.int32、torch.int64。其中 torch.int64 为默认的整数类型。
- 浮点型 torch.float16、torch.bfloat16、 torch.float32、torch.float64，其中 torch.float32 为默认的浮点数据类型。
- 布尔型 torch.bool
- PyTorch 里没有字符串类型，因为 Tensor 主要关注于数值计算，并不需要支持字符串类型。

Bool 类型在 PyTorch 里可以进行高效的索引选择，所以 PyTorch 支持 Bool 类型。比如 Bool 类型 tensor 进行索引操作示例如下：

```python
x = torch.tensor([1, 2, 3, 4, 5])
mask = x > 2  # 生成一个布尔掩码
print(mask)   # tensor([False, False,  True,  True,  True])

# 用布尔掩码选出大于 2 的值
filtered_x = x[mask]
print(filtered_x)  # tensor([3, 4, 5])

# 用布尔掩码选出大于 2 的值,并赋值为0
x[mask]=0
print(x) # tensor([1, 2, 0, 0, 0])
```

在创建 tensor 时，你还可以指定 tensor 的设备。如果你不指定，默认是在 CPU/内存上。如果你想创建一个 GPU/显存上的 tensor。可以通过把 device 关键字设定为“cuda”来指定。

```python
t_gpu = torch.tensor([1,2,3],device="cuda")
```

### Tensor 的属性
一个 tensor 有几个常用的关键属性，第一个是 tensor 的形状，第二个是 tensor 内元素的类型，第三个是 tensor 的设备，第四个是是否需要计算梯度。

```python
tensor = torch.rand(3,4)
print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")
print(f"Need grad: {tensor.requires_grad}")
```

### Tensor 的操作

#### 形状变换
reshape & permute(转置)， reshape 是按元素顺序重新组织维度，permute 会改变元素的顺序。
squeeze & unsqueeze

##### torch.squeeze(A，N)
torch.unsqueeze()函数的作用减少数组 A 指定位置 N 的维度，如果不指定位置参数 N，如果数组 A 的维度为（1，1，3）那么执行 torch.squeeze(A，1) 后 A 的维度变为 （1，3），中间的维度被删除。
注：
1. 如果指定的维度大于 1，那么将操作无效
2. 如果不指定维度 N，那么将删除所有维度为 1 的维度

```python
a=torch.randn(1,1,3)
print(a.shape)
b=torch.squeeze(a)
print(b.shape)
c=torch.squeeze(a,0)
print(c.shape)
d=torch.squeeze(a,1)
print(d.shape)
e=torch.squeeze(a,2)#如果去掉第三维，则数不够放了，所以直接保留
print(e.shape)

# 输出
torch.Size([1, 1, 3])
torch.Size([3])
torch.Size([1, 3])
torch.Size([1, 3])
torch.Size([1, 1, 3])
```

##### torch.unsqueeze(A，N)
torch.unsqueeze()函数的作用增加数组 A 指定位置 N 的维度，例如两行三列的数组 A 维度为(2，3)，那么这个数组就有三个位置可以增加维度，分别是（ [位置 0] 2，[位置 1] 3， [位置 2] ）或者是 （ [位置-3] 2，[位置-2] 3， [位置-1] ），如果执行 torch.unsqueeze(A，1)，数据的维度就变为了 （2，1，3）

```python
a=torch.randn(1,3)
print(a.shape)
b=torch.unsqueeze(a,0)
print(b.shape)
c=torch.unsqueeze(a,1)
print(c.shape)
d=torch.unsqueeze(a,2)
print(d.shape)

# 输出
torch.Size([1, 3])
torch.Size([1, 1, 3])
torch.Size([1, 1, 3])
torch.Size([1, 3, 1])
```

#### 数学运算

#### 统计函数
通过 tensor.sum()求和，通过 tensor.mean()求均值，通过 tensor.std()求标准差，通过 tensor.min()求最小值等。