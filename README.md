# Pytorch API学习及逻辑调用

tensor.item()

```python
# 对tensor取数值，返回张量元素的值，对于tensor内有一个数值才可调用此方法，多个元素报错
tensor.item()	
# 和item方法很类似，但numpy方法可以将tensor转换成ndarray格式，不限制tensor内元素个数
tensor.numpy()
```

tensor.eq()

```python
# tensor.eq()方法等同于tf.equal(),都是对tensor内的数据按元素对比（指定dim、axis），并且输出对应dim的布尔类型数据
torch.eq(tensor_a, tensor_b)	# --> def eq(self: Tensor, other: Tensor, *, out: Optional[Tensor]=None)
tensor_a.eq(tensor_b)
# tensor.equal()方法只输出两个tensor对比，是否相等
torch.equal(tensor_a, tensor_b)	# --> _bool:
tensor_a.equal(tensor_b)
```

tensor.numel()

```python
# tensor.numel()方法返回tensor占用内存大小
tensor_a.numel()
```

torch.set_default_tensor_type(torch.FloatTensor/torch.DoubleTensor)

```python
# torch.set_default_tensor_type()设定torch.Tensor()默认格式（如果不设置默认为FLostTensor）
torch.set_default_tensor_type()
```

torch.full([shape], num)

```python
# tensor.full()方法生成一个给定shape的全num型数组，给定输入需要是float不能是int
x = torch.full((3, 3), 5.3)
x
tensor([[5.3000, 5.3000, 5.3000],
        [5.3000, 5.3000, 5.3000],
        [5.3000, 5.3000, 5.3000]])
```

torch.linspace(start, end, steps=None)

```python
# torch.linspace(start, end, steps=None)方法生成等分的数组
x = torch.linspace(0, 3, steps=10)
x
tensor([0.0000, 0.3333, 0.6667, 1.0000, 1.3333, 1.6667, 2.0000, 2.3333, 2.6667,
        3.0000])
```

torch.logspace(start, end, steps=None)

```python
# torch.logspace(start: Number, end: Number, steps: _int=100, base: _float=10.0)生成以base为底的指数等分数组
x = torch.logspace(0, 1, steps=10, base=2)
x
tensor([1.0000, 1.0801, 1.1665, 1.2599, 1.3608, 1.4697, 1.5874, 1.7145, 1.8517,
        2.0000])
```

torch.randperm(Number)

```python
# randperm(Number)生成基于number大小的随机乱序数，相当于np.random.permutation(),用于数据维度shuffle
x = torch.randperm(10)
x
tensor([3, 6, 7, 5, 9, 2, 4, 0, 8, 1])
```

tensor[0, :, :, :]切片

```python
tensor[0, :, :, :] <=> tensor[0, ...]
tensor[...] <=> tensor[:, :, :, :]
tensor[..., :2] <=> tensor[:, :, :, :2]
```

tensor.view()和tensor.reshape()

```python
# 在pytorch0.3之前都是使用view函数对数组进行维度变换，但在0.4版本中为了保持与numpy一致，增加了reshape函数,两种函数用法完全相同
tensor_a.view(3, 3) <=> tensor_a.reshape(3,3)
```

tensor.unsqueeze()

```python
# dim -> 0   1   2   3
#		-4  -3  -2  -1
# unsqueeze用于在axis增加维度
x = torch.rand((2, 3, 28, 28))	# axis = 4的数据
y = x.unsqueeze(1)	# 表示在axis=1的维度之前插入一个维度（pytorch常用dim，tensorflow常用axis，两者表达相同）
y.shape
torch.Size([2, 1, 3, 28, 28])
```

torch.cat([tensor_a, tensor_b], dim=Number)

```python
# cat方法与tf.concat(), np.concatante()用法一致都用于在原有维度上进行拼接，不会增加维度
x = torch.rand((1,3,3))
y = torch.rand((1,3,3))
torch.cat([x, y], dim=0)
-> tensor([[[0.2267, 0.5636, 0.8269],
         [0.8434, 0.2561, 0.7266],
         [0.0459, 0.9119, 0.7341]],
        [[0.4417, 0.6696, 0.7375],
         [0.3352, 0.4405, 0.3100],
         [0.2330, 0.1244, 0.1808]]])
torch.cat([x, y], dim=0).shape
-> torch.Size([2, 3, 3])
```

torch.stack([tensor_a, tensor], dim=Number)  



```python
# stack方法与tf.stack(),np.stack()用法一致，用于将两个张量在一个新的维度拼接，会增加维度
# dim位置表示新增加的维度在什么位置
# stack位置之后的数组大小一定要相同，发否则报错
x = torch.rand((1,3,3))
y = torch.rand((1,3,3))
torch.stack([x, y], dim=1).shape
-> torch.Size([1, 2, 3, 3])
torch.stack([x, y], dim=0).shape
-> torch.Size([2, 1, 3, 3])
```

tensor.split(Number, dim=Number)

```python
# 方法按照给定dim将原张量按指定维度拆分成指定长度tensor，与tf.split()用法相同
# 示例为将z分为长度为2，2，1三个长度的张量
z = torch.rand((5,3,3))
a,b,c = z.split([2,2,1], dim=0)
a
tensor([[[0.6907, 0.1767, 0.2757],
         [0.0949, 0.6095, 0.0204],
         [0.3591, 0.7025, 0.1655]],
        [[0.8722, 0.5272, 0.8849],
         [0.5268, 0.4994, 0.4219],
         [0.1545, 0.1390, 0.5757]]])
b
tensor([[[0.4904, 0.2658, 0.3442],
         [0.5530, 0.2156, 0.0235],
         [0.1533, 0.5226, 0.0291]],
        [[0.1012, 0.4376, 0.9082],
         [0.8089, 0.9455, 0.0567],
         [0.8221, 0.6370, 0.9685]]])
c
tensor([[[0.1732, 0.1510, 0.9870],
         [0.7290, 0.7948, 0.9145],
         [0.9889, 0.4717, 0.8628]]])

```

tensor.chunk(num, dim=num)

```python
# 方法将原tensor按维度以给定数量拆分
z = torch.rand((6,3,3))
a, b = z.chunk(2, dim=0)
a.shape
-> torch.Size([3, 3, 3])
b.shape
-> torch.Size([3, 3, 3])
```

