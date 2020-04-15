# AdaGrad
在先前介绍的[梯度下降算法](https://blog.csdn.net/weixin_43378396/article/details/90723373)以及[动量方法](https://blog.csdn.net/weixin_43378396/article/details/90741645)都有一个共同点，即对于每一个参数都用相同的学习率（步长）进行更新。但是在实际应用中，各参数的重要性肯定是不同的，所以对于不同的参数要进行动态调整，采取不同的学习率，让目标函数能够更快地收敛。

本篇博客主要借鉴 [深度学习优化方法-AdaGrad](https://blog.csdn.net/program_developer/article/details/80756008) 以及《百面机器学习》，若构成侵权则立即删除。

【做法】：将每一个参数的每一次迭代的梯度取平方，然后累加并开方得到 r，最后用全局学习率除以 r，作为学习率的动态更新。

【计算过程】：令 α 表示全局学习率，r 为梯度累积变量，初始值为 0。
- 单独计算每一个参数在当前位置的梯度。
```math
g = \frac{\partial L(w)}{\partial w_i}
```
- 累积平方梯度，一般来说 g 是一个向量，而向量的平方通常写为 `$g^Tg$`。
```math
r = r + g^2 \quad \text{或} \quad r = r + g^Tg
```
- 更新参数。
```math
w = w - \frac{\alpha}{\sqrt{r}} g
```

上述式子存在一个问题，r 在计算过程中有可能变为 0，在代码中分母为零通常都会报错，因此我们需要想办法让分母不为零，同时不会影响到参数的更新。

怎么做呢？我们可以在分母上加一个极小的常数 `$\sigma$`，因为取值极小，即使开方后仍然不会影响参数的更新。通常，`$\sigma$` 大约设置为 10 的 -7 次方。

```math
w = w - \frac{\alpha}{\sigma + \sqrt{r}} g
```

从 AdaGrad 算法的计算过程中可以看出，随着不断地迭代，r 的值会越来越大（梯度的平方为正值），那么在每一轮迭代中学习率会越来越小，也就是说当前位置的梯度对参数的影响也越来越小。简单地讲，AdaGrad 算法在初始时鼓励收敛，随着迭代的深入慢慢变成惩罚收敛，速度也越来越慢。

【代码实现】：
```python
def AdaGrad(x, y, step=0.01, iter_count=500, batch_size=4):
    length, features = x.shape
    data = np.column_stack((x, np.ones((length, 1))))
    w = np.zeros((features + 1, 1))
    r, eta = 0, 10e-7
    start, end = 0, batch_size
    for i in range(iter_count):
        # 计算梯度
        dw = np.sum((np.dot(data[start:end], w) - y[start:end]) * data[start:end], axis=0) / length        
        # 计算梯度累积变量
        r = r + np.dot(dw, dw)
        # 更新参数
        w = w - (step / (eta + np.sqrt(r))) * dw.reshape((features + 1, 1))
        
        start = (start + batch_size) % length
        if start > length:
            start -= length
        end = (end + batch_size) % length
        if end > length:
            end -= length
    return w
    

print(AdaGrad(x, y, step=1, iter_count=1000))
# 输出：
array([[5.19133285],
       [1.35955132]])
```

【问题】：从训练开始时积累梯度平方会导致有效学习率过早和过量的减小。这也是为什么在上述代码示例的最后部分使用的全局学习率为 1。如果把 step 设置为 0.1 会发生什么？
```python
print(AdaGrad(x, y, step=0.1, iter_count=1000))
# 输出：
array([[3.37157325],
       [0.6519457 ]])
```
可以看到迭代 1000 次还没有收敛到最优点附近，且距离最优点还有一段距离。
```python
print(AdaGrad(x, y, step=0.1, iter_count=3000))
# 输出：
array([[4.72572017],
       [0.91424582]])
```
迭代 3000 次后距离最优点更近了一些。

为了避免这种情况的发生，我们可以在迭代一定次数后再开始累加 r。
```python
def AdaGrad(x, y, step=0.01, iter_count=500, step_count=100, batch_size=4):
    length, features = x.shape
    data = np.column_stack((x, np.ones((length, 1))))
    w = np.zeros((features + 1, 1))
    r, eta = 0, 10e-7
    start, end = 0, batch_size
    for i in range(iter_count):
        # 计算梯度
        dw = np.sum((np.dot(data[start:end], w) - y[start:end]) * data[start:end], axis=0) / length
        # 大于 step_count 时，更新梯度累积平方
        if i > step_count:
            r = r + np.dot(dw, dw)
            w = w - (step / (eta + np.sqrt(r))) * dw.reshape((features + 1, 1))
        else:
            w -= step * dw.reshape((features + 1, 1))        
        start = (start + batch_size) % length
        if start > length:
            start -= length
        end = (end + batch_size) % length
        if end > length:
            end -= length
    return w
```

此时，我们再调用 AdaGrad，就不会出现上述的情况了。
```python
print(AdaGrad(x, y, iter_count=500))
# 输出：
array([[5.24748173],
       [1.06459711]])
```
当然，我们也可以通过当前位置的梯度取值来进行判断。
```python
def AdaGrad(x, y, step=0.01, iter_count=500, step_threshold=30, batch_size=4):
    length, features = x.shape
    data = np.column_stack((x, np.ones((length, 1))))
    w = np.zeros((features + 1, 1))
    r, eta = 0, 10e-7
    start, end = 0, batch_size
    for i in range(iter_count):
        dw = np.sum((np.dot(data[start:end], w) - y[start:end]) * data[start:end], axis=0) / length
        dw2 = np.dot(dw, dw)
        if dw2 < step_threshold:
            r = r + dw2
            w = w - (step / (eta + np.sqrt(r))) * dw.reshape((features + 1, 1))
        else:
            w -= step * dw.reshape((features + 1, 1))        
        start = (start + batch_size) % length
        if start > length:
            start -= length
        end = (end + batch_size) % length
        if end > length:
            end -= length
    return w

    
print(AdaGrad(x, y, iter_count=500))
# 输出：
array([[5.12585752],
       [0.95310592]])
```

所有代码都可从 [传送门]() 内获得。

## 参考
- 《百面机器学习》
- 深度学习优化方法-AdaGrad：https://blog.csdn.net/program_developer/article/details/80756008
- Deep Learning 最优化方法之AdaGrad：https://blog.csdn.net/bvl10101111/article/details/72616097