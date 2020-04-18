# Adam
Adam（Adaptive Moment Estimation） 优化算法实质上是将 Momentum 和 RMSprop 进行结合。Momentum 具有保持惯性的优点，RMSprop 实际上根据参数来调整学习率的衰减，体现环境感知能力。Adam 结合 Momentum 和 RMSprop，因此同时拥有惯性保持和环境感知这两个优点，而这两个优点也是缓解山谷震荡和鞍部停滞的关键动力。

简单地介绍了什么是 Adam 之后，我们再来看看如何使用 Adam 算法。

【计算过程】：
- 计算每个参数的梯度。
```math
dw = \frac{\partial L(w)}{w}
```
- 计算速度更新量以及修正后的速度更新量。
```math
v = \beta_1 v + (1 - \beta_1)dw \quad v' = \frac{v}{1 - \beta_1^t}
```
- 计算梯度累积平方以及修正后的梯度累积平方。
```math
Sdw = \beta_2 Sdw + (1 - \beta_2)dw^2 \quad Sdw' = \frac{Sdw}{1 - \beta_2^t}
```
- 更新参数。
```math
w = w - \frac{\eta}{\sqrt{Sdw' + \sigma}}v'
```

需要注意的上，t 的初始值为 1，也就是说第一次迭代时，t 就等于 1，因此修正后的 v 和 Sdw 分母不会为零。

【问】：为什么要对 v 和 Sdw 进行修正？

假设，v 的初始值为 0，β1 = 0.9，第一轮迭代后的 dw = 30，根据 `$v = \beta_1 v + (1 - \beta_1)dw$` 计算所得的 v = 0.9 * 0 + 0.1 * 30 = 3，这和 dw = 30 差距太大了，我们知道梯度下降算法每次参数更新使用的公式为 `$w = w - \eta dw$`，在不考虑学习率的前提下，Adam 公式计算的结果与梯度下降算法的结果相差接近 10 倍！为什么会这样？

因为 v 的初始值为零，也就是说在第一轮迭代中，v 的计算并没有历史信息可以借鉴，根据 v 的计算公式可知，这一轮对 v 的影响程度只有 (1 - β)，通常只有 10%，而 90% 都落在历史信息上。正是基于该原因，v 在前几轮迭代过程中没有足够的历史信息，从而值较小，影响收敛速度。因此，我们需要想办法消除这种影响。

办法就是添加修正项，除上 `$(1 - \beta^t)$`。v = 3，v / (1 - 0.9) = 30，添加修正项后 v 的取值大致接近 dw。

解释了 Adam 算法为什么需要进行修正，我们再来讨论 Adam 算法中的参数该如何进行设置。

吴恩达老师在[深度学习课程](https://github.com/fengdu78/deeplearning_ai_books/blob/master/markdown/lesson2-week2.md)中建议：
- `$\beta_1$`：0.9
- `$\beta_2$`：0.999
- `$\sigma$`：10e-8

学习率则需要通过调参操作来选择合适的参数。

在其他的资料中，dw 往往被称为矩阵的第一阶矩（first moment），`$dw^2$` 被称为矩阵的第二阶矩（second moment），这也是 Adam 名字的由来。

【代码实现】：
```python
def Adam(x, y, step=0.01, iter_count=500, batch_size=4, beta1=0.9, beta2=0.999):
    length, features = x.shape
    
    # 初始化 v 和 Sdw 以及整合数据集
    data = np.column_stack((x, np.ones((length, 1))))
    w = np.zeros((features + 1, 1))
    v, Sdw, eta = 0, 0, 10e-8
    start, end = 0, batch_size
    
    # 开始迭代，从 1 开始
    for i in range(1, iter_count + 1):
        # 计算梯度
        dw = np.sum((np.dot(data[start:end], w) - y[start:end]) * data[start:end], axis=0).reshape((features + 1, 1)) / length
        
        # 计算速度更新量以及修正后的速度更新量
        v = beta1 * v + (1 - beta1) * dw
        v_corrected = v / (1 - beta1**i)
        
        # 计算梯度累积平方以及修正后的梯度累积平方
        Sdw = beta2 * Sdw + (1 - beta2) * np.dot(dw.T, dw)
        Sdw_corrected = Sdw / (1 - beta2**i)
        
        # 更新参数
        w = w - (step / np.sqrt(eta + Sdw_corrected)) * v_corrected
        
        start = (start + batch_size) % length
        if start > length:
            start -= length
        end = (end + batch_size) % length
        if end > length:
            end -= length
    return w
```

完整代码可从 [传送门](https://github.com/clvsit/Machine-Learning-Note/blob/master/%E6%A8%A1%E5%9E%8B%E4%BC%98%E5%8C%96/Adam.ipynb) 中获得。

## 参考
- 吴恩达老师的深度学习课程：
- 《百面机器学习》
- Deep Learning 最优化方法之Adam：https://blog.csdn.net/bvl10101111/article/details/72616516