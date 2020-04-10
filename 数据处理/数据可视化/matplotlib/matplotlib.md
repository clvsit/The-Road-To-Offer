matplotlib 是一个用于创建出版质量图表的桌面绘图包（主要是2D方面）。该项目是由 John Hunter 于 2002 年启动的，其目的是为 Python 构建一个 MATLAB 式的绘图接口。matplotlib 和 IPython 社区进行合作，简化了从 IPython shell（包括现在的Jupyter notebook）进行交互式绘图。matplotlib 支持各种操作系统上许多不同的 GUI 后端，而且还能将图片导出为各种常见的矢量（vector）和光栅（raster）图：PDF、SVG、JPG、PNG、BMP、GIF 等。

## 引入约定
matplotlib 的通常引入约定：
```
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
```

【注意】：使用 Jupyter notebook 有一点不同，即每个 Cell 重新执行后，图形会被重置。因此，对于复杂的图形，我们必须将所有的绘图命令存在一个 Cell 里。


## Figures 和 Subplot
matplotlib 的图像都位于 Figure 对象中，我们可以使用 plt.figure 创建一个新的 Figure。
```
fig = plt.figure()
```

需要注意的是，创建 fig 只是一个空窗口，我们不能通过空窗口进行绘图，必须调用 add_subplot() 函数创建一个或多个 subplot。
```
ax1 = fig.add_subplot(2, 2, 1)
```

上述代码表示创建一个 2 x 2 的图像（AxesSubplot 对象），并选中图像中的第 1 个（编号从左上角 1 开始）。此时，如果再把后面两个 subplot 也创建出来，最终得到的图像如下图所示。

![add_subplot()](https://markdown-picture-clvsit.oss-cn-hangzhou.aliyuncs.com/matplotlib%20add_subplot.jpg)

如果此时执行一条绘图命令，例如 `plt.plot([1.5, 3.5, -2, 1.6])`，matplotlib 会在最后一个用过的 subplot（如果没有则创建一个）上进行绘制，隐藏创建 figure 和 subplot 的过程。因此，如果我们执行下列命令，会得到如下图所示的结果。
```
plt.plot(np.random.randn(50).cumsum(), "k--")
```

![add_subplot() 2](https://markdown-picture-clvsit.oss-cn-hangzhou.aliyuncs.com/matplotlib%20add_subplot2.jpg)

“k--” 是一个线型选项，用于告知 matplotlib 绘制黑色虚线图。

上面代码中通过 `fig.add_subplot()` 创建的对象为 AxesSubplot，我们可以直接调用 AxesSubplot 的实例方法在它的区域范围内画图。
```
ax1.hist(np.random.randn(100), bins=20, color="k", alpha=0.3)
ax2.scatter(np.arange(30), np.arange(30) + 3 * np.random.randn(30))
```

![add_subplot() 3](https://markdown-picture-clvsit.oss-cn-hangzhou.aliyuncs.com/matplotlib%20add_subplot3.jpg)

除了单独创建每个 AxesSubplot 对象外，matplotlib 提供一个更为方便的方法 `plt.subplots()`，该方法允许创建一个新的 Figure，并返回一个含有以创建的 AxesSubplot 对象的 Numpy 数组。
```
fig, axes = plt.subplots(2, 3)
axes
```

![subplots()](https://markdown-picture-clvsit.oss-cn-hangzhou.aliyuncs.com/matplotlib%20subplots.jpg)

我们可以轻松地对 axes 数组进行索引，就好像是一个二维数组一样，例如 axes[0, 1]。

【subplots() 参数表】：

参数 | 说明
---|---
nrows | subplot 的行数
ncols | subplot 的列数
sharex | 所有 subplot 应该使用相同的 X 轴刻度（调节 xlim 将会影响所有 subplot）
sharey | 所有 subplot 应该私用相同的 Y 轴刻度（调节 ylim 将会影响所有 subplot）
subplot_kw | 用于创建各 subplot 的关键字字典
**fig_kw | 创建 figure 时的其他关键字

## 调整 Subplot 周围的间距
默认情况下，matplotlib 会在 subplot 外围留下一定的边距，并在 subplot 之间留下一定的间距。间距跟图像的高度和宽度有关，因此，如果调整了图像的大小，间距也会自动调整。我们可以使用 `subplots_adjust()` 方法修改间距。

【语法】：
```python
subplots_adjust(left=None, bottom=None, right=None, top=NOne, wspace=None, hspace=None)
```
其中，wspace 和 hspace 用于控制宽度和高度的百分比，可以用于调整 subplot 之间的间距。

【示例】：
```python
fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
for i in range(2):
    for j in range(2):
        axes[i, j].hist(np.random.randn(500), bins=50, color="k", alpha=0.5)
plt.subplots_adjust(wspace=0, hspace=0)
```

![subplots_adjust()](https://markdown-picture-clvsit.oss-cn-hangzhou.aliyuncs.com/matplotlib%20subplots_adjust.jpg)

上述代码将 subplot 之间的间距调整为了 0，因此 4 个 AxesSubplot 对象紧紧挨在一起。这会导致轴变迁重叠了，matplotlib 不会检查标签是否重叠，因此对于这种情况，我们需要自行设定刻度位置和刻度标签。下面就来讲讲刻度和标签。

## 刻度和标签
在设置刻度、标签前，我们先创建一个简单的图像，并绘制一段随机漫步。

【示例】：
```python
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(np.random.randn(1000).cumsum())
```

![刻度、标签和图例](https://markdown-picture-clvsit.oss-cn-hangzhou.aliyuncs.com/matplotlib%20%E5%88%BB%E5%BA%A6%E6%A0%87%E7%AD%BE%E5%92%8C%E5%9B%BE%E4%BE%8B.jpg)

- `set_xticks()`：设置刻度的具体数值。
- `set_xticklabels()`：设置刻度的标签值。

```python
ticks = ax.set_xticks([0, 250, 500, 750, 1000])
labels = ax.set_xticklabels(['one', 'two', 'three', 'four', 'five'], rotation=30, fontsize='small')
```

![set_xticks()、set_xticklabels()](https://markdown-picture-clvsit.oss-cn-hangzhou.aliyuncs.com/matplotlib%20set_xticks()%20set_xticklabels()jpg.jpg)
- rotation：顺时针旋转；
- fontsize：设置标签字体大小。

Y 轴的修改方式与此类似，只需将上述代码中的 x 替换为 y 即可。此外，我们也可以批量设定绘图选项。
```python
props = {
    "title": "My first matplotlib plot",
    "xlabel": "Stages"
}
ax.set(**props)
```

## 颜色、标记和线型
matplotlib 的 `plot()` 函数接受一组 X 和 Y 坐标，还可以接受一个表示颜色、标记和线型的字符串缩写。

【示例】：根据 x 和 y 绘制绿色虚线。
```
ax.plot(x, y, "go--")
```

![颜色、标记和线型1](https://markdown-picture-clvsit.oss-cn-hangzhou.aliyuncs.com/matplotlib%20%E9%A2%9C%E8%89%B2%E6%A0%87%E8%AE%B0%E5%92%8C%E7%BA%BF%E5%9E%8B1.jpg)

除了在字符串中同时指定颜色、标记和线型的方式外，还可以通过单独设定颜色、标记和线型的方式来设置。
```python
ax.plot(x, y, linestyle="--", color="g", marker="o")
```
- 颜色：可以使用颜色缩写，或通过十六进制的颜色标识码。
- 线型：可以通过查看 plot 的文档字符串来找到所有线型 `plt.plot?`。
- 标记：

【注意】：标记和线型必须放在颜色后面。

线型名称 | 线型写法 
---|:---:
solid | -
dashed | --
dashdot | -.
dotted | :

因为 matplotlib 可以创建连续线图，在线型图 plot 中，点之间按线性的方式插值，我们可以通过 drawstyle 参数进行修改。
```python
data = np.random.randn(30).cumsum()
plt.plot(data, "k--", label="default")
plt.plot(data, "k-", drawstyle="steps-post", label="steps-post")
plt.legend(loc="best")
```

![颜色标记和线型2](https://markdown-picture-clvsit.oss-cn-hangzhou.aliyuncs.com/matplotlib%20%E9%A2%9C%E8%89%B2%E6%A0%87%E8%AE%B0%E5%92%8C%E7%BA%BF%E5%9E%8B2.jpg)

## 图例
图例（legend）是另一种用于标识图表元素的重要工具。添加图例的方式有多种，最简单的方式是在添加 subplot 的时候传入 label 参数。

【示例】：
```python
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(np.random.randn(1000).cumsum(), "k", label="one")
ax.plot(np.random.randn(1000).cumsum(), "k--", label="two")
ax.plot(np.random.randn(1000).cumsum(), "k.", label="three")
ax.legend(loc="best")
```

![图例 1](https://markdown-picture-clvsit.oss-cn-hangzhou.aliyuncs.com/matplotlib%20%E5%9B%BE%E4%BE%8B1.jpg)

【loc 位置参数值】：
```
Location String   Location Code
===============   =============
'best'            0
'upper right'     1
'upper left'      2
'lower left'      3
'lower right'     4
'right'           5
'center left'     6
'center right'    7
'lower center'    8
'upper center'    9
'center'          10
```

## 将图表保存到文件
利用 `plt.savefig()` 可以将当前图表保存到文件。

【示例】：
```python
plt.savefig("figpath.svg")
```

保存图表时还可以设置 dpi（分辨率）和 bbox\_inches（剪除当前图表周围的空白部分）。
```python
plt.savefig("figpath.png", dpi=400, bbox_inches="tight")
```

【参数列表】：
- fname：含有文件路径的字符串或 Python 的文件型对象。
- dpi：图像分辨率，默认为 100。
- facecolor、edgecolor：图像的背景色，默认为“w”（白色）。
- format：显示设置文件格式，不写则从 fname 中的文件扩展名进行推断获得。
- bbox_inches：图表需要保存的部分，如果设置为 tight，则将尝试剪除图表周围的空白部分。

【其他】：savefig() 方法并非一定要写入磁盘，也可以写入任何文件型的对象，比如 BytesIO。
```python
from io import BytesIO

buffer = BytesIO()
plt.savefig(buffer)
plot_data = buffer.getvalue()
```

## matplotlib 配置
matplotlib 自带默认配置信息，但所有默认行为（图像大小、subplot 边距、配色方案、字体大小、网格类型等等）都能通过一组全局参数进行自定义。

【示例】：使用 rc 进行配置。
```
plt.rc("figure", figsize=(10, 10))
```
上述代码将全局的图像默认大小设置为 10 x 10。

【语法】：
```
plt.rc(object, **kwargs)
```
- object：自定义的对象，例如 figure、axes、xtick、ytick、grid、legend 等等。
- kwargs：待设置的关键字参数。

【示例】：配置字体。
```python
font_options = {
    "family": "monospace",
    "weight": "bold",
    "size": "small"
}
plt.rc("font", **font_options)
```

要了解全部的自定义选项，可以查阅 matplotlib 的配置文件 matplotlibrc（位于 matplotlib/mpl-data 目录中）。如果对该文件进行了自定义，并将其放在自己的 .matplotlibrc 目录中，则每次使用 matplotlib 时会自动加载该文件。