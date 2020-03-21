# 数据准备
> 特征选择是困难耗时的，也需要对需求的理解和专业知识的掌握。在机器学习的应用开发中，最基础的是特征工程。—— 吴恩达

吴恩达老师的这句话充分概括了特征工程的复杂度及其重要性。Kagglers 比赛和天池比赛的冠军其实在比赛中并没有用到很高深的算法，大多数都是在特征工程这个环节做了出色的工作，然后使用一些常见的算法，如逻辑回归，就能得到性能出色的模型。因此**特征工程是建立高准确度机器学习算法的基础**。所以也有机器学习方面的专家这样来概括机器学习：“使用正确的特征来构建正确的模型，以完成既定的任务”。实际上，在深度学习领域，训练数据的质量高低对模型的性能产生极大的影响，甚至演变成数据越多、质量越高、模型性能越好的局面。

## 数据预处理
数据预处理需要根据数据本身的特性进行，有不同的格式和不同的要求，有缺失值的要填，有无效数据的要剔，有冗余维的要选，这些步骤都和数据本身的特性紧密相关。数据预处理大致分为三个步骤：
- 数据的准备
- 数据的转换
- 数据的输出

数据处理是系统工程的基本环节，也是提高算法准确度的有效手段。因此，为了提高算法模型的准确度，在机器学习中也要根据算法的特征和数据的特征对数据进行转换。本章将利用 scikit-learn 来转换数据，以便我们将处理后的数据应用到算法中，这样也可以提高算法模型的准确度。

【数据转换方法】：
- 调整数据尺度（Rescale Data）
- 正态化数据（Standardize Data）
- 标准化数据（Normalize Data）
- 二值数据（Binarize Data）

### 为什么需要数据预处理
在开始机器学习的模型训练之前，需要对数据进行预处理，这是一个必需的过程。但是需要注意的是，不同的算法对数据有不同的假定，需要按照不同的方式转换数据。当然，如果按照算法的规则来准备数据，算法就可以产生一个准确度比较高的模型。

### 格式化数据
本章会介绍四种不同的方式来格式化数据。这四种方法都按照统一的流程来处理数据：
- 导入数据
- 按照算法的输入和输出整理数据
- 格式化输入数据
- 总结显示数据的变化

scikit-learn 提供了两种标准的格式化数据的方法，每一种方法都有适用的算法。利用这两种方法整理的数据，可以直接用来训练算法模型。在 scikit-learn 的说明文档中，也有对这两种方法的详细说明。
- 适合和多重变换（Fit and Multiple Transform）
- 适合和变换组合（Combined Fit-and-Transform）

推荐优先选择适合和多重变化（Fit and Multiple Transform）方法。首先调用 fit() 函数来准备数据转换的参数，然后调用 transform() 函数来做数据的预处理。适合和变换组合（Combined Fit-and-Transform）对绘图或汇总处理具有非常好的效果。

### 调整数据尺度
如果数据的各个属性按照不同的方式度量数据，那么通过调整数据的尺度让所有的属性按照相同的尺度来度量数据，就会给机器学习的算法模型训练带来极大的方便。这个方法通常会将数据的所有属性标准化，并将数据转换成 0 和 1 之间的值，这对于梯度下降等算法是非常有用的，对于回归算法、神经网络算法和 K 近邻算法的准确度提高也起到很重要的作用。

在统计学中，按照对事物描述的精确度，对所采用的尺度从低级到高级分成四个层次：
- 定类尺度：对事物类别属性的一种测度，按照事物的属性进行分组或分类。
- 定序尺度：对事物之间的等级或顺序的一种测度，可以比较优劣或排序。
- 定距尺度：对事物类别或次序之间间距的测量，不仅能将事物区分为不同的类型并进行排序，而且可以准确地指出类别之间的差距。
- 定比尺度：对事物类别或次序之间间距的测量，其他内容同定距尺度，差别在于定比尺度有一个固定的绝对“零”点。

在 scikit-learn 中，可以通过 MinMaxScaler 类来调整数据尺度，将不同计量单位的数据统一成相同的尺度，利于对事物的分类或分组。实际上，MinMaxScaler 是将属性缩放到一个指定范围，或者对数据进行标准化并将数据都聚集到 0 附近，方差为 1。数据尺度的统一，通常能够提高与距离相关的算法的准确度（如 K 近邻算法）。

```python
# 调整数据尺度（0..）
from pandas import read_csv
from numpy import set_printoptions
from sklearn.preprocessing import MinMaxScaler

# 导入数据
filename = "./data/pima_data.csv"
names = ["preg", "plas", "pres", "skin", "test", "mass", "pedi", "age", "class"]
data = read_csv(filename, names=names)

# 将数据分为输入数据和输出数据
array = data.values
X = array[:, 0:8]
Y = array[:, 8]
transformer = MinMaxScaler(feature_range=(0, 1))

# 数据转换
newX = transformer.fit_transform(X)

# 设定数据的打印格式
set_printoptions(precision=3)
print(newX)
```

    [[0.353 0.744 0.59  ... 0.501 0.234 0.483]
     [0.059 0.427 0.541 ... 0.396 0.117 0.167]
     [0.471 0.92  0.525 ... 0.347 0.254 0.183]
     ...
     [0.294 0.608 0.59  ... 0.39  0.071 0.15 ]
     [0.059 0.633 0.492 ... 0.449 0.116 0.433]
     [0.059 0.467 0.574 ... 0.453 0.101 0.033]]
    
调整完数据的尺度之后，所有的数据都按照设定的分布区间进行分布。

### 正态化数据
正态化数据（Standardize Data）是有效的处理符合高斯分布的数据的手段，输出结果以 0 为中位数，方差为 1，并作为假定数据符合高斯分布的算法的输入。这些算法有线性回归、逻辑回归和线性判别分析等。在这里可以通过 scikit-learn 提供的 StandardScaler 类来进行正态化数据处理。

```python
from sklearn.preprocessing import StandardScaler
transformer = StandardScaler().fit(X)

# 数据转换
newX = transformer.transform(X)
print(newX)
```

    [[ 0.64   0.848  0.15  ...  0.204  0.468  1.426]
     [-0.845 -1.123 -0.161 ... -0.684 -0.365 -0.191]
     [ 1.234  1.944 -0.264 ... -1.103  0.604 -0.106]
     ...
     [ 0.343  0.003  0.15  ... -0.735 -0.685 -0.276]
     [-0.845  0.16  -0.471 ... -0.24  -0.371  1.171]
     [-0.845 -0.873  0.046 ... -0.202 -0.474 -0.871]]
    
### 标准化数据
标准化数据（Normalize Data）处理是将每一行的数据的距离处理成 1（在线性代数中矢量距离为 1）的数据又叫作“归一元”处理，适合处理稀疏数据（具有喝多为 0 的数据）。归一元处理的数据对使用权重输入的神经网络和使用距离的 K 近邻算法的准确度的提升有显著作用。使用 scikit-learn 中的 Normalizer 类实现。

```python
# 标准化数据
from sklearn.preprocessing import Normalizer
transformer = Normalizer().fit(X)

# 数据转换
newX = transformer.transform(X)

# 设定数据的打印格式
set_printoptions(precision=3)
print(newX)
```

    [[0.034 0.828 0.403 ... 0.188 0.004 0.28 ]
     [0.008 0.716 0.556 ... 0.224 0.003 0.261]
     [0.04  0.924 0.323 ... 0.118 0.003 0.162]
     ...
     [0.027 0.651 0.388 ... 0.141 0.001 0.161]
     [0.007 0.838 0.399 ... 0.2   0.002 0.313]
     [0.008 0.736 0.554 ... 0.241 0.002 0.182]]
    
### 二值数据
二值数据（Binarize Data）是使用值将数据转化为二值，大于阈值设置为 1，小于阈值设置为 0。这个过程被叫作二分数据或阈值转换。在生成明确值或特征工程增加属性的时候使用，使用 scikit-learn 中的 Binarizer 类实现。

```python
# e二值数据
from sklearn.preprocessing import Binarizer
transformer = Binarizer(threshold=0.0).fit(X)

# 数据转换
newX = transformer.transform(X)

# 设定数据的打印格式
set_printoptions(precision=3)
print(newX)
```

    [[1. 1. 1. ... 1. 1. 1.]
     [1. 1. 1. ... 1. 1. 1.]
     [1. 1. 1. ... 1. 1. 1.]
     ...
     [1. 1. 1. ... 1. 1. 1.]
     [1. 1. 1. ... 1. 1. 1.]
     [1. 1. 1. ... 1. 1. 1.]]


## 数据特征选定
在做数据挖掘和数据分析时，数据是所有问题的基础，并且会影响整个项目的进程。相较于使用一些复杂的算法，灵活地处理数据经常会取到意想不到的效果。而处理数据不可避免地会使用到特征工程。那么特征工程是什么呢？有这么一句话在业界广为流传：**数据和特征决定了机器学习的上限，而模型和算法只是逼近这个上限而已**。因此，**特征工程的本质就是一项工程活动，目的是最大限度地从原始数据中提取合适的特征，以供算法和模型使用**。特征处理是特征工程的核心部分，scikit-learn 提供了较为完整的特征处理方法，包括数据预处理、特征选择、降维等。

本章将会学习通过 scikit-learn 来自动选择用于建立机器学习模型的数据特征的方法。本章将会介绍以下四个数据特征选择的方法：
- 单变量特征选定。
- 递归特征消除。
- 主要成分分析。
- 特征的重要性。

## 特征选定
特征选定是一个流程，能够选择有助于提高预测结果准确度的特征数据，或者有助于发现我们感兴趣的输出结果的特征数据。如果数据中包含无关的特征属性，会降低算法的准确度，对预测新数据造成干扰，尤其是线性相关算法（如线性回归算法和逻辑回归算法）。因此，在开始建立模型之前，执行特征选定有助于：
- **降低数据的拟合度**：较少的冗余数据，会使算法得出结论的机会更大。
- **提高算法精度**：较少的误导数据，能够提高算法的准确度。
- **减少训练时间**：越少的数据，训练模型所需要的时间越少。

可以在 scikit-learn 的特征选定文档中查看更多的信息（http://scikit-learn.org/stable/modules/feature_selection.html）。

## 单变量特征选定
统计分析可以用来分析选择对结果影响最大的数据特征。在 scikit-learn 中提供了 SelectKBest 类，可以使用一系列统计方法来选定数据特征，是对卡方检验的实现。经典的卡方检验是检验定性自变量对定性因变量的相关性的方法。假设自变量有 N 种取值，因变量有 M 种取值，考虑自变量等于 i 且因变量等于 j 的样本频数的观察值与期望值的差距，构建统计量。卡方检验就是统计样本的实际观测值与理论推断值之间的偏离程度，偏离程度决定了卡方值的大小，卡方值越大，越不符合；卡方值越小，偏差越小，越趋于符合；若两个值完全相等，卡方值就为 0，表明理论值完全符合。


```python
# 通过卡方检验选定数据特征
from pandas import read_csv
from numpy import set_printoptions
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

# 导入数据
filename = "./data/pima_data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names=names)
# 将数据分为输入数据和输出结果
array = data.values
x = array[:, 0:8]
y = array[:, 8]
# 特征选定
test = SelectKBest(score_func=chi2, k=4)
fit = test.fit(x, y)
set_printoptions(precision=3)
print(fit.scores_)
features = fit.transform(x)
print(features)
```

    [ 111.52  1411.887   17.605   53.108 2175.565  127.669    5.393  181.304]
    [[148.    0.   33.6  50. ]
     [ 85.    0.   26.6  31. ]
     [183.    0.   23.3  32. ]
     ...
     [121.  112.   26.2  30. ]
     [126.    0.   30.1  47. ]
     [ 93.    0.   30.4  23. ]]
    

通过设置 SelectKBest 的 score_func 参数，SelectKBest 不仅可以执行卡方检验来选择数据特征，还可以通过相关系数、互信息法等统计方法来选定数据特征。

## 递归特征消除
递归特征消除（RFE）使用一个基模型来进行多轮训练，每轮训练后消除若干权值系数的特征，再基于新的特征集进行下一轮训练。通过每一个基模型的精度，找到对最终的而预测结果影响最大的数据特征。


```python
# 通过递归消除来选定特征
from pandas import read_csv
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
# 导入数据
filename = "./data/pima_data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names=names)
# 将数据分为输入数据和输出结果
array = data.values
x = array[:, 0:8]
y = array[:, 8]
# 特征选定
model = LogisticRegression()
rfe = RFE(model, 3)
fit = rfe.fit(x, y)
print("特征个数：")
print(fit.n_features_)
print("被选定的特征：")
print(fit.support_)
print("特征排名：")
print(fit.ranking_)
```

    特征个数：
    3
    被选定的特征：
    [ True False False False False  True  True False]
    特征排名：
    [1 2 3 5 6 1 1 4]
    

执行后，可以看到 RFE 选定了 preg、mass 和 pedi 三个数据特征，它们在 support_ 中被标记为 True，在 ranking_ 中被标记为 1。

## 主要成分分析
主要成分分析（PCA）是使用线性代数来转换压缩数据，通常被称作数据降维。常见的降维方法除了主要成分分析（PCA），还有线性判别分析（LDA），它本身也是一个分类模型。PCA 和 LDA 有很多的相似之处，其本质是将原始的样本映射到维度更毒的样本空间中，但是 PCA 和 LDA 的映射目标不一样：
- PCA 是为了让映射后的样本具有最大的发散性；
- LDA 是为了让映射后的样本有最好的分类性能。

所以说，PCA 是一种无监督的降维方法，而 LDA 是一种有监督的降维方法。在聚类算法中，通常会利用 PCA 对数据进行降维处理，以利于对数据的简化分析和可视化。


```python
# 通过主要成分分析选定数据特征
from pandas import read_csv
from sklearn.decomposition import PCA
# 导入数据
filename = "./data/pima_data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names=names)
# 将数据分为输入数据和输出结果
array = data.values
x = array[:, 0:8]
y = array[:, 8]
# 特征选定
pca = PCA(n_components=3)
fit = pca.fit(x)
print("解释方差：%s" % fit.explained_variance_ratio_)
print(fit.components_)
```

    解释方差：[0.889 0.062 0.026]
    [[-2.022e-03  9.781e-02  1.609e-02  6.076e-02  9.931e-01  1.401e-02
       5.372e-04 -3.565e-03]
     [-2.265e-02 -9.722e-01 -1.419e-01  5.786e-02  9.463e-02 -4.697e-02
      -8.168e-04 -1.402e-01]
     [-2.246e-02  1.434e-01 -9.225e-01 -3.070e-01  2.098e-02 -1.324e-01
      -6.400e-04 -1.255e-01]]
    

## 特征重要性
袋装决策树算法（Bagged Decision Trees）、随机森林算法和极端随机树算法都可以用来计算数据特征的重要性。这三个算法都是集成算法中的袋装算法。


```python
# 通过决策树计算特征的重要性
from pandas import read_csv
from sklearn.ensemble import ExtraTreesClassifier
# 导入数据
filename = "./data/pima_data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names=names)
# 将数据分为输入数据和输出结果
array = data.values
x = array[:, 0:8]
y = array[:, 8]
# 特征选定
model = ExtraTreesClassifier()
fit = model.fit(x, y)
print(fit.feature_importances_)
```

```
[0.112 0.217 0.095 0.084 0.076 0.158 0.125 0.133]
```

## 总结

#### 数据预处理
学到了在 scikit-learn 中对数据进行预处理的四种方法。这四种方法适用于不同的场景，可以在实践中根据不同的算法模型来选择不同的数据预处理方法。

#### 数据特征选定
、介绍了四种选定数据特征的方法。通过选定数据特征来训练算法，得到一个能够提高准确度的模型。