输入一个正整数 target ，输出所有和为 target 的连续正整数序列（至少含有两个数）。

序列内的数字由小到大排列，不同序列按照首个数字从小到大排列。

【示例 1】：
```
输入：target = 9
输出：[[2,3,4],[4,5]]
```

【示例 2】：
```
输入：target = 15
输出：[[1,2,3,4,5],[4,5,6],[7,8]]
```

限制：
- 1 <= target <= 10^5

链接：https://leetcode-cn.com/problems/he-wei-sde-lian-xu-zheng-shu-xu-lie-lcof

【题目类型】：
- 前缀和
- 滑动窗口

## 方法 1：前缀和 + 双重循环
要求和为指定值的连续正数序列，通常我们会采用前缀和的方式进行处理，然后通过双重循环的方式找到所有的连续子数组，接着判断这些连续子数组的和是否为 s，若等于 s 则添加到 result 中，最终输出 result。

【代码实现】：
```python
class Solution:
    def findContinuousSequence(self, target: int) -> List[List[int]]:
        pre_sum_list = [0] * (target + 1)
        
        for i in range(1, target + 1):
            pre_sum_list[i] = pre_sum_list[i - 1] + i
        
        result = []
        for i in range(1, target):
            for j in range(i + 1, target + 1):
                pre_sum = pre_sum_list[j] - pre_sum_list[i - 1]
                if pre_sum == target:
                    result.append(list(range(i, j + 1)))
        
        return result
```

【执行效率】：
- 时间复杂度：O(n^2)；
- 空间复杂度：O(n)。

该方法在 LeetCode 上会超时。

## 方法 2：滑动窗口法
在没有固定长度的滑动窗口法中需要明确右边界如何扩张，以及左边界如何收缩。
- 右边界扩张：当滑动窗口中的总和小于 target 时；
- 左边界收缩：当滑动窗口中的总和大于 target 时。

【代码实现】：
```python
class Solution:
    def findContinuousSequence(self, target: int) -> List[List[int]]:
        sum_val = 0
        left = 1
        result = []

        for i in range(1, target):
            sum_val += i
            while sum_val > target:
                sum_val -= left
                left += 1

            if sum_val == target:
                result.append(list(range(left, i + 1)))
        
        return result
```

【执行效率】：
- 时间复杂度：O(n)；
- 空间复杂度：O(1)。
