统计一个数字在排序数组中出现的次数。

【示例 1】：
```
输入: nums = [5,7,7,8,8,10], target = 8
输出: 2
```

【示例 2】：
```
输入: nums = [5,7,7,8,8,10], target = 6
输出: 0
```

限制：
- 0 <= 数组长度 <= 50000

链接：https://leetcode-cn.com/problems/zai-pai-xu-shu-zu-zhong-cha-zhao-shu-zi-lcof

## 方法 1：二分查找 + 线性扫描
1. 首先，通过二分查找法找到目标元素的位置；
2. 然后，通过线性扫描的方式向前、向后找到其他元素，并统计个数；
3. 最终返回统计个数。

【代码实现】：
```python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        start, end = 0, len(nums) - 1
        count = 0

        while start <= end:
            mid = (start + end) // 2

            if target == nums[mid]:
                count += 1
                prev_index, next_index = mid - 1, mid + 1
                while prev_index >= 0 and target == nums[prev_index]:
                    count += 1
                    prev_index -= 1
                while next_index <= len(nums) - 1 and target == nums[next_index]:
                    count += 1
                    next_index += 1
                break
            elif target < nums[mid]:
                end = mid - 1
            else:
                start = mid + 1
        
        return count
```

【执行效率】：
- 时间复杂度：O(log n)；
- 空间复杂度：O(1)。
