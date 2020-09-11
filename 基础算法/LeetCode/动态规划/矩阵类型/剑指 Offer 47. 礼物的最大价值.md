在一个 m*n 的棋盘的每一格都放有一个礼物，每个礼物都有一定的价值（价值大于 0）。你可以从棋盘的左上角开始拿格子里的礼物，并每次向右或者向下移动一格、直到到达棋盘的右下角。给定一个棋盘及其上面的礼物的价值，请计算你最多能拿到多少价值的礼物？

【示例 1】：
```
输入: 
[
  [1,3,1],
  [1,5,1],
  [4,2,1]
]
输出: 12
解释: 路径 1→3→5→2→1 可以拿到最多价值的礼物
```

提示：
- 0 < grid.length <= 200
- 0 < grid[0].length <= 200

链接：https://leetcode-cn.com/problems/li-wu-de-zui-da-jie-zhi-lcof

## 方法 1：动态规划
因为每次只能向右或者向下移动，因此棋盘上的每一点只需要判断上方和左边哪一条路径的值较大，从而选择哪一条路径即可，类似于维比特算法。

我们可以使用 dp\_table 来记录走到当前位置的最大路径和，以示例 1 为例。
- 第一行第二列：此时只有左边一条路径，因此该点的路径和为 1 + 3 = 4；
- 第二行第二列：此时有两条路径，分别是 1-3-5 和 1-1-5，其中 1-3 和 1-1 这两条路径的和已经存储在 dp\_table 中，分别为 dp\_table[0][1] 和 dp\_table[1][0]，我们只需要比较这两个值，选择更大的那一个然后加上当前位置的值（5）作为当前位置的路径和。

【代码实现】：
```python
class Solution:
    def maxValue(self, grid: List[List[int]]) -> int:
        rows = len(grid)
        if rows == 0:
            return 0
        cols = len(grid[0])
        dp_table = []

        for row in range(rows + 1):
            dp_table.append([0] * (cols + 1))
        
        for row in range(rows):
            for col in range(cols):
                dp_table[row + 1][col + 1] = grid[row][col] + max(
                    dp_table[row][col + 1],
                    dp_table[row + 1][col]
                )
        
        return dp_table[-1][-1]
```

【执行效率】：
- 时间复杂度：O(nm)；
- 空间复杂度：O(nm)。
