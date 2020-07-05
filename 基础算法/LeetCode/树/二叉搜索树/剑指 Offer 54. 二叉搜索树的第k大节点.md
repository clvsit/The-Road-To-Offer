给定一棵二叉搜索树，请找出其中第k大的节点。

【示例 1】：
```
输入: root = [3,1,4,null,2], k = 1
   3
  / \
 1   4
  \
   2
输出: 4
```

【示例 2】：
```
输入: root = [5,3,6,2,4,null,null,1], k = 3
       5
      / \
     3   6
    / \
   2   4
  /
 1
输出: 4
```

限制：
- 1 ≤ k ≤ 二叉搜索树元素个数

链接：https://leetcode-cn.com/problems/er-cha-sou-suo-shu-de-di-kda-jie-dian-lcof

## 数据结构
```python
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
```

## 方法 1：中序遍历 + 有序数组
通过对二叉搜索树的中序遍历，我们可以得到递增的有序数组，此时只需要返回倒数第 k 个元素即可得到第 k 大节点。

【实现代码】：
```python
class Solution:
    def kthLargest(self, root: TreeNode, k: int) -> int:
        if not root:
            return None
        
        path = []
        self._in_order_traverse(root, path)
        return path[-k]

    def _in_order_traverse(self, node: TreeNode, path: List):
        if node:
            self._in_order_traverse(node.left, path)
            path.append(node.val)
            self._in_order_traverse(node.right, path)

```

【执行效率】：
- 时间复杂度：O(n)；
- 空间复杂度：O(2n)，创建有序数组需要 O(n)，遍历二叉搜索树需要 O(n) 的栈空间。

## 方法 2：递归法
直接一步到位，在中序遍历时找到第 k 大节点：首先遍历右子树，再遍历左子树，这样就能够按照递减的顺序遍历二叉搜索树，遍历到第 k 个节点返回即可。

需要一个额外的变量 index 来记录当前遍历节点的序号。

【实现代码】：
```python
class Solution:
    
    def __init__(self):
        self.index = 0

    def kthLargest(self, root: TreeNode, k: int) -> int:
        if not root:
            return None
        
        # 若找到第 k 大节点则直接返回，停止后续的遍历
        right_value = self.kthLargest(root.right, k)

        if right_value:
            return right_value

        # 中序处理
        self.index += 1
        if self.index == k:
            return root.val
        
        # 若找到第 k 大节点则直接返回，停止后续的遍历
        left_value = self.kthLargest(root.left, k)

        if left_value:
            return left_value        
        
        return None

```

【执行效率】：
- 时间复杂度：O(k)；
- 空间复杂度：O(k)。
