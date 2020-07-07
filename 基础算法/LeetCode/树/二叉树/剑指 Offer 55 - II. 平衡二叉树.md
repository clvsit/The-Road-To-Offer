输入一棵二叉树的根节点，判断该树是不是平衡二叉树。如果某二叉树中任意节点的左右子树的深度相差不超过1，那么它就是一棵平衡二叉树。

【示例 1】:
```
给定二叉树 [3,9,20,null,null,15,7]

    3
   / \
  9  20
    /  \
   15   7
返回 true 。
```

【示例 2】:
```
给定二叉树 [1,2,2,3,3,null,null,4,4]

       1
      / \
     2   2
    / \
   3   3
  / \
 4   4
返回 false 。
```

限制：
- 1 <= 树的结点个数 <= 10000

链接：https://leetcode-cn.com/problems/ping-heng-er-cha-shu-lcof

## 数据结构
```python
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

```

## 方法 1：递归法
采用后序遍历的方式来遍历二叉树，该题的难点在于函数的返回值，因为最终返回的结果是 bool，但每个子树的深度是一个 int。因此，一个简单的处理方式是创建一个后序遍历的函数，该函数用以遍历二叉树，并计算左子树与右子树的深度差。对于差超过 1 的，我们返回 -100，这表明这不是一棵平衡二叉树。否则返回深度更深的值。

【实现代码】：
```python
class Solution:
    def isBalanced(self, root: TreeNode) -> bool:

        def _post_traverse(node: TreeNode):
            if not node:
                return 0
            
            left_deepth = _post_traverse(node.left) + 1
            right_deepth = _post_traverse(node.right) + 1
            print(node.val, left_deepth, right_deepth)

            if abs(left_deepth - right_deepth) > 1:
                return -100
            else:
                return max([left_deepth, right_deepth])
        
        return True if _post_traverse(root) >= 0 else False

```

【执行效率】：
- 时间复杂度：O(n)；
- 空间复杂度：O(n)。
