输入一个字符串，打印出该字符串中字符的所有排列。你可以以任意顺序返回这个字符串数组，但里面不能有重复元素。

【示例 1】：
```
输入：s = "abc"
输出：["abc","acb","bac","bca","cab","cba"]
```

限制：
- 1 <= s 的长度 <= 8

链接：https://leetcode-cn.com/problems/zi-fu-chuan-de-pai-lie-lcof

## 方法 1：备忘录 + BFS

【代码实现】：
```python
class Solution:
    def permutation(self, s: str) -> List[str]:
        str_list = [("", s)]
        memo_set = set()
        result = []

        while len(str_list) > 0:
            char, choice_list = str_list.pop(0)
            if char in memo_set:
                continue
            if len(choice_list) == 0:
                result.append(char)
                continue

            for i, choice in enumerate(choice_list):
                str_list.append((char + choice, choice_list[:i] + choice_list[i + 1:]))
            
            memo_set.add(char)
            
        return result
```