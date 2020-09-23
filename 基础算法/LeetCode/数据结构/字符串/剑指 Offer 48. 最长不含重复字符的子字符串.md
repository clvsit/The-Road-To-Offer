请从字符串中找出一个最长的不包含重复字符的子字符串，计算该最长子字符串的长度。

【示例 1】：
```
输入: "abcabcbb"
输出: 3 
解释: 因为无重复字符的最长子串是 "abc"，所以其长度为 3。
```

【示例 2】：
```
输入: "bbbbb"
输出: 1
解释: 因为无重复字符的最长子串是 "b"，所以其长度为 1。
```

【示例 3】：
```
输入: "pwwkew"
输出: 3
解释: 因为无重复字符的最长子串是 "wke"，所以其长度为 3。
     请注意，你的答案必须是 子串 的长度，"pwke" 是一个子序列，不是子串。
```

提示：
- s.length <= 40000

注意：本题与主站 3 题相同：https://leetcode-cn.com/problems/longest-substring-without-repeating-characters/

链接：https://leetcode-cn.com/problems/zui-chang-bu-han-zhong-fu-zi-fu-de-zi-zi-fu-chuan-lcof

## 方法 1：滑动窗口
此题相当于求最大的滑动窗口，滑动窗口内没有重复的字符。因此，刚开始可以不断拓展滑动窗口的右边界，直到遇到重复的字符，此时再收缩左边界。在此过程中记录下最大的滑动窗口大小。

【代码实现】：
```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        if len(s) == 0:
            return 0
        
        char_dict = {}
        start = -1
        result = 0
        for i in range(len(s)):
            if s[i] in char_dict:
                start = max(char_dict[s[i]], start)
            char_dict[s[i]] = i
            result = max(result, i - start)
        return result
```

【执行效率】：
- 时间复杂度：O(n)；
- 空间复杂度：O(n)。

