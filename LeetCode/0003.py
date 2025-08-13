'''无重复字符的最长字符串
https://leetcode.cn/problems/longest-substring-without-repeating-characters/

'''
def lengthOfLongestSubstring(s: str) -> int:
    ans = 0
    result = [0]
    for i in range(len(s)):
        n = []
        for step in range(0, len(s)-i):
            if step != len(s) - i - 1:
                if s[i+step] not in n:
                    n.append(s[i+step])
                else:
                    ans = max(ans,step)
                    result.append(ans)
                    break
            else:
                if s[i+step] not in n:
                    ans = step + 1
                else:
                    ans = step
            #ans = max(ans,step)
            result.append(ans)
    return max(result)