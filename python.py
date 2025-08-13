def lengthOfLongestSubstring(s: str) -> int:
    '''
    # 哈希集合，记录每个字符是否出现过
    occ = set()
    n = len(s)
    # 右指针，初始值为 -1，相当于我们在字符串的左边界的左侧，还没有开始移动
    rk, ans = -1, 0
    for i in range(n):
        if i != 0:
            # 左指针向右移动一格，移除一个字符
            occ.remove(s[i - 1])
        while rk + 1 < n and s[rk + 1] not in occ:
            # 不断地移动右指针
            occ.add(s[rk + 1])
            rk += 1
        # 第 i 到 rk 个字符是一个极长的无重复字符子串
        ans = max(ans, rk - i + 1)
    return ans
    '''
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

s ="cau"

print(lengthOfLongestSubstring(s))