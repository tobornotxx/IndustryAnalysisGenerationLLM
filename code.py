def longestSubstring(s: str) -> int:
    index = {}
    left_bound = 0
    max_length = 0
    for i, char in enumerate(s):
        if char not in index:
            index[char] = i #第一次出现，加入index集合
        else:
            left_bound = max(left_bound, index[char] + 1) # 当前字符串已经出现过，如果上次出现的index>=left bound，必须丢掉，从他后面开始
            index[char] = i
        max_length = max(max_length, i - left_bound + 1)
    
    return max_length

print(longestSubstring("abcabcbb"))