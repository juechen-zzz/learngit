def combinationSum2(candidates, target):
    dp = [[] for _ in range(target + 1)]
    for val in candidates:
        for i in reversed(range(val, target + 1)):  # 返回一个反转的迭代器
            if dp[i - val]:
                for _list in dp[i - val]:
                    l = _list + [val]
                    dp[i].append(l)
            elif i == val:
                dp[i].append([val])
    res = []
    for i in dp[-1]:
        if sorted(i) not in res:
            res.append(sorted(i))
    return res

def main():
    candidates = [10, 1, 2, 7, 6, 1, 5]
    target = 6
    ans = combinationSum2(candidates, target)
    print(ans)

if __name__ == "__main__":
    main()