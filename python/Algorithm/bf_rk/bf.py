#!/usr/bin/python
# -*- coding: UTF-8 -*-

from time import time

def bf(main, pattern):
    """
    字符串匹配，bf暴力搜索
    :param main: 主串
    :param pattern: 模式串
    :return:
    """

    n = len(main)
    m = len(pattern)

    if n <= m:
        return 0 if pattern == main else -1

    for i in range(n-m+1):
        for j in range(m):
            if main[i+j] == pattern[j]:
                if j == m-1:
                    return i
                else:
                    continue
            else:
                break
    return -1

if __name__ == '__main__':
    m_str = 'a' * 10000
    p_str = "a" * 200 + 'b'

    print('---time consume---')
    t = time()
    print('[bf] result:', bf(m_str, p_str))
    print('[bf] time cost: {0:.5}s'.format(time()-t))

    print('')
    print('--- search ---')
    m_str = 'thequickbrownfoxjumpsoverthelazydog'
    p_str = 'jump'
    print('[bf] result:', bf(m_str, p_str))