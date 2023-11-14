def _combinations(N,k):
    if k == 0: return ['']

    if k == N or k < 0: return [''.join([str(x) for x in reversed(list(range(N)))])]

    if k > N: return []

    ret = []
    for j in reversed(list(range(k-1,N))):
        partial = _combinations(j,k-1)
        ret.extend([str(j)+cmb for cmb in partial])
    return ret

def combinations(N,k):
    cmbs = _combinations(N,k)
    ret = []
    for cmb in reversed(cmbs):
        ret.append([int(ch) for ch in cmb[::-1]])
    return ret

if __name__ == '__main__':

    for k in range(8):
        cmb = combinations(7,k)
        print(len(cmb), cmb)