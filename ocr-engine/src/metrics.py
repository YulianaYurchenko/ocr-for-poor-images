from numpy import zeros as np_zeros


def edit_distance(token1: str, token2: str) -> int:
    """
        Calculates edit distance aka Levenshtein distance between two strings.
        Returns:
            (int): calculated distance.
    """

    len1 = len(token1) + 1
    len2 = len(token2) + 1
    D = np_zeros((len1, len2))

    for i in range(len1):
        D[i][0] = i
    for j in range(len2):
        D[0][j] = j

    for i in range(1, len1):
        for j in range(1, len2):
            if token1[i - 1] == token2[j - 1]:
                cost = 0
            else:
                cost = 1
            D[i][j] = min(D[i - 1][j] + 1,
                          D[i][j - 1] + 1,
                          D[i - 1][j - 1] + cost)

    return int(D[len1 - 1][len2 - 1])
