def bfs(img, x, y, used, h, w):
    q = set()
    q.add((x, y))

    while len(q) != 0:
        i, j = q.pop()

        used[i][j] = True

        if i > 0 and img[i - 1][j] and not used[i - 1][j]:
            q.add((i - 1, j))

        if i < h - 1 and img[i + 1][j] and not used[i + 1][j]:
            q.add((i + 1, j))

        if j > 0 and img[i][j - 1] and not used[i][j - 1]:
            q.add((i, j - 1))

        if j < w - 1 and img[i][j + 1] and not used[i][j + 1]:
            q.add((i, j + 1))
