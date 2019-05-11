import numpy as np

cards = np.array(
    [[1, 2, 3],
     [1, 2, 4],
     [1, 2, 5],
     [1, 2, 6],
     [1, 3, 4],
     [1, 3, 5],
     [1, 3, 6],
     [1, 4, 5],
     [1, 4, 6],
     [1, 5, 6],
     [2, 3, 4],
     [2, 3, 5],
     [2, 3, 6],
     [2, 4, 5],
     [2, 4, 6],
     [2, 5, 6],
     [3, 4, 5],
     [3, 4, 6],
     [3, 5, 6],
     [4, 5, 6]
     ])

test = np.sum(np.array(cards == 1))

success = 0
exp = 100000
for i in range(exp):
    a = np.arange(20)
    np.random.shuffle(a)
    chosen = cards[a[:3]]

    success += ((1 in chosen[0]) or (1 in chosen[1]) or (
            1 in chosen[2]))
print(success / exp)

success = 0
exp = 10000
for i in range(exp):
    cube1 = np.random.randint(6) + 1
    cube2 = np.random.randint(6) + 1
    a = np.arange(20)
    np.random.shuffle(a)
    chosen = cards[a[:3]]

    success += ((cube1 in chosen[0] and cube2 in chosen[0]) or (cube1 in chosen[1] and cube2 in chosen[1]) or (
                cube1 in chosen[2] and cube2 in chosen[2]))
print(success / exp)


