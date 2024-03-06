def func(n):
    c = 0
    counter = 0
    while (n>=0):
        n = n - 2
        c = c + n - 2
        counter = counter + 1
    return c, counter

print(func(15000))