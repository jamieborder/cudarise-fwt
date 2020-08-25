


P = 3
N = 2**P

# s = 1
# for i in range(3):
    # for j in range(int(N)):
        # print(2 * j + s - 1)
    # s *= 2

    # print()

def pp(arr):
    for i in range(len(arr)):
        print("{0} ".format(arr[i]), end="")
    print()

s = [i for i in range(N)]
pp(s)

s1 = [s[i] % 2 for i in range(N)]
pp(s1)

s2 = [(s[i]-1) % 3 % 2 for i in range(N)]
pp(s2)

s3 = [(s[i]+1) % 4 for i in range(N)]
pp(s3)

print("----------")
s1 = []
for i in range(0,8,2):
    for j in range(1):
        s1.append(i + j)
pp(s1)

s2 = []
for i in range(0,8,3):
    for j in range(2):
        s2.append(i + j)
pp(s2)

# s1 = [i for i in range(0,8,2)]
# pp(s1)
# s2 = [i for i in range(0,8,4)]
# pp(s2)


t = [i for i in range(N)]
# tb = ["{0:3b}".format(
