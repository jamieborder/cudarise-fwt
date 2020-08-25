# sequency reordering for going from Hadamard matrix to Walsh matrix

Pa = 3
Na = 2**Pa

def print_bin_arr(arr, newLine=False, size=8):
    fmt = "{0:0%db} " % size
    if (newLine): fmt += "\n"
    for i in range(len(arr)):
        print(fmt.format(arr[i]), end="")
    if (not newLine): print()

# sequency:
s = [i for i in range(Na)]
print_bin_arr(s, False, 3)

# convert to binary:
# sb = sum([s[i] * 2**i for i in range(Na)])

# converted to Gray code:
# g = sb ^ (sb >> 1)
# print(bin(g))
g = [s[i] ^ (s[i]>>1) for i in range(Na)] + [0]
print_bin_arr(g, False, 3)

# bit-reverse g's to get k's:
k = [g[Na-1-i] ^ g[Na-i] for i in range(Na)]
# k = [(g[Na-1-i] << 1) ^ g[Na-1-i] for i in range(Na)]
print_bin_arr(k, False, 3)
print("000 100 110 010 011 111 101 001 :: soln")

for i in range(Na):
    # print("{0:3b} {1:3b} {2:3b}".format(g[Na-1-i], g[Na-i],
        # g[Na-1-i] ^ g[Na-i]))
    # print("{0:3b} {1:3b} {2:3b}".format(g[Na-1-i], g[Na-i],
        # (g[Na-1-i] << 1) ^ g[Na-i]))

    rev = 0
    n = g[i]
    while n>0:
        rev = rev << 1
        if (n & 1 == 1):
            rev = rev ^ 1
        n = n >> 1
    # print("{0:03b} {1:03b}".format(g[i], rev))


    count = 2
    tmp = g[i]
    num = g[i]
    num = num >> 1
    while (num):
        tmp = tmp << 1
        tmp = tmp | (num & 1)
        num = num >> 1
        count -= 1
    tmp = tmp << count
    # print("{0:03b} {1:03b}".format(g[i], tmp))
    print("{0:03b} {1}".format(g[i], "{0:03b}".format(tmp)[-3:]))
    print(int("{0:03b}".format(tmp)[-3:], 2))

# gb = sum([g[i] * 2**i for i in range(Na)])
# result = 0
# while gb:
    # result = (result << 1) + (gb & 1)
    # gb >>= 1
# print(bin(result))

