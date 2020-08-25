# sequency reordering for going from Hadamard matrix to Walsh matrix

Pa = 4
Na = 2**Pa

def print_bin_arr(arr, newLine=False, size=8):
    fmt = "{0:0%db} " % size
    if (newLine): fmt += "\n"
    for i in range(len(arr)):
        print(fmt.format(arr[i]), end="")
    if (not newLine): print()


# sequency:
s = [i for i in range(Na)]
print(s)
print_bin_arr(s, False, 3)

# convert to binary:
# sb = sum([s[i] * 2**i for i in range(Na)])

# converted to Gray code:
# g = sb ^ (sb >> 1)
# print(bin(g))
g = [s[i] ^ (s[i]>>1) for i in range(Na)] + [0]
print_bin_arr(g, False, 3)

def bit_reverse(val, nbits):
    fmt = "{0:0%db}" % nbits
    return int(fmt.format(val)[-1::-1], 2)

# bit-reverse g's to get k's:
k = [bit_reverse(g[i], Pa) for i in range(Na)]
print_bin_arr(k, False, 3)
print(k)
