import matplotlib.pyplot as plt
from numpy import sign

Na = 32

i1 = lambda a: int(2 * a) - 1
i2 = lambda a: i1(a) + 1

i1_0 = lambda a: i1(a + 1) - 1
i2_0 = lambda a: i2(a + 1) - 1

half_Na = int(Na / 2)

for a in range(0, half_Na):
    print("a={0:2d}\ti1={1:2d}\ti2={2:2d}".format(a, i1_0(a), i2_0(a)))


laneId = [i for i in range(Na)]


def shfl_xor(arr, laneMask):
    return [arr[i] ^ laneMask for i in range(len(arr))]


def print_bin(arr):
    for i in range(len(arr)):
        print("i:{0:2d} = {1:08b}".format(i, arr[i]))

def print_bin2(arr1, arr2):
    for i in range(len(arr1)):
        print("i:{0:2d} = {1:08b} :: {1:2d}".format(i, arr1[i]))
        print("  shfl {0:08b} :: {0:2d}".format(arr2[i]))


print("original array: ")
print_bin(laneId)

shfl1  = shfl_xor(laneId,  1)
shfl2  = shfl_xor(laneId,  2)
shfl4  = shfl_xor(laneId,  4)
shfl8  = shfl_xor(laneId,  8)
shfl16 = shfl_xor(laneId, 16)

# print("\nshufl by 2:")
# print_bin2(laneId, shfl2)
# print("\nshufl by 4:")
# print_bin2(laneId, shfl4)


if (False):
    plt.figure(1)

    for i in range(Na):
        plt.plot([laneId[i],  shfl1[i]], [1,2], "ro-", mfc='none')
        plt.plot([laneId[i],  shfl2[i]], [2.1,3.1], "bs-", mfc='none')
        plt.plot([laneId[i],  shfl4[i]], [3.2,4.2], "gD-", mfc='none')
        plt.plot([laneId[i],  shfl8[i]], [4.3,5.3], "kP-", mfc='none')
        plt.plot([laneId[i], shfl16[i]], [5.4,6.4], "c^-", mfc='none')

    plt.show()


def bfly_shfl(arr, Pa, plot=False):

    Na = 2**Pa

    laneId = [i for i in range(Na)]

    if (len(arr) == 0):
        arr    = [i for i in range(Na)]

    if (plot):
        plt.figure(1)
        start = 1
        s = ["r", "b", "g", "c", "k"]

        for i in range(Na):
            # plt.plot([laneId[i], arr[i]], [start, start+1],
                    # "ko-", mfc='none')
            plt.plot([start, start+1], [laneId[i], arr[i]], 
                    "ko-", mfc='none')
        start += 1.0

    mask = 1
    for p in range(Pa):
        arr = shfl_xor(arr, mask)
        mask *= 2

        if (plot):
            for i in range(Na):
                # plt.plot([laneId[i], arr[i]], [start, start+1],
                        # s[p%len(s)]+"o-", mfc='none')
                plt.plot([start, start+1], [laneId[i], arr[i]], 
                        s[p%len(s)]+"o-", mfc='none')
            start += 1.0

    if (plot):
        plt.show()

def gnoffo_FWT(arr, Pa, plot=False):

    Na = 2**Pa

    laneId = [i for i in range(Na)]

    if (len(arr) == 0):
        arr    = [i for i in range(Na)]

    h1 = [0] * Na
    h2 = [0] * Na

    plt.figure(1)
    start = 0
    s = ["r", "b"]

    # ...    = h1      h2
    # F1(i1) = f(i1) + f(i2)
    # F2(i2) = f(i1) - f(i2)

    for p in range(1):#Pa):
        for a in range(int(Na/2)):
            i1 = 2 * (a + 1) - 1
            i2 = i1 + 1

            h1[i1 - 1] = i1 - 1
            h2[i1 - 1] = i2 - 1

            h1[i2 - 1] = i1 - 1
            h2[i2 - 1] = -(i2 - 1)

            # arr[i1 - 1] = arr[i1 - 1] + f

            print("F1({0}) = f({0}) + f({1})".format(i1-1, i2-1))
            print("F2({1}) = f({0}) - f({1})".format(i1-1, i2-1))
            print()

        for i in range(Na):
            plt.plot([start, start+1], [h1[i], i], 
                    "bo-", mfc='none')
            plt.plot([start, start+1], [sign(h2[i])*h2[i], i], 
                    s[sign(h2[i]) % 3 % 2]+"o-", mfc='none')
        start += 1

    plt.gca().invert_yaxis()
    plt.show()

if __name__ == "__main__":
    bfly_shfl([], 3, True)
    # gnoffo_FWT([], 3, True)
