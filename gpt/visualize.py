import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 

#244_GDB9_f32True_grid0_backendipu_0_27768 +16 
#


cols = "SMILE,ATOMS,Position 0,Position 1,Position 2,Position 3,Position 4,Position 5,Position 6,Position 7,Position 8,Position 9,Position 10,Position 11,Position 12,Position 13,Position 14,Position 15,Position 16,Position 17,Position 18,Position 19,Position 20,Position 21,Position 22,Position 23,Position 24,Position 25,Position 26,Energy 0,Energy 1,Energy 2,Energy 3,Energy 4,Energy 5,Energy 6,Energy 7,Energy 8,Energy 9,Energy 10,Energy 11,Energy 12,Energy 13,Energy 14,Energy 15,Energy 16,Energy 17,Energy 18,Energy 19,Energy 20,Energy 21,Energy 22,Energy 23,Energy 24,Energy 25,Energy 26,Energy 27,Energy 28,Energy 29,Energy 30,Energy 31,Energy 32,Energy 33,Energy 34,Energy 35,Energy 36,Energy 37,Energy 38,Energy 39,Energy 40,Energy 41,Energy 42,Energy 43,Energy 44,Energy 45,Energy 46,Energy 47,Energy 48,Energy 49,Nuclear Energy,Time 0,Time 1,Time 2,Time 3,Time 4,Time 5,Time 6,Time 7,Time 8,Time 9,Time 10,Time 11".split(",")
ipu = pd.DataFrame(np.load("gdb9_f32_ipu_3_23_2023.npz", allow_pickle=True)["data"], columns=cols)

cols = "SMILE,ATOMS,Position 0,Position 1,Position 2,Position 3,Position 4,Position 5,Position 6,Position 7,Position 8,Position 9,Position 10,Position 11,Position 12,Position 13,Position 14,Position 15,Position 16,Position 17,Position 18,Position 19,Position 20,Position 21,Position 22,Position 23,Position 24,Position 25,Position 26,Energy 0,Energy 1,Energy 2,Energy 3,Energy 4,Energy 5,Energy 6,Energy 7,Energy 8,Energy 9,Energy 10,Energy 11,Energy 12,Energy 13,Energy 14,Energy 15,Energy 16,Energy 17,Energy 18,Energy 19,Energy 20,Energy 21,Energy 22,Energy 23,Energy 24,Energy 25,Energy 26,Energy 27,Energy 28,Energy 29,Energy 30,Energy 31,Energy 32,Energy 33,Energy 34,Energy 35,Energy 36,Energy 37,Energy 38,Energy 39,Energy 40,Energy 41,Energy 42,Energy 43,Energy 44,Energy 45,Energy 46,Energy 47,Energy 48,Energy 49,Nuclear Energy,Time 0,Time 1,Time 2,Time 3,Time 4,Time 5,Time 6,Time 7,Time 8,Time 9,Time 10,Time 11,Time PySCF,Energy PySCF".split(",")
cpu = pd.DataFrame(np.load("gdb9_f64_cpu_3_23_2023.npz", allow_pickle=True)["data"], columns=cols)

print(ipu.shape, cpu.shape)

fig, ax = plt.subplots()

print(cpu.head())# [:, 0])

plt.plot(cpu["Energy 49"].values, ipu["Energy 49"])
plt.savefig("test.jpg")


plt.plot(ipu[:])




