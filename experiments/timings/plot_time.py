import matplotlib.pyplot as plt 
import numpy as np 

# didn't set numa nodes 
'''nums_l3 = np.array([
  [10.178521156311035 , -325.8981140058157, "sto-3g", 45],
  [14.501192569732666 , -331.6632166797748, "6-31G" ,81],
  [16.437588214874268 , -332.0745229359578, "6-31G*", 126],
  [12.498451948165894 , -361.7615686765914, "sto-3g", 50],
  [15.663350820541382 , -367.9693026921675, "6-31G", 90],
  [17.75219464302063 , -368.3509005927019 ,"6-31G*" ,140],
  [16.051669597625732 , -397.7969270619933, "sto-3g" ,55],
  [17.100021600723267 , -404.61138353949093, "6-31G" ,99],
  [21.983044147491455 , -404.9970869501707 ,"6-31G*" ,154]])'''


nums_l0 = np.array([[0.6979353427886963 , -325.86678679395834,"sto-3g", 45],
[3.0375587940216064 , -331.69778389444355 ,"6-31G", 81],
[4.027990818023682 , -332.10846116006167 ,"6-31G*", 126],
[0.6185686588287354 , -361.7194020417239 ,"sto-3g", 50],
[3.620227575302124 , -367.9962752881727 ,"6-31G", 90],
[4.705349683761597 , -368.3784058009686 ,"6-31G*", 140],
[3.2274837493896484 , -397.75185858480245 ,"sto-3g", 55],
[6.759873151779175 , -404.639621951923 ,"6-31G",99],
[9.01852798461914 , -405.0250385723208 ,"6-31G*", 154]])


# from python density_functional_theory.py -backend ipu -float32 -its 30 -level 0 -plevel 0 -gdb {9,10,11} -generate -save 
# the following files contain times for generating thousands of DFTs, just toke an arbitarry one
# data/generated/0_GDB11_f32True_grid0_backendipu_1780985_2137181/data.csv
# data/generated/70_GDB9_f32True_grid0_backendipu_0_27768/data.csv
# data/generated/87_GDB10_f32True_grid0_backendipu_389254_583880/data.csv
us_gdb9 = [  1.5  , 1.6  , 0.3  ,27.   , 0.  , 33.2  , 1.5  , 0.2  , 2.2  , 0.  ,  4.8 ,129.9]
us_gdb10 = [  1.9  , 1.6  , 0.3  ,33.8 ,  0.  , 25.5  , 0.6  , 0.2 ,  2.7  , 0. ,   7.1, 157.2]
us_gdb11 = [  3.6   ,2.4  , 0.3  ,41.9  , 0.1  , 1.6  , 0.8  , 0.2 ,  2.8  , 0.  ,  6.7, 227.3]


xs = [0,1,2]


#plt.plot(xs, nums_l3[:3, 0].astype(np.float32), 'C0x-', label="pyscf GDB9 (grid3)")
#plt.plot(xs, nums_l3[3:6, 0].astype(np.float32), 'C1x-', label="pyscf GDB10 (grid3)")
#plt.plot(xs, nums_l3[6:9, 0].astype(np.float32), 'C2x-', label="pyscf GDB11 (grid3)")

plt.plot(xs, nums_l0[:3, 0].astype(np.float32), 'C0x--', label="pyscf GDB9 (grid0)")
plt.plot(xs, nums_l0[3:6, 0].astype(np.float32), 'C1x--', label="pyscf GDB10 (grid0)")
plt.plot(xs, nums_l0[6:9, 0].astype(np.float32), 'C2x--', label="pyscf GDB11 (grid0)")


plt.plot(xs[0], np.sum(us_gdb9)/1000, 'x-', label="jaxdft GDB9 (grid0)")
plt.plot(xs[0], np.sum(us_gdb10)/1000, 'x-', label="jaxdft GDB10 (grid0)")
plt.plot(xs[0], np.sum(us_gdb11)/1000, 'x-', label="jaxdft GDB11 (grid0)")

plt.xticks(xs, nums_l0[:3, 2])
plt.yscale("log")
plt.legend()
plt.savefig("time.jpg")
