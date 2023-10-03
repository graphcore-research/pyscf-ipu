import numpy as np
import jax
from tessellate_ipu_local import ipu_eigh
import jax.numpy as jnp


####################
def get_error(result_vec, result_val, reference_out, reference_in, prefix: str =""):
    vects = reference_out
    us_vects = np.asarray(result_vec)
    us_vals = np.asarray(result_val)
    err = us_vects - vects
    cumulated_err = np.sum(np.abs(err))
    abs_err = np.abs(us_vects) - np.abs(vects)
    cumulated_abs_err = np.sum(np.abs(abs_err))
    reconstruction_err = np.sum(np.abs(us_vects @ np.diag(us_vals) @ us_vects.T - reference_in))

    print(f'[{prefix}] iter: {i} cumulated abs err: {cumulated_abs_err} cumulated err: {cumulated_err}, reconstruction err: {reconstruction_err}')
    return cumulated_abs_err
####################
ITERATIONS = 12

# N = 36

# A = np.random.normal(0, 1, (N,N))
# A = A + A.T

# vals, vects = np.linalg.eigh(A)

# print("GROUND TRUTH SHAPE:", vals.shape, vects.shape, vals.reshape(N,1).shape)
# exit()

# pad = N % 2
# initial_guess = jnp.diag(vals), vects
# if pad:
#     x = jnp.pad(A, [(0, 1), (0, 1)], mode='constant')
#     initial_guess = (jnp.pad(initial_guess[0], ((0, 1), ), mode='constant'), jnp.pad(initial_guess[1], [(0, 1), (0, 1)], mode='constant'))
# else:
#     x = A


import matplotlib.pyplot as plt
import imageio

# Initialize list to store images
images = []

# name_inp = "../pyscf_ipu/nanoDFT/benzene_inp"
# name_ev = "../pyscf_ipu/nanoDFT/benzene_ev"
# name_vals = "../pyscf_ipu/nanoDFT/benzene_vals"

# name_inp = "../pyscf_ipu/nanoDFT/20_it_benz_inp"
# name_ev = "../pyscf_ipu/nanoDFT/20_it_benz_ev"
# name_vals = "../pyscf_ipu/nanoDFT/20_it_benz_vals"

name_inp = "../pyscf_ipu/nanoDFT/c20_inp"
name_ev = "../pyscf_ipu/nanoDFT/c20_ev"
name_vals = "../pyscf_ipu/nanoDFT/c20_vals"


num_its = 20
it=8 # iteration selected for comparison
assert it < num_its

refernce_input_list = []
reference_ev_list = []
reference_vals_list = []

for j in range(num_its):
    file_name = f'{name_inp}/input{j}.npy'
    refernce_input_list.append(np.load(file_name))
    file_name = f'{name_ev}/eigvect{j}.npy'
    reference_ev_list.append(np.load(file_name))
    file_name = f'{name_vals}/eigvals{j}.npy'
    reference_vals_list.append(np.load(file_name))
# print(data)

# exit()

refernce_input = refernce_input_list[it]
reference_ev = reference_ev_list[it]
reference_vals = reference_vals_list[it]

vals, vects = np.linalg.eigh(refernce_input)
### ALEX TESTED SIMILARITY OF EIGVALS AND EIGVECTS HERE
# us_vects, us_vals, _ = jax.jit(ipu_eigh)(refernce_input, num_iters=10)
# us_vects = np.asarray(us_vects)
# us_vals = np.asarray(us_vals)

# print("\n", vals, "\n", us_vals, "\n")
# print(np.max(np.abs(vects @ np.diag(vals) @ vects.T - refernce_input)))
# print(np.max(np.abs(us_vects @ np.diag(us_vals) @ us_vects.T - refernce_input)))
# print(np.max(np.abs(np.abs(vals)-np.abs(us_vals))))

# print(np.max(np.abs(vects)-np.abs(us_vects)))

# fig, ax = plt.subplots(1,3)
# ax[0].imshow(vects)
# ax[1].imshow(us_vects)
# ax[2].imshow(np.abs(vects)-np.abs(us_vects))
# plt.savefig("eigh_alex.jpg")
# exit()
######


x = refernce_input
n = x.shape[0]
pad = n % 2


print("--- default with no external initialization ---")
accum_errs = []
for i in range(ITERATIONS):

    us_vects, us_vals, unsorted_tuple = jax.jit(ipu_eigh)(x, num_iters=i)
    # us_vals, us_vects = unsorted_tuple
    # if pad:
    #     e1 = us_vects[-1:]
    #     col = jnp.argmax(e1)
    #     us_vects = jnp.roll(us_vects, -col-1)
    #     us_vects = us_vects[:, :-1]
    #     us_vects = jnp.roll(us_vects, -(-col))
    #     us_vects = us_vects[:-1]

    #     us_vals = jnp.roll(us_vals, -col-1)
    #     us_vals = us_vals[:-1]
    #     us_vals = jnp.roll(us_vals, -(-col))
    
    # _ = get_error(us_vects, vects, "linag")
    cumulated_abs_err = get_error(us_vects, us_vals, reference_ev, reference_vals, "refer")
    accum_errs.append(cumulated_abs_err)

# print("\n", reference_ev, "\n", vects)

plt.imshow(vects, cmap='viridis', animated=True)
plt.title(f'Reference eigenvectors from nanoDFT')
plt.colorbar()
plt.savefig("linalg_ev.png")
plt.clf()

plt.imshow(reference_ev, cmap='viridis', animated=True)
plt.title(f'Linalg eigenvectors computed with ipu_eigh input from nanoDFT')
plt.colorbar()
plt.savefig("reference.png")
plt.clf()

diff_ev = vects - reference_ev
# print(diff_ev)
plt.imshow(diff_ev, cmap='viridis', animated=True)
plt.title(f'Linalg and reference eigenvectors diff')
plt.colorbar()
plt.savefig("diff_ev.png")
plt.clf()

# print(reference_vals)
plt.imshow(np.diag(reference_vals), cmap='viridis', animated=True)
plt.title(f'Reference eigenvalues from nanoDFT')
plt.colorbar()
plt.savefig("reference_vals.png")
plt.clf()

initial_guess = x, reference_ev

# print("--- only reference eigenvectors ---")
# accum_errs_initialized = []
# for i in range(ITERATIONS):
#     us_vects, us_vals, _ = jax.jit(ipu_eigh)(x, num_iters=i, initial_guess=initial_guess)
#     us_vects = np.asarray(us_vects)
#     us_vals = np.asarray(us_vals)
    
#     cumulated_abs_err = get_error(us_vects, us_vals, reference_ev, reference_vals, "refer")
#     accum_errs_initialized.append(cumulated_abs_err)

print("--- lianlg eigenvectors and eigvals ---")
initial_guess_with_linalg = jnp.diag(vals), vects
accum_errs_linalg = []
for i in range(ITERATIONS):
    us_vects, us_vals, _ = jax.jit(ipu_eigh)(x, num_iters=i, initial_guess=initial_guess_with_linalg)
    us_vects = np.asarray(us_vects)
    us_vals = np.asarray(us_vals)
    
    cumulated_abs_err = get_error(us_vects, us_vals, reference_ev, reference_vals, "refer")
    accum_errs_linalg.append(cumulated_abs_err)

print("--- defaults initialized externaly ---")
initial_guess_with_defaults = x, np.eye(n)
accum_errs_defaults = []
for i in range(ITERATIONS):
    us_vects, us_vals, _ = jax.jit(ipu_eigh)(x, num_iters=i, initial_guess=initial_guess_with_defaults)
    us_vects = np.asarray(us_vects)
    us_vals = np.asarray(us_vals)
    
    cumulated_abs_err = get_error(us_vects, us_vals, reference_ev, reference_vals, "refer")
    accum_errs_defaults.append(cumulated_abs_err)

print("--- reference eigvals and eigvects ---")
initial_guess_with_reference = jnp.diag(reference_vals), reference_ev
accum_errs_reference = []
for i in range(ITERATIONS):
    us_vects, us_vals, _ = jax.jit(ipu_eigh)(x, num_iters=i, initial_guess=initial_guess_with_reference)
    us_vects = np.asarray(us_vects)
    us_vals = np.asarray(us_vals)
    
    cumulated_abs_err = get_error(us_vects, us_vals, reference_ev, reference_vals, "refer")
    accum_errs_reference.append(cumulated_abs_err)

# print("--- only reference eigenvals, eigvects as eye(n) ---")
# initial_guess_with_reference_eigvals = jnp.diag(reference_vals), np.eye(n)
# accum_errs_reference_vals = []
# for i in range(ITERATIONS):
#     us_vects, us_vals, _ = jax.jit(ipu_eigh)(x, num_iters=i, initial_guess=initial_guess_with_reference_eigvals)
#     us_vects = np.asarray(us_vects)
#     us_vals = np.asarray(us_vals)
    
#     cumulated_abs_err = get_error(us_vects, us_vals, reference_ev, reference_vals, "refer")
#     accum_errs_reference_vals.append(cumulated_abs_err)


###################################

prev_ev = reference_ev_list[it-1]
prev_vals = reference_vals_list[it-1]

print("--- previous eigvals and eigvect ---")
initial_guess_with_previous_iteration = jnp.diag(prev_vals), prev_ev
accum_errs_reference_previous_iteration = []
for i in range(ITERATIONS):
    us_vects, us_vals, _ = jax.jit(ipu_eigh)(x, num_iters=i, initial_guess=initial_guess_with_previous_iteration)
    us_vects = np.asarray(us_vects)
    us_vals = np.asarray(us_vals)
    
    cumulated_abs_err = get_error(us_vects, us_vals, reference_ev, reference_vals, "refer")
    accum_errs_reference_previous_iteration.append(cumulated_abs_err)

print("--- previous eigvects and eigvals replacing x diag ---")
initial_guess_with_previous_iteration_and_rest_of_x = x - jnp.diag(x) + jnp.diag(prev_vals), prev_ev
accum_errs_reference_previous_iteration_and_rest_of_x = []
for i in range(ITERATIONS):
    us_vects, us_vals, _ = jax.jit(ipu_eigh)(x, num_iters=i, initial_guess=initial_guess_with_previous_iteration_and_rest_of_x)
    us_vects = np.asarray(us_vects)
    us_vals = np.asarray(us_vals)
    
    cumulated_abs_err = get_error(us_vects, us_vals, reference_ev, reference_vals, "refer")
    accum_errs_reference_previous_iteration_and_rest_of_x.append(cumulated_abs_err)




from matplotlib import pyplot as plt

plt.figure()
plt.plot(accum_errs, label="default with no external initialization")
# plt.plot(accum_errs_initialized, label="initialized with only reference eigvect")
plt.plot(accum_errs_linalg, label="initialized with linalg eigval and eigvect")
# plt.plot(accum_errs_defaults, label="initialized externaly with default x and eye(N)")
plt.plot(accum_errs_reference, label="initialized reference diag(eigvals) and eigvects")
# plt.plot(accum_errs_reference_vals, label="initialized with only reference diag(eigvals) and eye(N)")
plt.plot(accum_errs_reference_previous_iteration, label="initialized with previous iteration diag(eigvals) and eigvects")
plt.plot(accum_errs_reference_previous_iteration_and_rest_of_x, label="initialized with previous iteration x - diag(x) + diag(eigvals) and eigvects")
plt.xlabel("Iterations of ipu_eigh")
plt.ylabel("Accumulated err of abs(us_vects) - abs(vects)")
plt.yscale('log')
plt.legend()
plt.savefig("eigh_err.png")
plt.clf()


ev_diffs = []
inp_diffs = []
vals_diffs = []

#add results with 
x_axis = range(1, num_its)

def get_abs_err(inp, ref):
    abs_err = np.abs(inp) - np.abs(ref)
    cumulated_abs_err = np.sum(np.abs(abs_err))
    return cumulated_abs_err

for j in range(1, num_its):
    ev_diffs.append(get_abs_err(reference_ev_list[j], reference_ev_list[j-1]))
    inp_diffs.append(get_abs_err(refernce_input_list[j], refernce_input_list[j-1]))
    vals_diffs.append(get_abs_err(reference_vals_list[j], reference_vals_list[j-1]))


plt.figure()
plt.xlim(0,num_its)
plt.plot(x_axis, ev_diffs, label="eigenvectors from ipu_eigh compared to prev")
plt.plot(x_axis, inp_diffs, label="inputs to ipu_eigh compared to prev")
plt.plot(x_axis, vals_diffs, label="eigenvalues from ipu_eigh compared to prev")

plt.xlabel("Iterations of nanoDFT")
plt.ylabel("Accumulated err of current iteration compared to previous")
plt.yscale('log')
plt.legend()
plt.savefig("iters_err.png")
plt.clf()


ev_diffs = []
inp_diffs = []
vals_diffs = []

def get_relative_err(inp, ref):
    abs_err = np.abs(np.abs(inp) - np.abs(ref))
    rel_err = np.divide(abs_err, ref)
    return np.sum(rel_err)

for j in range(1, num_its):
    ev_diffs.append(get_relative_err(reference_ev_list[j], reference_ev_list[j-1]))
    inp_diffs.append(get_relative_err(refernce_input_list[j], refernce_input_list[j-1]))
    vals_diffs.append(get_relative_err(reference_vals_list[j], reference_vals_list[j-1]))

plt.figure()
plt.xlim(0,num_its)
plt.plot(x_axis, ev_diffs, label="eigenvectors from ipu_eigh compared to prev")
plt.plot(x_axis, inp_diffs, label="inputs to ipu_eigh compared to prev")
plt.plot(x_axis, vals_diffs, label="eigenvalues from ipu_eigh compared to prev")

plt.xlabel("Iterations of nanoDFT")
plt.ylabel("Accumulated relative err of current iteration compared to previous")
plt.yscale('log')
plt.legend()
plt.savefig("iters_err_relative.png")
plt.clf()