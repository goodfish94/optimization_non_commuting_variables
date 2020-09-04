from primal_dual_interior_point_NT_scaling import primal_dual_interior_point_for_nc
import numpy as np
from aux_function import vectorize,get_vec_index,transform_vec_to_matrix



pi = 3.141592653
file = open("input.txt")
if not file:
    print("can't opten input.txt")
    exit()

line = file.readline()
line = line.rstrip("\n")
line = list( filter(None,line.split(" ")))

L = int( line[0] ) # lattice size
t = float(line[1] ) # hopping
U = float(line[2] ) # Hubbard interaction
iter = int(line[3] ) # Hubbard interaction

ham_vec = np.zeros( 2*L+6*L*L*L, dtype=np.float32)

print("initializing problem, may take a few minutes")

for i in range(0,L):
    ham_vec[i] = 2.0*t*np.cos(2*pi/float(L)*i)


for p in range(0,L):

    for q in range(0,L):
        for k in range(0,L):

            [sgn,index]=get_vec_index(q,k,"real",L)
            index = index + 2*L + 4*L*L*L + p*L*L

            ham_vec[index] = sgn*2.0*U*np.cos(2*pi/float(L)*(k-q))


solver = primal_dual_interior_point_for_nc( L, ham_vec,t,U,iter)

solver.main_solver()