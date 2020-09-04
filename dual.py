import numpy as np

from aux_function import vectorize,get_vec_index,transform_vec_to_matrix
from aux_function import SQRT2

class dual():
    ' class for dual variables '

    def __init__(self, L,  z1_vec, z2_ph_vec,z2_pair1_vec,z2_pair2_vec):
        """

        :param L: lattice size
        :param z1_vec: z1 in vector form length = L*2, real
        :param z2_ph_vec:   store particle hole matrix
        :param z2_pair1_vec:  store pair 1 matrix
        :param z2_pair2_vec:  store pair 2 matrix
        :return:
        """

        self.L = L


        self.vec_z = np.zeros(2 * L + 6 * L * L * L, dtype=np.float32)

        self.vec_z[0:2 * L] += z1_vec[:]
        self.vec_z[2 * L:2 * L + 4 * L * L * L] += z2_ph_vec[:]
        self.vec_z[2 * L + 4 * L * L * L: 2 * L + 4 * L * L * L + L * L * L] += z2_pair1_vec[:]
        self.vec_z[2 * L + 4 * L * L * L + L * L * L: 2 * L + 6 * L * L * L] += z2_pair2_vec[:]







    def get_z1(self,p):
        return self.vec_z[p]


    def generate_two_ph_dual_zmatrix(self,p):
        L = self.L
        index_start = 2 * L + int(p) * 4 * L * L
        index_end = 2 * L + int(p + 1) * 4 * L * L

        zmatrix = transform_vec_to_matrix( self.vec_z[index_start:index_end], 2*self.L)
        return zmatrix


    def generate_two_pair1_dual_zmatrix(self,p):

        L = self.L
        index_start = 2 * L + 4 * L * L * L + int(p) * L * L
        index_end = 2 * L + 4 * L * L * L + int(p + 1) * L * L

        zmatrix = transform_vec_to_matrix( self.vec_z[index_start:index_end], self.L)
        return zmatrix

    def generate_two_pair2_dual_zmatrix(self,p):

        L = self.L
        index_start = 2 * L + 5 * L * L * L + int(p) * L * L
        index_end = 2 * L + 5 * L * L * L + int(p + 1) * L * L

        zmatrix = transform_vec_to_matrix( self.vec_z[index_start:index_end], self.L)
        return zmatrix



    def if_feasible(self,z_new,lower_limit):


        L = self.L



        if(  any( z_new[0:2*self.L]<=lower_limit ) ):
            # some g1 less equan then zero
            return False



        for p in range(0,self.L):
            # ph
            index_start = 2 * L + int(p) * 4 * L * L
            index_end = 2 * L + int(p + 1) * 4 * L * L
            zmatrix = transform_vec_to_matrix(z_new[index_start:index_end], 2 * L) - lower_limit * np.identity(2*L,dtype=np.float32)


            try:
                np.linalg.cholesky(zmatrix)
            except:
                #if cholesky failed, then fmatrix is not positive definite
                return False

        for p in range(0,self.L):
            # pair1
            index_start = 2 * L + 4*L*L*L + int(p)  * L * L
            index_end = 2 * L + 4*L*L*L + int(p + 1)  * L * L
            zmatrix = transform_vec_to_matrix(z_new[index_start:index_end],  L) - lower_limit * np.identity(L,dtype=np.float32)

            try:
                np.linalg.cholesky(zmatrix)
            except:
                #if cholesky failed, then fmatrix is not positive definite
                return False

        for p in range(0,self.L):
            # pair2
            index_start = 2 * L + 5*L*L*L + int(p)  * L * L
            index_end = 2 * L + 5*L*L*L + int(p + 1)  * L * L
            zmatrix = transform_vec_to_matrix(z_new[index_start:index_end],  L) - lower_limit * np.identity(L,dtype=np.float32)

            try:
                np.linalg.cholesky(zmatrix)
            except:
                #if cholesky failed, then fmatrix is not positive definite
                return False

        return True

    def find_feasibile_alpha(self,del_z,N,lower_limit):
        """

        :param del_z:   check if the zmatrix is postivie definite
        :param N: z -> z+del_z_scale*j/N with j=N,N-1,...,1, if not feasible for j=1, replace del_z_scale with del_z_scale/N

        :return: alpha such that z+alpha*del_z is feasible
        """


        del_z_scale = del_z * float(N)
        scale_alpha = float(N)

        for sweep in range(0,10):

            scale_alpha = scale_alpha/float(N)
            del_z_scale = del_z_scale/float(N)

            for j in range(0,N):

                vec_z_new = self.vec_z + del_z_scale * float(N-j)/float(N)
                alpha = scale_alpha*float(N-j)/float(N)

                if( self.if_feasible(vec_z_new,lower_limit) ):
                    return alpha

        return 0.0


























