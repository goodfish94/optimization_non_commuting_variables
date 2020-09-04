import numpy as np

from aux_function import vectorize,get_vec_index,transform_vec_to_matrix

from aux_function import SQRT2


class primal():
    """
    primal variable
    """

    def __init__(self, L, g1_vec,g2_ph_vec,g2_pair1_vec,g2_pair2_vec):
        self.L = L

        #self.g1_vec = g1_vec #2*L
        #self.g2_ph_vec=g2_ph_vec #4*L*L*L
        #self.g2_pair1_vec=g2_pair1_vec #L*L*L
        #self.g2_pair2_vec=g2_pair2_vec #L*L*L

        self.vec_x = np.zeros( 2*L+6*L*L*L , dtype=np.float32)

        self.vec_x[0:2*L] += g1_vec[:]
        self.vec_x[2*L:2*L+4*L*L*L] += g2_ph_vec[:]
        self.vec_x[2*L+4*L*L*L : 2*L+5*L*L*L] += g2_pair1_vec[:]
        self.vec_x[2*L+5*L*L*L : 2*L + 6*L*L*L] += g2_pair2_vec[:]





    def get_g1(self,p):
        return self.vec_x[p]




















    def generate_two_ph_primal_matrix(self,p):
        L =self.L
        index_start = 2*L+ int(p)*4*L*L
        index_end = 2*L+ int(p+1)*4*L*L


        fmatrix = transform_vec_to_matrix(self.vec_x[index_start:index_end],2 * self.L)
        return fmatrix

    def generate_two_pair1_primal_matrix(self,p):

        L = self.L
        index_start = 2*L +  4*L*L*L + int(p)*L*L
        index_end = 2*L + 4*L*L*L + int(p+1)*L*L

        fmatrix =  transform_vec_to_matrix( self.vec_x[index_start:index_end], self.L)
        return fmatrix

    def generate_two_pair2_primal_matrix(self,p):

        L = self.L
        index_start = 2 * L + 5 * L * L * L + int(p) * L * L
        index_end = 2 * L + 5 * L * L * L + int(p + 1) * L * L



        fmatrix =  transform_vec_to_matrix( self.vec_x[index_start:index_end], self.L)
        return fmatrix





    def if_feasible(self,x_new,lower_limit):

        L = self.L

        #print("xnew",x_new[0:2*self.L])

        if(  any( x_new[0:2*self .L]<=lower_limit ) ):
            # some g1 less equal then zero
            return False



        for p in range(0,self.L):
            # ph
            index_start = 2 * L + int(p) * 4 * L * L
            index_end = 2 * L + int(p + 1) * 4 * L * L
            fmatrix = transform_vec_to_matrix(x_new[index_start:index_end], 2 * L) - lower_limit * np.identity(2*L,dtype=np.float32)

            try:
                np.linalg.cholesky(fmatrix)
            except:
                #if cholesky failed, then fmatrix is not positive definite
                return False

        for p in range(0,self.L):
            # pair1
            index_start = 2 * L + 4*L*L*L + int(p) * L * L
            index_end = 2 * L + 4*L*L*L + int(p + 1) * L * L
            fmatrix = transform_vec_to_matrix(x_new[index_start:index_end],  L)- lower_limit * np.identity(L,dtype=np.float32)

            try:
                np.linalg.cholesky(fmatrix)
            except:
                #if cholesky failed, then fmatrix is not positive definite
                return False

        for p in range(0,self.L):
            # pair2
            index_start = 2 * L + 5*L*L*L + int(p)  * L * L
            index_end = 2 * L + 5*L*L*L + int(p + 1)  * L * L
            fmatrix = transform_vec_to_matrix(x_new[index_start:index_end],  L)- lower_limit * np.identity(L,dtype=np.float32)

            try:
                np.linalg.cholesky(fmatrix)
            except:
                #if cholesky failed, then fmatrix is not positive definite
                return False

        return True









    def find_feasibile_alpha(self,del_x,N,lower_limit):
        """

        :param del_x:   check if the fmatrix is postivie definite
        :param N: x -> x+del_x_scale*j/N with j=N,N-1,...,1, if not feasible for j=1, replace del_x_scale with del_x/N
        :return: alpha such that x+alpha*del_x is feasible

        """


        del_x_scale = del_x * float(N)
        scale_alpha = float(N)

        for sweep in range(0,10):

            scale_alpha = scale_alpha/float(N)
            del_x_scale = del_x_scale/float(N)

            for j in range(0,N):

                vec_x_new = self.vec_x + del_x_scale * float(N-j)/float(N)
                alpha = scale_alpha*float(N-j)/float(N)


                if( self.if_feasible(vec_x_new,lower_limit) ):
                    return alpha


        return 0.0





































