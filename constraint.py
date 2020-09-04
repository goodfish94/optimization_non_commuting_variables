import numpy as np

from numpy.linalg import svd
from aux_function import vectorize,get_vec_index,print_mat


class constraint():
    """
    construct constraint matrix
    :return:
    """

    def __init__(self,L):
        """

        :param L: size of the problem

        """
        self.L = L
        self.truncation = 0.0001 # truncation for svd decomposition


        self.commut_sign = -1.0





    def constraint_for_g1(self):
        #constraint a^dag a +aa^dag =delta

        L =self.L
        num_of_const = L

        const_mat = np.zeros( (num_of_const ,2*L),dtype=np.float32)
        const_vec = np.zeros(num_of_const,dtype=np.float32)



        for alpha in range(0,self.L):
            const_mat[alpha,alpha] += -1.0*self.commut_sign
            const_mat[alpha,alpha+L] += 1.0
            const_vec[alpha] += 1.0

        u, s, vh = svd( const_mat )

        num_of_const_trunc = 0
        for i in range(0,len(s)):
            if( abs(s[i]) >self.truncation ):
                num_of_const_trunc += 1

        #print( "num of const for single",num_of_const)

        new_const_vec = np.dot(np.transpose(u), const_vec)

        const_mat_truncation = np.zeros( (num_of_const_trunc,2*L),dtype=np.float32)
        const_vec_truncation = np.zeros(num_of_const_trunc, dtype=np.float32)


        const_index = 0
        for i in range(0,len(s)):
            if( abs(s[i]) <=self.truncation ):
                continue
            else:
                const_mat_truncation[const_index,:] += s[i]*vh[i,:]
                const_vec_truncation[const_index] += new_const_vec[i]
                const_index += 1



        return [const_mat_truncation,const_vec_truncation,num_of_const_trunc]



    def constraint_for_f_at_momentum_p(self,p):
        #constraint F(p,k,q) + F(p,p-k,q) = 0.0

        L =self.L
        num_of_const = self.L*2 # diagonal

        for i in range(0,self.L):
            num_of_const += 2*(self.L - i-1) #off diag

        const_mat = np.zeros( (num_of_const ,self.L*self.L),dtype=np.float32)
        const_vec = np.zeros(num_of_const,dtype=np.float32)

        const_index=0



        for k in range(0, self.L ):

            [k_pk_sign,k_pk_ind] = get_vec_index((p-k)%L,k,"real", L)
            const_mat[const_index, k] += 1.0
            const_mat[const_index, k_pk_ind] += -k_pk_sign*self.commut_sign
            const_vec[const_index] = 0.0
            const_index += 1

            [k_pk_sign,k_pk_ind] = get_vec_index((p - k)%L,k, "imag",L)
            const_mat[const_index, k_pk_ind] += -k_pk_sign*self.commut_sign
            const_vec[const_index] = 0.0
            const_index += 1
            # for p-k ==k case, this still works

        for k in range(0,self.L-1):
            for q in range(k+1,self.L):
                [k_q_sign,k_q_ind] = get_vec_index(k,q,"real",L)
                [pk_q_sign,pk_q_ind ]= get_vec_index((p-k)%L,q,"real",L)

                const_mat[const_index,pk_q_ind] += -pk_q_sign*self.commut_sign
                const_mat[const_index,k_q_ind] += k_q_sign
                const_vec[const_index] = 0.0
                const_index += 1

                [k_q_sign, k_q_ind] = get_vec_index(k, q, "imag",L)
                [pk_q_sign, pk_q_ind] = get_vec_index((p - k) % L, q, "imag",L)
                const_mat[const_index, pk_q_ind] += -pk_q_sign*self.commut_sign
                const_mat[const_index, k_q_ind] += k_q_sign
                const_vec[const_index] = 0.0
                const_index += 1




        u, s, vh = svd( const_mat )

        num_of_const_trunc = 0
        for i in range(0,len(s)):
            if( abs(s[i]) >self.truncation ):
                num_of_const_trunc += 1

        #print( "num of const f at p",p,num_of_const_trunc)

        const_mat_truncation = np.zeros( (num_of_const_trunc,self.L*self.L),dtype=np.float32)

        new_const_vec = np.dot(np.transpose((u)), const_vec)
        const_vec_truncation = np.zeros(num_of_const_trunc, dtype=np.float32)

        const_index = 0
        for i in range(0,len(s)):
            if( abs(s[i]) <=self.truncation ):
                continue
            else:
                const_mat_truncation[const_index,:] += s[i]*vh[i,:]
                const_vec_truncation[const_index] += new_const_vec[i]
                const_index += 1



        return [const_mat_truncation,const_vec_truncation,num_of_const_trunc]



    def constraint_for_two_ph_at_momentum_p(self,p):

        # the x vector is [g1, g2_pair, g2_two_ph_block]

        L =self.L
        num_of_const = L*L*4 #


        const_mat = np.zeros( (num_of_const, L +L*L*L + 4*L*L) , dtype=np.float32)
        const_vec = np.zeros( num_of_const, dtype=np.float32)


        const_index=0


        for alpha in range(0,L):
            #diagonal part, up left block
            const_mat[const_index, alpha] += 1.0

            const_mat[const_index,L+L*L*L+alpha] -= 1.0
            const_mat[const_index,L+((2*alpha-p)%L)*L*L+alpha] += self.commut_sign


            const_index +=1

            # diagonal part, down right block
            if(p==0):
                const_mat[const_index,alpha] += 2.0*self.commut_sign
            const_mat[const_index,(alpha+p)%L] += 1.0

            const_mat[const_index, L+L*L*L+(alpha+L)] += -1.0

            const_mat[const_index, L+((2*alpha+p)%L)*L*L +(alpha+p)%L ] += self.commut_sign
            if(p==0):
                const_vec[const_index] += -1.0
            const_index += 1









        for alpha in range(0,self.L-1):
            for gamma in range(alpha+1,self.L):

                #-----------left up block real part------------------------


                [sign_tmp, index_tmp] = get_vec_index(alpha, gamma, "real", L)

                const_mat[const_index, index_tmp + L + L*L*( (alpha+gamma-p)%L  )] += self.commut_sign*sign_tmp

                [sign_tmp, index_tmp] = get_vec_index(alpha,gamma, "real", 2*L)

                const_mat[ const_index, index_tmp+L+L*L*L] += -sign_tmp

                const_index +=1


                # -----------left up block imag part------------------------

                [sign_tmp, index_tmp] = get_vec_index(alpha, gamma, "imag", L)

                const_mat[const_index, index_tmp + L + L * L * ((alpha + gamma - p) % L)] += self.commut_sign*sign_tmp

                [sign_tmp, index_tmp] = get_vec_index(alpha, gamma, "imag", 2 * L)

                const_mat[const_index, index_tmp + L + L * L * L ] += -sign_tmp

                const_index += 1

                #-----------right bottom block real part-----------------


                if(p==0):
                    const_vec[const_index] +=  -1.0


                if( p==0 ):
                    const_mat[const_index, alpha ] += self.commut_sign
                    const_mat[const_index, gamma ] += self.commut_sign


                [sign_tmp, index_tmp] = get_vec_index( (alpha+p)%L, (gamma+p)%L, "real", L)

                const_mat[const_index, index_tmp + L + L * L * ((alpha + gamma + p) % L)] += self.commut_sign*sign_tmp

                [sign_tmp, index_tmp] = get_vec_index(alpha+L, gamma+L, "real", 2 * L)

                const_mat[const_index, index_tmp + L + L * L * L ] += -sign_tmp

                const_index += 1

                # -----------right bottom block imag part-----------------

                [sign_tmp, index_tmp] = get_vec_index((alpha + p) % L, (gamma + p) % L, "imag", L)

                const_mat[const_index, index_tmp + L + L * L * ((alpha + gamma + p) % L)] += self.commut_sign*sign_tmp

                [sign_tmp, index_tmp] = get_vec_index(alpha + L, gamma + L, "imag", 2 * L)

                const_mat[const_index, index_tmp + L + L * L * L] += -sign_tmp

                const_index += 1





        for alpha in range(0, self.L ):
            for gamma in range(0, self.L):

                # -----------left bottom block real part-----------------

                if(p==0):
                    const_mat[const_index, alpha] += 1.0

                if( (alpha- gamma -p)%L==0):
                    const_mat[const_index, alpha] += self.commut_sign



                [sign_tmp, index_tmp] = get_vec_index( alpha, (gamma + p) % L, "real", L)

                const_mat[const_index, index_tmp + L + L * L * ((alpha + gamma ) % L)] += sign_tmp

                [sign_tmp, index_tmp] = get_vec_index(alpha , gamma + L, "real", 2 * L)

                const_mat[const_index, index_tmp + L + L * L * L ] += -sign_tmp

                const_index += 1


                # -----------left bottom block imag part-----------------

                [sign_tmp, index_tmp] = get_vec_index(alpha, (gamma + p) % L, "imag", L)

                const_mat[const_index, index_tmp + L + L * L * ((alpha + gamma) % L)] += sign_tmp

                [sign_tmp, index_tmp] = get_vec_index(alpha, gamma + L, "imag", 2 * L)

                const_mat[const_index, index_tmp + L + L * L * L ] += -sign_tmp

                const_index += 1



        u, s, vh = svd(const_mat)

        new_const_vec = np.dot( np.transpose( ( u )) , const_vec )



        num_of_const_trunc = 0
        for i in range(0, len(s)):
            if (abs(s[i]) > self.truncation):
                num_of_const_trunc += 1

        #print("num of const two ph at p:",p,num_of_const_trunc)

        const_mat_truncation = np.zeros( (num_of_const_trunc, L+L*L*L+ 4*L*L), dtype=np.float32)
        const_vec_truncation = np.zeros( num_of_const_trunc, dtype=np.float32)

        const_index = 0





        for i in range(0, len(s)):
            if (abs(s[i]) <= self.truncation):
                continue
            else:
                const_mat_truncation[const_index, :] += s[i] * vh[i, :]
                const_vec_truncation[const_index] += new_const_vec[i]
                const_index +=1




        return [const_mat_truncation,const_vec_truncation,num_of_const_trunc]



    def constraint_for_two_pair2_at_momentum_p(self, p):
        # the x vector is [g1, g2_pair, g2_pair_2_block]


        L = self.L
        num_of_const = L * L

        const_mat = np.zeros((num_of_const, L + L * L*L+L * L), dtype=np.float32)
        const_vec = np.zeros(num_of_const, dtype=np.float32)



        const_index = 0

        for alpha in range(0,L):
            #diagonal

            const_mat[const_index,alpha ] += self.commut_sign
            const_mat[const_index,(p-alpha)%L] += self.commut_sign

            const_vec[const_index] += -1.0

            if( (alpha+alpha-p)%L==0 ):

                const_mat[const_index,alpha] += 2.0

                const_vec[const_index] += -self.commut_sign

            const_mat[ const_index, L+ p*L*L+(p-alpha)%L] += 1.0

            const_mat[ const_index, L+L*L*L + alpha ] += -1.0

            const_index += 1


        for alpha in range(0,L):
            for gamma in range(alpha+1,L):
                #off diagonal

                #---------------real part -----------------

                if( (alpha+gamma -p)%L == 0 ):
                    const_mat[const_index,alpha] += 1.0
                    const_mat[const_index,gamma] += 1.0

                    const_vec[const_index] += -self.commut_sign

                [sign_tmp, index_tmp] = get_vec_index( (p-gamma)%L, (p-alpha) % L, "real", L)

                const_mat[const_index,L+p*L*L+index_tmp] += sign_tmp

                [sign_tmp, index_tmp] = get_vec_index(alpha, gamma, "real", L)

                const_mat[const_index,L+L*L*L+index_tmp] += -sign_tmp

                const_index += 1

                # ---------------imag part -----------------

                [sign_tmp, index_tmp] = get_vec_index((p - gamma) % L, (p - alpha) % L, "imag", L)
                const_mat[const_index, L + p * L * L + index_tmp] += sign_tmp

                [sign_tmp, index_tmp] = get_vec_index(alpha, gamma, "imag", L)
                const_mat[const_index, L + L * L * L + index_tmp] += -sign_tmp

                const_index += 1




        u, s, vh = svd(const_mat)

        new_const_vec = np.dot(np.transpose((u)), const_vec)

        num_of_const_trunc = 0
        for i in range(0, len(s)):
            if (abs(s[i]) > self.truncation):
                num_of_const_trunc += 1

        #print("num of const two pair2 at p",p,num_of_const_trunc)

        const_mat_truncation = np.zeros((num_of_const_trunc, L + L * L * L +  L * L), dtype=np.float32)
        const_vec_truncation = np.zeros(num_of_const_trunc, dtype=np.float32)

        const_index = 0
        for i in range(0, len(s)):
            if (abs(s[i]) <= self.truncation):
                continue
            else:
                const_mat_truncation[const_index, :] += s[i] * vh[i, :]
                const_vec_truncation[const_index] += new_const_vec[i]
                const_index += 1

        return [const_mat_truncation, const_vec_truncation, num_of_const_trunc]



    def generate_constraint_matrix(self):

        L = self.L


        # g1

        [const_mat_truncation, const_vec_truncation, num_of_const_trunc] = self.constraint_for_g1()

        self.mat_C =np.zeros( (num_of_const_trunc,2*L+6*L*L*L), dtype=np.float32 )
        self.mat_C[:, 0:2*L ] += const_mat_truncation
        self.vec_b = const_vec_truncation.copy()

        #print(num_of_const_trunc)
        #print("single")
        #print_mat(self.mat_C,2)
        # F matrix: pair_1
        for p in range(0,L):

            [const_mat_truncation, const_vec_truncation, num_of_const_trunc]=self.constraint_for_f_at_momentum_p(p)

            mat_C_block = np.zeros( (num_of_const_trunc,2*L+6*L*L*L), dtype=np.float32)
            mat_C_block[:,2*L+4*L*L*L+p*L*L:2*L+4*L*L*L+(p+1)*L*L] += const_mat_truncation

            self.mat_C=np.vstack((self.mat_C,mat_C_block))

            vec_b_block = const_vec_truncation.copy()

            self.vec_b = np.hstack((self.vec_b,vec_b_block))
            #print(num_of_const_trunc)
            #print("two pair 1")
            #print_mat(self.mat_C, 2)

        # two ph: g1,pair_1,two_ph
        for p in range(0,L):
            [const_mat_truncation, const_vec_truncation, num_of_const_trunc] = self.constraint_for_two_ph_at_momentum_p(p)

            mat_C_block = np.zeros((num_of_const_trunc,2 * L + 6 * L * L * L), dtype=np.float32)

            mat_C_block[:,0:L] += const_mat_truncation[:,0:L] #g1
            mat_C_block[:,2*L+L*L*L*4: 2*L+L*L*L*4+L*L*L] += const_mat_truncation[:,L:L+L*L*L]#pair 1
            mat_C_block[:,2*L+p*4*L*L:2*L+(p+1)*4*L*L] += const_mat_truncation[:,L+L*L*L:L+L*L*L+4*L*L]#two ph

            self.mat_C = np.vstack((self.mat_C, mat_C_block))
            vec_b_block = const_vec_truncation.copy()
            self.vec_b = np.hstack((self.vec_b, vec_b_block))
            #print(num_of_const_trunc)
            #print("two ph 1")
            #print_mat(self.mat_C, 2)

        #two pair 2: g1,pair_1,pair_2
        for p in range(0,L):
            [const_mat_truncation, const_vec_truncation, num_of_const_trunc] = self.constraint_for_two_pair2_at_momentum_p(p)

            mat_C_block = np.zeros((num_of_const_trunc, 2 * L + 6 * L * L * L), dtype=np.float32)

            mat_C_block[:,0:L] += const_mat_truncation[:,0:L] #g1
            mat_C_block[:, 2*L+L*L*L*4:2*L+4*L*L*L + L*L*L] += const_mat_truncation[:, L:L+L*L*L]  # pair 1
            mat_C_block[:,2*L+L*L*L*4+L*L*L+p*L*L:2*L+L*L*L*4+L*L*L+(p+1)*L*L] += const_mat_truncation[:,L+L*L*L:L+L*L*L+L*L] #pair 2

            self.mat_C = np.vstack((self.mat_C, mat_C_block))

            vec_b_block = const_vec_truncation.copy()
            self.vec_b = np.hstack((self.vec_b, vec_b_block))

            #print(num_of_const_trunc)
            #print("two pair2")
            #print_mat(self.mat_C, 2)

        self.dim_const = len( self.vec_b )

        #print_mat(self.mat_C,2)
        #print("vecb")
        #print(self.vec_b)
        #print("end")






        return 1


    def generate_constraint_matrix_test(self):

        L = self.L


        # g1

        [const_mat_truncation, const_vec_truncation, num_of_const_trunc] = self.constraint_for_g1()

        self.mat_C =np.zeros( (num_of_const_trunc,2*L+6*L*L*L), dtype=np.float32 )
        self.mat_C[:, 0:2*L ] += const_mat_truncation
        self.vec_b = const_vec_truncation.copy()

        self.dim_const = len(self.vec_b)





        return 1


















