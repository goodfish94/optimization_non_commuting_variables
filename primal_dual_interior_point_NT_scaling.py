from dual import dual
from primal import primal
from constraint import constraint
import numpy as np
import time


from numpy.linalg import inv
from numpy.linalg import eigvals
from numpy.linalg import cholesky
from numpy.linalg import svd
from numpy.linalg import eig

from aux_function import solve




from aux_function import vectorize, get_vec_index, transform_vec_to_matrix, generate_mat_basis, print_mat


class primal_dual_interior_point_for_nc():
    '  interior point solver '



    def __init__(self, L, ham_vec,t, U,max_iter):

        self.L = L
        self.max_iter = max_iter

        self.t_ = t
        self.U_ = U

        self.time_start = time.time()

        g1_vec = np.zeros(2 * self.L, dtype=np.float32)
        z1_vec = np.zeros(2 * self.L, dtype=np.float32)

        g2_ph_vec = np.zeros(4 * L * L * L, dtype=np.float32)
        z2_ph_vec = np.zeros(4 * L * L * L, dtype=np.float32)

        g2_pair1_vec = np.zeros(L * L * L, dtype=np.float32)
        z2_pair1_vec = np.zeros(L * L * L, dtype=np.float32)

        g2_pair2_vec = np.zeros(L * L * L, dtype=np.float32)
        z2_pair2_vec = np.zeros(L * L * L, dtype=np.float32)

        self.primal_var = primal(L, g1_vec, g2_ph_vec, g2_pair1_vec, g2_pair2_vec)
        self.dual_var = dual(L, z1_vec, z2_ph_vec, z2_pair1_vec, z2_pair2_vec)

        self.const_var = constraint(L)
        self.const_var.generate_constraint_matrix()
        #self.const_var.generate_constraint_matrix_test()

        self.duality_gap = np.zeros(5 * L)


        self.dim_var = len(g1_vec) + len(g2_ph_vec) + len(g2_pair1_vec) + len(g2_pair2_vec)
        self.dim_const = self.const_var.dim_const

        self.w = np.zeros( (self.dim_var,self.dim_var), dtype=np.float32 )
        self.inv_wT =  np.zeros( (self.dim_var,self.dim_var), dtype=np.float32 )




        self.vec_rx = np.zeros(self.dim_var, dtype=np.float32)
        self.vec_ry = np.zeros(self.dim_const, dtype=np.float32)


        self.vec_rz = np.zeros(self.dim_var, dtype=np.float32)
        self.vec_rd = np.zeros(self.dim_var, dtype=np.float32)

        self.wt_lambda_inv_I = np.zeros(self.dim_var,dtype=np.float32)

        self.var_y = np.zeros(self.dim_const, dtype=np.float32)

        self.del_x = np.zeros(self.dim_var, dtype=np.float32)
        self.del_z = np.zeros(self.dim_var, dtype=np.float32)
        self.del_y = np.zeros(self.const_var.dim_const, dtype=np.float32)

        self.vec_ham = ham_vec

        self.debug = False

        if (self.debug):
            self.eig_val_x = np.zeros(2 * self.L + self.L * self.L * 2 + self.L * self.L + self.L * self.L,
                                      dtype=np.complex64)
            self.eig_val_z = np.zeros(2 * self.L + self.L * self.L * 2 + self.L * self.L + self.L * self.L,
                                      dtype=np.complex64)

        self.if_combined_direction = True # if calculate combined direction


        self.convg_criteria = 0.001

        self.prev_error = 1000.0


        self.truncation = 0.001




    def single_block_solver(self):
        """

        :param self:

        :return: set up single particle duality gap, x_coeff,z_coeff , vec_rz
        """

        for p in range(0, self.L):

            xplus = self.primal_var.get_g1(p)
            xminus = self.primal_var.get_g1(p + self.L)
            zplus = self.dual_var.get_z1(p)
            zminus = self.dual_var.get_z1(p + self.L)

            if (self.debug):

                if (xplus < 0.0 or xminus < 0.0 or zplus < 0.0 or zminus < 0.0):
                    print("single block")
                    print(xplus, xminus, zplus, zminus)
                    exit(2)

                self.eig_val_x[p] = xplus
                self.eig_val_x[p + self.L] = xminus

                self.eig_val_z[p] = zplus
                self.eig_val_z[p + self.L] = zminus

            self.duality_gap[p] = np.real( xplus * zplus)

            self.duality_gap[p + self.L] = np.real(xminus * zminus)

            self.w[p,p]=np.sqrt(xplus/zplus)
            self.inv_wT[p,p] = 1.0/self.w[p,p]





            self.wt_lambda_inv_I[p] =1.0/zplus #* self.duality_gap[p]


            self.w[p+self.L, p+self.L] =np.sqrt( xminus / zminus )
            self.inv_wT[p+self.L,p+self.L] = 1.0/self.w[p+self.L, p+self.L]



            self.wt_lambda_inv_I[p+self.L] = 1.0/zminus #* self.duality_gap[p+self.L]




            self.vec_rz[p] +=  self.duality_gap[p]
            self.vec_rz[p + self.L] +=  self.duality_gap[p + self.L]

    def two_ph_block_solver(self):
        """

        :param self:

        :return: set up two_ph,  duality gap, x_coeff,z_coeff
        """

        for p in range(0, (self.L)):
            x_mat = self.primal_var.generate_two_ph_primal_matrix(p)
            z_mat = self.dual_var.generate_two_ph_dual_zmatrix(p)

            if (self.debug):
                try:
                    np.linalg.cholesky(x_mat)
                except:
                    print(" two ph xmat")
                    exit()

                try:
                    np.linalg.cholesky(z_mat)
                except:
                    print(" two ph zmat")
                    exit()

                ind_st = 2 * self.L + 2 * self.L * p
                ind_en = 2 * self.L + 2 * self.L * (p + 1)

                eig_tmp = eigvals(x_mat)
                self.eig_val_x[ind_st:ind_en] += eig_tmp[:]

                if (np.min(eig_tmp) < 0.001):
                    print("two ph xmat, too small eigvalue")
                    print(eig_tmp)
                    print("xmat")
                    print(x_mat)

                    exit()

                eig_tmp = eigvals(z_mat)
                self.eig_val_z[ind_st:ind_en] += eig_tmp[:]

                if (np.min(eig_tmp) < 0.001):
                    print("two ph zmat, too small eigvalue")
                    print(eig_tmp)
                    print("zmat")
                    print(x_mat)

                    exit()

            xz_dot_product = np.dot(x_mat, z_mat)
            xz_dot_product += np.transpose(np.conj(xz_dot_product))
            self.duality_gap[2 * self.L + p] += 0.5*np.real(np.trace(xz_dot_product)) / (2.0 * self.L)

            [block_w,block_inv_wT] = self.cal_scaling_mat(x_mat,z_mat,2*self.L)

            index_st = 2*self.L + 4*self.L*self.L * p
            index_en = 2*self.L + 4*self.L*self.L * (p+1)

            self.w[index_st:index_en, index_st:index_en] += block_w[:,:]
            self.inv_wT[index_st:index_en, index_st:index_en] += block_inv_wT[:, :]



            rz_mat =  np.identity(2 * self.L, dtype=np.complex64)
            vec_d_block = vectorize(rz_mat, 2 * self.L)

            self.vec_rz[index_st: index_en] += self.duality_gap[2 * self.L + p] *vec_d_block[:]

            vec_lambda = np.dot(block_w, self.dual_var.vec_z[index_st:index_en])
            for i in range(0,2*self.L):
                rz_mat[i,i] = rz_mat[i,i]/vec_lambda[i]

            self.wt_lambda_inv_I[index_st: index_en] += np.dot(np.transpose(block_w),vectorize(rz_mat, 2 * self.L))



    def two_pair1_block_solver(self):
        """
        :param self:
        :return: set up two_ph,  duality gap, x_coeff,z_coeff
        """

        for p in range(0, (self.L)):
            x_mat = self.primal_var.generate_two_pair1_primal_matrix(p)
            z_mat = self.dual_var.generate_two_pair1_dual_zmatrix(p)

            if (self.debug):

                try:
                    np.linalg.cholesky(x_mat)
                except:
                    print(" two pair1 xmat")
                    exit()

                try:
                    np.linalg.cholesky(z_mat)
                except:
                    print(" two pair1 zmat")
                    exit()

                ind_st = 2 * self.L + 2 * self.L * self.L + self.L * p
                ind_en = 2 * self.L + 2 * self.L * self.L + self.L * (p + 1)

                eig_tmp = eigvals(x_mat)
                self.eig_val_x[ind_st:ind_en] += eig_tmp[:]

                if (np.min(eig_tmp) < 0.00001):
                    print("two pair1 xmat, too small eigvalue")
                    print(eig_tmp)
                    print("xmat")
                    print(x_mat)
                    print(z_mat)
                    w,v=eig(x_mat)

                    print("eigval")
                    print(w)
                    print("eigvec")
                    print(v)

                    exit()

                eig_tmp = eigvals(z_mat)
                self.eig_val_z[ind_st:ind_en] += eig_tmp[:]

                if (np.min(eig_tmp) < 0.001):
                    print("two pair2 zmat, too small eigvalue")
                    print(eig_tmp)
                    print("zmat")
                    print(x_mat)

                    exit()

            xz_dot_product = np.dot(x_mat, z_mat)
            xz_dot_product += np.transpose(np.conj(xz_dot_product))
            self.duality_gap[2 * self.L + self.L + p] = 0.5*np.real(np.trace(xz_dot_product)) / float(self.L)

            [block_w,block_inv_wT] = self.cal_scaling_mat(x_mat, z_mat,  self.L)

            index_st = 2 * self.L + 4 * self.L * self.L * self.L + (p)*self.L*self.L
            index_en = 2 * self.L + 4 * self.L * self.L * self.L + (p+1)*self.L*self.L



            self.w[index_st:index_en, index_st:index_en] += block_w[:, :]
            self.inv_wT[index_st:index_en, index_st:index_en] += block_inv_wT[:, :]



            rz_mat =  np.identity(self.L,dtype=np.complex64)

            vec_d_block = vectorize(rz_mat, self.L)

            self.vec_rz[index_st:index_en] += self.duality_gap[2 * self.L + self.L + p] *vec_d_block[:]

            vec_lambda = np.dot(block_w, self.dual_var.vec_z[index_st:index_en])
            for i in range(0,  self.L):
                rz_mat[i, i] = rz_mat[i, i] / vec_lambda[i]

            self.wt_lambda_inv_I[index_st: index_en] += np.dot( np.transpose(block_w),vectorize(rz_mat,  self.L) )



    def two_pair2_block_solver(self):
        """
        :param self:
        :return: set up two_ph,  duality gap, x_coeff,z_coeff
        """

        for p in range(0, (self.L)):
            x_mat = self.primal_var.generate_two_pair2_primal_matrix(p)
            z_mat = self.dual_var.generate_two_pair2_dual_zmatrix(p)

            if (self.debug):

                try:
                    np.linalg.cholesky(x_mat)
                except:
                    print(" two pair2 xmat")
                    exit()

                try:
                    np.linalg.cholesky(z_mat)
                except:
                    print(" two pair2 zmat")
                    exit()

                ind_st = 2 * self.L + 3 * self.L * self.L + self.L * p
                ind_en = 2 * self.L + 3 * self.L * self.L + self.L * (p + 1)

                eig_tmp = eigvals(x_mat)
                self.eig_val_x[ind_st:ind_en] += eig_tmp[:]

                if (np.min(eig_tmp) < 0.001):
                    print("two pair2 xmat, too small eigvalue")
                    print(eig_tmp)
                    print("xmat")
                    print(x_mat)

                    exit()

                eig_tmp = eigvals(z_mat)
                self.eig_val_z[ind_st:ind_en] += eig_tmp[:]

                if (np.min(eig_tmp) < 0.001):
                    print("two pair2 zmat, too small eigvalue")
                    print(eig_tmp)
                    print("zmat")
                    print(x_mat)

                    exit()

            xz_dot_product = np.dot(x_mat, z_mat)
            xz_dot_product += np.transpose(np.conj(xz_dot_product))
            self.duality_gap[2 * self.L + self.L + self.L + p] = 0.5*np.real(np.trace(xz_dot_product)) / float(self.L)

            [block_w,block_inv_wT] = self.cal_scaling_mat(x_mat, z_mat, self.L)

            index_st = 2 * self.L + 5 * self.L * self.L * self.L + p * self.L * self.L
            index_en = 2 * self.L + 5 * self.L * self.L * self.L + (p+1) * self.L * self.L

            self.w[index_st:index_en, index_st:index_en] += block_w[:, :]
            self.inv_wT[index_st:index_en, index_st:index_en] += block_inv_wT[:, :]



            rz_mat =  np.identity(self.L,dtype=np.complex64)


            vec_d_block = vectorize(rz_mat, self.L)

            self.vec_rz[index_st:index_en] += self.duality_gap[2 * self.L + self.L + self.L + p] *vec_d_block[:]

            vec_lambda = np.dot(block_w, self.dual_var.vec_z[index_st:index_en])
            for i in range(0, self.L):
                rz_mat[i, i] = rz_mat[i, i] / vec_lambda[i]

            self.wt_lambda_inv_I[index_st: index_en] += np.dot( np.transpose(block_w),vectorize(rz_mat, self.L))






    def cal_scaling_mat(self, xmat,zmat,blocksize):

        block_w = np.zeros( (blocksize*blocksize,blocksize*blocksize), dtype=np.float32)
        block_inv_wT = np.zeros((blocksize * blocksize, blocksize * blocksize), dtype=np.float32)


        L1 = cholesky(xmat)
        L2 = cholesky(zmat)

        u, s, vh = svd(np.dot( np.conj(np.transpose(L2)), L1  ))

        R = np.dot(L1, np.transpose(np.conj(vh)))
        Rinv = np.dot( np.transpose(np.conj(u)), np.transpose(np.conj(L2)))

        for i in range(0,len(s)):
            Rinv[i,:] = Rinv[i,:]/np.sqrt(s[i])

        Rinvdag = np.transpose(np.conj(Rinv))

        for i in range(0,len(s)):
            R[:,i] = R[:,i] /np.sqrt(s[i])

        Rdag = np.transpose(np.conj(R))



        for alpha in range(0,blocksize*blocksize):
            basis_alpha = generate_mat_basis(alpha,blocksize)
            for gamma in range(0,blocksize*blocksize):
                basis_gamma = generate_mat_basis(gamma,blocksize)

                block_w[alpha,gamma] += np.real(np.trace( np.dot(np.dot( np.dot(Rdag,basis_gamma),R ), basis_alpha) ))
                block_inv_wT[alpha,gamma] += np.real(np.trace( np.dot(np.dot( np.dot(Rinv,basis_gamma),Rinvdag ), basis_alpha) ))



        return [block_w,block_inv_wT]



    def initialization(self):

        L=self.L
        mu=0.1


        linear_eq_A=np.dot(self.const_var.mat_C,np.transpose(self.const_var.mat_C))
        linear_eq_b = -self.const_var.vec_b - np.dot( self.const_var.mat_C, self.vec_ham)

        self.var_y = solve(linear_eq_A, linear_eq_b)



        temp_x = -np.dot(np.transpose(self.const_var.mat_C),self.var_y)-self.vec_ham

        self.var_y = self.var_y
        temp_x = temp_x


        for i in range(0,2*L):
            if(temp_x[i]>0.0):
                self.primal_var.vec_x[i]=temp_x[i]
                self.dual_var.vec_z[i] = mu/temp_x[i]
            else:
                self.primal_var.vec_x[i] = 1.0
                self.dual_var.vec_z[i] = mu



        for p in range(0,L):
            index_st=2*L+p*(L*L)*4
            index_en=index_st + 4*L*L
            temp_mat = transform_vec_to_matrix(temp_x[index_st:index_en],2*L)

            eig = eigvals(temp_mat)
            eig_min = np.min(eig)

            if(eig_min>self.truncation):
                self.primal_var.vec_x[index_st:index_en] = temp_x[index_st:index_en].copy()
            else:
                temp_mat = temp_mat + (1.0+abs(eig_min))*np.identity(2*L,dtype=np.complex64)
                self.primal_var.vec_x[index_st:index_en] = vectorize(temp_mat,2*L)

            self.dual_var.vec_z[index_st:index_en] = vectorize(mu*inv(temp_mat),2*L)


        for p in range(0, L):
            index_st = 2 * L +4* L*L*L + p * (L * L)
            index_en = index_st + (L * L)
            temp_mat = transform_vec_to_matrix(temp_x[index_st:index_en],  L)

            eig = eigvals(temp_mat)
            eig_min = np.min(eig)

            if (eig_min > self.truncation):
                self.primal_var.vec_x[index_st:index_en] = temp_x[index_st:index_en].copy()
            else:
                temp_mat = temp_mat + (1.0 + abs(eig_min)) * np.identity( L, dtype=np.complex64)
                self.primal_var.vec_x[index_st:index_en] = vectorize(temp_mat,  L)

            self.dual_var.vec_z[index_st:index_en] = vectorize(mu*inv(temp_mat), L)


        for p in range(0, L):
            index_st = 2 * L +5* L*L*L + p * (L * L)
            index_en = index_st + (L * L)
            temp_mat = transform_vec_to_matrix(temp_x[index_st:index_en],  L)

            eig = eigvals(temp_mat)
            eig_min = np.min(eig)


            if (eig_min > self.truncation):
                self.primal_var.vec_x[index_st:index_en] = temp_x[index_st:index_en].copy()
            else:
                temp_mat = temp_mat + (1.0 + abs(eig_min)) * np.identity( L, dtype=np.complex64)
                self.primal_var.vec_x[index_st:index_en] = vectorize(temp_mat,  L)

            self.dual_var.vec_z[index_st:index_en] = vectorize(mu*inv(temp_mat), L)


        rx = self.vec_ham + np.dot(np.transpose(self.const_var.mat_C), (self.var_y )) - ( self.dual_var.vec_z )

        ry = -self.const_var.vec_b + np.dot(self.const_var.mat_C, (self.primal_var.vec_x ))







    def if_del_solution(self):
        x = self.primal_var.vec_x +self.del_x
        z = self.dual_var.vec_z + self.del_z
        y = self.var_y + self.del_y

        rx = self.vec_ham +  np.dot(np.transpose(self.const_var.mat_C), y ) - z
        ry = np.dot(self.const_var.mat_C,x) - self.const_var.vec_b

        print("rx,ry if_solution")
        print(np.max(np.abs(rx)), np.max(np.abs(ry)))



    def test_direction(self,alpha):

        print("###################################")
        print("rx,ry,dual previous")

        rx_prev = self.vec_ham + np.dot(np.transpose(self.const_var.mat_C), (self.var_y )) - ( self.dual_var.vec_z )

        ry_prev = -self.const_var.vec_b + np.dot(self.const_var.mat_C, (self.primal_var.vec_x ))


        print(np.max(np.abs(rx_prev)),np.max(np.abs(ry_prev)))

        rx = self.vec_ham + np.dot(np.transpose(self.const_var.mat_C), (self.var_y + alpha*self.del_y)) - ( self.dual_var.vec_z + alpha*self.del_z)

        ry = -self.const_var.vec_b + np.dot(self.const_var.mat_C, (self.primal_var.vec_x + alpha*self.del_x))

        print("rx,ry,alpha=%f"%alpha)
        print(np.max(np.abs(rx)), np.max(np.abs(ry)))

        rx = self.vec_ham + np.dot(np.transpose(self.const_var.mat_C), (self.var_y+self.del_y)) - (self.dual_var.vec_z+self.del_z)

        ry = -self.const_var.vec_b + np.dot(self.const_var.mat_C, (self.primal_var.vec_x+self.del_x))

        print("rx,ry,alpha=1")


        print(np.max(np.abs(rx)),np.max(np.abs(ry)))



        if(self.if_debug and np.max(np.abs(rx)) - np.max(np.abs(rx_prev)) > 10.0*self.convg_criteria and np.max(np.abs(rx))>self.convg_criteria):
            print("rx failed")
            print(np.max(np.abs(rx)) - np.max(np.abs(rx_prev) ))
            self.if_del_solution()
            exit()

        if (self.if_debug and np.max(np.abs(ry)) - np.max(np.abs(ry_prev)) > 10.0*self.convg_criteria and np.max(np.abs(ry))>self.convg_criteria):
            print("ry failed")
            print(np.max(np.abs(ry)) - np.max(np.abs(ry_prev)))
            self.if_del_solution()
            exit()

        print("###################################")









    def each_iter_solver(self):

        self.duality_gap.fill(0.0)
        self.w.fill(0.0)
        self.inv_wT.fill(0.0)
        self.wt_lambda_inv_I.fill(0.0)


        self.vec_rx.fill(0.0)
        self.vec_ry.fill(0.0)
        self.vec_rz.fill(0.0)
        self.vec_rd.fill(0.0)

        self.del_x.fill(0.0)
        self.del_y.fill(0.0)
        self.del_z.fill(0.0)

        if (self.debug):
            self.eig_val_x.fill(0.0)
            self.eig_val_z.fill(0.0)

        self.single_block_solver()

        self.two_ph_block_solver()

        self.two_pair1_block_solver()

        self.two_pair2_block_solver()

        if (self.debug):
            print("eig for x", np.min(self.eig_val_x))
            print("eig for z", np.min(self.eig_val_z))

            print("eig pos for x", np.argmin(self.eig_val_x))
            print("eig pos for z", np.argmin(self.eig_val_z))











        self.vec_rx += self.vec_ham + np.dot(np.transpose(self.const_var.mat_C), self.var_y) - self.dual_var.vec_z

        self.vec_ry += -self.const_var.vec_b + np.dot(self.const_var.mat_C, self.primal_var.vec_x)

        convg_error = np.max([np.max(self.duality_gap), np.max(np.abs(self.vec_rx)), np.max(np.abs(self.vec_ry))])

        if(self.debug and convg_error-self.prev_error>self.convg_criteria):
            print(convg_error,self.prev_error)
            print(np.max(self.duality_gap))
            print("failed update")
            exit()

        self.prev_error = convg_error

        self.solve_kkt(self.vec_rx, self.vec_ry, self.primal_var.vec_x)










        alpha_x = self.primal_var.find_feasibile_alpha(self.del_x, 30,0.0001*self.convg_criteria)
        alpha_z = self.dual_var.find_feasibile_alpha(self.del_z, 30,0.0001*self.convg_criteria)



        alpha = np.min([alpha_x,alpha_z])

        if(self.debug):
            self.test_direction(alpha)

        if(self.if_combined_direction):
            rho = np.dot(self.primal_var.vec_x+ self.del_x *alpha, self.dual_var.vec_z+ self.del_z *alpha)

            rho = rho/np.dot(self.primal_var.vec_x , self.dual_var.vec_z )

            sigma = np.min([1.0,rho])
            sigma = np.max([0.0,sigma])

            sigma = np.power(sigma,3)


            mu = np.sum(self.duality_gap[0:2*self.L])+np.sum(self.duality_gap[2*self.L:3*self.L])*2*self.L + np.sum( self.duality_gap[3*self.L:5*self.L])*self.L
            mu = mu/(2*self.L+ 4*self.L*self.L)





            self.vec_rd =  self.primal_var.vec_x - sigma*self.wt_lambda_inv_I*mu

            self.res = np.max( [np.max(np.abs(self.vec_rx)), np.max(np.abs(self.vec_ry)) ] )

            self.vec_rx = (1.0-sigma) * self.vec_rx
            self.vec_ry = (1.0-sigma)*self.vec_ry



            self.solve_kkt(self.vec_rx,self.vec_ry,self.vec_rd)






            alpha_x = self.primal_var.find_feasibile_alpha(self.del_x, 100,0.0001*self.convg_criteria)
            alpha_z = self.dual_var.find_feasibile_alpha(self.del_z, 100,0.0001*self.convg_criteria)



            alpha = np.min([alpha_x, alpha_z])

            if(self.debug):
                self.test_direction(alpha)


        print("error=",convg_error)

        if(convg_error<self.convg_criteria):

            print("solved")
            print("time of calculation = %f"%(time.time()-self.time_start) )
            self.print_solution()
            self.print_density_density()

            exit()


        self.primal_var.vec_x += alpha*0.9 * self.del_x
        self.dual_var.vec_z += alpha*0.9 * self.del_z
        self.var_y += alpha*0.9 * self.del_y



    def main_solver(self):



        self.initialization()



        self.vec_rx.fill(0.0)
        self.vec_ry.fill(0.0)

        self.vec_rx += self.vec_ham + np.dot(np.transpose(self.const_var.mat_C), self.var_y) - self.dual_var.vec_z
        self.vec_ry += -self.const_var.vec_b + np.dot(self.const_var.mat_C, self.primal_var.vec_x)






        file = open("iteration_log.txt","w+")

        file.write("iteration_number \t maximum_of_duality_gap \t maximum_of_residues \t time \n ")

        for i in range(1, self.max_iter):
            print("iteration ", i)
            self.each_iter_solver()



            time_now = time.time() - self.time_start

            file.write(str(i) + "\t" + str(np.max(self.duality_gap)) + "\t" + str(self.res)+ "\t" +  str(time_now) + "\n" )

            if( i%10 ==0 ):
                self.print_solution()



        file.close()


        print("Doesn't converge after %d iterations"%self.max_iter)









    def solve_kkt(self,rx,ry,rz):

        wTw = np.dot(np.transpose(self.w),self.w)

        mat_A = np.zeros([2 * self.dim_var + self.dim_const, 2 * self.dim_var + self.dim_const], dtype=np.float32)

        mat_A[0:self.dim_var, self.dim_var:self.dim_var + self.dim_const] += np.transpose(self.const_var.mat_C)


        index_st = self.dim_var + self.dim_const
        index_en = 2 * self.dim_var + self.dim_const
        mat_A[0:self.dim_var, index_st:index_en] += -np.identity(self.dim_var, dtype=np.float32)


        index_st = self.dim_var
        index_en = self.dim_var + self.dim_const
        mat_A[index_st:index_en, 0:self.dim_var] += self.const_var.mat_C

        index_st = self.dim_var + self.dim_const
        index_en = 2 * self.dim_var + self.dim_const
        mat_A[index_st:index_en, 0:self.dim_var] += np.identity(self.dim_var, dtype=np.float32)

        index_st = self.dim_var + self.dim_const
        index_en = index_st + self.dim_var
        mat_A[index_st:index_en, index_st:index_en] = wTw

        vec_b = -rx
        vec_b = np.hstack( (vec_b,-ry) )
        vec_b = np.hstack( (vec_b,-rz) )

        x = solve(mat_A,vec_b)


        if(self.debug):
            tmp = np.dot(mat_A,x)-vec_b
            print("linear solver accuracy:",np.max(np.abs(tmp)))

        self.del_x.fill(0.0)
        self.del_y.fill(0.0)
        self.del_z.fill(0.0)

        self.del_x += x[0:self.dim_var]
        self.del_y += x[self.dim_var : self.dim_var + self.dim_const]
        self.del_z += x[self.dim_var+self.dim_const: 2*self.dim_var + self.dim_const]

        return 1

    #
    # def cal_Mehrotra_correction(self):
    #
    #     L= self.L
    #
    #     m= np.zeros(self.dim_var,dtype=np.float32)
    #
    #     winvTdel_x = np.dot(self.inv_wT,self.del_x)
    #     wdel_z = np.dot(self.w, self.del_z)
    #
    #
    #     lambda_ = np.dot(self.w, self.dual_var.vec_z)
    #
    #
    #     for i in range(0,2*self.L):
    #         m[i] += wdel_z[i]*winvTdel_x[i]
    #         m[i] = m[i]/lambda_[i]
    #
    #     for p in range(0,L):
    #
    #         index_st = 2*L+4*p*L*L
    #         index_en = index_st + 4*L*L
    #
    #         z_mat = transform_vec_to_matrix(wdel_z[index_st:index_en],2*L)
    #         x_mat = transform_vec_to_matrix(winvTdel_x[index_st:index_en],2*L)
    #
    #         xz_mat =0.5 * (np.dot(z_mat,x_mat) + np.dot(x_mat,z_mat))
    #
    #         vec_lambda = lambda_[index_st:index_st+2*L]
    #
    #         for alpha in range(0,2*L):
    #             for gamma in range(0,2*L):
    #                 xz_mat[alpha,gamma] = xz_mat[alpha,gamma]*2.0/ (vec_lambda[alpha] + vec_lambda[gamma])
    #
    #         m[index_st:index_en] += vectorize(xz_mat, 2*L)
    #
    #     for p in range(0, L):
    #
    #         index_st = 2 * L + 4 * L * L * L + p*L*L
    #         index_en = index_st + L*L
    #
    #         z_mat = transform_vec_to_matrix(wdel_z[index_st:index_en], L)
    #         x_mat = transform_vec_to_matrix(winvTdel_x[index_st:index_en],  L)
    #
    #         xz_mat = 0.5 * (np.dot(z_mat, x_mat) + np.dot(x_mat, z_mat))
    #
    #         vec_lambda = lambda_[index_st:index_st +  L]
    #
    #         for alpha in range(0,L):
    #             for gamma in range(0,L):
    #                 xz_mat[alpha,gamma] = xz_mat[alpha,gamma]*2.0/ (vec_lambda[alpha] + vec_lambda[gamma])
    #
    #
    #
    #         m[index_st:index_en] += vectorize(xz_mat, L)
    #
    #     for p in range(0, L):
    #
    #         index_st = 2 * L + 5 * L * L * L + p*L*L
    #         index_en = index_st + L*L
    #
    #         z_mat = transform_vec_to_matrix(wdel_z[index_st:index_en], L)
    #         x_mat = transform_vec_to_matrix(winvTdel_x[index_st:index_en],  L)
    #
    #         xz_mat = 0.5 * (np.dot(z_mat, x_mat) + np.dot(x_mat, z_mat))
    #
    #
    #         vec_lambda = lambda_[index_st:index_st +  L]
    #
    #         for alpha in range(0,L):
    #             for gamma in range(0,L):
    #                 xz_mat[alpha,gamma] = xz_mat[alpha,gamma]*2.0/ (vec_lambda[alpha] + vec_lambda[gamma])
    #
    #
    #
    #         m[index_st:index_en] = vectorize(xz_mat, L)
    #
    #
    #
    #     m = np.dot(np.transpose(self.w),m)
    #
    #     return m
    #
    #
    #
    #
    #
    #


    def print_solution(self):

        L= self.L

        file = open("solution.txt","w+")

        if( not file):
            print( " Can't create solution.txt file" )


        file.write("Spinless Fermi Hubbard Model\n")
        file.write("Lattice size L = %d \n"%self.L)
        file.write("Hopping amplitude t = %f \n "% self.t_)
        file.write("Hubbard interaction U = %f \n"%self.U_)


        energy = np.dot( self.primal_var.vec_x,self.vec_ham )

        file.write( "p^star(lowerst energy) = %f"%energy)


        file.write("\n\n\n")
        file.write("******************************************\n")
        file.write("\n\n\n")

        file.write("One particle green's function\n\n")



        for i in range(0,L):
            file.write("<a^dag(p) a(p)>=%f\n"%self.primal_var.vec_x[i])

        for i in range(0, L):
            file.write("<a(p) a^dag (p)>=%f \n" % self.primal_var.vec_x[i+L])

        file.write("\n\n\n\n")
        file.write("******************************************************")
        file.write("\n\n\n\n")

        file.write("Two particle green's function in the particle hole channel with basis { a^dag_{i}a_{i-p} , a_{i}a_{i+p}^dag }\n \n\n" )

        for p in range(0,L):
            file.write("\n\n---------At momentum p=%f------------ \n\n"% p)

            mat = self.primal_var.generate_two_ph_primal_matrix(p)

            for i in range(0,2*L):
                for j in range(0,2*L):
                    file.write(str(mat[i][j]) + " ")
                file.write("\n")

        file.write("\n\n\n\n")
        file.write("******************************************************")
        file.write("\n\n\n\n")

        file.write("Two particle green's function in the pariticle-partilce channel 1 with basis { a^dag_{i}a^\dag_{p-i} } \n\n")


        for p in range(0, L):
            file.write("\n\n---------At momentum p=%f------------ \n\n"% p)

            mat = self.primal_var.generate_two_pair1_primal_matrix(p)

            for i in range(0,  L):
                for j in range(0,  L):
                    file.write(str(mat[i][j]) + " ")
                file.write("\n")

        file.write("\n\n\n\n")
        file.write("******************************************************")
        file.write("\n\n\n\n")

        file.write( "Two particle green's function in the pariticle-partilce channel 2 with basis { a_{i}a_{p-i} } \n\n")


        for p in range(0, L):
            file.write("\n\n---------At momentum p=%f------------ \n\n"% p)

            mat = self.primal_var.generate_two_pair2_primal_matrix(p)

            for i in range(0, L):
                for j in range(0, L):
                    file.write(str(mat[i][j]) + " ")
                file.write("\n")
        file.close()









    def print_density_density(self):

        L=self.L
        Pi = 3.141592653

        nn = np.zeros(L,dtype=np.float32)

        for p in range(0,L):
            xmat = self.primal_var.generate_two_ph_primal_matrix(p)

            for x in range(0,L):
                for alpha in range(0,L):
                    for gamma in range(0,L):
                        nn[x] += np.real(xmat[alpha,gamma])*np.cos(2*Pi/float(L) * p*x)


        file = open("nn.txt","w+")
        for i in range(0,L):
            file.write( str(i) + " " + str(nn[i]/float(L)) + "\n")
        file.close()













