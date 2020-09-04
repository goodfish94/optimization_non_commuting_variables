import numpy as np
import scipy


SQRT2 = np.sqrt(2.0)


def vectorize(matrix, dim):
    """

    :param matrix:  Hermitian matrix
    :param dim: dimension of the matrix
    :return: dim^2 vector
    """

    vector = np.zeros( dim*dim ,dtype=np.float32)

    vector[0:dim] = np.real(matrix.diagonal())

    index = dim

    for i in range(0,dim-1):

        vector[ index:index + dim - i -1] = np.real(matrix[ i, i+1:dim ])*SQRT2

        index = index+dim-i-1

    for j in range(0,dim-1):
        vector[ index:index+dim-j-1] = -np.imag(matrix[j, j+1:dim])*SQRT2
        index = index + dim-j-1

    return vector

def get_vec_index( alpha, gamma,  real_or_imag,block_size):
    """

    :param alpha: row index
    :param gamma: col index
    :param real_or_imag: "real" or "imag
    :param block_size: size of the matrix block_size^2
    :return: the index in the vecotrize(mat)
    """




    if(alpha==gamma):
        # diagonal term
        if(real_or_imag=="real"):
            return [1.0,alpha]
        else:
            return [0.0,alpha]



    if(real_or_imag == "real"):
        index = block_size
        sgn = 1.0/SQRT2
    else:
        index = block_size + (block_size*(block_size-1))/2
        sgn = -1.0/SQRT2


    if(alpha>gamma):
        alpha,gamma = gamma,alpha
        sgn = 1.0/SQRT2




    index = index + block_size*alpha - ( (alpha+1)*alpha )/2 + gamma-alpha-1

    return [sgn,int(index)]



def transform_vec_to_matrix(vec, dim):
    """

    :param vec: vector with lenght dim*dim
    :param dim: dimension of the matrix
    :return: hermtion matrix
    """

    mat = np.zeros((dim,dim), dtype=np.complex64)


    for i in range(0,dim):
        mat[i,i]=vec[i]

    index = dim

    for i in range(0,dim-1):
        mat[i, i+1:dim] += vec[index:index+dim-i-1]/SQRT2
        mat[i+1:dim,i] += vec[index:index+dim-i-1]/SQRT2
        index += dim-i-1
    for i in range(0,dim-1):
        mat[i,i+1:dim] += -vec[index:index+dim-i-1]*np.complex(0.0,1.0)/SQRT2
        mat[i + 1:dim, i] += vec[index:index + dim - i - 1] * np.complex(0.0, 1.0)/SQRT2
        index += dim-i-1

    return mat



def generate_mat_basis(ind, dim):
    """

    :param ind: index of the vector
    :param dim: dimension of matrix
    :return: matrix basis
    """

    mat = np.zeros((dim,dim), dtype=np.complex64)

    if(ind<dim):
        #diagonal

        mat[ind,ind] = 1.0

        return mat

    count = dim

    for i in range(0,dim):
        if( ind>= count and ind< count + dim-1-i):
            j = ind -count +1 + i

            mat[i,j] = 1.0/SQRT2
            mat[j,i] = 1.0/SQRT2

            return mat
        count = count + dim-1-i

    for i in range(0,dim):
        if (ind >= count and ind < count + dim - 1 - i):
            j = ind - count + 1 + i

            mat[i, j] = -np.complex(0.0,1.0)/SQRT2
            mat[j, i] = np.complex(0.0,1.0)/SQRT2

            return mat
        count = count + dim - 1 - i

    print("generate mat basis error")
    exit(1)


def print_mat(mat,digit):
    print("-------------------------------")
    for i in range(0,len(mat)):
        for j in range(0,len(mat[i])):
            num = mat[i][j]
            print(round(num.real, digit) + round(num.imag, digit) * 1j , end=" ")

        print("\n")
    print("-------------------------------")


def foward_substitution(L,b):
    """
    solve Lx=b
    :param L:
    :param b:
    :return:
    """
    dim = len(b)

    x = np.zeros(dim,dtype=np.float32)

    for i in range(0,dim):
        tmp =b[i]
        for j in range(0,i):
            tmp = tmp - L[i][j]*x[j]
        x[i] = tmp/L[i][i]


    return x

def backward_substitution(U,b):
    """
    solve Ux=b
    :param U:
    :param b:
    :return:
    """
    dim = len(b)

    x = np.zeros(dim,dtype=np.float32)

    for i in range(0,dim):
        tmp =b[dim-1-i]
        for j in range(dim-1-i+1,dim):
            tmp = tmp - U[dim-1-i][j]*x[j]
        x[dim-1-i] = tmp/U[dim-1-i][dim-1-i]




    return x








def solve(A,b):
    """
    solve Ax=b
    :param A:
    :param b:
    :return:x
    """
    return np.linalg.solve(A,b)




    L=np.linalg.cholesky(A)

    x1 = foward_substitution(L,b)
    x2 = backward_substitution(np.transpose(L),x1)

    tmp = np.dot(A,x2)-b
    print("solve eq with cholesky")
    print(np.max(np.abs(tmp)))

    return x2






