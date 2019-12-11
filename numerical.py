import math
import itertools

'''
Vectors are represented by arrays of numbers.
Examples: [2] is a vector from R
          [1, 2] is a vector from R^2
          [0.5, 3e-6, 6, 7] is a vector from R^4

Matrices are represented by arrays of arrays of numbers,
  where each array of number has the same length

Example: The following is a 3-by-4 matrix:
[[1, 2, 3, 4],
 [5, 6, 7, 8],
 [9, 10, 11, 12]]

A permutation on n elements is represented by an array of n integers,
 where each integer is unique and in the interval [0, n-1]
Example: [0, 2, 1] is a permutation on 3 elements
         [4, 3, 0, 1, 2] is a permutation on 5 elements
'''

'''
VECTOR OPERATIONS
'''

def vector_sum(vector1, vector2):
    '''
    Given two vectors, calculate the sum of the two vectors.
    '''
    if len(vector1) != len(vector2):
        raise ValueError("Both vectors must be of the same length")

    return [a+b for (a, b) in zip(vector1, vector2)]

def vector_diff(vector1, vector2):
    '''
    Given two vectors, calculate the difference of the two vectors.
    '''
    if len(vector1) != len(vector2):
        raise ValueError("Both vectors must be of the same length")

    return [a-b for (a, b) in zip(vector1, vector2)]

def vector_scale(vector, scalar):
    '''
    Given a vector and a scalar,
      multiply all the components of the given vector by the given scalar.
    '''

    return [scalar*x for x in vector]

def l1Norm(vector):
    '''
    Calculate the L1-norm of the given vector.
    '''
    
    return sum([abs(x) for x in vector])

def l2Norm(vector):
    '''
    Calculate the L2-norm of the given vector.
    '''

    return math.sqrt(sum([x*x for x in vector]))

def lInfinityNorm(vector):
    '''
    Calculate the L Infinity-norm of the given vector.
    '''

    return max([abs(x) for x in vector])

def normalize(vector, norm=l2Norm):
    '''
    Scale the given vector so that it has norm 1.

    If the given vector has norm 0, then return the given vector.
    '''
    norm_of_vector = norm(vector)
    if norm_of_vector == 0:
        return vector
    return vector_scale(vector, 1/norm_of_vector)

def absolute_error(actual, computed, norm=l2Norm):
    '''
    Given the actual answer to a problem and the computed answer,
     compute the absolute error between the actual and computed.

    By default, this error is computed using the L2 norm
    '''
    
    return norm(vector_diff(actual, computed))

def relative_error(actual, computed, norm=l2Norm):
    '''
    Given the actual answer to a problem and the computed answer,
     compute the relative error between the actual and computed.

    By default, this error is computed using the L2 norm
    '''
    
    return norm(vector_diff(actual, computed))/norm(actual)

'''
MATRIX OPERATIONS
'''

def matrix_vector_prod(matrix, vector):
    '''
    Given a matrix and a vector, calculate their product.
    '''
    if len(matrix[0]) != len(vector):
        raise ValueError("The number of columns in the matrix must equal the length of the vector")

    return [
        sum([row[j]*vector[j] for j in range(len(vector))])
        for row in matrix
    ]

def matrix_matrix_prod(matrix1, matrix2):
    '''
    Given two matrices, calculate their product.
    '''
    if len(matrix1[0]) != len(matrix2):
        raise ValueError("The number of columns in the first matrix must equal the number of rows in the second matrix")

    return [
        [
            sum([matrix1[i][k]*matrix2[k][j] for k in range(len(matrix2))])
            for j in range(len(matrix2[0]))
        ]
        for i in range(len(matrix1))
    ]

def identity(n):
    '''
    Given a positive integer n,
     compute the identity matrix on R^n.
    '''

    return [
        [
            int(i == j)
            for j in range(n)
        ]
        for i in range(n)
    ]

def matrix_row_multiplication(matrix, row_index, scalar):
    '''
    Given a matrix, an integer representing the index of a row,
     and a scalar,
     multiply the appropriate row in the matrix by the given scalar.
    
    Note that this does not modify the given matrix in place,
     but rather returns a new matrix.
    '''
    if row_index < 0 or row_index >= len(matrix):
        raise ValueError("Row index does not make sense")

    return [
        [
            (scalar if (i == row_index) else 1)*matrix[i][j]
            for j in range(len(matrix[i]))
        ]
        for i in range(len(matrix))
    ]

def matrix_row_swap(matrix, row_index1, row_index2):
    '''
    Given a matrix and two integers representing the indices of
     two diferent rows, swap the two given rows.

    Note that this does not modify the given matrix in place,
     but rather returns a new matrix.
    '''
    if row_index1 < 0 or row_index1 >= len(matrix):
        raise ValueError("Row index does not make sense")
    if row_index2 < 0 or row_index2 >= len(matrix):
        raise ValueError("Row index does not make sense")
    
    return [
        [
            matrix[
                row_index2 if (i == row_index1)
                else row_index1 if (i == row_index2)
                else i
            ]
            [j]
            for j in range(len(matrix[i]))
        ]
        for i in range(len(matrix))
    ]

def matrix_column_swap(matrix, column_index1, column_index2):
    '''
    Given a matrix and two integers representing the indices of
     two diferent columns, swap the two given columns.

    Note that this does not modify the given matrix in place,
     but rather returns a new matrix.
    '''
    if column_index1 < 0 or column_index1 >= len(matrix[0]):
        raise ValueError("Column index does not make sense")
    if column_index2 < 0 or column_index2 >= len(matrix[0]):
        raise ValueError("Column index does not make sense")
    
    return [
        [
            matrix[i]
            [
                column_index2 if (j == column_index1)
                else column_index1 if (j == column_index2)
                else j
            ]
            for j in range(len(matrix[i]))
        ]
        for i in range(len(matrix))
    ]

def matrix_row_addition(matrix, row_index1, row_index2, scalar):
    '''
    Given a matrix, two integers representing the indices of
     two different rows, and a scalar,
     add the first row by the scalar times the second row.
    
    Note that this does not modify the given matrix in place,
     but rather returns a new matrix.
    '''
    if row_index1 < 0 or row_index1 >= len(matrix):
        raise ValueError("Row index does not make sense")
    if row_index2 < 0 or row_index2 >= len(matrix):
        raise ValueError("Row index does not make sense")
    
    return [
        [
            matrix[i][j]+
              (scalar if i == row_index1 else 0)*matrix[row_index2][j]
            for j in range(len(matrix[i]))
        ]
        for i in range(len(matrix))
    ]

def matrix_column_addition(matrix, column_index1, column_index2, scalar):
    '''
    Given a matrix, two integers representing the indices of
     two different columns, and a scalar,
     add the first column by the scalar times the second column.
    
    Note that this does not modify the given matrix in place,
     but rather returns a new matrix.
    '''
    if column_index1 < 0 or column_index1 >= len(matrix[0]):
        raise ValueError("Column index does not make sense")
    if column_index2 < 0 or column_index2 >= len(matrix[0]):
        raise ValueError("Column index does not make sense")
    
    return [
        [
            matrix[i][j]+
              (scalar if j == column_index1 else 0) *
                matrix[i][column_index2]
            for j in range(len(matrix[i]))
        ]
        for i in range(len(matrix))
    ]

def transpose(matrix):
    '''
    Compute the transpose of the given vector
    '''
    
    return [
        [
            matrix[i][j]
            for i in range(len(matrix))
        ]
        for j in range(len(matrix[0]))
    ]

def build_augmented_matrix(matrix, vector):
    '''
    Given a matrix A and a vector x,
     compute the augmented matrix [A | x]
    '''
    if len(matrix) != len(vector):
        raise ValueError("The number of rows in the matrix must equal the length of the vector")

    return [
        row+[vector[i]]
        for i, row in enumerate(matrix)
    ]

def split_augmented_matrix(aug_matrix):
    '''
    Given an augmented matrix [A | x],
     compute the tuple containing the matrix A and a vector x
    '''

    return (
        [row[:-1] for row in aug_matrix],
        [row[-1] for row in aug_matrix]
    )

'''
PERMUTATIONS
'''

def invert_permutation(permutation):
    '''
    Compute the inverse of the given permutation
    '''
    inverse = [None]*len(permutation)
    for in_, out in enumerate(permutation):
        inverse[out] = in_
    return inverse

def apply_permutation(permutation, list):
    '''
    Given a permutation and a list,
     permute the elements of the list according to that permutation
    '''
    if len(permutation) != len(list):
        raise ValueError("The length of the permutation must equal the length of the list")
    
    return [list[out] for out in permutation]

def permutation_sign(permutation):
    '''
    Compute the sign of the given permutation
    '''

    sign = 1
    marked = [False]*len(permutation)
    for in_, out in enumerate(permutation):
        if not marked[in_]:
            marked[in_] = True
            while not marked[out]:
                sign *= -1
                marked[out] = True
                out = permutation[out]
    return sign

'''
SOLVING SYSTEMS OF EQUATIONS
'''

TOLERANCE = 1e-6

def naive_pivot_strategy(matrix, row_ind, column_ind, ignore_last_column=False):
    '''
    Given a matrix A and an integer representing the index of a column,
     calculate the position of the pivot
     by finding the index of a row
     whose entry in the (column_ind)th column is non-zero

    Note that the pivot is assumed to be located
     in the row row_ind or a row after that row,
     and in the column column_ind.
    '''
    if ignore_last_column and column_ind == len(matrix[0])-1:
        raise ValueError("Tried to ignore last column, but column index specified was last column")
        
    for i in range(row_ind, len(matrix)):
        if abs(matrix[i][column_ind]) > TOLERANCE:
            return (i, column_ind)
    raise ValueError("Could not find pivot")

def simple_partial_pivoting(matrix, row_ind, column_ind, ignore_last_column=False):
    '''
    Given a matrix A and an integer representing the index of a column,
     calculate the position of the pivot
     by finding the index of the row
     whose entry in the (column_ind)th column has maximum absolute value

    Note that the pivot is assumed to be located
     in the row row_ind or a row after that row,
     and in the column column_ind.
    '''
    if ignore_last_column and column_ind == len(matrix[0])-1:
        raise ValueError("Tried to ignore last column, but column index specified was last column")
        
    desired_row = row_ind
    max_abs_so_far = 0
    for i in range(row_ind, len(matrix)):
        if abs(matrix[i][column_ind]) > max_abs_so_far:
            max_abs_so_far = abs(matrix[i][column_ind])
            desired_row = i
            
    if max_abs_so_far <= TOLERANCE:
        raise ValueError("Could not find pivot")
    return (desired_row, column_ind)

def scaled_partial_pivoting(matrix, row_ind, column_ind, ignore_last_column=False):
    '''
    Given a matrix A and an integer representing the index of a column,
     calculate the position of the pivot
     by finding the index of the row
     whose entry in the (column_ind)th column
     has maximum absolute value, relative to the biggest entry in that rwo

    Note that the pivot is assumed to be located
     in the row row_ind or a row after that row,
     and in the column column_ind.
    '''
    if ignore_last_column and column_ind == len(matrix[0])-1:
        raise ValueError("Tried to ignore last column, but column index specified was last column")
    
    desired_row = row_ind
    max_ratio_so_far = 0
    for i in range(row_ind, len(matrix)):
        if abs(matrix[i][column_ind]) <= TOLERANCE:
            continue
            
        biggest_element = max([abs(x) for x in matrix[i][column_ind:]])
        ratio = abs(matrix[i][column_ind])/biggest_element
        if ratio > max_ratio_so_far:
            max_ratio_so_far = ratio
            desired_row = i
            
    if abs(matrix[desired_row][column_ind]) <= TOLERANCE:
        raise ValueError("Could not find pivot")
    return (desired_row, column_ind)

def full_pivoting(matrix, row_ind, column_ind, ignore_last_column=False):
    '''
    Given a matrix A and an integer representing the index of a column,
     calculate the position of the pivot
     by finding the position in A
     whose entry has maximum absolute value

    Note that the pivot is assumed to be located
     in the row row_ind or a row after that row,
     and in the column column_ind or a column after that column.
    '''
    num_columns = len(matrix[0])
    if ignore_last_column: num_columns -= 1

    desired_pos = (row_ind, column_ind)
    max_abs_so_far = 0
    for i in range(row_ind, len(matrix)):
        for j in range(column_ind, num_columns):
            if abs(matrix[i][j]) > max_abs_so_far:
                desired_pos = (i, j)
                max_abs_so_far = abs(matrix[i][j])
            
    if max_abs_so_far <= TOLERANCE:
        raise ValueError("Could not find pivot")
    return desired_pos

def gaussian_elimination(matrix, find_pivot=naive_pivot_strategy):
    '''
    Row-reduce the given augmented matrix to row echelon form,
     using column swaps if the pivot strategy calls for it.
    Note that the pivot strategy should never swap
     the last column in the augmented matrix with another column,
     as the last column represents the constants on the right side
     of the equal sign in the system of equations.

    Note that this function returns a tuple containing:
     - the new augmented matrix
     - a permutation representing all of the column swaps
    '''

    column_permutation = list(range(len(matrix[0])-1))
    num_pivots_so_far = 0
    for i in range(len(matrix[0])):
        # Do not find more pivots than necessary:
        if num_pivots_so_far+1 >= len(matrix):
            break
        
        # Attempt to find pivot with non-zero entry,
        # but if no such pivot can be found,
        # then skip this column:
        try:
            pivot_row, pivot_column = find_pivot(
                matrix,
                row_ind=num_pivots_so_far,
                column_ind=i,
                ignore_last_column=(i < len(column_permutation))
            )
            matrix = matrix_row_swap(
                matrix,
                num_pivots_so_far,
                pivot_row
            )
            # Only adjust the column permutation
            #  if the current column is not the last column
            if i < len(column_permutation):
                if pivot_column == len(column_permutation):
                    raise ValueError("Pivot strategy tried to swap with last column")
                matrix = matrix_column_swap(
                    matrix,
                    i,
                    pivot_column
                )
                column_permutation[i], column_permutation[pivot_column] = \
                  column_permutation[pivot_column], column_permutation[i]
        except ValueError:
            continue
        for j in range(num_pivots_so_far+1, len(matrix)):
            matrix = matrix_row_addition(
                matrix,
                j,
                num_pivots_so_far,
                -matrix[j][i]/matrix[num_pivots_so_far][i]
            )
        num_pivots_so_far += 1
    return matrix, column_permutation

def back_substitution(aug_matrix):
    '''
    Given an augmented matrix in row echelon form,
     attempt to compute a solution to the system represented by that
     augmented matrix.
    '''

    solution = [0]*(len(aug_matrix[0])-1)
    for i in range(len(aug_matrix)-1, -1, -1):
        pivot_index = 0
        while pivot_index < len(solution):
            if abs(aug_matrix[i][pivot_index]) > TOLERANCE:
                break
            pivot_index += 1
        if pivot_index == len(solution):
            if abs(aug_matrix[i][-1]) > TOLERANCE:
                raise ValueError("System is inconsistent")
            else:
                continue
        solution[pivot_index] = (
            aug_matrix[i][-1]
            -
            sum([
                aug_matrix[i][j]*solution[j]
                for j in range(pivot_index+1, len(solution))
            ])
        ) / aug_matrix[i][pivot_index]
    return solution

def solve_system(matrix, vector, pivot_strategy=naive_pivot_strategy):
    '''
    Given a matrix A and a vector b,
     attempt to compute a solution to the equation A*x=b
    '''
    
    aug_matrix, column_permutation = gaussian_elimination(
        build_augmented_matrix(matrix, vector),
        pivot_strategy
    )
    return apply_permutation(
        invert_permutation(column_permutation),
        back_substitution(aug_matrix)
    )

'''
LU DECOMPOSITION
'''

def lu_decomposition(matrix, find_pivot=naive_pivot_strategy):
    '''
    Compute the LU decomposition of the given matrix

    Note that this function returns a tuple containing:
     - a permutation representing which row in the original matrix that each row in the upper-triangular matrix corresponds to
     - a lower-triangular matrix
     - an upper-triangular matrix
     - a permutation representing which column in the original matrix that each column in the upper-triangular matrix corresponds to
    '''

    row_permutation = list(range(len(matrix)))
    lower_triangular = identity(len(matrix))
    column_permutation = list(range(len(matrix[0])))
    num_pivots_so_far = 0
    for i in range(min(len(matrix), len(matrix[0]))):
        # Attempt to find pivot with non-zero entry,
        # but if no such pivot can be found,
        # then skip this column:
        try:
            pivot_row, pivot_column = find_pivot(
                matrix,
                row_ind=num_pivots_so_far,
                column_ind=i
            )
            matrix = matrix_row_swap(
                matrix,
                num_pivots_so_far,
                pivot_row
            )
            matrix = matrix_column_swap(
                matrix,
                i,
                pivot_column
            )
            row_permutation[num_pivots_so_far], \
              row_permutation[pivot_row] = \
              row_permutation[pivot_row], \
              row_permutation[num_pivots_so_far]
            column_permutation[i], column_permutation[pivot_column] = \
              column_permutation[pivot_column], column_permutation[i]
            # Also, adjust the lower triangular matrix to the row swap:
            lower_triangular = matrix_column_swap(
                lower_triangular,
                num_pivots_so_far,
                pivot_row
            )
            lower_triangular = matrix_row_swap(
                lower_triangular,
                num_pivots_so_far,
                pivot_row
            )
        except ValueError:
            continue
        for j in range(num_pivots_so_far+1, len(matrix)):
            lower_triangular[j][num_pivots_so_far] = \
              matrix[j][i]/matrix[num_pivots_so_far][i]
            matrix = matrix_row_addition(
                matrix,
                j,
                num_pivots_so_far,
                -matrix[j][i]/matrix[num_pivots_so_far][i]
            )
        num_pivots_so_far += 1
    return row_permutation, lower_triangular, matrix, column_permutation

def forward_substitution(matrix, vector):
    '''
    Given a lower-triangular matrix A and a vector b,
     compute the solution to the equation A*x=b

    Note that it is assumed that A is square,
     and that all of the diagonal elements of A are 1.
    '''

    solution = [0]*len(matrix)
    for i in range(len(matrix)):
        solution[i] = vector[i]-sum([matrix[i][j]*solution[j] for j in range(i)])
    return solution

def solve_system_using_lu_decomp(lu_decomp, vector):
    '''
    Given the LU-decomposition of a matrix A
     and a vector v,
     compute the solution to the equation A*x=b
    '''

    row_perm, lower, upper, column_perm = lu_decomp
    # Apply the row permutation to the output vector b:
    rearranged_vector = apply_permutation(row_perm, vector)
    # Compute (lower)^(-1)*rearranged_vector
    lower_inverse_times_vector = forward_substitution(
        lower,
        rearranged_vector
    )
    # Compute (upper)^(-1)*(lower)^(-1)*rearranged_vector
    solution = back_substitution(
        build_augmented_matrix(upper, lower_inverse_times_vector)
    )
    # Finally, apply the inverted column permutation to the solution x:
    return apply_permutation(
        invert_permutation(column_perm),
        solution
    )

def reconstruct_matrix(lu_decomp):
    '''
    Given the LU decomposition of a matrix A,
     compute the matrix A
    '''

    row_perm, lower, upper, column_perm = lu_decomp
    Q = apply_permutation(column_perm, identity(len(column_perm)))
    UQ = matrix_matrix_prod(upper, Q)
    LUQ = matrix_matrix_prod(lower, UQ)
    PLUQ = apply_permutation(
        invert_permutation(row_perm),
        LUQ
    )
    return PLUQ

'''
IMAGE, KERNEL, INVERSE, AND DETERMINANT
'''

def find_image_using_lu_decomp(lu_decomp):
    '''
    Given the LU-decomposition of a matrix A,
     compute a list of vectors which form a basis for the image of A
    '''

    pivots = []
    column_vectors = transpose(reconstruct_matrix(lu_decomp))
    
    row_perm, lower, upper, column_perm = lu_decomp
    for i in range(len(upper)):
        pivot_index = 0
        while pivot_index < len(upper[i]):
            if abs(upper[i][pivot_index]) > TOLERANCE:
                break
            pivot_index += 1
        if pivot_index == len(upper[i]):
            break
        pivots.append(pivot_index)
    return [
        column_vectors[index]
        for index in
        sorted([column_perm[pivot] for pivot in pivots])
    ]

def find_kernel_using_lu_decomp(lu_decomp):
    '''
    Given the LU-decomposition of a matrix A,
     return a list of vectors which form a basis for the kernel of A
    '''

    pivots_so_far = []
    last_pivot = -1
    basis = []
    row_perm, lower, upper, column_perm = lu_decomp

    def add_solution(free_var_index):
        '''
        Given an index representing a column in upper
         which corresponds to a free variable,
         add a solution to the equation A*x=0 to basis.
        '''
        i = len(pivots_so_far)
        
        solution = [0]*len(upper[0])
        solution[free_var_index] = 1
        # We get the solution
        #  by traversing the rows 0..(i-1) in reverse order
        #  and solving for the component of the solution
        #  corresponding to the pivot in that row
        for j, pivot in enumerate(reversed(pivots_so_far)):
            solution[pivot] = (
                -sum([
                    upper[i-j-1][pivots_so_far[-k-1]]*solution[pivots_so_far[-k-1]]
                    for k in range(j)
                ])
                -upper[i-j-1][free_var_index]
            ) / upper[i-j-1][pivot]
        basis.append(
            apply_permutation(
                invert_permutation(column_perm),
                solution
            )
        )
        
    for i in range(len(upper)):
        pivot_index = 0
        while pivot_index < len(upper[i]):
            if abs(upper[i][pivot_index]) > TOLERANCE:
                break
            pivot_index += 1
            
        for free_var_index in range(last_pivot+1, pivot_index):
            add_solution(free_var_index)
            
        last_pivot = pivot_index
        if pivot_index == len(upper[i]):
            break
        pivots_so_far.append(pivot_index)
        
    for free_var_index in range(last_pivot+1, len(upper[0])):
        add_solution(free_var_index)
    return basis

def is_singular_using_lu_decomp(lu_decomp):
    '''
    Given the LU-decomposition of a square matrix A,
     compute whether the matrix is singular or not
    '''
    row_perm, lower, upper, column_perm = lu_decomp
    if len(row_perm) != len(column_perm):
        raise ValueError("The matrix must be square")
    
    return len(find_kernel_using_lu_decomp(lu_decomp)) != 0

def find_inverse_using_lu_decomp(lu_decomp):
    '''
    Given the LU-decomposition of a square matrix A,
     compute whether the matrix is singular or not
    '''
    if is_singular_using_lu_decomp(lu_decomp):
        raise ValueError("The matrix must be non-singular")

    row_perm, lower, upper, column_perm = lu_decomp
    column_vectors = []
    for standard_basis_vector in identity(len(row_perm)):
        column_vectors.append(
            solve_system_using_lu_decomp(
                lu_decomp,
                standard_basis_vector
            )
        )
    return transpose(column_vectors)

def naive_determinant(matrix):
    '''
    Compute the determinant of the given matrix
     by looping through all the possible patterns
    '''
    if len(matrix) != len(matrix[0]):
        raise ValueError("The matrix must be square")

    determinant = 0
    for perm in itertools.permutations(range(len(matrix))):
        pattern = permutation_sign(perm)
        for i, j in enumerate(perm):
            pattern *= matrix[i][j]
        determinant += pattern
    return determinant

def find_determinant_using_lu_decomp(lu_decomp):
    '''
    Given the LU-decomposition of a square matrix A,
     compute the determinant of the matrix
    '''
    row_perm, lower, upper, column_perm = lu_decomp
    if len(row_perm) != len(column_perm):
        raise ValueError("The matrix must be square")
        
    determinant = permutation_sign(row_perm)*permutation_sign(column_perm)
    for i in range(len(row_perm)):
        determinant *= upper[i][i]
    return determinant

'''
QR DECOMPOSITION AND LEAST-SQUARE SOLUTIONS
'''

def qr_decomposition(matrix):
    '''
    Compute the QR decomposition of the given matrix
     using Householder reflections,
     assuming the matrix has at least as many rows than columns.

    Note that this function outputs a tuple containing:
     - An orthogonal (square) matrix
     - An upper-triangular matrix which has the same dimensions as the given matrix
    '''
    if len(matrix) < len(matrix[0]):
        raise ValueError("The matrix must have at least as many rows as columns")

    q = identity(len(matrix))
    r_so_far = matrix
    for i in range(len(matrix[0])):
        # alpha is the magnitude of the (i)th column vector,
        #  ignoring the first i components
        shortened_vector = [
            0 if j < i else r_so_far[j][i]
            for j in range(len(r_so_far))
        ]
        alpha = l2Norm(shortened_vector)
        # If alpha and r_so_far[j][i] have the same sign,
        #  switch the sign of alpha,
        #  so we don't get huge cancellation
        if (alpha*r_so_far[i][i]) > 0:
            alpha *= -1

        # Compute the difference between shortened_vector
        #  and alpha*e_i
        difference_vector = [
            shortened_vector[j]-(alpha if j == i else 0)
            for j in range(len(r_so_far))
        ]
        # Normalize this vector:
        diff_vector_normalized = normalize(difference_vector)
        # Compute the reflection matrix about diff_vector_normalized:
        new_q = [
            [
               (1 if i == j else 0)
               -2*diff_vector_normalized[i]*diff_vector_normalized[j]
               for j in range(len(r_so_far))
            ]
            for i in range(len(r_so_far))
        ]
        # Update R with this new reflection:
        r_so_far = matrix_matrix_prod(new_q, r_so_far)
        # Also, update Q by taking the inverse of the reflection:
        q = matrix_matrix_prod(q, transpose(new_q))
    return q, r_so_far

def find_least_squares_solution_using_qr_decomp(qr_decomp, vector):
    '''
    Given the QR decompositon of a matrix A, and a vector b,
     compute the least-squares solution to the equation A*x=b
     assuming that the columns of A are linearly independent

    Note that this assumption applies that R is in row echelon form,
     because it is an upper-triangular matrix with non-zero
     diagonal entries.
    '''

    q, r = qr_decomp
    vector_components = matrix_vector_prod(transpose(q), vector)
    # Ignore components of the vector
    #  which are perpendicular to the image of A:
    for i in range(len(r[0]), len(vector_components)):
        vector_components[i] = 0
    return back_substitution(build_augmented_matrix(r, vector_components))

def calc_best_fit_coefficients(funcs, points):
    '''
    Given a set of functions from R^n to R f1, f2, ..., fn,
     and a set of points (x, y) from the cartesian product of R^n and R,
     solve for the coefficients k1, k2, ..., kn
     that minimizes the sum of squared errors of the following model:
     y = k1*f1(x) + k2*f2(x) + ... + kn*fn(x)

    Assumes that the functions f1, f2, ..., fn
     are all linearly independent.

    Note that, in the points array, each x in a point (x, y)
     should be a member of R, in which case x is of type number,
     or should be a member of R^n for n >= 2,
     in which case x is a tuple of n numbers.
    '''

    matrix = [
        # Unwrap the input if the input is a tuple.
        # Otherwise, just apply the function normally.
        [func(*point[0]) if type(point[0]) == tuple else func(point[0])
         for func in funcs]
        for point in points
    ]
    return find_least_squares_solution_using_qr_decomp(
        qr_decomposition(matrix),
        [point[1] for point in points]
    )

'''
EIGENVALUES AND EIGENVECTORS
'''

def naive_qr_algorithm(matrix, tolerance=1e-10):
    '''
    Find the Schur decomposition of a matrix
     using a basic implementation of the QR algorithm

    Note that this function returns a tuple containing:
     - an orthogonal matrix Q
     - an upper triangular matrix T
    If the original matrix is A, then A=QT(Q^T),
     where Q^T denotes the transpose of Q.

    Assumes that the matrix has eigenvalues of distinct magnitudes,
     and that all of the matrix's eigenvalues are real.
    '''
    if len(matrix) != len(matrix[0]):
        raise ValueError("The matrix must be square")

    q = identity(len(matrix))
    t = matrix
    while True:
        # Exit if t is now upper-triangular
        for i in range(len(matrix)):
            is_upper = True
            for j in range(i):
                if abs(t[i][j]) > tolerance:
                    is_upper = False
                    break
            if not is_upper:
                break
        else:
            break
        # Find the QR decomposition of t
        #  and let the new value of t be RQ:
        new_q, new_r = qr_decomposition(t)
        t = matrix_matrix_prod(new_r, new_q)
        # t = q^T*matrix*q = new_q*new_r
        # new_t = new_r*new_q = new_q^T*t*new_q
        # -> new_t=(q*new_q)^T* matrix * (q*new_q)
        # Thus, multiply q by new_q:
        q = matrix_matrix_prod(q, new_q)
    return q, t

def find_eigendecomposition(schur_decomp):
    '''
    Given the Schur decomposition of a matrix,
     find all of the eigendecomposition of that matrix.

    Note that this function returns a tuple containing:
     - a non-singular matrix S
     - a diagonal matrix D
    If the original matrix is A, then A=SD(S^-1)

    Assumes that the matrix has a full eigenbasis.
    '''

    eigenbasis_so_far, upper = schur_decomp
    for i in range(len(upper)):
        for j in range(i):
            # Exit if we find a generalized eigenvector:
            # (i.e. if this matrix does not have a full eigenbasis
            if abs(upper[i][i]-upper[j][j]) <= TOLERANCE:
                if abs(upper[j][i]) > TOLERANCE:
                    raise ValueError("This matrix does not have a full eigenbasis")
                else:
                    continue
            # Eliminate the upper[j][i] using a similarity transformation:
            factor = upper[j][i]/(upper[i][i]-upper[j][j])
            upper = matrix_column_addition(
                upper,
                i,
                j,
                factor
            )
            upper = matrix_row_addition(
                upper,
                j,
                i,
                -factor
            )
            # Update the eigenbasis accordingly:
            eigenbasis_so_far = matrix_column_addition(
                eigenbasis_so_far,
                i,
                j,
                factor
            )
    return eigenbasis_so_far, upper

def find_roots_of_polynomial(poly_coefficients, find_schur_decomp=naive_qr_algorithm):
    '''
    Given an array of numbers [c0, c1, ..., c(n-1)],
     find the roots of the polynomial
     p(t)=c0 + c1*t + ... + c(n-1) * t^(n-1) + t^n

    Note that this function is only as good
     as the find_schur_decomp function.
    For example, if one uses naive_qr_algorithm,
     then this function will only work for polynomials
     such that all of their roots are real
     and have distinct magnitudes.
    '''

    degree = len(poly_coefficients)
    companion_matrix = [
        [
            1 if j == i+1 else
            -poly_coefficients[j] if i == degree-1 else 0
            for j in range(degree)
        ]
        for i in range(degree)
    ]
    q, upper = find_schur_decomp(companion_matrix)
    return sorted([upper[i][i] for i in range(degree)])