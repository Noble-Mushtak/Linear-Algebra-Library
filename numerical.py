import math

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