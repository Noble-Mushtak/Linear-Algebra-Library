from numerical import *
import unittest
import math

class TestCase(unittest.TestCase):
    def assertListAlmostEqual(self, lst1, lst2):
        self.assertEqual(len(lst1), len(lst2))
        for a, b in zip(lst1, lst2):
            self.assertAlmostEqual(a, b)
    
    def assertMatrixAlmostEqual(self, matrix1, matrix2):
        self.assertEqual(len(matrix1), len(matrix2))
        for row1, row2 in zip(matrix1, matrix2):
            self.assertListAlmostEqual(row1, row2)

class TestVectorOperations(TestCase):
    def test_sum(self):
        self.assertListAlmostEqual(vector_sum([1], [2]), [3])
        self.assertListAlmostEqual(vector_sum([1, 4], [-1, 3]), [0, 7])
        self.assertListAlmostEqual(
            vector_sum([7, -2, -8, 9], [4, 0, 4, -5]),
            [11, -2, -4, 4]
        )
        self.assertListAlmostEqual(
            vector_sum([7.3e-3, 5, 8.9, 1.1], [6.4, 8.5e5, 63, 7.2e-6]),
            [6.4073, 850005, 71.9, 1.1000072]
        )
    
    def test_diff(self):
        self.assertListAlmostEqual(vector_diff([1], [2]), [-1])
        self.assertListAlmostEqual(vector_diff([1, 4], [-1, 3]), [2, 1])
        self.assertListAlmostEqual(
            vector_diff([7, -2, -8, 9], [4, 0, 4, -5]),
            [3, -2, -12, 14]
        )
        self.assertListAlmostEqual(
            vector_diff([7.3e-3, 5, 8.9, 1.1], [6.4, 8.5e5, 63, 7.2e-6]),
            [-6.3927, -849995, -54.1, 1.0999928]
        )
    
    def test_scale(self):
        self.assertListAlmostEqual(vector_scale([1], 2), [2])
        self.assertListAlmostEqual(
            vector_scale([1, 2, 3], 0.5),
            [0.5, 1.0, 1.5]
        )
        self.assertListAlmostEqual(
            vector_scale([6.4, 8.5e5, 63, 7.2e-6], 4.7),
            [30.08, 3995000, 296.1, 3.384e-5]
        )
     
    def test_norms(self):
        self.assertAlmostEqual(l1Norm([-1, 2, 5]), 8)
        self.assertAlmostEqual(l2Norm([-1, 2, 5]), math.sqrt(30))
        self.assertAlmostEqual(lInfinityNorm([-1, 2, 5]), 5)
    
    def test_error(self):
        self.assertAlmostEqual(
            absolute_error([1, 1], [2, 0]),
            math.sqrt(2)
        )
        self.assertAlmostEqual(
            relative_error([1, 1], [2, 0]),
            1
        )
        self.assertAlmostEqual(
            absolute_error([57045/74998, -13975/37499, -4380/37499],
                           [0.76, -0.38, -0.12]),
            .00801479539
        )
        self.assertAlmostEqual(
            relative_error([57045/74998, -13975/37499, -4380/37499],
                           [0.76, -0.38, -0.12]),
            9.37371532388e-3
        )
        self.assertAlmostEqual(
            absolute_error([1, 1], [2, 0], norm=l1Norm),
            2
        )
        self.assertAlmostEqual(
            relative_error([1, 1], [2, 0], norm=l1Norm),
            1
        )

class TestMatrixOperations(TestCase):
    def test_prod(self):
        self.assertListAlmostEqual(
            matrix_vector_prod(
                [[1.01, 0.99],
                 [0.99, 1.01]],
                [1, 1]
            ),
            [2, 2]
        )
        self.assertListAlmostEqual(
            matrix_vector_prod(
                [[4.4, 5.2, 3.5],
                 [7.6, 9.8, 1.1],
                 [2.2, -4.5, 3.0]],
                [57045/74998, -13975/37499, -4380/37499]
            ),
            [1, 2, 3]
        )
        self.assertListAlmostEqual(
            matrix_vector_prod(
                [[1, 0, 0, 0],
                 [0, 1, 0, 0],
                 [0, 0, 1, 0]],
                [1, 2, 4, 5]
            ),
            [1, 2, 4]
        )
        self.assertMatrixAlmostEqual(
            matrix_matrix_prod(
                [[1,2],
                 [3,4],
                 [5,6]],
                [[1,2,3],
                 [4,5,6]]
            ),
            [[9,12,15],
             [19,26,33],
             [29,40,51]]
        )
        self.assertMatrixAlmostEqual(
            matrix_matrix_prod(
                [[1,2],
                 [3,4],
                 [5,6],
                 [7,8]],
                [[1,2,3],
                 [4,5,6]]
            ),
            [[9,12,15],
             [19,26,33],
             [29,40,51],
             [39,54,69]]
        )

    def test_identity(self):
        self.assertMatrixAlmostEqual(
            identity(2),
            [[1, 0],
             [0, 1]]
        )
        self.assertMatrixAlmostEqual(
            identity(3),
            [[1, 0, 0],
             [0, 1, 0],
             [0, 0, 1]]
        )
        self.assertMatrixAlmostEqual(
            matrix_matrix_prod(
                identity(4),
                [[7, 8, 9],
                 [4, 8, -6],
                 [-3, 2, 1],
                 [9, 5, 8]]
            ),
            [[7, 8, 9],
             [4, 8, -6],
             [-3, 2, 1],
             [9, 5, 8]]
        )
        self.assertMatrixAlmostEqual(
            matrix_matrix_prod(
                [[1, 5, 8, 4],
                 [-8, 9, 7, 6],
                 [20, -100, 45, 6.8]],
                identity(4)
            ),
            [[1, 5, 8, 4],
             [-8, 9, 7, 6],
             [20, -100, 45, 6.8]]
        )
     
    def test_row_ops(self):
        self.assertMatrixAlmostEqual(
            matrix_row_multiplication(
                [[1,2,3],
                 [4,5,6]],
                row_index=0,
                scalar=2
            ),
            [[2,4,6],
             [4,5,6]]
        )
        self.assertMatrixAlmostEqual(
            matrix_row_multiplication(
                [[1,2,3],
                 [4,5,6]],
                row_index=1,
                scalar=-0.25
            ),
            [[1,2,3],
             [-1, -1.25, -1.5]]
        )
        self.assertMatrixAlmostEqual(
            matrix_row_swap(
                [[1,2,3],
                 [4,5,6]],
                row_index1=0,
                row_index2=1
            ),
            [[4,5,6],
             [1,2,3]]
        )
        self.assertMatrixAlmostEqual(
            matrix_row_swap(
                [[1,2],
                 [4,5],
                 [7,8]],
                row_index1=2,
                row_index2=0
            ),
            [[7,8],
             [4,5],
             [1,2]]
        )
        self.assertMatrixAlmostEqual(
            matrix_column_swap(
                [[1,2],
                 [4,5],
                 [7,8]],
                column_index1=1,
                column_index2=0
            ),
            [[2,1],
             [5,4],
             [8,7]]
        )
        self.assertMatrixAlmostEqual(
            matrix_column_swap(
                [[1,2,-4,5],
                 [4,5,-3,1],
                 [7,8,9,0]],
                column_index1=1,
                column_index2=3
            ),
            [[1,5,-4,2],
             [4,1,-3,5],
             [7,0,9,8]]
        )
        self.assertMatrixAlmostEqual(
            matrix_row_addition(
                [[1,2],
                 [4,5],
                 [7,8]],
                row_index1=2,
                row_index2=0,
                scalar=-7
            ),
            [[1,2],
             [4,5],
             [0,-6]]
        )
        self.assertMatrixAlmostEqual(
            matrix_row_addition(
                [[1,2],
                 [4,5],
                 [7,8]],
                row_index1=0,
                row_index2=1,
                scalar=-0.5
            ),
            [[-1,-0.5],
             [4,5],
             [7,8]]
        )
    
    def test_transpose(self):
        self.assertMatrixAlmostEqual(
            transpose([[1, 2],
                       [2, 3],
                       [-1, 2]]),
            [[1,2,-1],
             [2,3,2]]
        )
        self.assertMatrixAlmostEqual(
            transpose([[3,4,5],
                       [7,-2,6]]),
            [[3,7],
             [4,-2],
             [5,6]]
        )
    
    def test_aug_matrix(self):
        self.assertMatrixAlmostEqual(
            build_augmented_matrix([[1, 2],
                                    [2, 3],
                                    [-1, 2]],
                                   [4, 5, 6]),
            [[1, 2, 4],
             [2, 3, 5],
             [-1, 2, 6]]
        )
        self.assertEqual(
            split_augmented_matrix([[1, 4, 9, 6],
                                    [7, 8, 9, -1],
                                    [4, 2, 5, 6]]),
            ([[1, 4, 9],
              [7, 8, 9],
              [4, 2, 5]],
             [6, -1, 6]
            )
        )

class TestPermutation(TestCase):
    def test_inverse(self):
        self.assertListEqual(
            invert_permutation(list(range(100))),
            list(range(100))
        )
        self.assertListEqual(
            invert_permutation([1, 0]),
            [1, 0]
        )
        self.assertListEqual(
            invert_permutation([4, 3, 1, 2, 0]),
            [4, 2, 3, 1, 0]
        )
    
    def test_sign(self):
        self.assertEqual(
            permutation_sign(list(range(100))),
            1
        )
        self.assertEqual(
            permutation_sign([1, 0]),
            -1
        )
        self.assertEqual(
            permutation_sign([4, 3, 1, 2, 0]),
            -1
        )

class TestSolvingSystems(TestCase):
    def test_naive_pivot(self):
        self.assertEqual(
            naive_pivot_strategy(
                [[1, 0, 1],
                 [3, 4, 2],
                 [0, 1, 3]],
                row_ind=0,
                column_ind=0
            ),
            (0, 0)
        )
        self.assertEqual(
            naive_pivot_strategy(
                [[1, 0, 1],
                 [0, 0, 2],
                 [0, 1, 3]],
                row_ind=1,
                column_ind=1
            ),
            (2, 1)
        )
    
    def test_simple_pivot(self):
        self.assertEqual(
            simple_partial_pivoting(
                [[1, 3, 5],
                 [0.5, 2, 6],
                 [0, 9, 5]],
                row_ind=0,
                column_ind=0
            ),
            (0, 0)
        )
        self.assertEqual(
            simple_partial_pivoting(
                [[1, 3, 5],
                 [-10, 2, 6],
                 [8, 9, 5]],
                row_ind=0,
                column_ind=0
            ),
            (1, 0)
        )
        self.assertEqual(
            simple_partial_pivoting(
                [[1, 3, 5],
                 [-10, 2, 11],
                 [9, 8, 5]],
                row_ind=0,
                column_ind=0
            ),
            (1, 0)
        )
    
    def test_scaled_pivot(self):
        self.assertEqual(
            scaled_partial_pivoting(
                [[1, 3, 5],
                 [0.5, 2, 6],
                 [0, 9, 5]],
                row_ind=0,
                column_ind=0
            ),
            (0, 0)
        )
        self.assertEqual(
            scaled_partial_pivoting(
                [[1, 3, 5],
                 [-10, 2, 11],
                 [9, 8, 5]],
                row_ind=0,
                column_ind=0
            ),
            (2, 0)
        )
    
    def test_full_pivot(self):
        self.assertEqual(
            full_pivoting(
                [[1, 3, 5],
                 [0.5, 2, 6],
                 [0, 9, 5]],
                row_ind=0,
                column_ind=0
            ),
            (2, 1)
        )
        self.assertEqual(
            full_pivoting(
                [[1, 3, 5],
                 [-10, 2, 11],
                 [9, 8, 5]],
                row_ind=0,
                column_ind=0
            ),
            (1, 2)
        )
        self.assertEqual(
            full_pivoting(
                [[1, 3, 5],
                 [-10, 2, 11],
                 [9, 8, 5]],
                row_ind=0,
                column_ind=0,
                ignore_last_column=True
            ),
            (1, 0)
        )
        self.assertEqual(
            full_pivoting(
                [[1, 3, 5],
                 [-10, 2, 7],
                 [9, 8, 9]],
                row_ind=0,
                column_ind=0
            ),
            (1, 0)
        )
        self.assertEqual(
            full_pivoting(
                [[1, 3, 5],
                 [-10, 2, 7],
                 [9, 8, 9]],
                row_ind=1,
                column_ind=1
            ),
            (2, 2)
        )
    
    def template_test_simple_elim(self, pivot_strategy):
        self.assertMatrixAlmostEqual(
            gaussian_elimination(
                [[1, 0, 3],
                 [0, 1, 2]],
                pivot_strategy
            )[0],
            [[1, 0, 3],
             [0, 1, 2]]
        )
        self.assertListEqual(
            gaussian_elimination(
                [[1, 0, 3],
                 [0, 1, 2]],
                pivot_strategy
            )[1],
            [0, 1]
        )
        self.assertMatrixAlmostEqual(
            gaussian_elimination(
                [[7, 3],
                 [4, 2]],
                pivot_strategy
            )[0],
            [[7, 3],
             [0, 2/7]]
        )
        self.assertListEqual(
            gaussian_elimination(
                [[7, 3],
                 [4, 2]],
                pivot_strategy
            )[1],
            [0, 1]
        )
        self.assertMatrixAlmostEqual(
            gaussian_elimination(
                [[1, 1, 1, 2],
                 [1, 1, 1, 2],
                 [1, 1, 1, 2]],
                pivot_strategy
            )[0],
            [[1, 1, 1, 2],
             [0, 0, 0, 0],
             [0, 0, 0, 0]]
        )
        self.assertListEqual(
            gaussian_elimination(
                [[1, 1, 1, 2],
                 [1, 1, 1, 2],
                 [1, 1, 1, 2]],
                pivot_strategy
            )[1],
            [0, 1, 2]
        )

    def test_simple_elim(self):
        self.template_test_simple_elim(naive_pivot_strategy)
        self.template_test_simple_elim(simple_partial_pivoting)
        self.template_test_simple_elim(scaled_partial_pivoting)
        self.template_test_simple_elim(full_pivoting)
    
    def test_naive_elim(self):
        self.assertMatrixAlmostEqual(
            gaussian_elimination(
                [[1, 1, 1, 1, 0],
                 [1, 1, 1, 2, 0],
                 [1, 1, 1, 3, 0]],
            )[0],
            [[1, 1, 1, 1, 0],
             [0, 0, 0, 1, 0],
             [0, 0, 0, 0, 0]]
        )
        self.assertMatrixAlmostEqual(
            gaussian_elimination(
                [[7, 9, 6, 3],
                 [4, 3, 5, 2]],
            )[0],
            [[7, 9, 6, 3],
             [0, -15/7, 11/7, 2/7]]
        )
        self.assertMatrixAlmostEqual(
            gaussian_elimination(
                [[7, 9, 6, 3],
                 [4, 3, 5, -1],
                 [8, 7, 2, 6]],
            )[0],
            [[7, 9, 6, 3],
             [0, -15/7, 11/7, -19/7],
             [0, 0, -218/30, 202/30]]
        )
        self.assertMatrixAlmostEqual(
            gaussian_elimination(
                [[1, 1, 2, 8, 2],
                 [1, 2, 3, -10, 2],
                 [1, 3, 4, 6, 2]],
            )[0],
            [[1, 1, 2, 8, 2],
             [0, 1, 1, -18, 0],
             [0, 0, 0, 34, 0]]
        )
    
    def test_simple_elim(self):
        # Here, the entry is 1 because the simple partial pivoting
        #  swaps the second and third row when looking for the best
        #  entry for the partial pivot.
        self.assertMatrixAlmostEqual(
            gaussian_elimination(
                [[1, 1, 1, 1, 0],
                 [1, 1, 1, 2, 0],
                 [1, 1, 1, 3, 0]],
                simple_partial_pivoting
            )[0],
            [[1, 1, 1, 1, 0],
             [0, 0, 0, 2, 0],
             [0, 0, 0, 0, 0]]
        )
        self.assertMatrixAlmostEqual(
            gaussian_elimination(
                [[7, 9, 6, 3],
                 [4, 3, 5, 2]],
                simple_partial_pivoting
            )[0],
            [[7, 9, 6, 3],
             [0, -15/7, 11/7, 2/7]]
        )
        self.assertMatrixAlmostEqual(
            gaussian_elimination(
                [[7, 9, 6, 3],
                 [4, 3, 5, -1],
                 [8, 7, 2, 6]],
                simple_partial_pivoting
            )[0],
            [[8, 7, 2, 6],
             [0, 23/8, 34/8, -18/8],
             [0, 0, 109/23, -101/23]]
        )
        self.assertMatrixAlmostEqual(
            gaussian_elimination(
                [[1, 1, 2, 8, 2],
                 [1, 2, 3, -10, 2],
                 [1, 3, 4, 6, 2]],
                simple_partial_pivoting
            )[0],
            [[1, 1, 2, 8, 2],
             [0, 2, 2, -2, 0],
             [0, 0, 0, -17, 0]]
        )
    
    def test_scaled_elim(self):
        self.assertMatrixAlmostEqual(
            gaussian_elimination(
                [[1, 1, 1, 1, 0],
                 [1, 1, 1, 2, 0],
                 [1, 1, 1, 3, 0]],
                scaled_partial_pivoting
            )[0],
            [[1, 1, 1, 1, 0],
             [0, 0, 0, 1, 0],
             [0, 0, 0, 0, 0]]
        )
        self.assertMatrixAlmostEqual(
            gaussian_elimination(
                [[7, 9, 6, 3],
                 [4, 3, 5, 2]],
                scaled_partial_pivoting
            )[0],
            [[4, 3, 5, 2],
             [0, 15/4, -11/4, -1/2]]
        )
        self.assertMatrixAlmostEqual(
            gaussian_elimination(
                [[7, 9, 6, 3],
                 [4, 3, 5, -1],
                 [8, 7, 2, 6]],
                scaled_partial_pivoting
            )[0],
            [[8, 7, 2, 6],
             [0, 23/8, 34/8, -18/8],
             [0, 0, 109/23, -101/23]]
        )
        self.assertMatrixAlmostEqual(
            gaussian_elimination(
                [[1, 1, 2, 8, 2],
                 [1, 2, 3, -10, 2],
                 [1, 3, 4, 6, 2]],
                scaled_partial_pivoting
            )[0],
            [[1, 3, 4, 6, 2],
             [0, -2, -2, 2, 0],
             [0, 0, 0, -17, 0]]
        )
    
    def test_full_elim(self):
        self.assertMatrixAlmostEqual(
            gaussian_elimination(
                [[1, 1, 1, 1, 0],
                 [1, 1, 1, 2, 0],
                 [1, 1, 1, 3, 0]],
                full_pivoting
            )[0],
            [[3, 1, 1, 1, 0],
             [0, 2/3, 2/3, 2/3, 0],
             [0, 0, 0, 0, 0]]
        )
        self.assertListEqual(
            gaussian_elimination(
                [[1, 1, 1, 1, 0],
                 [1, 1, 1, 2, 0],
                 [1, 1, 1, 3, 0]],
                full_pivoting
            )[1],
            [3, 1, 2, 0]
        )
        self.assertMatrixAlmostEqual(
            gaussian_elimination(
                [[7, 9, 6, 3],
                 [4, 3, 5, 2]],
                full_pivoting
            )[0],
            [[9, 7, 6, 3],
             [0, 5/3, 3, 1]]
        )
        self.assertListEqual(
            gaussian_elimination(
                [[7, 9, 6, 3],
                 [4, 3, 5, 2]],
                full_pivoting
            )[1],
            [1, 0, 2]
        )
        self.assertMatrixAlmostEqual(
            gaussian_elimination(
                [[7, 9, 6, 3],
                 [4, 3, 5, -1],
                 [8, 7, 2, 6]],
                full_pivoting
            )[0],
            [[9, 6, 7, 3],
             [0, 3, 5/3, -2],
             [0, 0, 109/27, 17/9]]
        )
        self.assertListEqual(
            gaussian_elimination(
                [[7, 9, 6, 3],
                 [4, 3, 5, -1],
                 [8, 7, 2, 6]],
                full_pivoting
            )[1],
            [1, 2, 0]
        )
        self.assertMatrixAlmostEqual(
            gaussian_elimination(
                [[1, 1, 2, 8, 2],
                 [1, 2, 3, -10, 2],
                 [1, 3, 4, 6, 2]],
                full_pivoting
            )[0],
            [[-10, 3, 2, 1, 2],
             [0, 5.8, 4.2, 1.6, 3.2],
             [0, 0, -17/29, 17/29, 34/29]]
        )
        self.assertListEqual(
            gaussian_elimination(
                [[1, 1, 2, 8, 2],
                 [1, 2, 3, -10, 2],
                 [1, 3, 4, 6, 2]],
                full_pivoting
            )[1],
            [3, 2, 1, 0]
        )
        
    def template_test_solve(self, pivot_strategy):
        self.assertListAlmostEqual(
            solve_system(
                [[1, 0],
                 [0, 1]],
                [3, 2],
                pivot_strategy
            ),
            [3, 2]
        )
        # NOTE: This represents solve_system
        #       attempting to solve an unsolvable system
        with self.assertRaises(ValueError):
            solve_system(
                [[7],
                 [4]],
                [3, 2],
                pivot_strategy
            )
        self.assertListAlmostEqual(
            solve_system(
                [[7, 9, 6],
                 [4, 3, 5]],
                [3, 2],
                pivot_strategy
            ),
            [3/5, -2/15, 0]
        )
        self.assertListAlmostEqual(
            solve_system(
                [[7, 9, 6],
                 [4, 3, 5],
                 [8, 7, 2]],
                [3, -1, 6],
                pivot_strategy
            ),
            [51/109, 64/109, -101/109]
        )
        self.assertListAlmostEqual(
            solve_system(
                [[1e-17, 1],
                 [1, 2]],
                [1,3],
                pivot_strategy
            ),
            [1, 1]
        )
        self.assertListAlmostEqual(
            solve_system(
                [[0.0003, 1.566],
                 [0.3454, -2.436]],
                [1.569, 1.018],
                pivot_strategy
            ),
            [10, 1]
        )
        self.assertListAlmostEqual(
            solve_system(
                [[1, 1, 1],
                 [1, 1, 1],
                 [1, 1, 1]],
                [2, 2, 2],
                pivot_strategy
            ),
            [2, 0, 0]
        )
        self.assertListAlmostEqual(
            matrix_vector_prod(
                [[1, 1, 2, 8],
                 [1, 2, 3, -10],
                 [1, 3, 4, 6]],
                solve_system(
                    [[1, 1, 2, 8],
                     [1, 2, 3, -10],
                     [1, 3, 4, 6]],
                    [2, 3, 4],
                    pivot_strategy
                )
            ),
            [2, 3, 4]
        )
        self.assertListAlmostEqual(
            matrix_vector_prod(
                [[1, 1, 2, 8],
                 [1, 2, 3, -10],
                 [1, 3, 4, 6]],
                solve_system(
                    [[1, 1, 2, 8],
                     [1, 2, 3, -10],
                     [1, 3, 4, 6]],
                    [-10, 4/5, 67.8],
                    pivot_strategy
                )
            ),
            [-10, 4/5, 67.8]
        )

    def test_solve_all(self):
        self.template_test_solve(naive_pivot_strategy)
        self.template_test_solve(simple_partial_pivoting)
        self.template_test_solve(scaled_partial_pivoting)
        self.template_test_solve(full_pivoting)

class TestLUDecomposition(TestCase):
    def test_naive_lu(self):
        self.assertListEqual(
            lu_decomposition(
                [[8, 9, 1],
                 [10, 2, 4],
                 [10, 3, 0]]
            )[0],
            [0, 1, 2]
        )
        self.assertMatrixAlmostEqual(
            lu_decomposition(
                [[8, 9, 1],
                 [10, 2, 4],
                 [10, 3, 0]]
            )[1],
            [[1, 0, 0],
             [5/4, 1, 0],
             [5/4, 33/37, 1]]
        )
        self.assertMatrixAlmostEqual(
            lu_decomposition(
                [[8, 9, 1],
                 [10, 2, 4],
                 [10, 3, 0]]
            )[2],
            [[8, 9, 1],
             [0, -37/4, 11/4],
             [0, 0, -137/37]]
        )
        self.assertListEqual(
            lu_decomposition(
                [[8, 9, 1],
                 [10, 2, 4],
                 [10, 3, 0]]
            )[3],
            [0, 1, 2]
        )
        
        self.assertListEqual(
            lu_decomposition(
                [[0, 0, 2, 9],
                 [1, -5, 6, 7],
                 [4, 6, 7, 8],
                 [5, 1, 13, 15]]
            )[0],
            [1, 2, 0, 3]
        )
        self.assertMatrixAlmostEqual(
            lu_decomposition(
                [[0, 0, 2, 9],
                 [1, -5, 6, 7],
                 [4, 6, 7, 8],
                 [5, 1, 13, 15]]
            )[1],
            [[1, 0, 0, 0],
             [4, 1, 0, 0],
             [0, 0, 1, 0],
             [5, 1, 0, 1]]
        )
        self.assertMatrixAlmostEqual(
            lu_decomposition(
                [[0, 0, 2, 9],
                 [1, -5, 6, 7],
                 [4, 6, 7, 8],
                 [5, 1, 13, 15]]
            )[2],
            [[1, -5, 6, 7],
             [0, 26, -17, -20],
             [0, 0, 2, 9],
             [0, 0, 0, 0]]
        )
        self.assertListEqual(
            lu_decomposition(
                [[0, 0, 2, 9],
                 [1, -5, 6, 7],
                 [4, 6, 7, 8],
                 [5, 1, 13, 15]]
            )[3],
            [0, 1, 2, 3]
        )
        
        self.assertListEqual(
            lu_decomposition(
                [[0, 5, 0, -4],
                 [-5, 0, 4, 4],
                 [3, 4, -3, -4],
                 [1, 0, -5, -4],
                 [-1, -5, -2, -3]]
            )[0],
            [1, 0, 2, 3, 4]
        )
        self.assertMatrixAlmostEqual(
            lu_decomposition(
                [[0, 5, 0, -4],
                 [-5, 0, 4, 4],
                 [3, 4, -3, -4],
                 [1, 0, -5, -4],
                 [-1, -5, -2, -3]]
            )[1],
            [[1, 0, 0, 0, 0],
             [0, 1, 0, 0, 0],
             [-3/5, 4/5, 1, 0, 0],
             [-1/5, 0, 7, 1, 0],
             [1/5, -1, 14/3, 229/216, 1]]
        )
        self.assertMatrixAlmostEqual(
            lu_decomposition(
                [[0, 5, 0, -4],
                 [-5, 0, 4, 4],
                 [3, 4, -3, -4],
                 [1, 0, -5, -4],
                 [-1, -5, -2, -3]]
            )[2],
            [[-5, 0, 4, 4],
             [0, 5, 0, -4],
             [0, 0, -3/5, 8/5],
             [0, 0, 0, -72/5],
             [0, 0, 0, 0]]
        )
        self.assertListEqual(
            lu_decomposition(
                [[0, 5, 0, -4],
                 [-5, 0, 4, 4],
                 [3, 4, -3, -4],
                 [1, 0, -5, -4],
                 [-1, -5, -2, -3]]
            )[3],
            [0, 1, 2, 3]
        )

        self.assertListEqual(
            lu_decomposition(
                [[1, 5, 2, 4],
                 [1, 5, 3, 7],
                 [1, 5, 9, 8],
                 [1, 5, 2, 6]]
            )[0],
            [0, 1, 2, 3]
        )
        self.assertMatrixAlmostEqual(
            lu_decomposition(
                [[1, 5, 2, 4],
                 [1, 5, 3, 7],
                 [1, 5, 9, 8],
                 [1, 5, 2, 6]]
            )[1],
            [[1, 0, 0, 0],
             [1, 1, 0, 0],
             [1, 7, 1, 0],
             [1, 0, -2/17, 1]]
        )
        self.assertMatrixAlmostEqual(
            lu_decomposition(
                [[1, 5, 2, 4],
                 [1, 5, 3, 7],
                 [1, 5, 9, 8],
                 [1, 5, 2, 6]]
            )[2],
            [[1, 5, 2, 4],
             [0, 0, 1, 3],
             [0, 0, 0, -17],
             [0, 0, 0, 0]]
        )
        self.assertListEqual(
            lu_decomposition(
                [[1, 5, 2, 4],
                 [1, 5, 3, 7],
                 [1, 5, 9, 8],
                 [1, 5, 2, 6]]
            )[3],
            [0, 1, 2, 3]
        )
        
    def test_full_pivot_lu(self):
        self.assertListEqual(
            lu_decomposition(
                [[8, 9, 1],
                 [10, 2, 4],
                 [10, 3, 0]],
                full_pivoting
            )[0],
            [1, 0, 2]
        )
        self.assertMatrixAlmostEqual(
            lu_decomposition(
                [[8, 9, 1],
                 [10, 2, 4],
                 [10, 3, 0]],
                 full_pivoting
            )[1],
            [[1, 0, 0],
             [4/5, 1, 0],
             [1, 5/37, 1]]
        )
        self.assertMatrixAlmostEqual(
            lu_decomposition(
                [[8, 9, 1],
                 [10, 2, 4],
                 [10, 3, 0]],
                 full_pivoting
            )[2],
            [[10, 2, 4],
             [0, 7.4, -2.2],
             [0, 0, -137/37]]
        )
        self.assertListEqual(
            lu_decomposition(
                [[8, 9, 1],
                 [10, 2, 4],
                 [10, 3, 0]],
                 full_pivoting
            )[3],
            [0, 1, 2]
        )
        
        self.assertListEqual(
            lu_decomposition(
                [[0, 0, 2, 9],
                 [1, -5, 6, 7],
                 [4, 6, 7, 8],
                 [5, 1, 13, 15]],
                 full_pivoting
            )[0],
            [3, 0, 2, 1]
        )
        self.assertMatrixAlmostEqual(
            lu_decomposition(
                [[0, 0, 2, 9],
                 [1, -5, 6, 7],
                 [4, 6, 7, 8],
                 [5, 1, 13, 15]],
                 full_pivoting
            )[1],
            [[1, 0, 0, 0],
             [9/15, 1, 0, 0],
             [8/15, -1/87, 1, 0],
             [7/15, 1/87, -1, 1]]
        )
        self.assertMatrixAlmostEqual(
            lu_decomposition(
                [[0, 0, 2, 9],
                 [1, -5, 6, 7],
                 [4, 6, 7, 8],
                 [5, 1, 13, 15]],
                 full_pivoting
            )[2],
            [[15, 13, 1, 5],
             [0, -5.8, -0.6, -3.0],
             [0, 0, 475/87, 113/87],
             [0, 0, 0, 0]]
        )
        self.assertListEqual(
            lu_decomposition(
                [[0, 0, 2, 9],
                 [1, -5, 6, 7],
                 [4, 6, 7, 8],
                 [5, 1, 13, 15]],
                 full_pivoting
            )[3],
            [3, 2, 1, 0]
        )
        
        self.assertListEqual(
            lu_decomposition(
                [[0, 5, 0, -4],
                 [-5, 0, 4, 4],
                 [3, 4, -3, -4],
                 [1, 0, -5, -4],
                 [-1, -5, -2, -3]],
                full_pivoting
            )[0],
            [0, 4, 1, 3, 2]
        )
        self.assertMatrixAlmostEqual(
            lu_decomposition(
                [[0, 5, 0, -4],
                 [-5, 0, 4, 4],
                 [3, 4, -3, -4],
                 [1, 0, -5, -4],
                 [-1, -5, -2, -3]],
                full_pivoting
            )[1],
            [[1, 0, 0, 0, 0],
             [-1, 1, 0, 0, 0],
             [0, -4/7, 1, 0, 0],
             [0, 4/7, -11/39, 1, 0],
             [4/5, 4/35, -109/195, 45.8/119, 1]]
        )
        self.assertMatrixAlmostEqual(
            lu_decomposition(
                [[0, 5, 0, -4],
                 [-5, 0, 4, 4],
                 [3, 4, -3, -4],
                 [1, 0, -5, -4],
                 [-1, -5, -2, -3]],
                full_pivoting
            )[2],
            [[5, -4, 0, 0],
             [0, -7, -1, -2],
             [0, 0, -39/7, 20/7],
             [0, 0, 0, -119/39],
             [0, 0, 0, 0]]
        )
        self.assertListEqual(
            lu_decomposition(
                [[0, 5, 0, -4],
                 [-5, 0, 4, 4],
                 [3, 4, -3, -4],
                 [1, 0, -5, -4],
                 [-1, -5, -2, -3]],
                full_pivoting
            )[3],
            [1, 3, 0, 2]
        )
        
        self.assertListEqual(
            lu_decomposition(
                [[1, 5, 2, 4],
                 [1, 5, 3, 7],
                 [1, 5, 9, 8],
                 [1, 5, 2, 6]],
                full_pivoting
            )[0],
            [2, 1, 0, 3]
        )
        self.assertMatrixAlmostEqual(
            lu_decomposition(
                [[1, 5, 2, 4],
                 [1, 5, 3, 7],
                 [1, 5, 9, 8],
                 [1, 5, 2, 6]],
                full_pivoting
            )[1],
            [[1, 0, 0, 0],
             [1/3, 1, 0, 0],
             [2/9, 20/39, 1, 0],
             [2/9, 38/39, 5/17, 1]]
        )
        self.assertMatrixAlmostEqual(
            lu_decomposition(
                [[1, 5, 2, 4],
                 [1, 5, 3, 7],
                 [1, 5, 9, 8],
                 [1, 5, 2, 6]],
                full_pivoting
            )[2],
            [[9, 8, 5, 1],
             [0, 13/3, 10/3, 2/3],
             [0, 0, 85/39, 17/39],
             [0, 0, 0, 0]]
        )
        self.assertListEqual(
            lu_decomposition(
                [[1, 5, 2, 4],
                 [1, 5, 3, 7],
                 [1, 5, 9, 8],
                 [1, 5, 2, 6]],
                full_pivoting
            )[3],
            [2, 3, 1, 0]
        )
        
    def test_forward_substitution(self):
        self.assertListAlmostEqual(
            forward_substitution(
                [[1, 0, 0, 0],
                 [9, 1, 0, 0],
                 [4, -7, 1, 0],
                 [9, 8, 5, 1]],
                [2, 3, 9, -10]
            ),
            [2, -15, -104, 612]
        )
        self.assertListAlmostEqual(
            forward_substitution(
                [[1, 0, 0, 0],
                 [1, 1, 0, 0],
                 [0, 1, 1, 0],
                 [2, 4, 1, 1]],
                [8, 1, -100, 2]
            ),
            [8, -7, -93, 107]
        )
     
    def template_test_lu_solve(self, pivot_strategy):
        self.assertListAlmostEqual(
            solve_system_using_lu_decomp(
                lu_decomposition(
                    [[1, 0],
                     [0, 1]],
                    pivot_strategy
                ),
                [3, 2]
            ),
            [3, 2]
        )
        # NOTE: This represents solve_system_using_lu_decomp
        #       attempting to solve an unsolvable system
        with self.assertRaises(ValueError):
            solve_system_using_lu_decomp(
                lu_decomposition(
                    [[7],
                     [4]],
                    pivot_strategy
                ),
                [3, 2]
            )
        self.assertListAlmostEqual(
            matrix_vector_prod(
                [[7, 9, 6],
                 [4, 3, 5]],
                solve_system_using_lu_decomp(
                    lu_decomposition(
                        [[7, 9, 6],
                         [4, 3, 5]],
                        pivot_strategy
                    ),
                    [3, 2]
                )
            ),
            [3, 2]
        )
        self.assertListAlmostEqual(
            solve_system_using_lu_decomp(
                lu_decomposition(
                    [[7, 9, 6],
                     [4, 3, 5],
                     [8, 7, 2]],
                    pivot_strategy
                ),
                [3, -1, 6]
            ),
            [51/109, 64/109, -101/109]
        )
        self.assertListAlmostEqual(
            solve_system_using_lu_decomp(
                lu_decomposition(
                    [[1e-17, 1],
                     [1, 2]],
                    pivot_strategy
                ),
                [1,3]
            ),
            [1, 1]
        )
        self.assertListAlmostEqual(
            solve_system_using_lu_decomp(
                lu_decomposition(
                    [[0.0003, 1.566],
                     [0.3454, -2.436]],
                    pivot_strategy
                ),
                [1.569, 1.018]
            ),
            [10, 1]
        )
        self.assertListAlmostEqual(
            solve_system_using_lu_decomp(
                lu_decomposition(
                    [[1, 1, 1],
                     [1, 1, 1],
                     [1, 1, 1]],
                    pivot_strategy
                ),
                [2, 2, 2]
            ),
            [2, 0, 0]
        )
        self.assertListAlmostEqual(
            matrix_vector_prod(
                [[1, 1, 2, 8],
                 [1, 2, 3, -10],
                 [1, 3, 4, 6]],
                solve_system_using_lu_decomp(
                    lu_decomposition(
                        [[1, 1, 2, 8],
                         [1, 2, 3, -10],
                         [1, 3, 4, 6]],
                        pivot_strategy
                    ),
                    [2, 3, 4]
                )
            ),
            [2, 3, 4]
        )
        self.assertListAlmostEqual(
            matrix_vector_prod(
                [[1, 1, 2, 8],
                 [1, 2, 3, -10],
                 [1, 3, 4, 6]],
                solve_system_using_lu_decomp(
                    lu_decomposition(
                        [[1, 1, 2, 8],
                         [1, 2, 3, -10],
                         [1, 3, 4, 6]],
                        pivot_strategy
                    ),
                    [-10, 4/5, 67.8]
                )
            ),
            [-10, 4/5, 67.8]
        )
        self.assertListAlmostEqual(
            matrix_vector_prod(
                [[0, 0, 2, 9],
                 [1, -5, 6, 7],
                 [4, 6, 7, 8],
                 [5, 1, 13, 15]],
                solve_system_using_lu_decomp(
                    lu_decomposition(
                        [[0, 0, 2, 9],
                         [1, -5, 6, 7],
                         [4, 6, 7, 8],
                         [5, 1, 13, 15]],
                        pivot_strategy
                    ),
                    [65, -11, 68, 57]
                )
            ),
            [65, -11, 68, 57]
        )

    def test_lu_solve(self):
        self.template_test_lu_solve(naive_pivot_strategy)
        self.template_test_lu_solve(simple_partial_pivoting)
        self.template_test_lu_solve(scaled_partial_pivoting)
        self.template_test_lu_solve(full_pivoting)
    
    def test_reconstruct(self):
        self.assertMatrixAlmostEqual(
            reconstruct_matrix(
                lu_decomposition(
                    [[8, 9, 1],
                     [10, 2, 4],
                     [10, 3, 0]],
                    full_pivoting
                )
            ),
            [[8, 9, 1],
             [10, 2, 4],
             [10, 3, 0]]
        )
        self.assertListEqual(
            reconstruct_matrix(
                lu_decomposition(
                    [[0, 0, 2, 9],
                     [1, -5, 6, 7],
                     [4, 6, 7, 8],
                     [5, 1, 13, 15]],
                     full_pivoting
                )
            ),
            [[0, 0, 2, 9],
             [1, -5, 6, 7],
             [4, 6, 7, 8],
             [5, 1, 13, 15]]
        )
        self.assertListEqual(
            reconstruct_matrix(
                lu_decomposition(
                    [[0, 5, 0, -4],
                     [-5, 0, 4, 4],
                     [3, 4, -3, -4],
                     [1, 0, -5, -4],
                     [-1, -5, -2, -3]],
                    full_pivoting
                )
            ),
            [[0, 5, 0, -4],
             [-5, 0, 4, 4],
             [3, 4, -3, -4],
             [1, 0, -5, -4],
             [-1, -5, -2, -3]]
        )
        self.assertListEqual(
            reconstruct_matrix(
                lu_decomposition(
                    [[1, 5, 2, 4],
                     [1, 5, 3, 7],
                     [1, 5, 9, 8],
                     [1, 5, 2, 6]],
                    full_pivoting
                )
            ),
            [[1, 5, 2, 4],
             [1, 5, 3, 7],
             [1, 5, 9, 8],
             [1, 5, 2, 6]]
        )
        self.assertListEqual(
            reconstruct_matrix(
                lu_decomposition(
                    [[7, -7, 5, -5],
                     [-2, -8, 4, -1],
                     [7, 5, -4, -5]],
                    full_pivoting
                )
            ),
            [[7, -7, 5, -5],
             [-2, -8, 4, -1],
             [7, 5, -4, -5]]
        )

class TestImageKernelInverse(TestCase):
    def test_image(self):
        self.assertMatrixAlmostEqual(
            find_image_using_lu_decomp(
                lu_decomposition(
                    [[0, 0, 0],
                     [0, 0, 0],
                     [0, 0, 0]],
                    full_pivoting
                )
            ),
            []
        )
        self.assertMatrixAlmostEqual(
            find_image_using_lu_decomp(
                lu_decomposition(
                    [[0, 0, 0],
                     [0, 2, 0],
                     [0, 3, 0]],
                    full_pivoting
                )
            ),
            [[0, 2, 3]]
        )
        self.assertMatrixAlmostEqual(
            find_image_using_lu_decomp(
                lu_decomposition(
                    [[0, 0, 0],
                     [0, 2, 3],
                     [0, 0, 0]],
                    full_pivoting
                )
            ),
            [[0, 3, 0]]
        )
        self.assertMatrixAlmostEqual(
            find_image_using_lu_decomp(
                lu_decomposition(
                    [[1, 8, 7, 4],
                     [5, 3, -2, 6],
                     [8, 9, 1, 3],
                     [4, 7, 3, 2]]
                )
            ),
            [[1, 5, 8, 4],
             [8, 3, 9, 7],
             [4, 6, 3, 2]]
        )
        self.assertMatrixAlmostEqual(
            find_image_using_lu_decomp(
                lu_decomposition(
                    [[1, 8, 7, 4],
                     [5, 3, -2, 6],
                     [8, 9, 1, 3],
                     [4, 7, 3, 2]],
                    full_pivoting
                )
            ),
            [[1, 5, 8, 4],
             [8, 3, 9, 7],
             [4, 6, 3, 2]]
        )
        self.assertMatrixAlmostEqual(
            find_image_using_lu_decomp(
                lu_decomposition(
                    [[8, 9, 1],
                     [10, 2, 4],
                     [10, 3, 0]],
                    full_pivoting
                )
            ),
            [[8, 10, 10],
             [9, 2, 3],
             [1, 4, 0]]
        )
        self.assertMatrixAlmostEqual(
            find_image_using_lu_decomp(
                lu_decomposition(
                    [[0, 0, 2, 9],
                     [1, -5, 6, 7],
                     [4, 6, 7, 8],
                     [5, 1, 13, 15]],
                     full_pivoting
                )
            ),
            [[0, -5, 6, 1],
             [2, 6, 7, 13],
             [9, 7, 8, 15]]
        )
        self.assertMatrixAlmostEqual(
            find_image_using_lu_decomp(
                lu_decomposition(
                    [[0, 5, 0, -4],
                     [-5, 0, 4, 4],
                     [3, 4, -3, -4],
                     [1, 0, -5, -4],
                     [-1, -5, -2, -3]],
                    full_pivoting
                )
            ),
            [[0, -5, 3, 1, -1],
             [5, 0, 4, 0, -5],
             [0, 4, -3, -5, -2],
             [-4, 4, -4, -4, -3]]
        )
        self.assertMatrixAlmostEqual(
            find_image_using_lu_decomp(
                lu_decomposition(
                    [[1, 5, 2, 4],
                     [1, 5, 3, 7],
                     [1, 5, 9, 8],
                     [1, 5, 2, 6]]
                )
            ),
            [[1, 1, 1, 1],
             [2, 3, 9, 2],
             [4, 7, 8, 6]]
        )
        self.assertMatrixAlmostEqual(
            find_image_using_lu_decomp(
                lu_decomposition(
                    [[1, 5, 2, 4],
                     [1, 5, 3, 7],
                     [1, 5, 9, 8],
                     [1, 5, 2, 6]],
                    full_pivoting
                )
            ),
            [[5, 5, 5, 5],
             [2, 3, 9, 2],
             [4, 7, 8, 6]]
        )
        self.assertMatrixAlmostEqual(
            find_image_using_lu_decomp(
                lu_decomposition(
                    [[7, -7, 5, -5],
                     [-2, -8, 4, -1],
                     [7, 5, -4, -5]],
                    full_pivoting
                )
            ),
            [[7, -2, 7],
             [-7, -8, 5],
             [-5, -1, -5]]
        )
        
    def test_kernel(self):
        self.assertMatrixAlmostEqual(
            find_kernel_using_lu_decomp(
                lu_decomposition(
                    [[0, 0, 0],
                     [0, 0, 0],
                     [0, 0, 0]],
                    full_pivoting
                )
            ),
            [[1, 0, 0],
             [0, 1, 0],
             [0, 0, 1]]
        )
        self.assertMatrixAlmostEqual(
            find_kernel_using_lu_decomp(
                lu_decomposition(
                    [[0, 0, 0],
                     [0, 2, 0],
                     [0, 3, 0]],
                    full_pivoting
                )
            ),
            [[1, 0, 0],
             [0, 0, 1]]
        )
        self.assertMatrixAlmostEqual(
            find_kernel_using_lu_decomp(
                lu_decomposition(
                    [[0, 0, 0],
                     [0, 2, 3],
                     [0, 0, 0]],
                    full_pivoting
                )
            ),
            [[0, 1, -2/3],
             [1, 0, 0]]
        )
        self.assertMatrixAlmostEqual(
            find_kernel_using_lu_decomp(
                lu_decomposition(
                    [[1, 8, 7, 4],
                     [5, 3, -2, 6],
                     [8, 9, 1, 3],
                     [4, 7, 3, 2]]
                )
            ),
            [[1, -1, 1, 0]]
        )
        self.assertMatrixAlmostEqual(
            find_kernel_using_lu_decomp(
                lu_decomposition(
                    [[1, 8, 7, 4],
                     [5, 3, -2, 6],
                     [8, 9, 1, 3],
                     [4, 7, 3, 2]],
                    full_pivoting
                )
            ),
            [[1, -1, 1, 0]]
        )
        self.assertMatrixAlmostEqual(
            find_kernel_using_lu_decomp(
                lu_decomposition(
                    [[8, 9, 1],
                     [10, 2, 4],
                     [10, 3, 0]],
                    full_pivoting
                )
            ),
            []
        )
        self.assertMatrixAlmostEqual(
            find_kernel_using_lu_decomp(
                lu_decomposition(
                    [[0, 0, 2, 9],
                     [1, -5, 6, 7],
                     [4, 6, 7, 8],
                     [5, 1, 13, 15]]
                )
            ),
            [[237.5/26, -56.5/26, -9/2, 1]]
        )
        self.assertMatrixAlmostEqual(
            find_kernel_using_lu_decomp(
                lu_decomposition(
                    [[0, 5, 0, -4],
                     [-5, 0, 4, 4],
                     [3, 4, -3, -4],
                     [1, 0, -5, -4],
                     [-1, -5, -2, -3]],
                    full_pivoting
                )
            ),
            []
        )
        self.assertMatrixAlmostEqual(
            find_kernel_using_lu_decomp(
                lu_decomposition(
                    [[1, 5, 2, 4],
                     [1, 5, 3, 7],
                     [1, 5, 9, 8],
                     [1, 5, 2, 6]]
                )
            ),
            [[-5, 1, 0, 0]]
        )
        self.assertMatrixAlmostEqual(
            find_kernel_using_lu_decomp(
                lu_decomposition(
                    [[1, 5, 2, 4],
                     [1, 5, 3, 7],
                     [1, 5, 9, 8],
                     [1, 5, 2, 6]],
                    full_pivoting
                )
            ),
            [[1, -1/5, 0, 0]]
        )
        self.assertMatrixAlmostEqual(
            find_kernel_using_lu_decomp(
                lu_decomposition(
                    [[7, -7, 5, -5],
                     [-2, -8, 4, -1],
                     [7, 5, -4, -5]],
                    full_pivoting
                )
            ),
            [[-39/68, 51/68, 1, -58/68]]
        )
    
    def template_test_singular(self, pivot_strategy):
        self.assertEqual(
            is_singular_using_lu_decomp(
                lu_decomposition(
                    identity(10),
                    pivot_strategy
                )
            ),
            False
        )
        self.assertEqual(
            is_singular_using_lu_decomp(
                lu_decomposition(
                    [[8, 9, 1],
                     [10, 2, 4],
                     [10, 3, 0]],
                    pivot_strategy
                )
            ),
            False
        )
        self.assertEqual(
            is_singular_using_lu_decomp(
                lu_decomposition(
                    [[-2, -8, -10, -7, -4],
                     [2, 2, 10, 9, -7],
                     [-3, 5, 8, -4, -10],
                     [-6, 7, 0, 2, -1],
                     [-5, 4, -7, 3, 9]],
                    pivot_strategy
                )
            ),
            False
        )
        self.assertEqual(
            is_singular_using_lu_decomp(
                lu_decomposition(
                    [[9, 9, -10, 8, -4],
                     [-5, 2, -2, -2, 7],
                     [8, -4, 7, 7, -6],
                     [-9, -7, 2, 1, 8],
                     [3, 1, -9, 6, -2]],
                    pivot_strategy
                )
            ),
            False
        )
        self.assertEqual(
            is_singular_using_lu_decomp(
                lu_decomposition(
                    [[0, 0, 0],
                     [0, 0, 0],
                     [0, 0, 0]],
                    pivot_strategy
                )
            ),
            True
        )
        self.assertEqual(
            is_singular_using_lu_decomp(
                lu_decomposition(
                    [[0, 0, 0],
                     [0, 2, 0],
                     [0, 3, 0]],
                    pivot_strategy
                )
            ),
            True
        )
        self.assertEqual(
            is_singular_using_lu_decomp(
                lu_decomposition(
                    [[0, 0, 0],
                     [0, 2, 3],
                     [0, 0, 0]],
                    pivot_strategy
                )
            ),
            True
        )
        self.assertEqual(
            is_singular_using_lu_decomp(
                lu_decomposition(
                    [[1, 8, 7, 4],
                     [5, 3, -2, 6],
                     [8, 9, 1, 3],
                     [4, 7, 3, 2]],
                    pivot_strategy
                )
            ),
            True
        )
        self.assertEqual(
            is_singular_using_lu_decomp(
                lu_decomposition(
                    [[1, 8, 7, 4],
                     [5, 3, -2, 6],
                     [8, 9, 1, 3],
                     [4, 7, 3, 2]],
                    pivot_strategy
                )
            ),
            True
        )
        self.assertEqual(
            is_singular_using_lu_decomp(
                lu_decomposition(
                    [[0, 0, 2, 9],
                     [1, -5, 6, 7],
                     [4, 6, 7, 8],
                     [5, 1, 13, 15]],
                    pivot_strategy
                )
            ),
            True
        )
        self.assertEqual(
            is_singular_using_lu_decomp(
                lu_decomposition(
                    [[1, 5, 2, 4],
                     [1, 5, 3, 7],
                     [1, 5, 9, 8],
                     [1, 5, 2, 6]],
                    pivot_strategy
                )
            ),
            True
        )

    def test_singular(self):
        self.template_test_singular(naive_pivot_strategy)
        self.template_test_singular(simple_partial_pivoting)
        self.template_test_singular(scaled_partial_pivoting)
        self.template_test_singular(full_pivoting)

    def template_test_inverse(self, pivot_strategy):
        self.assertMatrixAlmostEqual(
            find_inverse_using_lu_decomp(
                lu_decomposition(
                    identity(10),
                    pivot_strategy
                )
            ),
            identity(10)
        )
        self.assertMatrixAlmostEqual(
            find_inverse_using_lu_decomp(
                lu_decomposition(
                    [[8, 9, 1],
                     [10, 2, 4],
                     [10, 3, 0]],
                    pivot_strategy
                )
            ),
            [[-12/274, 3/274, 34/274],
             [40/274, -10/274, -22/274],
             [10/274, 66/274, -74/274]]
        )
        self.assertMatrixAlmostEqual(
            find_inverse_using_lu_decomp(
                lu_decomposition(
                    [[-2, -8, -10, -7, -4],
                     [2, 2, 10, 9, -7],
                     [-3, 5, 8, -4, -10],
                     [-6, 7, 0, 2, -1],
                     [-5, 4, -7, 3, 9]],
                    pivot_strategy
                )
            ),
            [[-2025/11965, -3645/11965, -7705/11965, 10585/11965, -11120/11965],
             [-1825/11965, -3285/11965, -5910/11965, 9835/11965, -8840/11965],
             [479/11965, 2298/11965, 5675/11965, -8377/11965, 7375/11965],
             [293/11965, 1006/11965, -900/11965, 696/11965, -10/11965],
             [-39/11965, 887/11965, 3060/11965, -5238/11965, 4820/11965]]
        )
        self.assertMatrixAlmostEqual(
            find_inverse_using_lu_decomp(
                lu_decomposition(
                    apply_permutation(
                        [3, 1, 2, 4, 0],
                        identity(5)
                    ),
                    pivot_strategy
                )
            ),
            apply_permutation(
                invert_permutation([3, 1, 2, 4, 0]),
                identity(5)
            )
        )

    def test_inverse(self):
        self.template_test_inverse(naive_pivot_strategy)
        self.template_test_inverse(simple_partial_pivoting)
        self.template_test_inverse(scaled_partial_pivoting)
        self.template_test_inverse(full_pivoting)
    
    def template_test_determinant(self, pivot_strategy):
        self.assertAlmostEqual(
            find_determinant_using_lu_decomp(
                lu_decomposition(
                    identity(10),
                    pivot_strategy
                )
            ),
            1
        )
        self.assertAlmostEqual(
            find_determinant_using_lu_decomp(
                lu_decomposition(
                    [[8, 9, 1],
                     [10, 2, 4],
                     [10, 3, 0]],
                    pivot_strategy
                )
            ),
            274
        )
        self.assertAlmostEqual(
            find_determinant_using_lu_decomp(
                lu_decomposition(
                    [[-2, -8, -10, -7, -4],
                     [2, 2, 10, 9, -7],
                     [-3, 5, 8, -4, -10],
                     [-6, 7, 0, 2, -1],
                     [-5, 4, -7, 3, 9]],
                    pivot_strategy
                )
            ),
            11965
        )
        self.assertAlmostEqual(
            find_determinant_using_lu_decomp(
                lu_decomposition(
                    [[9, 9, -10, 8, -4],
                     [-5, 2, -2, -2, 7],
                     [8, -4, 7, 7, -6],
                     [-9, -7, 2, 1, 8],
                     [3, 1, -9, 6, -2]],
                    pivot_strategy
                )
            ),
            9848
        )
        self.assertAlmostEqual(
            find_determinant_using_lu_decomp(
                lu_decomposition(
                    [[0, 0, 0],
                     [0, 0, 0],
                     [0, 0, 0]],
                    pivot_strategy
                )
            ),
            0
        )
        self.assertAlmostEqual(
            find_determinant_using_lu_decomp(
                lu_decomposition(
                    [[0, 0, 0],
                     [0, 2, 0],
                     [0, 3, 0]],
                    pivot_strategy
                )
            ),
            0
        )
        self.assertAlmostEqual(
            find_determinant_using_lu_decomp(
                lu_decomposition(
                    [[0, 0, 0],
                     [0, 2, 3],
                     [0, 0, 0]],
                    pivot_strategy
                )
            ),
            0
        )
        self.assertAlmostEqual(
            find_determinant_using_lu_decomp(
                lu_decomposition(
                    [[1, 8, 7, 4],
                     [5, 3, -2, 6],
                     [8, 9, 1, 3],
                     [4, 7, 3, 2]],
                    pivot_strategy
                )
            ),
            0
        )
        self.assertAlmostEqual(
            find_determinant_using_lu_decomp(
                lu_decomposition(
                    [[1, 8, 7, 4],
                     [5, 3, -2, 6],
                     [8, 9, 1, 3],
                     [4, 7, 3, 2]],
                    pivot_strategy
                )
            ),
            0
        )
        self.assertAlmostEqual(
            find_determinant_using_lu_decomp(
                lu_decomposition(
                    [[0, 0, 2, 9],
                     [1, -5, 6, 7],
                     [4, 6, 7, 8],
                     [5, 1, 13, 15]],
                    pivot_strategy
                )
            ),
            0
        )
        self.assertAlmostEqual(
            find_determinant_using_lu_decomp(
                lu_decomposition(
                    [[1, 5, 2, 4],
                     [1, 5, 3, 7],
                     [1, 5, 9, 8],
                     [1, 5, 2, 6]],
                    pivot_strategy
                )
            ),
            0
        )

    def test_determinant(self):
        self.template_test_determinant(naive_pivot_strategy)
        self.template_test_determinant(simple_partial_pivoting)
        self.template_test_determinant(scaled_partial_pivoting)
        self.template_test_determinant(full_pivoting)
    
    def test_naive_determinant(self):
        self.assertAlmostEqual(
            find_naive_determinant(
                identity(5)
            ),
            1
        )
        self.assertAlmostEqual(
            find_naive_determinant(
                [[8, 9, 1],
                 [10, 2, 4],
                 [10, 3, 0]]
            ),
            274
        )
        self.assertAlmostEqual(
            find_naive_determinant(
                [[-2, -8, -10, -7, -4],
                 [2, 2, 10, 9, -7],
                 [-3, 5, 8, -4, -10],
                 [-6, 7, 0, 2, -1],
                 [-5, 4, -7, 3, 9]]
            ),
            11965
        )
        self.assertAlmostEqual(
            find_naive_determinant(
                [[9, 9, -10, 8, -4],
                 [-5, 2, -2, -2, 7],
                 [8, -4, 7, 7, -6],
                 [-9, -7, 2, 1, 8],
                 [3, 1, -9, 6, -2]]
            ),
            9848
        )
        self.assertAlmostEqual(
            find_naive_determinant(
                [[0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0]]
            ),
            0
        )
        self.assertAlmostEqual(
            find_naive_determinant(
                [[0, 0, 0],
                 [0, 2, 0],
                 [0, 3, 0]]
            ),
            0
        )
        self.assertAlmostEqual(
            find_naive_determinant(
                [[0, 0, 0],
                 [0, 2, 3],
                 [0, 0, 0]]
            ),
            0
        )
        self.assertAlmostEqual(
            find_naive_determinant(
                [[1, 8, 7, 4],
                 [5, 3, -2, 6],
                 [8, 9, 1, 3],
                 [4, 7, 3, 2]]
            ),
            0
        )
        self.assertAlmostEqual(
            find_naive_determinant(
                [[1, 8, 7, 4],
                 [5, 3, -2, 6],
                 [8, 9, 1, 3],
                 [4, 7, 3, 2]]
            ),
            0
        )
        self.assertAlmostEqual(
            find_naive_determinant(
                [[0, 0, 2, 9],
                 [1, -5, 6, 7],
                 [4, 6, 7, 8],
                 [5, 1, 13, 15]]
            ),
            0
        )
        self.assertAlmostEqual(
            find_naive_determinant(
                [[1, 5, 2, 4],
                 [1, 5, 3, 7],
                 [1, 5, 9, 8],
                 [1, 5, 2, 6]]
            ),
            0
        )

class TestQRDecomposition(TestCase):
    def template_test_qr(self, matrix):
        q, r = qr_decomposition(matrix)
        for i in range(len(r)):
            for j in range(min(len(r[i]), i)):
                self.assertAlmostEqual(r[i][j], 0)
        self.assertMatrixAlmostEqual(
            matrix_matrix_prod(q, transpose(q)),
            identity(len(matrix))
        )
        self.assertMatrixAlmostEqual(
            matrix_matrix_prod(q, r),
            matrix
        )
        
    def test_qr(self):
        self.template_test_qr(
            [[1, 5, 2, 4],
             [1, 5, 3, 7],
             [1, 5, 9, 8],
             [1, 5, 2, 6]]
        )
        self.template_test_qr(
            [[1, 8, 7, 4],
             [5, 3, -2, 6],
             [8, 9, 1, 3],
             [4, 7, 3, 2]]
        )
        self.template_test_qr(
            [[-2, -8, -10, -7, -4],
             [2, 2, 10, 9, -7],
             [-3, 5, 8, -4, -10],
             [-6, 7, 0, 2, -1],
             [-5, 4, -7, 3, 9]]
        )
        self.template_test_qr(
            [[9, 9, -10, 8, -4],
             [-5, 2, -2, -2, 7],
             [8, -4, 7, 7, -6],
             [-9, -7, 2, 1, 8],
             [3, 1, -9, 6, -2]]
        )
        self.template_test_qr(
            [[2, 2],
             [2, 1],
             [1, 5]]
        )
        self.template_test_qr(
            [[5, 3, -2],
             [3, -2, -4],
             [2, -3, -3],
             [-1, 4, 4],
             [-5, 2, -4],
             [-2, -4, 3],
             [-5, 1, 3],
             [0, 1, 3],
             [-1, 1, -5],
             [2, 4, 0]]
        )
        self.template_test_qr(
            [[8, 1, 8, 3],
             [-9, -10, -7, 1],
             [8, 4, 10, -6],
             [3, -1, 3, 10],
             [-5, -4, -2, -5],
             [-5, 2, -4, 7],
             [2, 8, 1, -6],
             [-6, -10, -4, -7]]
        )

    def test_least_squares(self):
        self.assertListAlmostEqual(
            find_least_squares_solution_using_qr_decomp(
                qr_decomposition(
                    [[-2, -8, -10, -7, -4],
                     [2, 2, 10, 9, -7],
                     [-3, 5, 8, -4, -10],
                     [-6, 7, 0, 2, -1],
                     [-5, 4, -7, 3, 9]]
                ),
                [1, -3, 4, 5, 0]
            ),
            solve_system(
                [[-2, -8, -10, -7, -4],
                 [2, 2, 10, 9, -7],
                 [-3, 5, 8, -4, -10],
                 [-6, 7, 0, 2, -1],
                 [-5, 4, -7, 3, 9]],
                [1, -3, 4, 5, 0]
            )
        )
        self.assertListAlmostEqual(
            find_least_squares_solution_using_qr_decomp(
                qr_decomposition(
                    [[9, 9, -10, 8, -4],
                     [-5, 2, -2, -2, 7],
                     [8, -4, 7, 7, -6],
                     [-9, -7, 2, 1, 8],
                     [3, 1, -9, 6, -2]]
                ),
                [4, 5, 6, -2, 1]
            ),
            solve_system(
                [[9, 9, -10, 8, -4],
                 [-5, 2, -2, -2, 7],
                 [8, -4, 7, 7, -6],
                 [-9, -7, 2, 1, 8],
                 [3, 1, -9, 6, -2]],
                [4, 5, 6, -2, 1]
            )
        )
        self.assertListAlmostEqual(
            find_least_squares_solution_using_qr_decomp(
                qr_decomposition(
                    [[2, 2],
                     [2, 1],
                     [1, 5]]
                ),
                [1, -3, 4]
            ),
            [-1.4026845637583893, 1.1476510067114094]
        )
        self.assertListAlmostEqual(
            find_least_squares_solution_using_qr_decomp(
                qr_decomposition(
                    [[5, 3, -2],
                     [3, -2, -4],
                     [2, -3, -3],
                     [-1, 4, 4],
                     [-5, 2, -4],
                     [-2, -4, 3],
                     [-5, 1, 3],
                     [0, 1, 3],
                     [-1, 1, -5],
                     [2, 4, 0]]
                ),
                [1, -2, 3, 4, 7, 8, 100, 4, 4, 3]
            ),
            [-5.212920257428537, 1.305850891348219, 1.279570138594825]
        )
    
    def test_best_fit(self):
        self.assertListAlmostEqual(
            calc_best_fit_coefficients(
                [lambda x: 1, lambda x: x],
                [[0, 5],
                 [1, 4],
                 [9, 8],
                 [7, 6],
                 [-4, -5]]
            ),
            [1.5053003533568918, 0.8056537102473497]
        )
        self.assertListAlmostEqual(
            calc_best_fit_coefficients(
                [lambda x: 1, lambda x: x, lambda x: x**2],
                [[0, 5],
                 [1, 4],
                 [9, 8],
                 [7, 6],
                 [-4, -5]]
            ),
            [3.2741558338617645, 1.458455532656943, -0.11789592580849712]
        )
        self.assertListAlmostEqual(
            calc_best_fit_coefficients(
                [lambda x: 1, lambda x: x, lambda x: x**2, lambda x: x**3],
                [[0, 5],
                 [1, 4],
                 [9, 8],
                 [7, 6],
                 [-4, -5]]
            ),
            [4.182845260598198, 0.906243151424816, -0.25343414362002603, 0.02235820796891005]
        )

if __name__ == "__main__":
    unittest.main()