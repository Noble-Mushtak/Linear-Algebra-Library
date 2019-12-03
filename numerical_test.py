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

if __name__ == "__main__":
    unittest.main()