import pytest
import torch
from test_utils import assert_expected, set_rng_seed
import sys
sys.path.append('..')
from kernels.flashalibi_algo import dynamic_distance_bias_matrix

@pytest.fixture(autouse=True)
def set_seed():
    set_rng_seed(2020)



class TestFlashAlibiMask:
    @pytest.fixture
    def N(self,):
        return 2048
    
    @pytest.fixture
    def step_row(self,):
        return 128
    
    @pytest.fixture
    def step_col(self,):
        return 64
    
    @pytest.fixture(autouse=True)
    def mask(self, N):
        return dynamic_distance_bias_matrix((0,N), (0,N)) 
    

    def test_walk_left_side(self,step_row, step_col, mask, N):
        start_row = 0
        start_col = 0
        assert N % step_row ==0, f"mismatch in row step size vs N"
        for stop_row in range(step_row, N, step_row):
            #print(f"{start_row=}, {stop_row=}, {start_col=}, {step_col=}")
            expected = mask[start_row:stop_row, start_col:step_col]
            actual = dynamic_distance_bias_matrix((start_row, stop_row), (start_col,step_col))
            #print(f"{actual=}, {expected=}")
            assert torch.equal(actual, expected)
            start_row+= step_row
    
    def test_walk_right_side(self, mask, N, step_row, step_col):
        start_row = 0
        start_col = N - step_col
        stop_col = N
        assert N % step_row ==0, f"mismatch in row step size vs N"
        for stop_row in range(step_row, N, step_row):
            #print(f"{start_row=}, {stop_row=}, {start_col=}, {stop_col=}")
            expected = mask[start_row:stop_row, start_col:stop_col]
            actual = dynamic_distance_bias_matrix((start_row, stop_row), (start_col,stop_col))
            #print(f"{actual.shape=}, {expected.shape=}")
            assert torch.equal(actual, expected)
            start_row += step_row

    def test_walk_diagonal(self, mask, N, step_row, step_col):
        start_row = 0
        start_col = 0
        stop_col = step_col

        assert N % step_row ==0, f"mismatch in row step size vs N"
        assert N % step_col ==0, f"mismatch in col step size vs N"
        for stop_row in range(step_row, N, step_row):
            #print(f"{start_row=}, {stop_row=}, {start_col=}, {stop_col=}")
            expected = mask[start_row:stop_row, start_col:stop_col]
            actual = dynamic_distance_bias_matrix((start_row, stop_row), (start_col,stop_col))
            #print(f"{actual.shape=}, {expected.shape=}")
            assert torch.equal(actual, expected)
            start_row += step_row
            start_col += step_col
            stop_col += step_col
            if stop_col > N:
                # cols don't necessarily fit evenly relative to rows
                break

    def test_walk_reverse_diagonal(self, mask, N, step_row, step_col):
        start_row = 0
        start_col = N-step_col
        stop_col = N

        assert N % step_row ==0, f"mismatch in row step size vs N"
        assert N % step_col ==0, f"mismatch in col step size vs N"
        for stop_row in range(step_row, N, step_row):
            #print(f"{start_row=}, {stop_row=}, {start_col=}, {stop_col=}")
            expected = mask[start_row:stop_row, start_col:stop_col]
            actual = dynamic_distance_bias_matrix((start_row, stop_row), (start_col,stop_col))
            #print(f"{actual.shape=}, {expected.shape=}")
            assert torch.equal(actual, expected)
            start_row += step_row
            start_col -= step_col
            stop_col -= step_col
            if stop_col < 0:
                # cols don't necessarily fit evenly relative to rows
                break

    def test_walk_all_rows(self, mask, N, step_row, step_col):
        start_row = 0
        start_col = 0
        stop_col = step_col

        assert N % step_row ==0, f"mismatch in row step size vs N"
        assert N % step_col ==0, f"mismatch in col step size vs N"
        block_counter=0
        for stop_row in range(step_row, N, step_row):
            for stop_col in range(step_col,N, step_col):

                #print(f"{start_row=}, {stop_row=}, {start_col=}, {stop_col=}")
                expected = mask[start_row:stop_row, start_col:stop_col]
                actual = dynamic_distance_bias_matrix((start_row, stop_row), (start_col,stop_col))
                #print(f"{actual.shape=}, {expected.shape=}")
                assert torch.equal(actual, expected)
                
                start_col += step_col
                block_counter+=1
            
            start_row += step_row
            start_col = 0
        print(f"checked {block_counter} blocks")

    def test_walk_all_cols(self, mask, N, step_row, step_col):
        start_row = 0
        start_col = 0
        stop_col = step_col

        assert N % step_row ==0, f"mismatch in row step size vs N"
        assert N % step_col ==0, f"mismatch in col step size vs N"
        block_counter=0
        
        for stop_col in range(step_col,N, step_col):
            for stop_row in range(step_row, N, step_row):

                #print(f"{start_row=}, {stop_row=}, {start_col=}, {stop_col=}")
                expected = mask[start_row:stop_row, start_col:stop_col]
                actual = dynamic_distance_bias_matrix((start_row, stop_row), (start_col,stop_col))
                #print(f"{actual.shape=}, {expected.shape=}")
                assert torch.equal(actual, expected)
                
                start_row += step_row
                block_counter+=1
            
            start_col += step_col
            start_row = 0
        print(f"checked {block_counter} blocks")
    