from src.logreg_train import formula
from pytest import raises

def test_logreg_train_return_1_if_weight_1():
    values = [1,1]
    weights = [1,1,1]

    result = formula(values, weights)
    assert result == 3

def test_logreg_train_return_1_if_bias_1():
    values = [0,0]
    weights = [0,0,1] 

    result = formula(values, weights)
    assert result == 1

def test_logreg_train_return_error_with_different_length():
    values = [0,0]
    weights = [0,0] 

    with raises(ValueError):
        formula(values, weights)