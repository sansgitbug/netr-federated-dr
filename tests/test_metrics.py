import numpy as np

from netra.model import decode, quadratic_weighted_kappa, referable_sens_spec


def test_kappa_perfect_agreement_is_one():
    cm = np.diag([10, 5, 8, 3, 2])
    assert quadratic_weighted_kappa(cm) == 1.0


def test_kappa_penalises_far_errors_more():
    near = np.diag([10, 10, 10, 10, 10]).astype(float)
    far = near.copy()
    near[2, 2] -= 5; near[2, 3] += 5   # off by one grade
    far[2, 2] -= 5; far[2, 4] += 5     # off by two grades
    assert quadratic_weighted_kappa(near) > quadratic_weighted_kappa(far)


def test_kappa_empty_matrix_is_zero():
    assert quadratic_weighted_kappa(np.zeros((5, 5))) == 0.0


def test_referable_sensitivity_specificity():
    cm = np.zeros((5, 5))
    cm[0, 0] = 8; cm[0, 2] = 2      # 2 of 10 non-referable called referable
    cm[3, 3] = 3; cm[3, 1] = 1      # 1 of 4 referable missed
    sens, spec = referable_sens_spec(cm)
    assert sens == 0.75
    assert spec == 0.8


def test_decode_covers_all_five_grades():
    for g in range(5):
        probs = np.full(5, 0.05)
        probs[g] = 0.8
        grade, conf, referable = decode(probs)
        assert grade == g
        assert abs(conf - 0.8) < 1e-9
        assert referable == (g >= 2)
