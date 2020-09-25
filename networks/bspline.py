import torch
import torch.nn as nn


def _bspline_basis0(xi, knots):
    return torch.where((knots[:-1] <=  xi, xi < knots[1:]), 1.0, 0.0)


def bspline_basis(xi, p, knots):
    if p == 0:
        return _bspline_basis0(xi, knots)
    else:
        basis_p_minus_1 = bspline_basis(xi, p - 1, knots)

    first_term_numerator = xi - knots[:-p]
    first_term_denominator = knots[p:] - knots[:-p]

    second_term_numerator = knots[(p + 1):] - xi
    second_term_denominator = (knots[(p + 1):] -
                                knots[1:-p])

    first_term = torch.where(first_term_denominator != 0.0,
                            (first_term_numerator /
                            first_term_denominator), 0.0)
    second_term = torch.where(second_term_denominator != 0.0,
                            (second_term_numerator /
                            second_term_denominator), 0.0)

    return  (first_term[:-1] * basis_p_minus_1[:-1] +
                second_term * basis_p_minus_1[1:])


def sample_curve(degree, knots, cp, num_points=10):
    t = torch.linspace(0.0, 1.0, num_points)
    N = bspline_basis(t, degree, knots)
    batch_size = knots.size(0)
    P = torch.zeros(batch_size, num_points, 3)
    for i in range(cp.size(1)):
        P += N[:, i].unsqueeze * cp[:, i, :, :, :]
    return P


class BSplineCurveSampler(nn.Module):
    def __init__(self, step_size=0.1):
        super(BSplineCurveSampler, self).__init__()
        #self.step = torch.tensor(-2.2).float()  # sigmoid(-2.2) ~= 0.1
        self.step_size = step_size
    
    def forward(self, degree, knots, control_points):
        num_points = 1.0 / self.step_size
        return sample_curve(degree, knots, control_points, num_points)