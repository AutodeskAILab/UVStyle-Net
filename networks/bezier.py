import torch
import torch.nn as nn


def bernstein_cubic(t):
    """
    Computes the cubic Bernstein basis functions at t [0.0, 1.0]
    :return Values of the 4 cubic Bernstein basis functions
    """
    return torch.stack((torch.pow(1.0 - t, 3), 3.0 * t * torch.pow(1 - t, 2), 3 * t * t * (1 - t), torch.pow(t, 3)), dim=0)


def sample_curve(cp, num_points=10):
    """
    Evaluates a set of samples on the cubic Bezier curve
    :param cp Batch of control points (N x dim x 4 ) of any dimension dim
    """
    device = cp.device
    batch_size = cp.size(0)
    t = torch.linspace(0.0, 1.0, num_points).to(device)
    B = bernstein_cubic(t).unsqueeze(0).repeat(batch_size, 1, 1)
    return torch.bmm(cp, B)


class CubicBezierSampler(nn.Module):
    """
    Samples the given cubic Bezier curve at uniformly spaced parameters
    """
    def __init__(self, num_points=10):
        """
        :param num_points Number of points to sample along the curve
        """
        super(CubicBezierSampler, self).__init__()
        self.num_points = num_points
    
    def forward(self, cp):
        """
        Forward pass
        :param cp Control points of size N x dim x 4
        """
        return sample_curve(cp, self.num_points)


def sample_surface(cp, num_points_u=10, num_points_v=10):
    """
    TODO(pradeep): function must be fixed to work with minibatches
    Evaluates a set of samples on the bicubic Bezier surface
    :param cp Batch of 3D control points (N x 4 x 4 x 3 )
    """
    u = torch.linspace(0.0, 1.0, num_points_u)
    v = torch.linspace(0.0, 1.0, num_points_v)
    Bu = bernstein_cubic(u)
    Bv = bernstein_cubic(v)
    points = torch.zeros((cp.size(0), num_points_u, num_points_v, 3), dtype=torch.float32)
    for i in range(4):
        for j in range(4):
            points += Bu[i] * Bv[j] * cp[..., i, j, :]
    return points


if __name__ == "__main__":
    # Test curve sampling
    cp = torch.tensor([[0, 0], [0, 1], [1, 1], [1, 0]]).float().transpose(0, 1)
    cp = cp.cuda()
    cp = cp.unsqueeze(0)
    pts = sample_curve(cp)
    #print(pts)
    layer = CubicBezierSampler(10)
    #print(layer(cp))
