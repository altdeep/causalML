import torch
import torch.distributions as td

from .multivariate import MultivariateDistribution
from deepscm.util import matvec, posdef_inverse


class MultivariateNormal(MultivariateDistribution, td.MultivariateNormal):
    @property
    def num_variables(self):
        return self.event_shape[0]

    @property
    def variable_shapes(self):
        return [1] * self.num_variables

    def _marginalise_single(self, index):
        mx = self.loc[..., index]
        Sxx = self.covariance_matrix[..., index, index]
        return td.Normal(mx, torch.sqrt(Sxx))

    def _marginalise_multi(self, indices):
        indices = torch.as_tensor(indices)
        mx = self.loc[..., indices]  # (..., Dx)
        Sxx = self.covariance_matrix[..., indices.unsqueeze(-1), indices.unsqueeze(-2)]  # (..., Dx, Dx)
        return MultivariateNormal(mx, covariance_matrix=Sxx)

    def _condition(self, y_dims, x_dims, x, squeeze):
        """Conditional distribution of Y|X"""
        x_dims = torch.as_tensor(x_dims)  # .unsqueeze(0)
        y_dims = torch.as_tensor(y_dims)  # .unsqueeze(0)
        x = torch.cat(x, dim=-1)
        # TODO: Correctly handle single dimensions (int)
        m, S = self.loc, self.covariance_matrix
        mx = m[..., x_dims]  # (..., Dx)
        my = m[..., y_dims]  # (..., Dy)
        Sxx = S[..., x_dims.unsqueeze(-1), x_dims.unsqueeze(-2)]  # (..., Dx, Dx)
        Sxy = S[..., x_dims.unsqueeze(-1), y_dims.unsqueeze(-2)]  # (..., Dx, Dy)
        Syy = S[..., y_dims.unsqueeze(-1), y_dims.unsqueeze(-2)]  # (..., Dy, Dy)
        Syx = Sxy.transpose(-2, -1)  # (..., Dy, Dx)
        Syx_iSxx = Syx @ posdef_inverse(Sxx)[0]  # (..., Dy, Dx)
        # if x_dims.dim() > 0:
        #     Syx_iSxx = Syx @ posdef_inverse(Sxx)[0]  # (..., Dy, Dx)
        # else:
        #     Syx_iSxx = Syx / Sxx  # (..., Dy, Dx)
        Syy_x = Syy - Syx_iSxx @ Sxy  # (..., Dy, Dy)
        my_x = my + matvec(Syx_iSxx, x - mx)  # (..., Dy)
        if squeeze:
            return td.Normal(my_x.squeeze(-1), torch.sqrt(Syy_x).reshape(Syy_x.shape[:-2]))
        return MultivariateNormal(my_x, covariance_matrix=Syy_x)


if __name__ == '__main__':
    D = 5
    mean = torch.zeros(D)
    cov = torch.eye(D)
    mvn = MultivariateNormal(mean, cov, var_names='abcdefghij'[:D])
    values = torch.zeros(7, D)
    # print(mvn.marginalise(0))
    # print(mvn.marginalise([0]).batch_shape)
    # print(mvn.condition((1,), (0,), values[:, 0]).batch_shape)
    # # print(mvn.condition((1,), 0, mean[0]))
    # # print(mvn.condition(1, (0,), mean[0]))
    # print(mvn.condition([1, 2], [0], values[:, 0]))
    # print(mvn.condition([1, 2], [0, 3], values[:, [0, 1]]))
    # print(mvn.condition([1], [0, 3], values[:, [0, 1]]))

    for x_dims in [(0,), [0], [0, 3]]:
        for y_dims in [(1,), [1], [1, 2]]:
            result = mvn._condition(y_dims, x_dims, [values[:, x_dims]], squeeze=False)
            print(result)
            assert result.variable_names == mvn.variable_names
            assert result.batch_shape == values.shape[:-1]
            assert result.event_shape == (len(y_dims),)
