import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

class FEM_Solver:
    def __init__(self, mesh, mu_func, beta_func, sigma_func, f_func, alpha, u_left=0.0, u_right=0.0):
        self.mesh = mesh
        self.mu_func = mu_func
        self.beta_func = beta_func
        self.sigma_func = sigma_func
        self.f_func = f_func
        self.alpha = alpha
        self.u_left = u_left   
        self.u_right = u_right 
        self.update_element_values()

    def update_element_values(self):
        centers = self.mesh.centers
        self.mu = np.array([self.mu_func(x) for x in centers])
        self.beta = np.array([self.beta_func(x) for x in centers])
        self.sigma = np.array([self.sigma_func(x) for x in centers])
        self.f = np.array([self.f_func(x) for x in centers])

    def compute_local_Pe_Sh(self, elem_idx):
        h = self.mesh.h[elem_idx]
        mu = self.mu[elem_idx]
        beta = self.beta[elem_idx]
        sigma = self.sigma[elem_idx]

        Pe = h * beta / mu
        Sh = h**2 * sigma / mu
        return Pe, Sh

    def assemble_system(self):
        N = self.mesh.N

        A = lil_matrix((N + 1, N + 1))
        b = np.zeros(N + 1)

        A[0, 0] = 1.0
        b[0] = self.u_left 

        for i in range(1, N):
            elem_left = i - 1
            elem_right = i

            h_left = self.mesh.h[elem_left]
            mu_left = self.mu[elem_left]
            f_left = self.f[elem_left]
            Pe_left, Sh_left = self.compute_local_Pe_Sh(elem_left)

            h_right = self.mesh.h[elem_right]
            mu_right = self.mu[elem_right]
            f_right = self.f[elem_right]
            Pe_right, Sh_right = self.compute_local_Pe_Sh(elem_right)

            coeff_im1 = (mu_left / h_left) * (-1 + (1.0/6.0) * (Sh_left - 3.0 * Pe_left))
            A[i, i-1] = coeff_im1

            term_left = (mu_left / h_left) * (1 + (1.0/6.0) * (2.0 * Sh_left + 3.0 * Pe_left))
            term_right = (mu_right / h_right) * (1 + (1.0/6.0) * (2.0 * Sh_right - 3.0 * Pe_right))
            A[i, i] = term_left + term_right

            coeff_ip1 = (mu_right / h_right) * (-1 + (1.0/6.0) * (Sh_right + 3.0 * Pe_right))
            A[i, i+1] = coeff_ip1

            b[i] = 0.5 * (h_left * f_left + h_right * f_right)

        elem_last = N - 1
        h_last = self.mesh.h[elem_last]
        mu_last = self.mu[elem_last]
        f_last = self.f[elem_last]
        Pe_last, Sh_last = self.compute_local_Pe_Sh(elem_last)

        A[N, N-1] = (mu_last / h_last) * (-1 + (1.0/6.0) * (Sh_last - 3.0 * Pe_last))
        
        fem_part = (mu_last / h_last) * (1 + (1.0/6.0) * (2.0 * Sh_last + 3.0 * Pe_last))
        A[N, N] = fem_part + self.alpha

        b[N] = 0.5 * h_last * f_last + self.alpha * self.u_right

        return A.tocsr(), b

    def solve(self):
        self.update_element_values()
        A, b = self.assemble_system()
        q = spsolve(A, b)
        return q

    def evaluate_solution(self, q, x_eval):
        return np.interp(x_eval, self.mesh.nodes, q)