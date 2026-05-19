import numpy as np
class Error_Estimator:
    def __init__(self, estimator_type='quadratic'):
        if estimator_type not in ['quadratic', 'linear']:
            raise ValueError("estimator_type має бути 'quadratic' або 'linear'")

        self.estimator_type = estimator_type

    def compute_q_dot_and_q_bar(self, q, mesh):
        N = mesh.N

        q_dot = np.zeros(N)  
        q_bar = np.zeros(N)  
        for i in range(N):
            h = mesh.h[i] 

            val_left = q[i]
            val_right = q[i+1]

            q_dot[i] = (val_right - val_left) / h
            q_bar[i] = (val_right + val_left) / 2.0
        return q_dot, q_bar

    def compute_mu_dot(self, solver):
        mesh = solver.mesh 
        N = mesh.N
        mu_dot = np.zeros(N)

        for i in range(N):
            h = mesh.h[i]
            x_left = mesh.nodes[i]
            x_right = mesh.nodes[i+1]
            
            mu_left = solver.mu_func(x_left)
            mu_right = solver.mu_func(x_right)
            
            mu_dot[i] = (mu_right - mu_left) / h
        return mu_dot

    def compute_lambda_coefficients(self, solver, q):
        mesh = solver.mesh
        N = mesh.N
        q_dot, q_bar = self.compute_q_dot_and_q_bar(q, mesh)
        lambda_coef = np.zeros(N)

        if self.estimator_type == 'linear':
            mu_dot = self.compute_mu_dot(solver)

        for i in range(N):
            h = mesh.h[i]          
            mu = solver.mu[i]      
            beta = solver.beta[i]  
            sigma = solver.sigma[i]
            f = solver.f[i]        

            Pe, Sh = solver.compute_local_Pe_Sh(i)

            if self.estimator_type == 'quadratic':
                residual = f - beta * q_dot[i] - sigma * q_bar[i]
                numerator = (h**2) * residual
                denominator = mu * (10 + Pe * Sh)
                lambda_coef[i] = (5.0 / 4.0) * numerator / denominator

            else:  
                residual = f - sigma * q_bar[i] - (beta - mu_dot[i]) * q_dot[i]
                numerator = (h**2) * residual
                denominator = mu * (12 + Sh)
                lambda_coef[i] = (3.0 / 2.0) * numerator / denominator
        
        return lambda_coef

    def compute_error_norm_squared(self, solver, q):
        mesh = solver.mesh
        N = mesh.N

        q_dot, q_bar = self.compute_q_dot_and_q_bar(q, mesh)
        
        element_errors = np.zeros(N)

        if self.estimator_type == 'linear':
            mu_dot = self.compute_mu_dot(solver)

        for i in range(N):
            h = mesh.h[i]
            mu = solver.mu[i]
            beta = solver.beta[i]
            sigma = solver.sigma[i]
            f = solver.f[i]
            
            Pe, Sh = solver.compute_local_Pe_Sh(i)
            
            if self.estimator_type == 'quadratic':
                residual = f - beta * q_dot[i] - sigma * q_bar[i]
                numerator = (h**3) * (residual**2)
                denominator = mu * (10 + Pe * Sh)
                element_errors[i] = (5.0 / 6.0) * numerator / denominator
            else:  
                residual = f - (beta + mu_dot[i]) * q_dot[i] - sigma * q_bar[i]
                numerator = (h**3) * (residual**2)
                denominator = mu * (12 + Sh)
                element_errors[i] = (3.0 / 4.0) * numerator / denominator
        error_norm_sq = np.sum(element_errors)
        return error_norm_sq, element_errors

    def compute_solution_norm_squared(self, q, mesh):
        N = mesh.N 
        q_dot, _ = self.compute_q_dot_and_q_bar(q, mesh) 

        element_norms = np.zeros(N)
        for i in range(N):
            h = mesh.h[i]          
            slope = q_dot[i]       
            element_norms[i] = h * (slope**2)
        return element_norms
        
    def compute_solution_norm_squared2(self, q, solver):
        mesh = solver.mesh
        N = mesh.N 
        q_dot, q_bar = self.compute_q_dot_and_q_bar(q, mesh) 

        element_norms = np.zeros(N)
        for i in range(N):
            h = mesh.h[i]          
            slope = q_dot[i]
            avg_q = q_bar[i]
            
            mu = solver.mu[i]
            beta = solver.beta[i]
            sigma = solver.sigma[i]
            
            u_i = q[i]
            u_ip1 = q[i+1]
            
            term_mu = mu * h * (slope**2)
            
            term_beta = beta * h * slope * avg_q
            
            term_sigma = sigma * (h / 3.0) * (u_i**2 + u_i * u_ip1 + u_ip1**2)
            
            element_norms[i] = term_mu + term_beta + term_sigma
            
            if i == N - 1 and hasattr(solver, 'alpha'):
                element_norms[i] += solver.alpha * (u_ip1 ** 2)
                
        return element_norms
    def compute_error_indicators(self, solver, q):
        mesh = solver.mesh
        N = mesh.N 
        _, element_errors = self.compute_error_norm_squared(solver, q)
        element_norms = self.compute_solution_norm_squared(q, mesh)
        total_norm = np.sqrt(np.sum(element_norms + element_errors))
        indicators = np.zeros(N)
        for i in range(N):
            error_val = np.sqrt(element_errors[i])
            numerator = np.sqrt(N) * error_val

            if total_norm > 1e-15: 
                indicators[i] = (numerator / total_norm) * 100.0
            else:
                indicators[i] = 0.0

        return indicators