import numpy as np
class Adaptivity:
    def __init__(self, solver, estimator, tolerance=10.0, max_iterations=20):
        self.solver = solver
        self.estimator = estimator
        self.tolerance = tolerance
        self.max_iterations = max_iterations

        self.history = {
            'iterations': [],
            'num_elements': [],
            'error_norms': [],
            'max_indicators': [],
            'solutions': [],
            'grids': [],
            'indicators': [],
            'real_errors': [],
            'relative_errors': [],       
            'prof_relative_errors': [],   
            'u_norms': [],            
            'convergence_rates': []
        }
    
    def adapt(self, exact_solution=None,exact_solution_deriv=None, verbose=True):
        q = None

        for iteration in range(self.max_iterations):

            q = self.solver.solve()

            error_norm_sq, _ = self.estimator.compute_error_norm_squared(
                self.solver, q
            )
            error_norm = np.sqrt(error_norm_sq)

            indicators = self.estimator.compute_error_indicators(
                self.solver, q
            )
            max_indicator = np.max(indicators)

            u_norm_sq_array = self.estimator.compute_solution_norm_squared(q, self.solver.mesh)
            u_norm = np.sqrt(np.sum(u_norm_sq_array))
            
            standard_rel_error = 0.0
            prof_rel_error = 0.0
            if u_norm > 1e-15:
                standard_rel_error = (error_norm / u_norm) * 100.0
            if (u_norm + error_norm) > 1e-15:
                prof_rel_error = (error_norm / (u_norm + error_norm)) * 100.0

            d_N = 0.0
            if iteration > 0:
                E_prev = self.history['error_norms'][-1]
                N_prev = self.history['num_elements'][-1]
                E_curr = error_norm
                N_curr = self.solver.mesh.N
                
                if N_curr > N_prev and E_curr > 0 and E_prev > 0:
                    d_N = (np.log(E_prev) - np.log(E_curr)) / (np.log(N_curr) - np.log(N_prev))
            
            real_error_sq = 0.0
            for i in range(self.solver.mesh.N):
                h = self.solver.mesh.h[i]
                mu_val = self.solver.mu[i]
                beta_val = self.solver.beta[i]
                sigma_val = self.solver.sigma[i]
                
                x_l = self.solver.mesh.nodes[i]
                x_r = self.solver.mesh.nodes[i+1]
                x_fine = np.linspace(x_l, x_r, 5000) 

                due_dx = exact_solution_deriv(x_fine)
                u_ex = exact_solution(x_fine)
                
                duh_dx = (q[i+1] - q[i]) / h
                u_h = np.interp(x_fine, [x_l, x_r], [q[i], q[i+1]])

                e = u_ex - u_h
                de_dx = due_dx - duh_dx
                
                integrand = mu_val * de_dx**2
                
                real_error_sq += np.trapz(integrand, x_fine)

            real_error = np.sqrt(real_error_sq)
            self.history['iterations'].append(iteration)
            self.history['num_elements'].append(self.solver.mesh.N)
            self.history['error_norms'].append(error_norm)
            self.history['max_indicators'].append(max_indicator)
            self.history['solutions'].append(q.copy())
            self.history['grids'].append(self.solver.mesh.nodes.copy())
            self.history['indicators'].append(indicators.copy())
            
            self.history['u_norms'].append(u_norm)
            self.history['real_errors'].append(real_error)
            self.history['relative_errors'].append(standard_rel_error)
            self.history['prof_relative_errors'].append(prof_rel_error)
            self.history['convergence_rates'].append(d_N)

            if verbose:
                print(
                    f"Ітер {iteration + 1}: "
                    f"N={self.solver.mesh.N}, "
                    f"||ε_h||_V={error_norm:.4f}, "
                    f"max(η)={max_indicator:.2f}%, "
                )

            if max_indicator <= self.tolerance:
                if verbose:
                    print(
                        f"\nЗбіжність досягнуто! "
                        f"Всі елементи мають η ≤ {self.tolerance}%"
                    )
                break

            elements_to_refine = [
                i for i, val in enumerate(indicators)
                if val > self.tolerance
            ]

            if not elements_to_refine:
                if verbose:
                    print(
                        "\nЗбіжність досягнуто "
                        "(немає елементів для розбиття)."
                    )
                break

            self.solver.mesh.refine_elements(elements_to_refine)
            self.solver.update_element_values()

        return q

    def get_history(self):
        return self.history