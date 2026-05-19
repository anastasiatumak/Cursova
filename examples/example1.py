import sys
import os
import io
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from src import Mesh, FEM_Solver, Error_Estimator, Adaptivity, Visualizer

mu = 1.0 
sigma = 2.0
alpha = 10.0**8

u_left = 0.0
u_right = 0.0

mu_func = lambda x: mu
beta_func = lambda x: x
sigma_func = lambda x: sigma
    
def exact_solution(x):
    return 7.0 * x * np.exp(-50.0 * x**2)

x_sym = sp.Symbol('x')
w_sym = 7.0 * x_sym * sp.exp(-50.0 * x_sym**2)
w_prime_sym = sp.diff(w_sym, x_sym)
w_double_prime_sym = sp.diff(w_prime_sym, x_sym)

beta_sym = x_sym

f_sym = sp.simplify(-w_double_prime_sym * mu + beta_sym * w_prime_sym + sigma * w_sym)

exact_solution_auto = sp.lambdify(x_sym, w_sym, modules=['numpy'])
exact_solution_deriv_auto = sp.lambdify(x_sym, w_prime_sym, modules=['numpy'])
f_func_auto = sp.lambdify(x_sym, sp.simplify(-w_double_prime_sym * mu + x_sym * w_prime_sym + sigma * w_sym), modules=['numpy'])

def f_func(x):
    return f_func_auto(x)

def exact_solution_deriv(x):
    return exact_solution_deriv_auto(x)

def run_example(estimator_type='quadratic', N0=10, tolerance=10.0, save_folder=None):
    print("="*85)
    print(f"ЗАПУСК: {estimator_type.upper()} ОЦІНЮВАЧ")
    print("="*85)
    print(f"  μ(x) = 1, β(x) = x, σ(x) = 2")
    print(f"  f(x) = {f_sym}, Область: [-1, 1]")
    print(f"  Початкова сітка: N₀ = {N0}, Допустима похибка: {tolerance}%")

    mesh = Mesh(N0=N0)
    mesh.nodes = np.linspace(-1, 1, N0 + 1)
    mesh.update_elements()
    
    
    solver = FEM_Solver(
        mesh=mesh, mu_func=mu_func, beta_func=beta_func, 
        sigma_func=sigma_func, f_func=f_func, alpha=alpha, 
        u_left=u_left, u_right=u_right
    )
    
    estimator = Error_Estimator(estimator_type=estimator_type)
    
    adaptivity = Adaptivity(
        solver=solver, estimator=estimator, 
        tolerance=tolerance, max_iterations=50
    )

    q = adaptivity.adapt(exact_solution=exact_solution, exact_solution_deriv=exact_solution_deriv, verbose=True)
    visualizer = Visualizer()
    
    if save_folder:
        visualizer.save_iteration_plots(adaptivity.get_history(), output_folder=save_folder)

    final_indicators = estimator.compute_error_indicators(solver, q)
    visualizer.plot_summary(
        mesh=solver.mesh, q=q, indicators=final_indicators, 
        history=adaptivity.get_history(), exact_solution=exact_solution
    )
    plt.suptitle(f'Приклад 1: {estimator_type.capitalize()} Estimator', fontsize=12)

    main_filename = f"example1_{estimator_type}.png"
    visualizer.save_figure(main_filename)

    x_test = np.linspace(-1, 1, 1000)          
    u_h = solver.evaluate_solution(q, x_test) 

    history = adaptivity.get_history()
    filename = f"example1_{estimator_type}.txt"
    
    with open(filename, "w", encoding="utf-8") as f_out:
        f_out.write("="*110 + "\n")
        f_out.write(f"ЗАПУСК: {estimator_type.upper()} ОЦІНЮВАЧ\n")
        f_out.write("="*110 + "\n")
        f_out.write(f"  μ(x) = 1, β(x) = x, σ(x) = 2\n")
        f_out.write(f"  f(x) = {f_sym}, Область: [-1, 1]\n")
        f_out.write(f"  Початкова сітка: N₀ = {N0}, Допустима похибка: {tolerance}%\n\n")
        f_out.write(f"  α = {alpha}, u(0) = {u_left}, u(1) = {u_right}\n")
        f_out.write("="*110 + "\n")
        f_out.write(f"| {'Ітер.':<5} | {'Елем.':<5} | {'||ε_h||_V':<9} | {'max(η),%':<8} | {'||e_h||_V':<10} | {'Індекс еф':<10} | {'Відн(С),%':<9} | {'Відн(у),%':<9} | {'d_N':<9} |\n")
        f_out.write("="*110 + "\n")
        
        for i, it in enumerate(history['iterations']):
            elems = history['num_elements'][i]
            err_norm = history['error_norms'][i]
            max_ind = history['max_indicators'][i]
            r_err = history['real_errors'][i]
            rel_s = history['relative_errors'][i]
            rel_p = history['prof_relative_errors'][i]
            d_N = history['convergence_rates'][i]
            d_N_str = "-" if i == 0 else f"{d_N:.4f}"
            eff_index = err_norm / r_err if r_err > 1e-15 else 0.0

            f_out.write(f"| {it + 1:<5} | {elems:<5} | {err_norm:<9.4f} | {max_ind:<8.2f} | {r_err:<10.4f} | {eff_index:<10.4f} | {rel_s:<9.2f} | {rel_p:<9.2f} | {d_N_str:<9} |\n")
        f_out.write("="*110 + "\n\n")
        
        if history['max_indicators'][-1] <= tolerance:
            f_out.write(f"Збіжність досягнуто! Всі елементи мають η ≤ {tolerance}%\n\n")

    return solver, q, history

if __name__ == "__main__":
    solver_quad, q_quad, history_quad = run_example(
        estimator_type='quadratic',
        N0=10, 
        save_folder='plots1_quadratic' 
    )
    solver_lin, q_lin, history_lin = run_example(
        estimator_type='linear',
        N0=10, 
        save_folder='plots1_linear'   
    )
    visualizer = Visualizer()
    visualizer.plot_error_comparison(history_lin, history_quad, filename='error_norm_comparison_1.png')
    visualizer.plot_rel_errors_s_and_p(history_lin, estimator_name='Лінійний', filename='rel_errors_s_p_linear1.png')
    visualizer.plot_rel_errors_s_and_p(history_quad, estimator_name='Квадратичний', filename='rel_errors_s_p_quadratic1.png')
    visualizer.plot_exact_solution(-1, 1, exact_solution, title=f"Точний розв'язок", filename=f'exact_solution_1.png')
    visualizer.plot_element_growth_comparison(history_lin, history_quad, filename='element_growth_comparison1.png')
    visualizer.plot_solution_comparison(history_lin, history_quad, exact_solution=exact_solution, filename='solution_comparison1.png')
    visualizer.plot_indicator_comparison(history_lin, history_quad, filename='indicator_comparison1.png')
    visualizer.plot_effectivity_index(history_lin, history_quad, filename='effectivity_index_1.png')
    visualizer.plot_convergence_rate_comparison(history_lin, history_quad, filename='convergence_rate_comparison_1.png')
    plt.show()