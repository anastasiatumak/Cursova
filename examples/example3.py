import sys
import os
import io
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import matplotlib.pyplot as plt
from src import Mesh, FEM_Solver, Error_Estimator, Adaptivity, Visualizer

mu = 1.0
beta = 100.0   
sigma = 100.0
alpha = 1000.0
u_left = 0.0
u_right = 0.0

mu_func = lambda x: mu
beta_func = lambda x: beta
sigma_func = lambda x: sigma

def f_func(x):
    return 500.0 * np.exp(-200.0 * x**2)

def exact_solution(x):
    return np.zeros_like(x)

def exact_solution_deriv(x):
    return np.zeros_like(x)

def run_example(estimator_type='quadratic', N0=10, tolerance=10.0, save_folder=None):
    print("="*85)
    print(f"ЗАПУСК: {estimator_type.upper()} ОЦІНЮВАЧ")
    print("="*85)
    print(f"  μ = {mu}, β = {beta}, σ = {sigma}")
    print(f"  f(x) = 500*exp(-200x^2), Область: [-1, 1]")
    print(f"  Початкова сітка: N₀ = {N0}, Допустима похибка: {tolerance}%")

    mesh = Mesh(N0=N0)
    mesh.nodes = np.linspace(-1, 1, N0 + 1)
    mesh.update_elements()
    
    solver = FEM_Solver(
        mesh=mesh, mu_func=mu_func, beta_func=beta_func, 
        sigma_func=sigma_func, f_func=f_func, alpha=alpha, u_left=u_left, u_right=u_right
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
    plt.suptitle(f'Приклад 3: {estimator_type.capitalize()} Estimator', fontsize=12)

    main_filename = f"example3_{estimator_type}.png"
    visualizer.save_figure(main_filename)

    history = adaptivity.get_history()
    filename = f"example3_{estimator_type}.txt"
    
    with open(filename, "w", encoding="utf-8") as f_out:
        f_out.write("="*110 + "\n")
        f_out.write(f"ЗАПУСК: {estimator_type.upper()} ОЦІНЮВАЧ\n")
        f_out.write("="*110 + "\n")
        f_out.write(f"  μ = {mu}, β = {beta}, σ = {sigma}\n")
        f_out.write(f"  f(x) = 500*exp(-200x^2), Область: [-1, 1]\n")
        f_out.write(f"  Початкова сітка: N₀ = {N0}, Допустима похибка: {tolerance}%\n\n")
        
        f_out.write("="*110 + "\n")
        f_out.write(f"| {'Ітер.':<5} | {'Елем.':<5} | {'||ε_h||_V':<9} | {'max(η),%':<8} | {'||u_h||_V':<9} | {'Відн(С),%':<9} | {'Відн(П),%':<9} | {'Безладність':<10}|\n")
        f_out.write("="*110 + "\n")
        
        for i, it in enumerate(history['iterations']):
            elems = history['num_elements'][i]
            err_norm = history['error_norms'][i]
            max_ind = history['max_indicators'][i]
            u_n = history['u_norms'][i]
            rel_s = history['relative_errors'][i]
            rel_p = history['prof_relative_errors'][i]
            
            nodes = history['grids'][i]
            h_vals = np.diff(nodes)
            irregularity = np.max(h_vals) / np.min(h_vals)
            
            f_out.write(f"| {it + 1:<5} | {elems:<5} | {err_norm:<9.4f} | {max_ind:<8.2f} | {u_n:<9.4f} | {rel_s:<9.2f} | {rel_p:<9.2f} | {irregularity:<8.2f}|\n")
        f_out.write("="*110 + "\n\n")
        
        if history['max_indicators'][-1] <= tolerance:
            f_out.write(f"Збіжність досягнуто! Всі елементи мають η ≤ {tolerance}%\n\n")

    return solver, q, history

if __name__ == "__main__":
    solver_quad, q_quad, history_quad = run_example(
        estimator_type='quadratic',
        N0=10, 
        save_folder='plots3_quadratic' 
    )
    solver_lin, q_lin, history_lin = run_example(
        estimator_type='linear',
        N0=10, 
        save_folder='plots3_linear'   
    )
    visualizer = Visualizer()
    visualizer.plot_error_comparison(history_lin, history_quad, filename='error_norm_comparison3.png')
    visualizer.plot_indicator_comparison(history_lin, history_quad, filename='indicator_comparison3.png')
    visualizer.plot_solution_comparison(history_lin, history_quad, exact_solution=exact_solution, filename='solution_comparison3.png')
    visualizer.plot_element_growth_comparison(history_lin, history_quad, filename='element_growth_comparison3.png')
    visualizer.plot_rel_errors_s_and_p(history_lin, estimator_name='Лінійний', filename='rel_errors_s_p_linear3.png')
    visualizer.plot_rel_errors_s_and_p(history_quad, estimator_name='Квадратичний', filename='rel_errors_s_p_quadratic3.png')
    visualizer.plot_superimposed_iterations(history_lin, filename='superimposed_iterations_linear_3.png')
    visualizer.plot_superimposed_iterations(history_quad, filename='superimposed_iterations_quadratic_3.png')
    
    visualizer.plot_combined_linear_superimposed_and_indicators(
        history_lin, history_quad, 
        filename='combined_linear_superimposed_and_indicators_3.png'
    )
    plt.show()
