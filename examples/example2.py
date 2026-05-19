import sys
import os
import io
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import matplotlib.pyplot as plt
from src import Mesh, FEM_Solver, Error_Estimator, Adaptivity, Visualizer

epsilon = 0.1  

mu = epsilon    
beta = -1.0      
sigma = 0.0     
alpha = 1000.0 
u_left = 0.0    
u_right = 1.0    

mu_func = lambda x: mu
beta_func = lambda x: beta
sigma_func = lambda x: sigma

def f_func(x):
    return -1.0 - 2.0 * x

def exact_solution(x):
    term1 = x * (x + 1.0 - 2.0 * epsilon)
    
    numerator = (2.0 * epsilon - 1.0) * (1.0 - np.exp(-x / epsilon))
    denominator = 1.0 - np.exp(-1.0 / epsilon)
    term2 = numerator / denominator
    
    return term1 + term2
def exact_solution_deriv(x):
    
    term1 = 2.0 * x + 1.0 - 2.0 * epsilon
    numerator = (2.0 * epsilon - 1.0) * np.exp(-x / epsilon)
    denominator = epsilon * (1.0 - np.exp(-1.0 / epsilon))
    term2 = numerator / denominator
    
    return term1 + term2

def run_example(estimator_type='quadratic', N0=4, tolerance=1.0, save_folder=None):
    print("="*70)
    print(f"ЗАПУСК: {estimator_type.upper()} ОЦІНЮВАЧ (ПРИКЛАД 6.2)")
    print("="*70)
    print(f"\nПараметри:")
    print(f"  ε (μ) = {mu}, β = {beta}, σ = {sigma}")
    print(f"  f(x) = 1 + 2x")
    print(f"  α = {alpha}, u(0) = {u_left}, u(1) = {u_right}")
    print(f"  Початкова сітка: N₀ = {N0}, Допустима похибка: {tolerance}%")
    
    mesh = Mesh(N0=N0)

    solver = FEM_Solver(
        mesh=mesh,
        mu_func=mu_func,
        beta_func=beta_func,
        sigma_func=sigma_func,
        f_func=f_func,
        alpha=alpha,
        u_left=u_left,   
        u_right=u_right   
    )

    estimator = Error_Estimator(estimator_type=estimator_type)

    adaptivity = Adaptivity(
        solver=solver,
        estimator=estimator,
        tolerance=tolerance,
        max_iterations=50
    )

    q = adaptivity.adapt(exact_solution=exact_solution, exact_solution_deriv=exact_solution_deriv, verbose=True)

    visualizer = Visualizer(figsize=(14, 10))

    if save_folder:
        visualizer.save_iteration_plots(adaptivity.get_history(), output_folder=save_folder)

    final_indicators = estimator.compute_error_indicators(solver, q)
    visualizer.plot_summary(
        mesh=solver.mesh,
        q=q,
        indicators=final_indicators,
        history=adaptivity.get_history(),
        exact_solution=exact_solution
    )

    plt.suptitle(f'Приклад 6.2 (ε={epsilon}): {estimator_type.capitalize()} ОЦІНЮВАЧ',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()

    main_filename = f'example_2_{estimator_type}.png'
    visualizer.save_figure(main_filename, dpi=300)

    history = adaptivity.get_history()
    filename = f"example_2_{estimator_type}.txt"
    with open(filename, "w", encoding="utf-8") as f_out:
        f_out.write("="*61 + "\n")
        f_out.write(f"ЗАПУСК: {estimator_type.upper()} ОЦІНЮВАЧ (ПРИКЛАД 6.2)\n")
        f_out.write("="*61 + "\n")
        f_out.write(f"Параметри:\n")
        f_out.write(f"  ε (μ) = {mu}, β = {beta}, σ = {sigma}\n")
        f_out.write(f"  f(x) = 1 + 2x\n")
        f_out.write(f"  α = {alpha}, u(0) = {u_left}, u(1) = {u_right}\n")
        f_out.write(f"  Початкова сітка: N₀ = {N0}, Допустима похибка: {tolerance}%\n\n")
        
        f_out.write("="*130 + "\n")
        f_out.write(f"| {'Ітерація':<8} | {'Елементів':<9} | {'||ε_h||_V':<10} | {'max(η),%':<9} | {'||e_h||_V':<10} | {'Індекс еф':<10} | {'Станд.,%':<13} | {'Уточн.,%':<13} | {'||u_h||_V':<13} | {'d_N':<6} |\n")
        f_out.write("="*130 + "\n")
        for i, it in enumerate(history['iterations']):
            elems = history['num_elements'][i]
            err_norm = history['error_norms'][i]
            max_ind = history['max_indicators'][i]
            r_err = history['real_errors'][i]
            rel_s = history['relative_errors'][i]
            rel_p = history['prof_relative_errors'][i]
            u_n = history['u_norms'][i]
            d_N = history['convergence_rates'][i]
            eff_index = err_norm / r_err if r_err > 1e-15 else 0.0
            
            f_out.write(f"| {it + 1:<8} | {elems:<9} | {err_norm:<10.4f} | {max_ind:<9.2f} | {r_err:<10.4f} | {eff_index:<10.4f} | {rel_s:<13.2f} | {rel_p:<13.2f} | {u_n:<6.4f} | {d_N:<6.2f} |\n")
        f_out.write("="*130 + "\n\n")
        
        if history['max_indicators'][-1] <= tolerance:
            f_out.write(f"Збіжність досягнуто! Всі елементи мають η ≤ {tolerance}%\n\n")
        

    return solver, q, history

if __name__ == "__main__":
    solver_quad, q_quad, history_quad = run_example(
        estimator_type='quadratic',
        save_folder='plots2_quadratic'  
    )
    solver_lin, q_lin, history_lin = run_example(
        estimator_type='linear',
        save_folder='plots2_linear'    
    )
    
    visualizer = Visualizer()
    visualizer.plot_error_comparison(history_lin, history_quad, filename='error_norm_comparison_2.png')
    visualizer.plot_rel_errors_s_and_p(history_lin, estimator_name='Лінійний', filename='rel_errors_s_p_linear2.png')
    visualizer.plot_rel_errors_s_and_p(history_quad, estimator_name='Квадратичний', filename='rel_errors_s_p_quadratic2.png')
    visualizer.plot_exact_solution(0, 1, exact_solution, title=f"Точний розв'язок (ε={epsilon})", filename=f'exact_solution_2.png')
    visualizer.plot_element_growth_comparison(history_lin, history_quad, filename='element_growth_comparison2.png')
    visualizer.plot_solution_comparison(history_lin, history_quad, exact_solution=exact_solution, filename='solution_comparison2.png')
    visualizer.plot_indicator_comparison(history_lin, history_quad, filename='indicator_comparison2.png')
    visualizer.plot_effectivity_index(history_lin, history_quad, filename='effectivity_index_2.png')
    visualizer.plot_convergence_rate_comparison(history_lin, history_quad, filename='convergence_rate_comparison_2.png')
    plt.show()
