import numpy as np
import matplotlib.pyplot as plt
import os
class Visualizer:
    def __init__(self, figsize=(12, 6)):
        self.figsize = figsize
        plt.style.use('default')
        
    def save_iteration_plots(self, history, output_folder='iteration_plots'):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        num_steps = len(history['iterations'])

        for i in range(num_steps):
            iter_num = history['iterations'][i] + 1 
            nodes = history['grids'][i]            
            q = history['solutions'][i]            
            indicators = history['indicators'][i]  

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

            x_smooth = np.linspace(nodes[0], nodes[-1], 1000)
            u_smooth = np.interp(x_smooth, nodes, q)
            ax1.plot(x_smooth, u_smooth, 'k-', linewidth=1.5, label="Наближений розв'язок")
            ax1.plot(nodes, q, 'k.', markersize=8) 
            ax1.set_title("Графік наближеного розв'язку")
            ax1.set_xlabel('x')
            ax1.set_ylabel('u(x)')
            ax1.grid(True)
            ax1.legend()

            indices = np.arange(0.5, len(indicators) + 0.5)
            ax2.plot(indices, indicators, 'k.-', linewidth=1, markersize=10)
            ax2.set_title("Розподіл значень індикаторів якості")
            ax2.set_xlabel('Номер елемента')
            ax2.set_ylabel(r'$\eta$ (%)')
            ax2.set_xlim(0, len(indicators) + 1)
            ax2.set_ylim(bottom=0) 
            ax2.grid(True)
            fig.suptitle(f'Ітерація {iter_num}: Кількість елементів N = {len(nodes)-1}', 
                         fontsize=14, fontweight='bold')

            filename = os.path.join(output_folder, f"iteration_{iter_num:02d}.png")
            
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close(fig)

    def plot_summary(self, mesh, q, indicators, history, exact_solution=None):
        fig = plt.figure(figsize=(14, 10))
        iterations = np.array(history['iterations']) + 1
        ax1 = plt.subplot(2, 2, 1)
        x_plot = np.linspace(mesh.nodes[0], mesh.nodes[-1], 1000)
        u_h = np.interp(x_plot, mesh.nodes, q)
        ax1.plot(x_plot, u_h, 'b-', linewidth=2, label="МСЕ розв'язок")
        if exact_solution:
            ax1.plot(x_plot, exact_solution(x_plot), 'g--',linewidth=3, label="Точний")
        
        ax1.set_title(f"Фінальний розв'язок (N={mesh.N})")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2 = plt.subplot(2, 2, 2)
        indices = np.arange(0.5, len(indicators) + 0.5)
        #indices = np.arange(1, len(indicators) + 1)
        ax2.plot(indices, indicators, 'k.-', linewidth=1, markersize=6)
        ax2.set_title('Фінальний розподіл індикаторів якості')
        ax2.set_xlabel('Номер елемента')
        ax2.set_xlim(0, len(indicators) + 1)
        ax2.set_ylim(bottom=0)
        ax2.grid(True, alpha=0.3)

        ax3 = plt.subplot(2, 2, 3)
        ax3.plot(iterations, history['num_elements'], 'bo-', linewidth=2)
        ax3.set_xlabel('Ітерація')
        ax3.set_ylabel('Кількість елементів N')
        ax3.set_title('Динаміка зростання кількості елементів')
        ax3.grid(True, alpha=0.3)

        ax4 = plt.subplot(2, 2, 4)
        ax4.plot(iterations, history['error_norms'], 'ro-', linewidth=2)

        ax4.set_xlabel('Ітерація')
        ax4.set_ylabel('$||\\varepsilon_h||_V$') 
        ax4.set_title('Збіжність норми похибки')
        ax4.grid(True, alpha=0.3)
        plt.tight_layout()

    def plot_error_comparison(self, history_linear, history_quadratic,
                          filename='error_comparison.png'):

        iterations_lin = np.array(history_linear['iterations']) + 1
        iterations_quad = np.array(history_quadratic['iterations']) + 1

        est_lin = np.array(history_linear['error_norms'])
        est_quad = np.array(history_quadratic['error_norms'])

        real_lin = np.array(history_linear['real_errors'])
        real_quad = np.array(history_quadratic['real_errors'])

        plt.figure(figsize=(10, 7))

        plt.plot(iterations_lin, est_lin, 'ro-', linewidth=2, markersize=8,
             label='Лінійний оцінювач')
    
        plt.plot(iterations_quad, est_quad, 'bs-', linewidth=2, markersize=8,
             label='Квадратичний оцінювач')

        #plt.plot(iterations_lin, real_lin, 'r^--', linewidth=2, markersize=8, alpha=0.7,
             #label='Істинна (Лінійний)')

        #plt.plot(iterations_quad, real_quad, 'b*--', linewidth=2, markersize=8, alpha=0.7,
             #label='Істинна (Квадратичний)')

        plt.xlabel('Ітерація', fontsize=12)
        plt.ylabel('Похибка', fontsize=12)
        plt.title('Порівняння оціненої та істинної похибки', fontsize=14, fontweight='bold')

        plt.grid(True, which='both', linestyle='--', alpha=0.5)
        plt.legend(fontsize=10, loc='best')

        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_indicator_comparison(self, history_linear, history_quadratic, filename='indicator_comparison.png'):
        ind_lin = history_linear['indicators'][-1]
        ind_quad = history_quadratic['indicators'][-1]

        plt.figure(figsize=(10, 6))

        indices_lin = np.arange(0.5, len(ind_lin) + 0.5)
        plt.plot(indices_lin, ind_lin, 'ro-', linewidth=1.5, markersize=5, label='Лінійний оцінювач')

        indices_quad = np.arange(0.5, len(ind_quad) + 0.5)
        plt.plot(indices_quad, ind_quad, 'bs--', linewidth=1.5, markersize=5, label='Квадратичний оцінювач')

        plt.title('Порівняння фінального розподілу індикаторів якості')
        plt.xlabel('Номер елемента')
        plt.ylabel(r'$\eta$ (%)')
        plt.xlim(0, max(len(ind_lin), len(ind_quad)) + 1)
        plt.ylim(bottom=0)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
    
        
    def plot_rel_errors_s_and_p(self, history, estimator_name='', filename='rel_errors_comparison.png'):
        elements = np.array(history['num_elements'])
        rel_s = np.array(history['relative_errors'])
        rel_p = np.array(history['prof_relative_errors'])

        plt.figure(figsize=(8,6))

        plt.plot(elements, rel_s, 'ro-', linewidth=2, label='Стандартна')
        plt.plot(elements, rel_p, 'bs--', linewidth=2, label='Уточнена')

        plt.xlabel('Кількість елементів N')
        plt.ylabel('Відносна похибка (%)')
        plt.title(f'Порівняння Стандартної та Уточненої відносних похибок ({estimator_name})')

        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()


    def plot_relative_error(self, history_linear, history_quadratic,
                        filename='relative_error.png'):

        elements_lin = np.array(history_linear['num_elements'])
        elements_quad = np.array(history_quadratic['num_elements'])

        rel_lin = np.array(history_linear['relative_errors'])
        rel_quad = np.array(history_quadratic['relative_errors'])

        plt.figure(figsize=(8,6))

        plt.plot(elements_lin, rel_lin, 'ro-', linewidth=2,
             label='Лінійний оцінювач')

        plt.plot(elements_quad, rel_quad, 'bs-', linewidth=2,
             label='Квадратичний оцінювач')

        plt.xlabel('Кількість елементів N')
        plt.ylabel('Відносна похибка')
        plt.title('Збіжність відносної похибки')

        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()

    def plot_solution_comparison(self, history_linear, history_quadratic, exact_solution=None, filename='solution_comparison.png'):
        nodes_lin = history_linear['grids'][-1]
        q_lin = history_linear['solutions'][-1]

        nodes_quad = history_quadratic['grids'][-1]
        q_quad = history_quadratic['solutions'][-1]

        plt.figure(figsize=(10, 6))

        x_plot_lin = np.linspace(nodes_lin[0], nodes_lin[-1], 1000)
        u_h_lin = np.interp(x_plot_lin, nodes_lin, q_lin)
        plt.plot(x_plot_lin, u_h_lin, 'r-', linewidth=2, label="МСЕ розв'язок (Лінійний)")
        x_plot_quad = np.linspace(nodes_quad[0], nodes_quad[-1], 1000)
        u_h_quad = np.interp(x_plot_quad, nodes_quad, q_quad)
        plt.plot(x_plot_quad, u_h_quad, 'b--', linewidth=2, label="МСЕ розв'язок (Квадратичний)")
        if exact_solution:
         x_exact = np.linspace(min(nodes_lin[0], nodes_quad[0]), max(nodes_lin[-1], nodes_quad[-1]), 1000)
         plt.plot(x_exact, exact_solution(x_exact), 'k:', linewidth=3, label="Точний розв'язок")

        plt.title("Порівняння МСЕ розв'язків")
        plt.xlabel('x')
        plt.ylabel('u(x)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_element_growth_comparison(self, history_linear, history_quadratic, filename='element_growth_comparison.png'):
        iterations_lin = np.array(history_linear['iterations']) + 1
        elements_lin = np.array(history_linear['num_elements'])

        iterations_quad = np.array(history_quadratic['iterations']) + 1
        elements_quad = np.array(history_quadratic['num_elements'])

        plt.figure(figsize=(8, 6))

        plt.plot(iterations_lin, elements_lin, 'ro-', linewidth=2, label='Лінійний оцінювач')
        plt.plot(iterations_quad, elements_quad, 'bs-', linewidth=2, label='Квадратичний оцінювач')

        plt.xlabel('Ітерація')
        plt.ylabel('Кількість елементів N')
        plt.title('Динаміка зростання кількості елементів')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_exact_solution(self, x_min, x_max, exact_solution, title="Точний розв'язок", 
                           figsize=(10, 6), filename=None):
        fig, ax = plt.subplots(figsize=figsize)
        
        x_smooth = np.linspace(x_min, x_max, 1000)
        u_exact = exact_solution(x_smooth)
        
        ax.plot(x_smooth, u_exact, 'g-', linewidth=2.5, label="Точний розв'язок u(x)")
        
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('u(x)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11)
        
        plt.tight_layout()
        
        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close(fig)
        
        return fig, ax
        
    def plot_effectivity_index(self, history_linear, history_quadratic, filename='effectivity_index.png'):
        iterations_lin = np.array(history_linear['iterations']) + 1
        iterations_quad = np.array(history_quadratic['iterations']) + 1

        est_lin = np.array(history_linear['error_norms'])
        est_quad = np.array(history_quadratic['error_norms'])

        real_lin = np.array(history_linear['real_errors'])
        real_quad = np.array(history_quadratic['real_errors'])

        eff_index_lin = est_lin / real_lin
        eff_index_quad = est_quad / real_quad

        plt.figure(figsize=(10, 6))
        plt.plot(iterations_lin, eff_index_lin, 'ro-', linewidth=2, markersize=8, label='Лінійний оцінювач')
        plt.plot(iterations_quad, eff_index_quad, 'bs-', linewidth=2, markersize=8, label='Квадратичний оцінювач')

        plt.axhline(y=1.0, color='k', linestyle='--', linewidth=2, alpha=0.7, label='Ідеал (Індекс = 1.0)')

        plt.xlabel('Ітерація', fontsize=12)
        plt.ylabel('Індекс ефективності', fontsize=12)
        plt.title('Порівняння індексу ефективності оцінювачів', fontsize=14, fontweight='bold')

        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=11)

        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_convergence_rate_comparison(self, history_linear, history_quadratic, filename='convergence_rate_comparison.png'):
        iters_lin = np.array(history_linear['iterations'])[1:] + 1
        rate_lin = np.array(history_linear['convergence_rates'])[1:]

        iters_quad = np.array(history_quadratic['iterations'])[1:] + 1
        rate_quad = np.array(history_quadratic['convergence_rates'])[1:]

        plt.figure(figsize=(10, 6))

        plt.plot(iters_lin, rate_lin, 'ro-', linewidth=2, markersize=8, label='Лінійний оцінювач')
        plt.plot(iters_quad, rate_quad, 'bs-', linewidth=2, markersize=8, label='Квадратичний оцінювач')

        plt.xlabel('Ітерація', fontsize=12)
        plt.ylabel('Порядок збіжності (d_N)', fontsize=12)
        plt.title('Порівняння порядку збіжності апроксимацій (d_N)', fontsize=14, fontweight='bold')
        
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=11)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_figure(self, filename, dpi=300):
        plt.savefig(filename, dpi=dpi, bbox_inches='tight')

    def plot_superimposed_iterations(self, history, filename='superimposed_iterations.png'):
        iters = history['iterations']
        if len(iters) < 2:
            print("Попередження: Недостатньо ітерацій для побудови суміщеного графіка.")
            return

        idx_2 = 1
        nodes_2 = history['grids'][idx_2]
        q_2 = history['solutions'][idx_2]
        idx_final = -1
        nodes_final = history['grids'][idx_final]
        q_final = history['solutions'][idx_final]
        iter_final = iters[idx_final] + 1

        plt.figure(figsize=(10, 6))

        x_plot_2 = np.linspace(nodes_2[0], nodes_2[-1], 1000)
        u_h_2 = np.interp(x_plot_2, nodes_2, q_2)
        plt.plot(x_plot_2, u_h_2, 'r--', linewidth=2, label=f"Ітерація 2 (N={len(nodes_2)-1})")

        x_plot_final = np.linspace(nodes_final[0], nodes_final[-1], 1000)
        u_h_final = np.interp(x_plot_final, nodes_final, q_final)
        plt.plot(x_plot_final, u_h_final, 'b-', linewidth=2, label=f"Фінальна ітерація (N={len(nodes_final)-1})")

        plt.title("Порівняння розв'язків на 2-й та фінальній ітераціях", fontsize=14, fontweight='bold')
        plt.xlabel('x', fontsize=12)
        plt.ylabel('u(x)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=11)
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_combined_linear_superimposed_and_indicators(self, history_linear, history_quadratic, filename='combined_linear_superimposed_indicators.png'):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        iters_lin = history_linear['iterations']
        if len(iters_lin) >= 2:
            idx_2 = 1
            nodes_2 = history_linear['grids'][idx_2]
            q_2 = history_linear['solutions'][idx_2]

            idx_final = -1
            nodes_final = history_linear['grids'][idx_final]
            q_final = history_linear['solutions'][idx_final]

            x_plot_2 = np.linspace(nodes_2[0], nodes_2[-1], 1000)
            u_h_2 = np.interp(x_plot_2, nodes_2, q_2)
            ax1.plot(x_plot_2, u_h_2, 'r--', linewidth=2, label=f"Ітерація 2 (N={len(nodes_2)-1})")

            x_plot_final = np.linspace(nodes_final[0], nodes_final[-1], 1000)
            u_h_final = np.interp(x_plot_final, nodes_final, q_final)
            ax1.plot(x_plot_final, u_h_final, 'b-', linewidth=2, label=f"Фінальна ітерація (N={len(nodes_final)-1})")

            ax1.set_title("Лінійний оцінювач: наближений розв'язок на 2-й та фінальній ітераціях", fontsize=12, fontweight='bold')
            ax1.set_xlabel('x', fontsize=12)
            ax1.set_ylabel('u(x)', fontsize=12)
            ax1.grid(True, linestyle='--', alpha=0.7)
            ax1.legend(fontsize=11)
        else:
            ax1.text(0.5, 0.5, 'Недостатньо ітерацій', horizontalalignment='center', verticalalignment='center')
            ax1.set_title("Лінійний оцінювач: розв'язки", fontsize=12, fontweight='bold')

        ind_lin = history_linear['indicators'][-1]
        ind_quad = history_quadratic['indicators'][-1]

        indices_lin = np.arange(0.5, len(ind_lin) + 0.5)
        ax2.plot(indices_lin, ind_lin, 'ro-', linewidth=1.5, markersize=5, label='Лінійний оцінювач')

        indices_quad = np.arange(0.5, len(ind_quad) + 0.5)
        ax2.plot(indices_quad, ind_quad, 'bs--', linewidth=1.5, markersize=5, label='Квадратичний оцінювач')

        ax2.set_title('Порівняння фінального розподілу індикаторів похибок', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Номер елемента', fontsize=12)
        ax2.set_ylabel(r'$\eta$ (%)', fontsize=12)
        ax2.set_xlim(0, max(len(ind_lin), len(ind_quad)) + 1)
        ax2.set_ylim(bottom=0)
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=11)

        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()