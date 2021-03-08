import skopt
from skopt.plots import plot_convergence
from scripts import train_evaluate
from npfd import data
import matplotlib.pyplot as plt

SPACE = [
    skopt.space.Real(1e-5, 1e-1, name='variance_floor', prior='log-uniform'),
    skopt.space.Real(1e-8, 1e-1, name='minimum_variance', prior='log-uniform'),
    skopt.space.Real(0.0, 8.0, name='word_insertion_penalty', prior='uniform'),
    skopt.space.Real(0.0, 6.0, name='grammar_scale_factor', prior='uniform')
]


@skopt.utils.use_named_args(SPACE)
def objective(**params):
    return -1.0 * train_evaluate(params)


results = skopt.gp_minimize(objective, SPACE, n_calls=200)
# results = skopt.forest_minimize(objective, SPACE, n_calls=5, n_random_starts=3)
best_auc = -1.0 * results.fun
best_params = results.x

plot_convergence(results)
plt.show()

skopt.plots.plot_evaluations(results)
plt.show()

skopt.plots.plot_objective(results)
plt.show()

print('best result: ', best_auc)
print('best parameters: ', best_params)
