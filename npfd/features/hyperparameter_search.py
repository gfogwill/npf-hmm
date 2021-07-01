import skopt
from skopt.plots import plot_convergence
from npfd.features.scripts import train_evaluate
from npfd import data
import matplotlib.pyplot as plt


SPACE = [
    skopt.space.Real(1e-3, 2, name='variance_floor', prior='log-uniform'),
    skopt.space.Real(1e-3, 1, name='minimum_variance', prior='log-uniform'),
    skopt.space.Real(0.0, 8.0, name='word_insertion_penalty', prior='uniform'),
    skopt.space.Real(0.0, 6.0, name='grammar_scale_factor', prior='uniform')
]

X_train, X_val, y_train, y_val = data.dataset.make_dataset('db3', clean_interim_dir=True)
X_adapt, X_test, y_adapt, y_test = data.dataset.make_dataset('db3', test_size=0.3, clean_interim_dir=False)


@skopt.utils.use_named_args(SPACE)
def objective(**params):
    return -1.0 * train_evaluate(params, X_train, X_val, y_train, y_val, X_adapt, X_test, y_adapt, y_test)


results = skopt.gp_minimize(objective, SPACE, n_calls=500)
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
