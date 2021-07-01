import skopt
from skopt.plots import plot_convergence
from npfd.features.scripts import train_evaluate
from npfd import data
import matplotlib.pyplot as plt

space = [
    skopt.space.Real(1e-6, 0.1, name='variance_floor', prior='log-uniform'),
    skopt.space.Real(1e-6, 0.1, name='minimum_variance', prior='log-uniform'),
    skopt.space.Real(0.0, 50.0, name='word_insertion_penalty', prior='uniform'),
    skopt.space.Real(0.0, 50.0, name='grammar_scale_factor', prior='uniform'),
    skopt.space.Integer(2, 7, name='gaussian_duplication_times')
]

X_train, X_val, y_train, y_val = data.dataset.make_dataset('db1', clean_interim_dir=True)
X_adapt, X_test, y_adapt, y_test = data.dataset.make_dataset('db3', test_size=0.4, clean_interim_dir=False)


@skopt.utils.use_named_args(space)
def objective(**params):
    return -1.0 * train_evaluate(X_train, X_val, y_train, y_val, X_adapt, X_test, y_adapt, y_test, **params)


results = skopt.gp_minimize(objective, space, n_calls=500)

best_auc = -1.0 * results.fun
best_params = results.x

print("""Best parameters:
- variance_floor=%.6f
- minimum_variance=%.6f
- word_insertion_penalty=%.2f
- grammar_scale_factor=%.2f
- gaussian_duplication_times=%d""" % (results.x[0], results.x[1],
                                      results.x[2], results.x[3],
                                      results.x[4]))

plot_convergence(results)
plt.show()

skopt.plots.plot_evaluations(results)
plt.show()

skopt.plots.plot_objective(results)
plt.show()

print('best result: ', best_auc)
print('best parameters: ', best_params)
