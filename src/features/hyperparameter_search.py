import skopt
from skopt.plots import plot_convergence
from src.features.scripts import train_evaluate_with_adaptation, train_evaluate
from src import data
import matplotlib.pyplot as plt

vf = [0, 10]
mv = [0, 10]
wip = [0.0, 5]
gsf = [15, 40]
gdt = [4, 7]

space = [
    skopt.space.Real(vf[0], vf[1], name='variance_floor', prior='uniform'),
    skopt.space.Real(mv[0], mv[1], name='minimum_variance', prior='uniform'),
    # skopt.space.Real(wip[0], wip[1], name='word_insertion_penalty', prior='uniform'),
    skopt.space.Real(gsf[0], gsf[1], name='grammar_scale_factor', prior='uniform')
    # skopt.space.Integer(gdt[0], gdt[1], name='gaussian_duplication_times')
]

X_train, X_val, y_train, y_val = data.dataset.make_dataset('db3', clean_interim_dir=True)
X_adapt, X_test, y_adapt, y_test = data.dataset.make_dataset('db3', test_size=0.4, clean_interim_dir=False)


@skopt.utils.use_named_args(space)
def objective(**params):
    return -1.0 * train_evaluate_with_adaptation(X_train, X_val, y_train, y_val, X_adapt, X_test, y_adapt, y_test,
                                                 **params)


results = skopt.gp_minimize(objective, space, n_calls=40)

best_auc = -1.0 * results.fun
best_params = results.x

print('best result: ', best_auc)
print('best parameters: ', best_params)

plot_convergence(results)
plt.show()

skopt.plots.plot_evaluations(results)
plt.show()

skopt.plots.plot_objective(results)
plt.show()

print("""Best parameters:
- variance_floor=%.6f
- minimum_variance=%.6f
- word_insertion_penalty=%.2f
- grammar_scale_factor=%.2f
- gaussian_duplication_times=%d""" % (results.x[0], results.x[1],
                                      results.x[2], results.x[3],
                                      results.x[4]))


