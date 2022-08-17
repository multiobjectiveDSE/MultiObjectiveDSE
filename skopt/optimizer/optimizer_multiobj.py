import copy
import csv
import socket
import sys
import warnings
from math import log
from numbers import Number
import numpy as np
import pandas
import torch
import random
from copy import deepcopy

from numpy import VisibleDeprecationWarning
from scipy import stats

# from botorch.acquisition.multi_objective import qNoisyExpectedHypervolumeImprovement
from botorch.models import SingleTaskGP
from scipy.optimize import fmin_l_bfgs_b

from sklearn.base import clone
from sklearn.base import is_regressor
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning
from sklearn.multioutput import MultiOutputRegressor
from sklearn.utils import check_random_state
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

from botorch_model_wrapper import botorch_model_wrapper
from config import mape_line_analysis, case_name, N_SAMPLES_INIT, N_SAMPLES_ALL
from ..acquisition import _gaussian_acquisition
from ..acquisition import gaussian_acquisition_1D
from ..learning import GaussianProcessRegressor
from ..space import Categorical
from ..space import Space
from ..utils import check_x_in_space
from ..utils import cook_estimator
from ..utils import create_result, create_result_multiobj
from ..utils import has_gradients
from ..utils import is_listlike
from ..utils import is_2Dlistlike
from ..utils import normalize_dimensions
from ..utils import cook_initial_point_generator

from botorch.utils.multi_objective.box_decompositions.dominated import DominatedPartitioning

from get_real_pareto_frontier import is_pareto_efficient_dumb, scale_for_hv
from simulation_metrics import evaluate_ADRS, CPI_metric, power_metric, get_orthogonal_array, evaluate_non_uniformity, \
    distance_f, get_di
from config import plot_pareto

warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=VisibleDeprecationWarning)


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


import matplotlib.pyplot as plt
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.size'] = '10'
plt.rcParams['font.weight'] = 'bold'
font = {'family': 'sans-serif',
        'weight': 'bold',
        'size': 100,
        }

class OptimizerMultiobj(object):
    """Run bayesian optimisation loop.

    An `Optimizer` represents the steps of a bayesian optimisation loop. To
    use it you need to provide your own loop mechanism. The various
    optimisers provided by `skopt` use this class under the hood.

    Use this class directly if you want to control the iterations of your
    bayesian optimisation loop.

    Parameters
    ----------
    dimensions : list, shape (n_dims,)
        List of search space dimensions.
        Each search dimension can be defined either as

        - a `(lower_bound, upper_bound)` tuple (for `Real` or `Integer`
          dimensions),
        - a `(lower_bound, upper_bound, "prior")` tuple (for `Real`
          dimensions),
        - as a list of categories (for `Categorical` dimensions), or
        - an instance of a `Dimension` object (`Real`, `Integer` or
          `Categorical`).

    base_estimator : `"GP"`, `"RF"`, `"ET"`, `"GBRT"` or sklearn regressor, \
            default: `"GP"`
        Should inherit from :obj:`sklearn.base.RegressorMixin`.
        In addition the `predict` method, should have an optional `return_std`
        argument, which returns `std(Y | x)` along with `E[Y | x]`.
        If base_estimator is one of ["GP", "RF", "ET", "GBRT"], a default
        surrogate model of the corresponding type is used corresponding to what
        is used in the minimize functions.

    n_random_starts : int, default: 10
        .. deprecated:: 0.6
            use `n_initial_points` instead.

    n_initial_points : int, default: 10
        Number of evaluations of `func` with initialization points
        before approximating it with `base_estimator`. Initial point
        generator can be changed by setting `initial_point_generator`.

    initial_point_generator : str, InitialPointGenerator instance, \
            default: `"random"`
        Sets a initial points generator. Can be either

        - `"random"` for uniform random numbers,
        - `"sobol"` for a Sobol' sequence,
        - `"halton"` for a Halton sequence,
        - `"hammersly"` for a Hammersly sequence,
        - `"lhs"` for a latin hypercube sequence,
        - `"grid"` for a uniform grid sequence

    acq_func : string, default: `"gp_hedge"`
        Function to minimize over the posterior distribution. Can be either

        - `"LCB"` for lower confidence bound.
        - `"EI"` for negative expected improvement.
        - `"PI"` for negative probability of improvement.
        - `"gp_hedge"` Probabilistically choose one of the above three
          acquisition functions at every iteration.

          - The gains `g_i` are initialized to zero.
          - At every iteration,

            - Each acquisition function is optimised independently to
              propose an candidate point `X_i`.
            - Out of all these candidate points, the next point `X_best` is
              chosen by :math:`softmax(\\eta g_i)`
            - After fitting the surrogate model with `(X_best, y_best)`,
              the gains are updated such that :math:`g_i -= \\mu(X_i)`

        - `"EIps"` for negated expected improvement per second to take into
          account the function compute time. Then, the objective function is
          assumed to return two values, the first being the objective value and
          the second being the time taken in seconds.
        - `"PIps"` for negated probability of improvement per second. The
          return type of the objective function is assumed to be similar to
          that of `"EIps"`

    acq_optimizer : string, `"sampling"` or `"lbfgs"`, default: `"auto"`
        Method to minimize the acquisition function. The fit model
        is updated with the optimal value obtained by optimizing `acq_func`
        with `acq_optimizer`.

        - If set to `"auto"`, then `acq_optimizer` is configured on the
          basis of the base_estimator and the space searched over.
          If the space is Categorical or if the estimator provided based on
          tree-models then this is set to be `"sampling"`.
        - If set to `"sampling"`, then `acq_func` is optimized by computing
          `acq_func` at `n_points` randomly sampled points.
        - If set to `"lbfgs"`, then `acq_func` is optimized by

          - Sampling `n_restarts_optimizer` points randomly.
          - `"lbfgs"` is run for 20 iterations with these points as initial
            points to find local minima.
          - The optimal of these local minima is used to update the prior.

    random_state : int, RandomState instance, or None (default)
        Set random state to something other than None for reproducible
        results.

    n_jobs : int, default: 1
        The number of jobs to run in parallel in the base_estimator,
        if the base_estimator supports n_jobs as parameter and
        base_estimator was given as string.
        If -1, then the number of jobs is set to the number of cores.

    acq_func_kwargs : dict
        Additional arguments to be passed to the acquisition function.

    acq_optimizer_kwargs : dict
        Additional arguments to be passed to the acquisition optimizer.

    model_queue_size : int or None, default: None
        Keeps list of models only as long as the argument given. In the
        case of None, the list has no capped length.

    Attributes
    ----------
    Xi : list
        Points at which objective has been evaluated.
    yi : scalar
        Values of objective at corresponding points in `Xi`.
    models : list
        Regression models used to fit observations and compute acquisition
        function.
    space : Space
        An instance of :class:`skopt.space.Space`. Stores parameter search
        space used to sample points, bounds, and type of parameters.

    """

    def __init__(self, dimensions, base_estimator="gp", base_estimator2="gp",
                 n_random_starts=None, n_initial_points=10,
                 initial_point_generator="random",
                 n_jobs=1, acq_func=None,
                 acq_optimizer="auto",
                 random_state=None,
                 model_queue_size=None,
                 acq_func_kwargs=None,
                 acq_optimizer_kwargs=None,
                 real_pareto_data=None,
                 surrogate_model_dict=None,
                 n_generation_points=0):
        args = locals().copy()
        del args['self']
        self.specs = {"args": args,
                      "function": "Optimizer"}
        self.rng = check_random_state(random_state)

        # Configure acquisition function

        # Store and creat acquisition function set
        self.acq_func = acq_func
        self.acq_func_kwargs = acq_func_kwargs

        # allowed_acq_funcs = ["gp_hedge", "EI", "LCB", "PI", "EIps", "PIps"]
        allowed_acq_funcs = ["EHVI", "HVI", "cv_ranking"]
        if self.acq_func not in allowed_acq_funcs:
            raise ValueError("expected acq_func to be in %s, got %s" %
                             (",".join(allowed_acq_funcs), self.acq_func))

        # treat hedging method separately
        if self.acq_func == "gp_hedge":
            self.cand_acq_funcs_ = ["EI", "LCB", "PI"]
            self.gains_ = np.zeros(3)
        else:
            self.cand_acq_funcs_ = [self.acq_func]

        if acq_func_kwargs is None:
            acq_func_kwargs = dict()
        self.eta = acq_func_kwargs.get("eta", 1.0)

        # Configure counters of points

        # Check `n_random_starts` deprecation first
        if n_random_starts is not None:
            warnings.warn(("n_random_starts will be removed in favour of "
                           "n_initial_points."),
                          DeprecationWarning)
            n_initial_points = n_random_starts

        if n_initial_points < 0:
            raise ValueError(
                "Expected `n_initial_points` >= 0, got %d" % n_initial_points)
        self._n_initial_points = n_initial_points
        self.n_initial_points_ = n_initial_points

        # Configure estimator

        # build base_estimator if doesn't exist
        if isinstance(base_estimator, str):
            base_estimator = cook_estimator(
                base_estimator, space=dimensions,
                random_state=self.rng.randint(0, np.iinfo(np.int32).max),
                n_jobs=n_jobs)
            base_estimator2 = cook_estimator(
                base_estimator2, space=dimensions,
                random_state=self.rng.randint(0, np.iinfo(np.int32).max),
                n_jobs=n_jobs)

        # check if regressor
        if not is_regressor(base_estimator) and base_estimator is not None:
            raise ValueError(
                "%s has to be a regressor." % base_estimator)

        # treat per second acqusition function specially
        is_multi_regressor = isinstance(base_estimator, MultiOutputRegressor)
        if "ps" in self.acq_func and not is_multi_regressor:
            self.base_estimator_ = MultiOutputRegressor(base_estimator)
        else:
            self.base_estimator_ = base_estimator
            self.base_estimator2_ = base_estimator2

        # Configure optimizer

        # decide optimizer based on gradient information
        if acq_optimizer == "auto":
            if has_gradients(self.base_estimator_):
                acq_optimizer = "lbfgs"
            else:
                acq_optimizer = "sampling"

        if acq_optimizer not in ["lbfgs", "sampling", "full"]:
            raise ValueError("Expected acq_optimizer to be 'lbfgs' or "
                             "'sampling', got {0}".format(acq_optimizer))

        self.acq_optimizer = acq_optimizer

        # record other arguments
        if acq_optimizer_kwargs is None:
            acq_optimizer_kwargs = dict()

        self.n_points = acq_optimizer_kwargs.get("n_points", 10000)
        self.n_restarts_optimizer = acq_optimizer_kwargs.get(
            "n_restarts_optimizer", 5)
        self.n_jobs = acq_optimizer_kwargs.get("n_jobs", 1)
        self.acq_optimizer_kwargs = acq_optimizer_kwargs

        # Configure search space

        # normalize space if GP regressor
        if isinstance(self.base_estimator_, GaussianProcessRegressor):
            dimensions = normalize_dimensions(dimensions)
        self.space = Space(dimensions)

        self._initial_samples = None
        if "orthogonal" == initial_point_generator:
            self._initial_point_generator = cook_initial_point_generator("sobol")
        else:
            self._initial_point_generator = cook_initial_point_generator(initial_point_generator)

        if self._initial_point_generator is not None:
            transformer = self.space.get_transformer()
            if "orthogonal" == initial_point_generator:
                self._initial_samples = get_orthogonal_array()
            else:
                self._initial_samples = self._initial_point_generator.generate(
                    self.space.dimensions, n_initial_points,
                    random_state=self.rng.randint(0, np.iinfo(np.int32).max))
            self.space.set_transformer(transformer)

        # record categorical and non-categorical indices
        self._cat_inds = []
        self._non_cat_inds = []
        for ind, dim in enumerate(self.space.dimensions):
            if isinstance(dim, Categorical):
                self._cat_inds.append(ind)
            else:
                self._non_cat_inds.append(ind)

        # Initialize storage for optimization
        if not isinstance(model_queue_size, (int, type(None))):
            raise TypeError("model_queue_size should be an int or None, "
                            "got {}".format(type(model_queue_size)))
        self.max_model_queue_size = model_queue_size
        self.models = []
        self.Xi = []
        self.yi = []
        self.yi2 = []
        self.current_pareto_x = []
        self.current_pareto_y = []
        self.hv = 0.0
        self.func_vals = []
        self.predict_errors = []
        self.predict_error_unlabelled_mapes = []
        self.non_uniformitys = []
        self.coverage = []
        if surrogate_model_dict['hv_scale_v']:
            self.ref_point_scale = None

        self.ref_point = torch.Tensor([100, 100])
        self.real_pareto_data = real_pareto_data
        self.surrogate_model_dict = surrogate_model_dict
        self.X_all = self.get_all_design_point()
        self.X_unlabeled = copy.deepcopy(self.X_all)
        self._n_generation_points = n_generation_points
        self.hv_last_iter_num = 0
        self.adrs_last_iter_num = 0
        self.cv_mode = 0
        self.feature_importances = None
        self.statistics = {'semi_train_accumulation': 0,
                           'ucb': 0,
                           'cv_ranking_beta': 0,
                           'hv_acq_stuck': 0,
                           'hv_last_iter': 0,
                           'non_uniformity_explore': 0,
                           'hv': [0.0],
                           }
        self.real_pareto_data_non_uniformity = -1 # evaluate_non_uniformity(np.reshape(real_pareto_data, (len(real_pareto_data[0]), 2)))
        #print(f"real_pareto_data_non_uniformity={self.real_pareto_data_non_uniformity}")

        hostname = socket.getfqdn(socket.gethostname())
        if "SC-202005121725" in hostname:
            # bookpad
            self.print_info = False
        else:
            self.print_info = False

        # Initialize cache for `ask` method responses
        # This ensures that multiple calls to `ask` with n_points set
        # return same sets of points. Reset to {} at every call to `tell`.
        self.cache_ = {}

    def copy(self, random_state=None):
        """Create a shallow copy of an instance of the optimizer.

        Parameters
        ----------
        random_state : int, RandomState instance, or None (default)
            Set the random state of the copy.
        """

        print("error ‘copy’ not fixed now")
        optimizer = OptimizerMultiobj(
            dimensions=self.space.dimensions,
            base_estimator=self.base_estimator_,
            base_estimator2=self.base_estimator2_,
            n_initial_points=self.n_initial_points_,
            initial_point_generator=self._initial_point_generator,
            acq_func=self.acq_func,
            acq_optimizer=self.acq_optimizer,
            acq_func_kwargs=self.acq_func_kwargs,
            acq_optimizer_kwargs=self.acq_optimizer_kwargs,
            random_state=random_state
        )
        optimizer._initial_samples = self._initial_samples
        if hasattr(self, "gains_"):
            optimizer.gains_ = np.copy(self.gains_)
        if self.Xi:
            optimizer._tell(self.Xi, self.yi, self.yi2)

        return optimizer

    def ask(self, n_points=None, strategy="cl_min"):
        """Query point or multiple points at which objective should be evaluated.

        n_points : int or None, default: None
            Number of points returned by the ask method.
            If the value is None, a single point to evaluate is returned.
            Otherwise a list of points to evaluate is returned of size
            n_points. This is useful if you can evaluate your objective in
            parallel, and thus obtain more objective function evaluations per
            unit of time.

        strategy : string, default: "cl_min"
            Method to use to sample multiple points (see also `n_points`
            description). This parameter is ignored if n_points = None.
            Supported options are `"cl_min"`, `"cl_mean"` or `"cl_max"`.

            - If set to `"cl_min"`, then constant liar strategy is used
               with lie objective value being minimum of observed objective
               values. `"cl_mean"` and `"cl_max"` means mean and max of values
               respectively. For details on this strategy see:

               https://hal.archives-ouvertes.fr/hal-00732512/document

               With this strategy a copy of optimizer is created, which is
               then asked for a point, and the point is told to the copy of
               optimizer with some fake objective (lie), the next point is
               asked from copy, it is also told to the copy with fake
               objective and so on. The type of lie defines different
               flavours of `cl_x` strategies.

        """
        if n_points is None:
            return self._ask()

        supported_strategies = ["cl_min", "cl_mean", "cl_max"]

        if not (isinstance(n_points, int) and n_points > 0):
            raise ValueError(
                "n_points should be int > 0, got " + str(n_points)
            )

        if strategy not in supported_strategies:
            raise ValueError(
                "Expected parallel_strategy to be one of " +
                str(supported_strategies) + ", " + "got %s" % strategy
            )

        # Caching the result with n_points not None. If some new parameters
        # are provided to the ask, the cache_ is not used.
        if (n_points, strategy) in self.cache_:
            return self.cache_[(n_points, strategy)]

        # Copy of the optimizer is made in order to manage the
        # deletion of points with "lie" objective (the copy of
        # oiptimizer is simply discarded)
        opt = self.copy(random_state=self.rng.randint(0,
                                                      np.iinfo(np.int32).max))

        X = []
        for i in range(n_points):
            x = opt.ask()
            X.append(x)

            ti_available = "ps" in self.acq_func and len(opt.yi) > 0
            ti = [t for (_, t) in opt.yi] if ti_available else None

            if strategy == "cl_min":
                y_lie = np.min(opt.yi) if opt.yi else 0.0  # CL-min lie
                t_lie = np.min(ti) if ti is not None else log(sys.float_info.max)
            elif strategy == "cl_mean":
                y_lie = np.mean(opt.yi) if opt.yi else 0.0  # CL-mean lie
                t_lie = np.mean(ti) if ti is not None else log(sys.float_info.max)
            else:
                y_lie = np.max(opt.yi) if opt.yi else 0.0  # CL-max lie
                t_lie = np.max(ti) if ti is not None else log(sys.float_info.max)

            # Lie to the optimizer.
            if "ps" in self.acq_func:
                # Use `_tell()` instead of `tell()` to prevent repeated
                # log transformations of the computation times.
                opt._tell(x, (y_lie, t_lie))
            else:
                opt._tell(x, y_lie)

        self.cache_ = {(n_points, strategy): X}  # cache_ the result

        return X

    def _ask(self):
        """Suggest next point at which to evaluate the objective.

        Return a random point while not at least `n_initial_points`
        observations have been `tell`ed, after that `base_estimator` is used
        to determine the next point.
        """
        if self._n_initial_points > 0 or self.base_estimator_ is None:
            # this will not make a copy of `self.rng` and hence keep advancing
            # our random state.
            if self._initial_samples is None:
                return self.space.rvs(random_state=self.rng)[0]
            elif self._n_initial_points <= len(self._initial_samples):
                # The samples are evaluated starting form initial_samples[0]
                # print(f"_initial_samples={self._initial_samples} self._n_initial_points={self._n_initial_points}")
                return self._initial_samples[
                    len(self._initial_samples) - self._n_initial_points]
            else:
                return self.space.rvs(random_state=self.rng)[0]

        else:
            if not self.models:
                raise RuntimeError("Random evaluations exhausted and no "
                                   "model has been fit.")

            next_x = self._next_x
            min_delta_x = min([self.space.distance(next_x, xi)
                               for xi in self.Xi])
            if abs(min_delta_x) <= 1e-8:
                warnings.warn("The objective has been evaluated "
                              "at this point before.")
                print(f"warining sampled next_x={next_x}")

            # return point computed from last call to tell()
            # print(f"ask next_x={next_x}")
            return next_x

    def calculate_exploration_part(self, values_mean, values_std, median_obj=0, stuck_retry=0):
        if 13 == self.surrogate_model_dict['ucb_v']:
            diff_value = self.surrogate_model_dict['ucb-iter'] * values_std / values_mean
        elif 14 == self.surrogate_model_dict['ucb_v']:
            diff_value = self.surrogate_model_dict['ucb-iter'] * values_std * abs(values_mean - median_obj)
        elif 15 == self.surrogate_model_dict['ucb_v']:
            diff_value = self.surrogate_model_dict['ucb-iter'] * (1 + self.hv_last_iter_num) * values_std * abs(values_mean- median_obj)
        elif 16 == self.surrogate_model_dict['ucb_v']:
            diff_value = self.surrogate_model_dict['ucb-iter'] * values_std \
                         * pow(1 + abs(values_mean - median_obj)/median_obj/2,  1 + self.hv_last_iter_num + stuck_retry + self.statistics['hv_acq_stuck'])
        elif 17 == self.surrogate_model_dict['ucb_v']:
            diff_value = self.surrogate_model_dict['ucb-iter'] * (1 + self.hv_last_iter_num) * values_std
        elif 18 == self.surrogate_model_dict['ucb_v']:
            diff_value = self.surrogate_model_dict['ucb-iter'] * (1 + self.hv_last_iter_num) * values_std
        elif 20 == self.surrogate_model_dict['ucb_v']:
            diff_value = self.surrogate_model_dict['ucb-iter'] * values_std
        else:
            # 12 == self.surrogate_model_dict['ucb_v']:
            diff_value = self.surrogate_model_dict['ucb-iter'] * values_std
        return diff_value

    def get_hvi(self, X_origin, est, est2, stuck_retry=0):
        ref_point = 0 - self.ref_point
        hypervolume_acq_values = []
        X_transform = self.space.transform(X_origin)
        if self.surrogate_model_dict['ucb']:
            self.statistics['ucb'] += 1
            values = np.asarray([each_predict for each_predict in est.staged_predict(X_transform)])
            values2 = np.asarray([each_predict for each_predict in est2.staged_predict(X_transform)])
            values_mean = values.mean(axis=0)
            values2_mean = values2.mean(axis=0)
            values_std = values.std(axis=0)
            values2_std = values2.std(axis=0)
            if 0:
                cv = values_std/values_mean
                cv2 = values2_std/values2_mean
                print(f"CPI std/mean={np.min(cv), np.max(cv)}, power std/mean={np.min(cv2), np.max(cv2)}")
        else:
            values_mean = est.predict(X_transform)
            values2_mean = est2.predict(X_transform)
        median_obj = np.median(self.current_pareto_y[: ,0])
        median_obj2 = np.median(self.current_pareto_y[: ,1])
        if self.surrogate_model_dict['ucb-iter']:
            if (20 == self.surrogate_model_dict['ucb_v']) and (0 == stuck_retry):
                diff_values = None
                diff_values2 = None
            else:
                diff_values = self.calculate_exploration_part(values_mean=values_mean, values_std=values_std, median_obj=median_obj, stuck_retry=stuck_retry)
                diff_values2 = self.calculate_exploration_part(values_mean=values2_mean, values_std=values2_std, median_obj=median_obj2, stuck_retry=stuck_retry)
        else:
            diff_values = None
            diff_values2 = None
        self.ref_point_scale = [max(max(self.yi), max(values_mean))*1.1, max(max(self.yi2), max(values2_mean))*1.1]
        for candidate_x_index, candidate_x in enumerate(X_origin):
            # print(f"\tcandidate_x={candidate_x}")
            candidate_y = values_mean[candidate_x_index]
            candidate_y2 = values2_mean[candidate_x_index]
            diff_value = diff_values[candidate_x_index] if diff_values is not None else 0
            diff_value2 = diff_values2[candidate_x_index] if diff_values2 is not None else 0
            candidate_y -= diff_value
            candidate_y2 -= diff_value2
            try_predict_x = np.append(copy.deepcopy(self.current_pareto_x), np.asarray([candidate_x]), axis=0)
            try_predict_y = np.append(copy.deepcopy(self.current_pareto_y),
                                      np.asarray([[candidate_y, candidate_y2]]), axis=0)
            # print(f"\ttry_predict_y={try_predict_y}")
            try_new_pareto_frontier_x, try_new_pareto_frontier_y = is_pareto_efficient_dumb(try_predict_x, try_predict_y)
            if self.surrogate_model_dict['hv_scale_v']:
                try_new_pareto_frontier_y_scale = scale_for_hv(try_new_pareto_frontier_y, self.ref_point_scale)
                hypervolume = DominatedPartitioning(
                    ref_point=torch.Tensor([-1,-1]),
                    Y=0 - torch.Tensor(try_new_pareto_frontier_y_scale)
                ).compute_hypervolume().item()
            else:
                hypervolume = DominatedPartitioning(
                    ref_point=0 - torch.Tensor(self.ref_point),
                    Y=0 - torch.Tensor(try_new_pareto_frontier_y)
                ).compute_hypervolume().item()

            hypervolume_acq_values.append(hypervolume)
        # if self.print_info:
        # print("hv acq count=", np.count_nonzero(self.hv < np.asarray(hypervolume_acq_values)))
        # print(f"\thypervolume_acq_values={hypervolume_acq_values}")
        # plt.plot(range(len(hypervolume_acq_values)), hypervolume_acq_values)
        # plt.show()
        next_index = np.argmax(hypervolume_acq_values)
        # next_y = values[next_index]
        # next_y2 = values2[next_index]
        hv_try_stuck = ((hypervolume_acq_values[next_index] - self.hv) < 0.00001)

        candi_point_hv = None
        if (18 == self.surrogate_model_dict['ucb_v']) \
                and ((hypervolume_acq_values[next_index] - self.hv) < 0.00001) \
                and (0 == self.surrogate_model_dict['non_uniformity_explore']):
            print("hv struck and ucb_v == 18")
            candi_point_hv = (np.asarray(values_mean) - np.ones(len(values_mean)) * int(self.ref_point[0])) * \
                            (np.asarray(values2_mean) - np.ones(len(values2_mean)) * int(self.ref_point[1]))
            next_index = np.argmax(candi_point_hv)
        elif (19 == self.surrogate_model_dict['ucb_v']) \
                and ((hypervolume_acq_values[next_index] - self.hv) < 0.00001) \
                and (0 == self.surrogate_model_dict['non_uniformity_explore']):
            print("hv struck and ucb_v == 19")
            candi_point_hv = (np.asarray(values_mean)/max(values_mean) - np.ones(len(values_mean))) * \
                            (np.asarray(values2_mean)/max(values2_mean) - np.ones(len(values2_mean)))
            next_index = np.argmax(candi_point_hv)
        elif (25 == self.surrogate_model_dict['ucb_v']) \
                and ((hypervolume_acq_values[next_index] - self.hv) < 0.00001) \
                and (0 == self.surrogate_model_dict['non_uniformity_explore'] or (self.hv_last_iter_num < 3)):
            print("hv struck and ucb_v == 25")
            candi_point_hv = (np.asarray(values_mean)/self.ref_point_scale[0] - np.ones(len(values_mean))) * \
                        (np.asarray(values2_mean)/self.ref_point_scale[1] - np.ones(len(values2_mean)))
            next_index = np.argmax(candi_point_hv)
        elif (27 == self.surrogate_model_dict['ucb_v']) \
                and ((hypervolume_acq_values[next_index] - self.hv) < 0.00001) \
                and (0 == self.surrogate_model_dict['non_uniformity_explore'] or (self.hv_last_iter_num < 2)):
            print("hv struck and ucb_v == 27")
            candi_point_hv = (np.asarray(values_mean)/self.ref_point_scale[0] - np.ones(len(values_mean))) * \
                        (np.asarray(values2_mean)/self.ref_point_scale[1] - np.ones(len(values2_mean)))
            next_index = np.argmax(candi_point_hv)

        if plot_pareto and (
                #self.hv_last_iter_num or (0 == self._n_initial_points & 0xF) or
                ((- self._n_generation_points) == self._n_initial_points)
                or (hv_try_stuck)
        ):
            try:
                import matplotlib.pyplot as plt
                X_pareto = self.space.transform(self.real_pareto_data[2])
                Y_pareto = np.asarray([each_predict for each_predict in est.staged_predict(X_pareto)])
                Y2_pareto = np.asarray([each_predict for each_predict in est2.staged_predict(X_pareto)])
                Y_pareto_mean = Y_pareto.mean(axis=0)
                Y2_pareto_mean = Y2_pareto.mean(axis=0)
                Y_pareto_std = Y_pareto.std(axis=0)
                Y2_pareto_std = Y2_pareto.std(axis=0)
                if 0:
                    Y_pareto_diff = self.calculate_exploration_part(values_mean=Y_pareto_mean, values_std=Y_pareto_std, median_obj=median_obj, stuck_retry=stuck_retry)
                    Y2_pareto_diff = self.calculate_exploration_part(values_mean=Y2_pareto_mean, values_std=Y2_pareto_std, median_obj=median_obj2, stuck_retry=stuck_retry)
                else:
                    Y_pareto_diff = Y_pareto_std
                    Y2_pareto_diff = Y2_pareto_std
                plt.errorbar(x=Y_pareto_mean, y=Y2_pareto_mean,
                             xerr=Y_pareto_diff, yerr=Y2_pareto_diff,
                             fmt='m^', capsize=2, elinewidth=1, ms=2, label='Pareto Predict')
                pareto_point_hv = (np.asarray(Y_pareto_mean) / self.ref_point_scale[0] - np.ones(len(Y_pareto_mean))) * \
                                 (np.asarray(Y2_pareto_mean) / self.ref_point_scale[1] - np.ones(len(Y2_pareto_mean)))
                #for each_pareto_iter, each_pareto_hv in enumerate(pareto_point_hv):
                #    plt.text(x=Y_pareto_mean[each_pareto_iter] + 0.1, y=Y2_pareto_mean[each_pareto_iter] + 0.1, s='%.2f' % each_pareto_hv, rotation=45, fontsize=4, color='brown')
                plt.scatter(self.yi, self.yi2, c='Gray', s=4, label="Sampled")

                if candi_point_hv is not None:
                    hypervolume_acq_values = candi_point_hv
                hypervolume_acq_values_sort_index = np.argsort(- np.asarray(hypervolume_acq_values))[:10]
                for candi, try_new_pareto_frontier_y_index in enumerate(hypervolume_acq_values_sort_index):
                    # plt.scatter(try_new_pareto_frontier_y[:, 0], try_new_pareto_frontier_y[:, 1], c='Blue', s=6)
                    fmt = "ko" if 0 == candi else "co"
                    ms = 4 if 0 == candi else 2
                    plt.errorbar(x=values_mean[try_new_pareto_frontier_y_index],
                                 y=values2_mean[try_new_pareto_frontier_y_index],
                                 xerr=diff_values[try_new_pareto_frontier_y_index] if diff_values is not None else 0,
                                 yerr=diff_values2[try_new_pareto_frontier_y_index] if diff_values2 is not None else 0,
                                 fmt=fmt, capsize=2, elinewidth=1, ms=ms)
                plt.scatter(np.asarray(self.current_pareto_y[:, 0:1]), np.asarray(self.current_pareto_y[:, 1:2]),
                            marker='*', c='orange', s=10, label='current')
                plt.scatter(values_mean[next_index], values2_mean[next_index],
                            marker='o', c='white', s=13, alpha=0.5, edgecolors='green', label='origin w/ max_hv')
                #if candi_point_hv is not None:
                #    plt.text(x=values_mean[next_index]+0.1, y=values2_mean[next_index]+0.1, s='%.2f' % candi_point_hv[next_index], rotation=45, fontsize=4, color='green')
                plt.scatter(self.real_pareto_data[0], self.real_pareto_data[1], c='red', s=4, marker='+', label='Real')

                plt.ylabel("Power")
                plt.xlabel("CPI")
                plt.grid()
                #plt.xscale('log')
                #plt.yscale('log')
                plt.legend()

                info_str = case_name + "/" + self.surrogate_model_dict['tag'] + '-' + str(-self._n_initial_points)
                plt.title(info_str)
                plt.savefig("fig_pareto_debug/" + info_str + ".png")
                plt.close('all')
            except:
                print("plt savefig except")
                plt.close('all')
            
            #exit(1)
        if ((hypervolume_acq_values[next_index] - self.hv) < 0.00001):  # or (2 < self.hv_last_iter_num):
            self.statistics['hv_acq_stuck'] += 1
            if self.surrogate_model_dict['non_uniformity_explore']:
                next_index = self.get_non_uniformity_explore(X_origin, values_mean, values2_mean)
            elif (16 == self.surrogate_model_dict['ucb_v']) and (stuck_retry < 4):
                return self.get_hvi(X_origin, est, est2, stuck_retry=stuck_retry + 1)
            elif (20 == self.surrogate_model_dict['ucb_v']) and (stuck_retry < 1):
                return self.get_hvi(X_origin, est, est2, stuck_retry=stuck_retry + 1)
        return next_index

    def get_ehvi(self, X_origin, est, est2):
        ref_point = 0 - self.ref_point
        hypervolume_acq_values = []
        hypervolume_max_this_iteration = self.hv
        integral_num = 128
        integral_standard_deviation_num = 3
        integral_value = integral_standard_deviation_num * 2 / integral_num
        for candidate_x in X_origin:
            # print(f"\tcandidate_x={candidate_x}")
            candidate_x_transform = self.space.transform([candidate_x])
            if "BOOM" in self.surrogate_model_dict['tag']:
                value_mean, value_std = est.predict(candidate_x_transform, return_std=True)
                value2_mean, value2_std = est2.predict(candidate_x_transform, return_std=True)
            else:
                values = np.asarray([each_predict for each_predict in est.staged_predict(candidate_x_transform)])
                values2 = np.asarray([each_predict for each_predict in est2.staged_predict(candidate_x_transform)])
                if 2 < len(est.estimators_):
                    value_mean = values.mean(axis=0)[0]
                    value2_mean = values2.mean(axis=0)[0]
                    value_std = values.std(axis=0)[0]
                    value2_std = values2.std(axis=0)[0]
                else:
                    print(f"no staged_predict std")

            hypervolume = 0
            try_predict_x = np.append(copy.deepcopy(self.current_pareto_x), np.asarray([candidate_x]), axis=0)
            # try with the max hv possible, if it done't improve the hypervolume, ignore it
            try_predict_y = np.append(copy.deepcopy(self.current_pareto_y),
                                      np.asarray([[value_mean - value_std, value2_mean - value2_std]]), axis=0)
            try_new_pareto_frontier_x, try_new_pareto_frontier_y = is_pareto_efficient_dumb(try_predict_x,
                                                                                            try_predict_y)
            hypervolume_max_prob = DominatedPartitioning(
                ref_point=ref_point,
                Y=0 - torch.Tensor(try_new_pareto_frontier_y)
            ).compute_hypervolume().item()
            if hypervolume_max_this_iteration < hypervolume_max_prob:
                prob_y1 = np.linspace(value_mean - value_std * integral_standard_deviation_num,
                                      value_mean + value_std * integral_standard_deviation_num,
                                      integral_num)
                prob_y1_norm = stats.norm.pdf(prob_y1, loc=value_mean, scale=value_std) * value_std
                # print(f"prob_y1_norm={prob_y1_norm}")
                prob_y2 = np.linspace(value2_mean - value2_std * integral_standard_deviation_num,
                                      value2_mean + value2_std * integral_standard_deviation_num,
                                      integral_num)
                prob_y2_norm = stats.norm.pdf(prob_y2, loc=value2_mean, scale=value2_std) * value2_std
                if 0:
                    import matplotlib.pyplot as plt
                    plt.plot(prob_y1, prob_y1_norm, c='b')
                    plt.plot(prob_y2, prob_y2_norm, c='r')
                    print(f"prob_y1_norm sum={np.sum(prob_y1_norm) * value_std * integral_value}")
                    print(f"prob_y2_norm sum={np.sum(prob_y2_norm) * value2_std * integral_value}")
                    plt.show()
                prob_sum = 0
                for candidate_y, prob_y1_pdf in zip(prob_y1, prob_y1_norm):
                    # prob_y1_pdf = prob_y1_norm.pdf(candidate_y)
                    # print(f"candidate_y={candidate_y} prob_y1_pdf={prob_y1_pdf}")
                    for candidate_y2, prob_y2_pdf in zip(prob_y2, prob_y2_norm):
                        # prob_y2_pdf = prob_y2_norm.pdf(candidate_y2)
                        # print(f"candidate_y2={candidate_y2} prob_y2_pdf={prob_y2_pdf}")
                        try_predict_y = np.append(copy.deepcopy(self.current_pareto_y),
                                                  np.asarray([[candidate_y, candidate_y2]]), axis=0)
                        _, try_new_pareto_frontier_y = is_pareto_efficient_dumb(try_predict_x, try_predict_y)
                        hypervolume_prob = DominatedPartitioning(
                            ref_point=ref_point,
                            Y=0 - torch.Tensor(try_new_pareto_frontier_y)
                        ).compute_hypervolume().item()
                        # print(f"hypervolume_prob={hypervolume_prob} prob_pdf={prob_y1_pdf}, {prob_y2_pdf}")
                        prob = prob_y1_pdf * prob_y2_pdf
                        hypervolume += hypervolume_prob * prob
                        prob_sum += prob
                hypervolume *= value_std * integral_value * value2_std * integral_value
                hypervolume_max_this_iteration = max(hypervolume_max_this_iteration, hypervolume)
                prob_sum *= value_std * integral_value * value2_std * integral_value
                # print(f"NEHVI_hypervolume= {hypervolume}")
                # print(f"prob_sum={prob_sum}")
                hypervolume /= prob_sum
            else:
                hypervolume = hypervolume_max_prob
                # print(f"ignore try_predict_x")
            if 0:
                try:
                    # model = self.base_estimator_
                    # model = SingleTaskGP(torch.Tensor(self.current_pareto_x), 0 - torch.Tensor(self.current_pareto_y))
                    model = botorch_model_wrapper(torch.Tensor(self.current_pareto_x),
                                                  0 - torch.Tensor(self.current_pareto_y), self.base_estimator_)
                    # print(f"current_pareto_x= {torch.Tensor(self.current_pareto_x)}")
                    qNEHVI_func = qNoisyExpectedHypervolumeImprovement(
                        model=model,
                        ref_point=0 - self.ref_point,
                        X_baseline=torch.Tensor(self.current_pareto_x))
                    qNEHVI_value = qNEHVI_func(torch.Tensor(try_predict_x))
                    print(f"qNEHVI_value= {qNEHVI_value}")
                except:
                    print("qNoisyExpectedHypervolumeImprovement failed")

            hypervolume_acq_values.append(hypervolume)
        # print(f"\thypervolume_acq_values={hypervolume_acq_values}")
        plt_show = False
        if plt_show:
            import matplotlib.pyplot as plt
            plt.plot(range(len(hypervolume_acq_values)), hypervolume_acq_values)
            plt.title("hypervolume_acq_values")
            plt.show()
        next_index = np.argmax(hypervolume_acq_values)
        next_y = values_mean[next_index]
        next_y2 = values2_mean[next_index]
        if False:
            plt.scatter(np.asarray(self.current_pareto_y[:, 0:1]), np.asarray(self.current_pareto_y[:, 1:2]))
            plt.show()
            exit(1)
        return next_index, next_y, next_y2, max(hypervolume_acq_values)

    def roulette_wheel_method(self, di, power_num):
        di = np.asarray(di)
        if 15 == self.surrogate_model_dict['non_uniformity_explore']:
            candi_num = min(int(power_num/2+2), len(di)-1)
        else:
            candi_num = min(power_num, len(di) - 1)
        di_sort_index = np.argsort(-di)[:candi_num]
        di_head = di[di_sort_index]
        di_prob_1 = di_head / sum(di_head)
        di_pow = pow(di_prob_1, 1.0/power_num)
        prob = di_pow / sum(di_pow)
        #print(f"di_prob={prob*100}")
        di_acc = 0
        for di_choose_iter, di_choose_index in enumerate(di_sort_index):
            di_acc += prob[di_choose_iter]
            if random.random() < di_acc:
                di_max_index = di_choose_index
                return di_choose_index
        return di_sort_index[-1]

    def get_non_uniformity_explore(self, X_origin, try_y, try_y2):
        self.statistics['non_uniformity_explore'] += 1
        print(f"hv_acq_stuck num={self.statistics['hv_acq_stuck']} iter={-self._n_initial_points}")

        #x_transform = self.space.transform(X_origin)
        #try_y = est.predict(x_transform)
        #try_y2 = est2.predict(x_transform)

        pareto_for_di = []
        for each_y, each_x in zip(self.current_pareto_y, self.current_pareto_x):
            pareto_for_di.append([each_y[0], each_y[1], each_x])

        if 12 == self.surrogate_model_dict['non_uniformity_explore']:
            pareto_for_di = np.append(pareto_for_di,
                                      [[min(self.current_pareto_y[:, 0]) * 0.9, max(self.current_pareto_y[:, 1]) * 1.1, []]],
                                      axis=0)
            pareto_for_di = np.append(pareto_for_di,
                                      [[max(self.current_pareto_y[:, 0]) * 1.1, min(self.current_pareto_y[:, 1]) * 0.9, []]],
                                      axis=0)
        elif 13 == self.surrogate_model_dict['non_uniformity_explore']:
            pareto_for_di = np.append(pareto_for_di,
                                      [[min(self.current_pareto_y[:, 0]) * 0.8, max(self.current_pareto_y[:, 1]) * 1.2, []]],
                                      axis=0)
            pareto_for_di = np.append(pareto_for_di,
                                      [[max(self.current_pareto_y[:, 0]) * 1.2, min(self.current_pareto_y[:, 1]) * 0.8, []]],
                                      axis=0)
        elif 15 == self.surrogate_model_dict['non_uniformity_explore']:
            pareto_for_di = np.append(pareto_for_di,
                                      [[min(self.current_pareto_y[:, 0]) * 0.9, max(self.current_pareto_y[:, 1]) * 1.2, []]],
                                      axis=0)
            pareto_for_di = np.append(pareto_for_di,
                                      [[max(self.current_pareto_y[:, 0]) * 1.1, min(self.current_pareto_y[:, 1]) * 0.9, []]],
                                      axis=0)       
        elif 16 == self.surrogate_model_dict['non_uniformity_explore']:
            pareto_for_di = np.append(pareto_for_di,
                                      [[min(self.current_pareto_y[:, 0]) * 0.9, max(self.current_pareto_y[:, 1]) * 1.2, []]],
                                      axis=0)
            pareto_for_di = np.append(pareto_for_di,
                                      [[max(self.current_pareto_y[:, 0]) * 1.1, min(self.current_pareto_y[:, 1]) * 0.9, []]],
                                      axis=0)      
        else:
            pareto_for_di = np.append(pareto_for_di,
                                      [[min(min(try_y), min(self.yi)) * 0.99, max(max(try_y2), max(self.yi2)) * 1.01, []]],
                                      axis=0)
            pareto_for_di = np.append(pareto_for_di,
                                      [[max(max(try_y), max(self.yi)) * 1.01, min(min(try_y2), min(self.yi2)) * 0.99, []]],
                                      axis=0)

        di, learned_pareto_optimal_sets_y_sort = get_di(learned_pareto_optimal_sets_y_unsort=pareto_for_di)
        # di_max_index = np.argmax(di)

        di_max_index = self.roulette_wheel_method(di, self.hv_last_iter_num + 1)
        # print(f"non_uniformity_explore choose id={di_max_index}")

        di_max = learned_pareto_optimal_sets_y_sort[di_max_index:di_max_index + 2]
        di_max_found = di_max[:, 0:2]
        #if 0 == di_max_index or len(learned_pareto_optimal_sets_y_sort) == di_max_index:
        di_x = di_max[:, 2]

        if 7 <= self.surrogate_model_dict['non_uniformity_explore'] <= 8:
            if 0 == len(di_x[0]):
                choose_di_x = di_x[1]
            else:
                choose_di_x = di_x[0]
            choose_di_x_trans = self.space.transform([choose_di_x])[0]
            candi_x = self.get_near_pareto_points()
            distance = []
            for each in candi_x:
                distance.append(- np.sum(np.square(self.space.transform([each])[0] - choose_di_x_trans)))

            next_x_index = np.argmax(distance)
            next_x = candi_x[next_x_index]
        elif 16 == self.surrogate_model_dict['non_uniformity_explore']:
            distance_1 = np.square(try_y - di_max[0][0]) + np.square(try_y2 - di_max[0][1])
            distance_2 = np.square(try_y - di_max[1][0]) + np.square(try_y2 - di_max[1][1])
            distance = []
            for each_1, each_2 in zip(distance_1, distance_2):
                distance.append(min(each_1, each_2))
            distance_sign = ((try_y < di_max[1][0]) & (try_y2 < di_max[0][1]))
            for each_index, each_sign in enumerate(distance_sign):
                if each_sign:
                    distance[each_index] *= -1

            next_x_index = np.argmin(distance)
            next_x = X_origin[next_x_index]
        else:
            # 6 == self.surrogate_model_dict['non_uniformity_explore']:

            distance_1 = np.square(try_y - di_max[0][0]) + np.square(try_y2 - di_max[0][1])
            distance_2 = np.square(try_y - di_max[1][0]) + np.square(try_y2 - di_max[1][1])
            distance = distance_1 + distance_2
            distance_sign = ((try_y < di_max[1][0]) & (try_y2 < di_max[0][1]))
            for each_index, each_sign in enumerate(distance_sign):
                if not each_sign:
                    distance[each_index] *= -1

            next_x_index = np.argmax(distance)
            next_x = X_origin[next_x_index]

        #if next_x_index < 0:
        #    next_x_index = random.randint(0, len(X_origin))
        #    next_x = X_origin[next_x_index]
        #    print(f"random choose next_x")

        if plot_pareto:
            try:
                import matplotlib.pyplot as plt
                plt.scatter(self.yi, self.yi2, c='Gray', s=4, label="Sampled")
                if di_max_found is not None:
                    plt.scatter(di_max_found[:, 0], di_max_found[:, 1], c='white', alpha=0.5, edgecolors='Black', s=20, marker='o', label="Max non_uniformity")
                    #plt.axvline(di_max_found[0][0], c="g", ls="--", lw=1)
                plt.scatter(try_y[next_x_index], try_y2[next_x_index], c='Black', s=15, label="Choose " + str(di_max_index))
                plt.scatter(np.asarray(self.current_pareto_y[:, 0:1]), np.asarray(self.current_pareto_y[:, 1:2]),
                            marker='*', c='orange', s=10, label='current')
                plt.scatter(self.real_pareto_data[0], self.real_pareto_data[1], c='red', s=4, marker='+', label='Real')

                plt.ylabel("Power")
                plt.xlabel("CPI")
                #plt.xscale('log')
                #plt.yscale('log')
                plt.legend()
                plt.title(case_name.split('.')[0] + " " + self.surrogate_model_dict['tag'] + ' iter-' + str(-self._n_initial_points), fontsize=8)
                plt.savefig("fig_pareto_debug/" + case_name + "/" + self.surrogate_model_dict['tag'] + '-' + str(-self._n_initial_points) + "_uniformity.png")
                #plt.show()
                #plt.close()    
                #exit(1)
            except:
                print("plt plot_pareto except")
            plt.close('all')
            if 0:
                filename = "csv/" + "fig_pareto_debug/" + case_name + "/" + self.surrogate_model_dict['tag'] + '-' + str(-self._n_initial_points)
                with open(filename + '.csv', mode='w', encoding='utf-8-sig', newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["di_max_found"])
                    writer.writerows(di_max_found)
                    writer.writerow(["try_y"])
                    writer.writerow([try_y[next_x_index], try_y2[next_x_index]])
                    writer.writerow(["current"])
                    writer.writerows(self.current_pareto_y)
                    writer.writerow(["real_pareto_data"])
                    writer.writerows(self.real_pareto_data)

        return next_x_index

    def cv_ranking(self, X_origin, est, est2,
                   metric_filter=False,
                   cv_ranking="minsort",
                   X_sampled_transform=None, Y_sampled=None, Y2_sampled=None):
        candidate_x_transform = self.space.transform(X_origin)
        try:
            values = np.asarray([each_predict for each_predict in est.staged_predict(candidate_x_transform)])
            values2 = np.asarray([each_predict for each_predict in est2.staged_predict(candidate_x_transform)])
        except:
            print(f"surrogate model dost not support staged_predict \n model_1={est} model_2={est2} ")
            exit(1)
        if len(est.estimators_) < 2:
            next_index = np.random.randint(low=0, high=len(candidate_x_transform))
            next_y = est.predict([candidate_x_transform[next_index]])
            next_y2 = est2.predict([candidate_x_transform[next_index]])
        else:
            values_mean = values.mean(axis=0)
            values2_mean = values2.mean(axis=0)
            self.cv_mode = 0 if self.cv_mode else 1
            if self.cv_mode:
                values_iter = values2_mean
                values_iter_std = values2.std(axis=0)
            else:
                values_iter = values_mean
                values_iter_std = values.std(axis=0)
            coefficient_of_variation = values_iter_std / values_iter
            if 'minsort' == cv_ranking:
                cv_ranks = np.argsort(coefficient_of_variation)
            else:
                # max_sort
                cv_ranks = np.argsort(- coefficient_of_variation)

            next_index = cv_ranks[0]
            if metric_filter:
                sampled_y_error = mean_absolute_percentage_error(y_true=Y_sampled,
                                                                 y_pred=est.predict(X_sampled_transform))
                sampled_y_error2 = mean_absolute_percentage_error(y_true=Y2_sampled,
                                                                  y_pred=est2.predict(X_sampled_transform))
                for cv_ranking_iter, cv_rank in enumerate(cv_ranks):
                    # if 0.01 < coefficient_of_variation[cv_rank]:
                    #    break
                    try_x = np.append(copy.deepcopy(X_sampled_transform), [candidate_x_transform[cv_rank]], axis=0)
                    try_y = copy.deepcopy(Y_sampled)
                    try_y.append(values_mean[cv_rank])
                    try_y2 = copy.deepcopy(Y2_sampled)
                    try_y2.append(values2_mean[cv_rank])
                    try_sampled_y = clone(est).fit(try_x, try_y).predict(X_sampled_transform)
                    try_sampled_y2 = clone(est2).fit(try_x, try_y2).predict(X_sampled_transform)
                    try_sampled_y_error = mean_absolute_percentage_error(y_true=Y_sampled, y_pred=try_sampled_y)
                    try_sampled_y_error2 = mean_absolute_percentage_error(y_true=Y2_sampled, y_pred=try_sampled_y2)
                    if (try_sampled_y_error < sampled_y_error) and (try_sampled_y_error2 < sampled_y_error2):
                        next_index = cv_rank
                        if self.print_info:
                            print(f"rank={cv_ranking_iter} mode={self.cv_mode} "
                                  f"cv={coefficient_of_variation[cv_rank] * 100} percent = "
                                  f"(std={values_iter_std[cv_rank]} / mean={values_iter[cv_rank]})")
                        # print(f"iter= {cv_ranking_iter} "
                        #     f"EL'={try_sampled_y_error} < EL={sampled_y_error}"
                        #      f" or EL'={try_sampled_y_error2} < EL={sampled_y_error2}")
                        break
            else:
                # randomly choose N=4 from top W=16 candidate, here just choose N=1
                next_index = cv_ranks[random.randint(0, 16)]
            next_y = values_mean[next_index]
            next_y2 = values2_mean[next_index]
        next_x = X_origin[next_index]
        return next_x, next_index, next_y, next_y2

    def semi_train(self, X_origin, est, est2):
        random.shuffle(X_origin)
        self.statistics['semi_train_accumulation'] += 1
        X_pool = deepcopy(X_origin[:self.surrogate_model_dict['cv_pool_size']])
        X_sampled_transform = self.space.transform(deepcopy(self.Xi))
        Y_sampled = deepcopy(self.yi)
        Y2_sampled = deepcopy(self.yi2)
        for semi_train_iter in range(self.surrogate_model_dict['semi_train_iter_max']):
            most_confident_x, most_confident_x_index, most_confident_y1, most_confident_y2 = self.cv_ranking(
                X_origin=X_pool,
                est=est,
                est2=est2,
                metric_filter=True,
                cv_ranking="minsort",
                X_sampled_transform=X_sampled_transform,
                Y_sampled=Y_sampled,
                Y2_sampled=Y2_sampled,
            )
            X_sampled_transform = np.append(X_sampled_transform, np.asarray(self.space.transform([most_confident_x])),
                                            axis=0)
            Y_sampled.append(float(most_confident_y1))
            Y2_sampled.append(float(most_confident_y2))
            X_pool.pop(most_confident_x_index)
            est.fit(X_sampled_transform, Y_sampled)
            est2.fit(X_sampled_transform, Y2_sampled)
            X_pool.append(X_origin[self.surrogate_model_dict['cv_pool_size'] + semi_train_iter])
        return est, est2

    def get_accuracy_on_real_pareto_data(self, est, est2, real_pareto_data, labeled_predict_mode=None):
        [real_pareto_points_obj1, real_pareto_points_obj2, real_pareto] = real_pareto_data
        real_pareto_transform = self.space.transform(real_pareto)
        # print(f"real_pareto={real_pareto}, real_pareto_transform={real_pareto_transform}")
        if labeled_predict_mode is None:
            labeled_predict_mode = self.surrogate_model_dict['labeled_predict_mode']

        if labeled_predict_mode:
            predict_value_obj1 = (est.predict(real_pareto_transform))
            predict_value_obj2 = (est2.predict(real_pareto_transform))
        else:
            predict_value_obj1 = []
            predict_value_obj2 = []
            for real_pareto_each in real_pareto:
                predict_value_obj1_ = None
                predict_value_obj2_ = None
                for each_labelled, each_labelled_y1, each_labelled_y2 in zip(self.Xi, self.yi, self.yi2):
                    if real_pareto_each == each_labelled:
                        predict_value_obj1_ = each_labelled_y1
                        predict_value_obj2_ = each_labelled_y2
                        break
                if predict_value_obj1_ is None:
                    predict_value_obj1_ = (est.predict([real_pareto_each]))[0]
                    predict_value_obj2_ = (est2.predict([real_pareto_each]))[0]

                predict_value_obj1.append(predict_value_obj1_)
                predict_value_obj2.append(predict_value_obj2_)
        # print(f"predict_value_obj2={predict_value_obj2}, real_pareto_points_obj2={real_pareto_points_obj2}")
        predict_error_mape = [mean_absolute_percentage_error(y_true=real_pareto_points_obj1, y_pred=predict_value_obj1),
                              mean_absolute_percentage_error(y_true=real_pareto_points_obj2, y_pred=predict_value_obj2)]
        return predict_error_mape

    def get_accuracy(self, est, est2, labeled_predict_mode=None):
        if labeled_predict_mode is None:
            labeled_predict_mode = self.surrogate_model_dict['labeled_predict_mode']

        if labeled_predict_mode:
            try_predict_x = self.X_all
            X_all_transform = self.space.transform(try_predict_x)
            predict_value_obj1 = est.predict(X_all_transform)
            predict_value_obj2 = est2.predict(X_all_transform)
        else:
            try_predict_x = self.Xi + self.X_unlabeled
            X_all_transform = self.space.transform(self.X_unlabeled)
            try_predict_y1 = est.predict(X_all_transform)
            try_predict_y2 = est2.predict(X_all_transform)
            predict_value_obj1 = self.yi
            predict_value_obj1 = np.append(predict_value_obj1, try_predict_y1)
            predict_value_obj2 = self.yi2
            predict_value_obj2 = np.append(predict_value_obj2, try_predict_y2)

        Y_all = [CPI_metric(x) for x in try_predict_x]
        Y2_all = [power_metric(x) for x in try_predict_x]
        predict_error_mape = [mean_absolute_percentage_error(y_true=Y_all, y_pred=predict_value_obj1),
                              mean_absolute_percentage_error(y_true=Y2_all, y_pred=predict_value_obj2)]
        # predict_error_mse = [mean_squared_error(y_true=Y_all, y_pred=predict_value_obj1),
        #                     mean_squared_error(y_true=Y2_all, y_pred=predict_value_obj2)]
        return predict_error_mape  # , predict_error_mse

    def get_predict_pareto_adrs(self, est, est2, labeled_predict_mode=None):
        # print("get_predict_pareto_adrs")
        if labeled_predict_mode is None:
            labeled_predict_mode = self.surrogate_model_dict['labeled_predict_mode']

        if labeled_predict_mode:
            label = 'Predict All'
            try_predict_x = self.X_all
            X_all_transform = self.space.transform(try_predict_x)
            try_predict_y1 = est.predict(X_all_transform)
            try_predict_y2 = est2.predict(X_all_transform)
            try_predict_y = np.asarray(
                [[each_y1, each_y2] for each_y1, each_y2 in zip(try_predict_y1, try_predict_y2)])
        else:
            label = 'Predict unlabelled'
            try_predict_x = self.Xi + self.X_unlabeled
            X_all_transform = self.space.transform(self.X_unlabeled)
            try_predict_y1 = est.predict(X_all_transform)
            try_predict_y2 = est2.predict(X_all_transform)
            try_predict_y = np.asarray(
                [[each_y1, each_y2] for each_y1, each_y2 in zip(self.yi, self.yi2)])
            try_predict_y_unlabeled = np.asarray(
                [[each_y1, each_y2] for each_y1, each_y2 in zip(try_predict_y1, try_predict_y2)])
            try_predict_y = np.append(try_predict_y, try_predict_y_unlabeled, axis=0)

        learned_pareto_optimal_sets, try_new_pareto_frontier_y = is_pareto_efficient_dumb(
            try_predict_x,
            try_predict_y)
        metric_ADRS_predict = evaluate_ADRS(self.real_pareto_data[2], learned_pareto_optimal_sets)

        return metric_ADRS_predict, learned_pareto_optimal_sets, try_new_pareto_frontier_y

    def get_all_design_point(self):
        X = []
        import itertools
        var_list = [[] for i in range(8)]
        for var_index, var_iter in enumerate(self.space):
            var_list[var_index] = [value for value in range(var_iter.high + 1)]
        for each in itertools.product(var_list[0], var_list[1], var_list[2], var_list[3], var_list[4], var_list[5],
                                      var_list[6], var_list[7]):
            X.append(np.asarray(each))
        # print(f"X_length={len(X)}")
        return X

    def get_near_pareto_points(self):
        distances_pool = []
        for point in self.X_unlabeled:
            distances = []
            point_trans = self.space.transform([point])[0]
            for real_pareto_optimal in self.current_pareto_x:
                distances.append(distance_f(self.space.transform([real_pareto_optimal])[0], point_trans))
            min_distance = min(distances)
            distances_pool.append(min_distance)
        pool_index = np.argsort(distances_pool)[:self.surrogate_model_dict['cv_pool_size']]
        pool = np.asarray(self.X_unlabeled)[pool_index]
        return pool

    def register_result(self, est, est2):
        # predict_error_mape, predict_error_mse = self.get_accuracy(est, est2)
        predict_error_mape = self.get_accuracy_on_real_pareto_data(est, est2, self.real_pareto_data)
        self.predict_errors.append(predict_error_mape)
        # if skopt_verbose:
        # print(f"\tpredict_error_mape={predict_error_mape}"
        # f", predict_error_mse={predict_error_mse}"
        # )
        predict_error_unlabelled_mape = self.get_accuracy(est, est2)
        self.predict_error_unlabelled_mapes.append(predict_error_unlabelled_mape)
        # if skopt_verbose:
        # print(f"\tpredict_error_unlabelled_mape={predicpredict_error_unlabelled_mapet_error_mape}"
        # f", predict_error_mse={predict_error_mse}"
        # )

    def tell(self, x, y, y2, fit=True):
        """Record an observation (or several) of the objective function.

        Provide values of the objective function at points suggested by
        `ask()` or other points. By default a new model will be fit to all
        observations. The new model is used to suggest the next point at
        which to evaluate the objective. This point can be retrieved by calling
        `ask()`.

        To add observations without fitting a new model set `fit` to False.

        To add multiple observations in a batch pass a list-of-lists for `x`
        and a list of scalars for `y`.

        Parameters
        ----------
        x : list or list-of-lists
            Point at which objective was evaluated.

        y : scalar or list
            Value of objective at `x`.

        fit : bool, default: True
            Fit a model to observed evaluations of the objective. A model will
            only be fitted after `n_initial_points` points have been told to
            the optimizer irrespective of the value of `fit`.
        """
        check_x_in_space(x, self.space)
        self._check_y_is_valid(x, y)

        # take the logarithm of the computation times
        if "ps" in self.acq_func:
            if is_2Dlistlike(x):
                y = [[val, log(t)] for (val, t) in y]
            elif is_listlike(x):
                y = list(y)
                y[1] = log(y[1])

        return self._tell(x, y, y2, fit=fit)

    def _tell(self, x, y, y2, fit=True):
        """Perform the actual work of incorporating one or more new points.
        See `tell()` for the full description.

        This method exists to give access to the internals of adding points
        by side stepping all input validation and transformation."""

        if "ps" in self.acq_func:
            if is_2Dlistlike(x):
                print("not support now, maybe wrong")
                self.Xi.extend(x)
                self.yi.extend(y)
                self.yi2.append(y2)
                self._n_initial_points -= len(y)
            elif is_listlike(x):
                self.Xi.append(x)
                self.yi.append(y)
                self.yi2.append(y2)
                self._n_initial_points -= 1
        # if y isn't a scalar it means we have been handed a batch of points
        elif is_listlike(y) and is_2Dlistlike(x):
            print("not support now, maybe wrong")
            self.Xi.extend(x)
            self.yi.extend(y)
            self.yi2.append(y2)
            self._n_initial_points -= len(y)
        elif is_listlike(x):
            self.Xi.append(x)
            self.yi.append(y)
            self.yi2.append(y2)
            self._n_initial_points -= 1
            for X_unlabeled_index, X_unlabeled_sample in enumerate(self.X_unlabeled):
                if (x == X_unlabeled_sample).all():
                    self.X_unlabeled.pop(X_unlabeled_index)
                    break
        else:
            raise ValueError("Type of arguments `x` (%s) and `y` (%s) "
                             "not compatible." % (type(x), type(y)))

        # optimizer learned something new - discard cache
        self.cache_ = {}

        skopt_verbose = False
        skopt_train_info = False

        # if skopt_verbose and False:
        #print(f"fit={fit}, _n_initial_points={self._n_initial_points}")

        if mape_line_analysis and not self.surrogate_model_dict['model_is_iterative']:
            if self._n_initial_points <= (N_SAMPLES_ALL - N_SAMPLES_INIT):
                sample_array_several_index = np.linspace(0, N_SAMPLES_ALL-N_SAMPLES_INIT- 1, num=20, dtype=int)
                sample_partly = (-1 == self.surrogate_model_dict['tag'].find("Semi"))
                if (sample_partly) or ((not sample_partly) and (self._n_initial_points in sample_array_several_index)):
                    print(f"iter={self._n_initial_points} mape_line_analysis")
                    est = clone(self.base_estimator_)
                    est2 = clone(self.base_estimator2_)
                    Xi_transform = self.space.transform(self.Xi)
                    est.fit(Xi_transform, self.yi)
                    est2.fit(Xi_transform, self.yi2)
                    #print(est.kernel)

                    if self.surrogate_model_dict['semi_train']:
                        print("mape_line_analysis and not iterative, semi_train")
                        est, est2 = self.semi_train(X_origin=self.X_unlabeled, est=est, est2=est2)

                    metric_ADRS_predict, learned_pareto_optimal_sets, try_new_pareto_frontier_y = \
                        self.get_predict_pareto_adrs(est=est, est2=est2)
                    self.current_pareto_x, self.current_pareto_y = learned_pareto_optimal_sets, try_new_pareto_frontier_y
                    self.register_result(est=est, est2=est2)
                else:
                    metric_ADRS_predict = -1
                    self.predict_errors.append([-1, -1])
                    self.predict_error_unlabelled_mapes.append([-1, -1])

                self.func_vals.append(metric_ADRS_predict)

                if skopt_verbose:
                    print(f"\tmetric_ADRS_predict= {metric_ADRS_predict}")

                if skopt_train_info:
                    # #mean_squared_error
                    try_predict_y1_labeled = est.predict(Xi_transform)
                    try_predict_y2_labeled = est2.predict(Xi_transform)
                    score = mean_absolute_percentage_error(try_predict_y1_labeled, self.yi)
                    score2 = mean_absolute_percentage_error(try_predict_y2_labeled, self.yi2)
                    print(f"\ttrain_fit_MAPE={score, score2}")

        if self._n_initial_points == 0:
            if self.surrogate_model_dict['kernel_train']:
                if skopt_verbose:
                    print(f"\t##training kernel")
                self.base_estimator_.kernel.my_train(self.Xi, self.yi)
                self.base_estimator2_.kernel.my_train(self.Xi, self.yi2)

            if not self.surrogate_model_dict['model_is_iterative']:
                # print("_n_initial_points == 0")
                est = clone(self.base_estimator_)
                est2 = clone(self.base_estimator2_)
                Xi_transform = self.space.transform(self.Xi)
                est.fit(Xi_transform, self.yi)
                est2.fit(Xi_transform, self.yi2)

                if self.surrogate_model_dict['semi_train']:
                    print("semi_train")
                    est, est2 = self.semi_train(X_origin=self.X_unlabeled, est=est, est2=est2)

                metric_ADRS_predict, learned_pareto_optimal_sets, try_new_pareto_frontier_y = \
                    self.get_predict_pareto_adrs(est=est, est2=est2)
                self.func_vals.append(metric_ADRS_predict)
                self.current_pareto_x, self.current_pareto_y = learned_pareto_optimal_sets, try_new_pareto_frontier_y

                self.register_result(est=est, est2=est2)

                if skopt_verbose:
                    print(f"\tmetric_ADRS_predict= {metric_ADRS_predict}")

                if skopt_train_info:
                    # #mean_squared_error
                    try_predict_y1_labeled = est.predict(Xi_transform)
                    try_predict_y2_labeled = est2.predict(Xi_transform)
                    score = mean_absolute_percentage_error(try_predict_y1_labeled, self.yi)
                    score2 = mean_absolute_percentage_error(try_predict_y2_labeled, self.yi2)
                    print(f"\ttrain_fit_MAPE={score, score2}")

        if self._n_initial_points <= 0:
            y = np.asarray([[y1_value, y2_value] for y1_value, y2_value in zip(self.yi, self.yi2)])
            self.current_pareto_x, self.current_pareto_y = is_pareto_efficient_dumb(np.asarray(self.Xi), y)
            hv_last = self.hv
            if self.surrogate_model_dict['hv_scale_v']:
                if 0 == self._n_initial_points:
                    self.ref_point_scale = [max(self.yi), max(self.yi2)]
                current_pareto_y_scale = scale_for_hv(self.current_pareto_y, self.ref_point_scale)
                self.hv = DominatedPartitioning(ref_point=torch.Tensor([-1,-1]),
                                                Y=0 - torch.Tensor(current_pareto_y_scale)).compute_hypervolume().item()
            else:
                self.hv = DominatedPartitioning(ref_point=0 - self.ref_point,
                                                Y=0 - torch.Tensor(self.current_pareto_y)).compute_hypervolume().item()
            self.statistics['hv'].append(self.hv)
            if (self.hv - hv_last) < 0.00001:
                self.hv_last_iter_num += 1
                #print(f"hv_last_iter_num={self.hv_last_iter_num}")
                self.statistics['hv_last_iter'] += 1
            else:
                self.hv_last_iter_num = 0
            # if self.print_info:
            # print(f"\thv={self.hv} ({self.hv - hv_last})")
            # print(f"\t_n_initial_points={self._n_initial_points}")
            # print(f"\thv={self.hv}")
            # print(f"X={self.Xi}")
            # print(f"y={y}")
            # print(f"pareto_x={self.current_pareto_x}, pareto_y={self.current_pareto_y}")

            # if (0.2 < (-self._n_initial_points / N_SAMPLES_ALL)):# or (2 < self.hv_last_iter_num):
            ucb_enable = False
            if self.surrogate_model_dict['ucb']:
                if (11 == self.surrogate_model_dict['ucb_v']) and (0 < self.hv_last_iter_num):
                    ucb_enable = True
                elif 12 <= self.surrogate_model_dict['ucb_v'] <= 20:
                    ucb_enable = True
                elif (21 == self.surrogate_model_dict['ucb_v']) and ((-self._n_initial_points / N_SAMPLES_ALL) < 0.2):
                    ucb_enable = True
                elif (22 == self.surrogate_model_dict['ucb_v']) and ((-self._n_initial_points / N_SAMPLES_ALL) < 0.2):
                    ucb_enable = True
                elif (23 == self.surrogate_model_dict['ucb_v']) and ((-self._n_initial_points / N_SAMPLES_ALL) < 0.2):
                    ucb_enable = True
                elif (24 == self.surrogate_model_dict['ucb_v']) and ((-self._n_initial_points / N_SAMPLES_ALL) < 0.2):
                    ucb_enable = True
                elif (25 == self.surrogate_model_dict['ucb_v']) and ((-self._n_initial_points / N_SAMPLES_ALL) < 0.2):
                    ucb_enable = True
                elif (27 == self.surrogate_model_dict['ucb_v']) and ((-self._n_initial_points / N_SAMPLES_ALL) < 0.1):
                    ucb_enable = True

            if ucb_enable:
                # self.surrogate_model_dict['ucb'] *= self.surrogate_model_dict['ucb_scale']
                if 17 == self.surrogate_model_dict['ucb_v']:
                    self.surrogate_model_dict['ucb-iter'] = self.surrogate_model_dict['ucb']
                elif 18 == self.surrogate_model_dict['ucb_v']:
                    self.surrogate_model_dict['ucb-iter'] = self.surrogate_model_dict['ucb']
                elif 23 == self.surrogate_model_dict['ucb_v']:
                    self.surrogate_model_dict['ucb-iter'] = self.surrogate_model_dict['ucb']
                elif 24 == self.surrogate_model_dict['ucb_v']:
                    self.surrogate_model_dict['ucb-iter'] = -self.surrogate_model_dict['ucb']
                elif 25 == self.surrogate_model_dict['ucb_v']:
                    self.surrogate_model_dict['ucb-iter'] = -self.surrogate_model_dict['ucb']
                elif 27 == self.surrogate_model_dict['ucb_v']:
                    self.surrogate_model_dict['ucb-iter'] = self.surrogate_model_dict['ucb']                    
                elif 0.0 < self.surrogate_model_dict['ucb']:
                    self.surrogate_model_dict['ucb-iter'] = self.surrogate_model_dict['ucb'] \
                                                            * np.sqrt(np.log(2 - self._n_initial_points))
                else:
                    self.surrogate_model_dict['ucb-iter'] = self.surrogate_model_dict['ucb'] \
                                                            / np.sqrt(np.log(2 - self._n_initial_points))
            else:
                self.surrogate_model_dict['ucb-iter'] = 0

        # after being "told" n_initial_points we switch from sampling
        # random points to using a surrogate model
        if (fit and self._n_initial_points <= 0 and self.surrogate_model_dict['model_is_iterative'] and
                self.base_estimator_ is not None):
            transformed_bounds = np.array(self.space.transformed_bounds)
            est = clone(self.base_estimator_)
            est2 = clone(self.base_estimator2_)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                Xi_transform = self.space.transform(self.Xi)
                est.fit(Xi_transform, self.yi)
                est2.fit(Xi_transform, self.yi2)
                # if False:
                #    print(
                #        f"\test={est.kernel_}, theta={est.kernel_.theta}, likelihood,grad={est.log_marginal_likelihood(est.kernel_.theta, eval_gradient=True)}")
                #    print(f"\test2={est2.kernel_}, theta={est2.kernel_.theta}")
                if skopt_train_info:
                    # print(f"\treal_y={self.yi, self.yi2}")
                    score = est.score(Xi_transform, self.yi)
                    score2 = est2.score(Xi_transform, self.yi2)
                    print(f"\ttrain_fit_score={score, score2}")

            if hasattr(self, "next_xs_") and self.acq_func == "gp_hedge":
                self.gains_ -= est.predict(np.vstack(self.next_xs_))

            if self.max_model_queue_size is None:
                self.models.append(est)
            elif len(self.models) < self.max_model_queue_size:
                self.models.append(est)
            else:
                # Maximum list size obtained, remove oldest model.
                self.models.pop(0)
                self.models.append(est)

            self.next_xs_ = []
            for cand_acq_func in self.cand_acq_funcs_:
                # Find the minimum of the acquisition function by randomly
                # sampling points from the space
                if self.acq_optimizer == "full":
                    X_origin = self.X_unlabeled
                    random.shuffle(X_origin)
                    # print(f"X_origin={X_origin}")

                    non_uniformity = -1 #evaluate_non_uniformity(learned_pareto_optimal_sets_y_unsort=self.current_pareto_y)
                    self.non_uniformitys.append(non_uniformity)

                    # if 3 < self.hv_last_iter_num:
                    if ((self.surrogate_model_dict['semi_train'] == -1)
                            or ((1000 < self.surrogate_model_dict['semi_train']) and (
                                    self.surrogate_model_dict['semi_train'] - 1000 < self.adrs_last_iter_num))
                            or ((0 < self.surrogate_model_dict['semi_train']) and (
                                    0 == (self._n_initial_points % self.surrogate_model_dict['semi_train'])))
                    ):
                        print(f"semi_train={self.surrogate_model_dict['semi_train']}")
                        est, est2 = self.semi_train(X_origin=self.X_unlabeled, est=est, est2=est2)

                    acq_func = self.acq_func
                    cv_ranking = self.surrogate_model_dict['cv_ranking']
                    pool = X_origin[:self.surrogate_model_dict['cv_pool_size']]
                    # if (0.2 < (-self._n_initial_points / N_SAMPLES_ALL)):
                    # or (2 < self.hv_last_iter_num): cvbetav10
                    cv_enable = False
                    if self.surrogate_model_dict[
                        'cv_ranking_beta']:  # and (np.random.rand() < self.surrogate_model_dict['cv_ranking_beta']):
                        if (12 == self.surrogate_model_dict['cv_ranking_beta_v']) and (0 < self.hv_last_iter_num):
                            cv_enable = True

                    if cv_enable:
                        pool = self.get_near_pareto_points()
                        acq_func = "cv_ranking"
                        cv_ranking = "maxsort"
                        self.statistics['cv_ranking_beta'] += 1
                        # print("#cv_ranking beta")

                    #if (self.surrogate_model_dict['non_uniformity_explore']):#  and (0 < self.hv_last_iter_num):
                    #    acq_func = "non_uniformity_explore"

                    if acq_func == "cv_ranking":
                        next_x_origin, next_x_index, _, _ = self.cv_ranking(
                            X_origin=pool,
                            est=est,
                            est2=est2,
                            metric_filter=False,
                            cv_ranking=cv_ranking,
                        )
                    elif acq_func == "non_uniformity_explore":
                        next_x_index = self.get_non_uniformity_explore(X_origin, est, est2)
                        next_x_origin = X_origin[next_x_index]
                    elif acq_func == "EHVI":
                        next_x_index, next_y, next_y2, predict_hv = self.get_ehvi(X_origin, est, est2)
                        next_x_origin = X_origin[next_x_index]
                    else:
                        next_x_index = self.get_hvi(X_origin, est, est2)
                        next_x_origin = X_origin[next_x_index]
                    next_x = self.space.transform([next_x_origin])

                    last_adrs = self.func_vals[-1] if self._n_initial_points < 0 else 99

                    metric_ADRS, coverage_percent = evaluate_ADRS(real_pareto_optimal_sets=self.real_pareto_data[2],
                                                                  learned_pareto_optimal_sets=self.current_pareto_x,
                                                                  coverage=True)


                    # print(f"coverage_percent={coverage_percent}")
                    if (- self._n_generation_points) == self._n_initial_points:
                        # last iteration and add semi_train phase
                        # print(f"_n_initial_points={self._n_initial_points} semi_train debug")
                        print(f"\tbefore predict_last ADRS= {metric_ADRS} ")
                        if self.surrogate_model_dict['semi_train']:
                            est, est2 = self.semi_train(X_origin=self.X_unlabeled, est=est, est2=est2)
                        metric_ADRS, learned_pareto_optimal_sets, try_new_pareto_frontier_y = \
                            self.get_predict_pareto_adrs(est=est, est2=est2)
                        self.current_pareto_x = learned_pareto_optimal_sets
                        self.current_pareto_y = try_new_pareto_frontier_y
                        non_uniformity = evaluate_non_uniformity(
                            learned_pareto_optimal_sets_y_unsort=self.current_pareto_y)
                        try:
                            self.feature_importances = [est.feature_importances_ * 100, est2.feature_importances_ * 100]
                        except:
                            self.feature_importances = None
                        # print(f"[OUT] feature_importances= {self.feature_importances}")

                        if ((- self._n_generation_points) == self._n_initial_points) \
                                and plot_pareto \
                                and ("-exp0" in self.surrogate_model_dict["tag"]):
                            # global plot_pareto
                            # plot_pareto = False
                            import matplotlib.pyplot as plt
                            plt.scatter(self.yi, self.yi2, c='Gray', s=4, label="Sampled")
                            plt.scatter(try_new_pareto_frontier_y[:, 0], try_new_pareto_frontier_y[:, 1], c='Blue', s=6,
                                        label="Predicted")
                            plt.scatter(self.real_pareto_data[0], self.real_pareto_data[1], c='red', s=3, marker='+',
                                        label='Real')
                            plt.ylabel("Power", fontsize=12, font=font)
                            plt.xlabel("CPI", fontsize=12, font=font)
                            plt.legend()
                            plt.savefig(
                                "fig_pareto/pareto-" + case_name + '-' + self.surrogate_model_dict['tag'] + ".png")
                            #plt.close()
                            plt.close('all')
                            # plt.show()

                    if (last_adrs - metric_ADRS) < 0.001:
                        # print(f"last_adrs={last_adrs} - metric_ADRS={metric_ADRS} = {last_adrs - metric_ADRS}")
                        self.adrs_last_iter_num += 1
                    else:
                        self.adrs_last_iter_num = 0

                    self.func_vals.append(metric_ADRS)
                    self.coverage.append(coverage_percent)
                    if skopt_verbose:
                        print(f"\tADRS={metric_ADRS}")

                    self.register_result(est=est, est2=est2)

                elif self.acq_optimizer == "sampling":
                    print("error !!!!! no implement now")
                    exit(1)
                    X = self.space.transform(self.space.rvs(
                        n_samples=self.n_points, random_state=self.rng))

                # lbfgs should handle this but just in case there are
                # precision errors.
                if not self.space.is_categorical:
                    next_x = np.clip(
                        next_x, transformed_bounds[:, 0],
                        transformed_bounds[:, 1])
                self.next_xs_.append(next_x)

            if self.acq_func == "gp_hedge":
                logits = np.array(self.gains_)
                logits -= np.max(logits)
                exp_logits = np.exp(self.eta * logits)
                probs = exp_logits / np.sum(exp_logits)
                next_x = self.next_xs_[np.argmax(self.rng.multinomial(1, probs))]
            else:
                next_x = self.next_xs_[0]

            # note the need for [0] at the end
            self._next_x = self.space.inverse_transform(
                next_x.reshape((1, -1)))[0]

        # Pack results
        result = create_result_multiobj(self.Xi, self.yi, self.yi2,
                                        self.current_pareto_x, self.func_vals,
                                        self.predict_errors, self.predict_error_unlabelled_mapes,
                                        self.non_uniformitys, self.coverage,
                                        self.space, self.rng,
                                        models=self.models)

        result.specs = self.specs
        return result

    def _check_y_is_valid(self, x, y):
        """Check if the shape and types of x and y are consistent."""

        if "ps" in self.acq_func:
            if is_2Dlistlike(x):
                if not (np.ndim(y) == 2 and np.shape(y)[1] == 2):
                    raise TypeError("expected y to be a list of (func_val, t)")
            elif is_listlike(x):
                if not (np.ndim(y) == 1 and len(y) == 2):
                    raise TypeError("expected y to be (func_val, t)")

        # if y isn't a scalar it means we have been handed a batch of points
        elif is_listlike(y) and is_2Dlistlike(x):
            for y_value in y:
                if not isinstance(y_value, Number):
                    raise ValueError("expected y to be a list of scalars")

        elif is_listlike(x):
            if not isinstance(y, Number):
                raise ValueError("`func` should return a scalar")

        else:
            raise ValueError("Type of arguments `x` (%s) and `y` (%s) "
                             "not compatible." % (type(x), type(y)))

    def run(self, func, func2, n_iter=1):
        """Execute ask() + tell() `n_iter` times"""
        for _ in range(n_iter):
            x = self.ask()
            self.tell(x, func(x), func2(x))

        result = create_result_multiobj(self.Xi, self.yi, self.yi2,
                                        self.current_pareto_x, self.func_vals,
                                        self.predict_errors, self.predict_error_unlabelled_mapes,
                                        self.non_uniformitys, self.coverage,
                                        self.space, self.rng,
                                        models=self.models)
        result.specs = self.specs
        return result

    def update_next(self):
        """Updates the value returned by opt.ask(). Useful if a parameter
        was updated after ask was called."""
        self.cache_ = {}
        # Ask for a new next_x.
        # We only need to overwrite _next_x if it exists.
        if hasattr(self, '_next_x'):
            opt = self.copy(random_state=self.rng)
            self._next_x = opt._next_x

    def get_result(self):
        """Returns the same result that would be returned by opt.tell()
        but without calling tell

        Returns
        -------
        res : `OptimizeResult`, scipy object
            OptimizeResult instance with the required information.

        """
        result = create_result_multiobj(self.Xi, self.yi, self.yi2,
                                        self.current_pareto_x, self.func_vals,
                                        self.predict_errors, self.predict_error_unlabelled_mapes,
                                        self.non_uniformitys, self.coverage,
                                        self.space, self.rng,
                                        models=self.models)
        result.specs = self.specs
        return result
