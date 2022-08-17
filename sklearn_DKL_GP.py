import numpy as np
import torch
from sklearn.gaussian_process.kernels import Matern, Hyperparameter
from sklearn.neural_network import MLPRegressor

from ANN_model import Loss_Fun, MLP_Predictor


class LargeFeatureExtractor(torch.nn.Sequential):
    def __init__(self):
        super(LargeFeatureExtractor, self).__init__()
        data_dim = 8
        self.output_dim = 2
        self.add_module('linear1', torch.nn.Linear(data_dim, 1000))
        self.add_module('relu1', torch.nn.ReLU())
        self.add_module('linear2', torch.nn.Linear(1000, 500))
        self.add_module('relu2', torch.nn.ReLU())
        self.add_module('linear3', torch.nn.Linear(500, 50))
        self.add_module('relu3', torch.nn.ReLU())
        self.add_module('linear4', torch.nn.Linear(50, self.output_dim))


class Sklearn_DKL_GP(Matern):
    def __init__(self, length_scale=1.0, length_scale_bounds=(1e-5, 1e5), nu=1.5):
        # print(f"debugging Sklearn_DKL_GP init length_scale={length_scale}")
        super().__init__(length_scale=length_scale, length_scale_bounds=length_scale_bounds, nu=nu)
        self.mlp_layers = 3
        self.train_flag = False
        self.use_sklearn_mlp = False
        if self.use_sklearn_mlp:
            self.mlp = MLPRegressor(hidden_layer_sizes=(10, 1),
                                    max_iter=2,
                                    solver='sgd',  # ['adam', 'sgd', 'lbfgs'],
                                    activation='relu',
                                    )
        else:
            self.mlp = MLP_Predictor(
                in_channel=8
                , out_channel=1
                , drop_rate=0.01, use_bias=True, use_drop=False
                , initial_lr=0.001
                , momentum=0.4
                , loss_fun=torch.nn.MSELoss()  # Loss_Fun()
            )

    '''
    @property
    def hyperparameter_mlp_weight(self):
        return Hyperparameter("mlp_weight", "numeric", self.mlp_weight_bounds)
    '''

    def __call__(self, X, Y=None, eval_gradient=False):
        if self.use_sklearn_mlp:
            x_mlp = self.mlp.predict(X)
        else:
            x_mlp = self.mlp.predict(torch.Tensor(X)).detach().numpy()
        #print(f"x_mlp={x_mlp}")
        if Y is None:
            y_mlp = None
        else:
            y_mlp = self.mlp.predict(torch.Tensor(Y)).detach().numpy()
        if eval_gradient:
            K, K_gradient = super().__call__(x_mlp, y_mlp, eval_gradient=eval_gradient)
            # print(f"debugging Sklearn_DKL_GP X size= {np.shape(X)} x_mlp size ={np.shape(x_mlp)}")
            # print(f"K_gradient {K_gradient}")
            return K, K_gradient
        else:
            K = super().__call__(x_mlp, y_mlp, eval_gradient=eval_gradient)
            return K

    def __repr__(self):
        if self.anisotropic:
            return "{0}(length_scale=[{1}], nu={2:.3g}, mlp_layers={3})".format(
                self.__class__.__name__,
                ", ".join(map("{0:.3g}".format, self.length_scale)),
                self.nu,
                self.mlp_layers,
            )
        else:
            return "{0}(length_scale={1:.3g}, nu={2:.3g}, mlp_layers={3} train_flag={4})".format(
                self.__class__.__name__, np.ravel(self.length_scale)[0], self.nu, self.mlp_layers, self.train_flag
            )

    def my_train(self, X, X_value):
        if self.use_sklearn_mlp:
            self.mlp.fit(X, X_value)
        else:
            self.mlp.my_train(X, X_value)
        self.train_flag = True
