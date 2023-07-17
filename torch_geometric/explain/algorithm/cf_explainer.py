from math import sqrt
from typing import Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor
from torch.nn.parameter import Parameter
import torch.nn.functional as F


from torch_geometric.explain import ExplainerConfig, Explanation, ModelConfig
from torch_geometric.explain.algorithm import ExplainerAlgorithm
from torch_geometric.explain.algorithm.utils import clear_masks, set_masks
from torch_geometric.explain.config import MaskType, ModelMode, ModelTaskLevel
from torch_geometric.utils import to_dense_adj
#from torch_geometric.utils import dense_adjacency


class CFExplainer(ExplainerAlgorithm):
    r"""The CF-Explainer model from the `"CF-GNNExplainer: Counterfactual Explanations for Graph Neural
Networks"
    <https://arxiv.org/abs/2102.03322>`_ paper for generating CF explanations for GNNs: 
    the minimal perturbation to the input (graph) data such that the prediction changes.

    .. note::

        For an example of using :class:`GNNExplainer`, see
        `examples/explain/gnn_explainer.py <https://github.com/pyg-team/
        pytorch_geometric/blob/master/examples/explain/gnn_explainer.py>`_,
        `examples/explain/gnn_explainer_ba_shapes.py <https://github.com/
        pyg-team/pytorch_geometric/blob/master/examples/
        explain/gnn_explainer_ba_shapes.py>`_, and `examples/explain/
        gnn_explainer_link_pred.py <https://github.com/pyg-team/
        pytorch_geometric/blob/master/examples/explain/gnn_explainer_link_pred.py>`_.

    Args:
        epochs (int, optional): The number of epochs to train.
            (default: :obj:`100`)
        lr (float, optional): The learning rate to apply.
            (default: :obj:`0.01`)
        **kwargs (optional): Additional hyper-parameters to override default
            settings in
            :attr:`~torch_geometric.explain.algorithm.GNNExplainer.coeffs`.
    """

    coeffs = {
        'edge_size': 0.005,
        'edge_reduction': 'sum',
        'beta' : .001,
        'node_feat_size': 1.0,
        'node_feat_reduction': 'mean',
        'edge_ent': 1.0,
        'node_feat_ent': 0.1,
        'EPS': 1e-15,
    }

    def __init__(self, epochs: int = 100, lr: float = 0.01, cf_optimizer = "SGD", n_momentum = 0, **kwargs):
        super().__init__()
        self.epochs = epochs
        self.lr = lr
        self.cf_optimizer = cf_optimizer
        self.n_momentum = n_momentum
        self.coeffs.update(kwargs)
        self.node_mask = self.hard_node_mask = None
        self.edge_mask = self.hard_edge_mask = None
        self.best_cf_example = None
        self.best_loss = np.inf
        
    def forward(
        self,
        model: torch.nn.Module,
        x: Tensor,
        edge_index: Tensor,
        *,
        index: int = None,
        **kwargs,
    ) -> Explanation:
        if isinstance(x, dict) or isinstance(edge_index, dict):
            raise ValueError(f"Heterogeneous graphs not yet supported in "
                             f"'{self.__class__.__name__}'")

        self._train(model, x, edge_index, index=index, **kwargs)

        # node_mask = self._post_process_mask(
        #     self.best_cf_example[0],
        #     self.hard_node_mask,
        #     apply_sigmoid=True,
        # )
        node_mask = self._post_process_mask(
            self.node_mask,
            self.hard_node_mask,
            apply_sigmoid=True,
        )
        edge_mask = self._post_process_mask(
            self.edge_mask,
            self.hard_edge_mask,
            apply_sigmoid=True,
        )

        self._clean_model(model)

        return Explanation(node_mask=node_mask, edge_mask=edge_mask)

    def supports(self) -> bool:
        return True
    
    def _train(
        self,
        model: torch.nn.Module,
        x: Tensor,
        edge_index: Tensor,
        *,
        target: Tensor,
        index: int = None,
        **kwargs,
    ):
        self._initialize_masks(x, edge_index)
        parameters = []
        if self.node_mask is not None:
            parameters.append(self.node_mask)
        if self.edge_mask is not None:
            set_masks(model, self.edge_mask, edge_index, apply_sigmoid=True)
            parameters.append(self.edge_mask)
    
        if self.cf_optimizer == "SGD" and self.n_momentum == 0.0:
            optimizer = torch.optim.SGD(parameters, lr=self.lr)
        elif self.cf_optimizer == "SGD" and self.n_momentum != 0.0:
            optimizer = torch.optim.SGD(parameters, lr=self.lr, nesterov=True, momentum=n_momentum)
        elif self.cf_optimizer == "Adadelta":
            optimizer = torch.optim.Adadelta(parameters, lr=self.lr)
        else:
            raise Exception("Optimizer is not currently supported.")
        
        original_prediction  = model(x, edge_index, **kwargs)
        for i in range(self.epochs):
            optimizer.zero_grad()
            h = x # if self.node_mask is None else x * self.node_mask.sigmoid()
            #discrete_edge_mask = torch.where(torch.sigmoid(self.edge_mask)>=0.5, 1, 0)
            set_masks(model, self.edge_mask, edge_index, apply_sigmoid=True)
            y_hat, y = model(h, edge_index, **kwargs), original_prediction
            y_hat_discrete, y_discrete = y_hat.argmax(dim=-1), y.argmax(dim=-1)
            set_masks(model, self.edge_mask, edge_index, apply_sigmoid=True)

            if index is not None:
                y_hat, y = y_hat[index], y[index]

            loss = self._loss(y_hat, y, edge_index, index=index)
            
            loss.backward()
            optimizer.step()

            # In the first iteration, we collect the nodes and edges that are
            # involved into making the prediction. These are all the nodes and
            # edges with gradient != 0 (without regularization applied).
            if i == 0 and self.node_mask is not None:
                if self.node_mask.grad is None:
                    raise ValueError("Could not compute gradients for node "
                                     "features. Please make sure that node "
                                     "features are used inside the model or "
                                     "disable it via `node_mask_type=None`.")
                self.hard_node_mask = self.node_mask.grad != 0.0
            if i == 0 and self.edge_mask is not None:
                if self.edge_mask.grad is None:
                    raise ValueError("Could not compute gradients for edges. "
                                     "Please make sure that edges are used "
                                     "via message passing inside the model or "
                                     "disable it via `edge_mask_type=None`.")
                self.hard_edge_mask = self.edge_mask.grad != 0.0

    def _initialize_masks(self, x: Tensor, edge_index: Tensor):
        node_mask_type = None
        edge_mask_type = self.explainer_config.edge_mask_type

        device = x.device
        (N, F), E = x.size(), edge_index.size(1)

        if node_mask_type is None:
            self.node_mask = None
        elif node_mask_type == MaskType.object:
            self.node_mask = Parameter(torch.ones(N, 1, device=device))
        elif node_mask_type == MaskType.attributes:
            self.node_mask = Parameter(torch.ones(N, F, device=device))
        elif node_mask_type == MaskType.common_attributes:
            self.node_mask = Parameter(torch.ones(1, F, device=device))
        else:
            assert False

        if edge_mask_type is None:
            self.edge_mask = None
        elif edge_mask_type == MaskType.object:
            self.edge_mask = Parameter(torch.ones(E, device=device))
        else:
            assert False

    # def _initialize_masks(self, x: Tensor, edge_index: Tensor):
    #     node_mask_type = self.explainer_config.node_mask_type
    #     edge_mask_type = self.explainer_config.edge_mask_type

    #     device = x.device
    #     (N, F), E = x.size(), edge_index.size(1)

    #     if node_mask_type is None:
    #         self.node_mask = None
    #     elif node_mask_type == MaskType.object:
    #         self.node_mask = Parameter(torch.ones(N, 1, device=device))
    #     elif node_mask_type == MaskType.attributes:
    #         self.node_mask = Parameter(torch.ones(N, F, device=device))
    #     elif node_mask_type == MaskType.common_attributes:
    #         self.node_mask = Parameter(torch.ones(1, F, device=device))
    #     else:
    #         assert False


    #     if edge_mask_type is None:
    #         self.edge_mask = None
    #     elif edge_mask_type == MaskType.object:
    #         self.edge_mask = Parameter(torch.ones(E, device=device))
    #     else:
    #         assert False


    def _loss(self, y_hat: Tensor, y: Tensor, edge_index, index) -> Tensor:
        y_hat_discrete = torch.argmax(y_hat, dim=-1)  # Compute argmax along the class dimension
        y_discrete = torch.argmax(y, dim=-1)  # Compute argmax along the class dimension

        pred_same = (y_hat_discrete == y_discrete).float()
        if self.model_config.mode == ModelMode.binary_classification:
            loss_pred = - self._loss_binary_classification(y_hat, y.long())
        elif self.model_config.mode == ModelMode.multiclass_classification:
            loss_pred = - self._loss_multiclass_classification(y_hat, y.long())
#         elif self.model_config.mode == ModelMode.regression:
#             loss_pred = - self._loss_regression(y_hat, y)
        else:
            assert False

        discrete_edge_mask = torch.where(torch.sigmoid(self.edge_mask) > 0.5, torch.tensor(1.0), torch.tensor(0.0))
        new_edge_index = edge_index * discrete_edge_mask

        loss_graph_dist = torch.sum(torch.abs(new_edge_index - edge_index)) / 2

        loss_total = pred_same * loss_pred + 10000* self.coeffs['beta'] * loss_graph_dist
        print("loss_total", loss_total)

        return loss_total

    def _clean_model(self, model):
        clear_masks(model)
        self.node_mask = self.hard_node_mask = None
        self.edge_mask = self.hard_edge_mask = None
