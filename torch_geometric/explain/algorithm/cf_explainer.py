from math import sqrt
from typing import Optional, Tuple, Union

import torch
from torch import Tensor
from torch.nn.parameter import Parameter

from torch_geometric.explain import ExplainerConfig, Explanation, ModelConfig
from torch_geometric.explain.algorithm import ExplainerAlgorithm
from torch_geometric.explain.algorithm.utils import clear_masks, set_masks
from torch_geometric.explain.config import MaskType, ModelMode, ModelTaskLevel
from torch_geometric.explain.algorithm.GCNSyntheticPerturb import GCNSyntheticPerturb


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

    def forward(
        self,
        model: torch.nn.Module,
        x: Tensor,
        edge_index: Tensor,
        *,
        target: Tensor,
        index: Optional[Union[int, Tensor]] = None,
        **kwargs,
    ) -> Explanation:
        if isinstance(x, dict) or isinstance(edge_index, dict):
            raise ValueError(f"Heterogeneous graphs not yet supported in "
                             f"'{self.__class__.__name__}'")

        self._train(model, x, edge_index, target=target, index=index, **kwargs)

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
        index: Optional[Union[int, Tensor]] = None,
        cf_optimizer = "SGD",
        **kwargs,
    ):
        self._initialize_masks(x, edge_index)

        parameters = []
        if self.node_mask is not None:
            parameters.append(self.node_mask)
        if self.edge_mask is not None:
            set_masks(model, self.edge_mask, edge_index, apply_sigmoid=True)
            parameters.append(self.edge_mask)
        # self.node_idx = node_idx
		# self.new_idx = new_idx

		# self.x = self.sub_feat
		# self.A_x = self.sub_adj
		# self.D_x = get_degree_matrix(self.A_x)

        if self.cf_optimizer == "SGD" and n_momentum == 0.0:
            self.cf_optimizer = optim.SGD(self.cf_model.parameters(), lr=self.lr)
        elif self.cf_optimizer == "SGD" and n_momentum != 0.0:
            self.cf_optimizer = optim.SGD(self.cf_model.parameters(), lr=self.lr, nesterov=True, momentum=n_momentum)
        elif self.cf_optimizer == "Adadelta":
            self.cf_optimizer = optim.Adadelta(self.cf_model.parameters(), lr=self.lr)
        else:
            self.cf_optimizer = optim.SGD(self.cf_model.parameters(), lr=self.lr)
        best_cf_example = []
        best_loss = np.inf
        num_cf_examples = 0
        for epoch in range(self.epochs):
            new_example, loss_total = self.train(epoch)
            model.train()
            self.cf_optimizer.zero_grad()

		    # output uses differentiable P_hat ==> adjacency matrix not binary, but needed for training
		    # output_actual uses thresholded P ==> binary adjacency matrix ==> gives actual prediction
            output = model.forward(self.x, self.A_x)
            output_actual, self.P = model.forward_prediction(self.x)

		    # Need to use new_idx from now on since sub_adj is reindexed
            y_pred_new = torch.argmax(output[self.new_idx])
            y_pred_new_actual = torch.argmax(output_actual[self.new_idx])
            # loss_pred indicator should be based on y_pred_new_actual NOT y_pred_new!
            loss_total, loss_pred, loss_graph_dist, cf_adj = self.cf_model.loss(output[self.new_idx], self.y_pred_orig, y_pred_new_actual)
            loss_total.backward()
            clip_grad_norm(self.cf_model.parameters(), 2.0)
            self.cf_optimizer.step()
            print('Node idx: {}'.format(self.node_idx),
		      'New idx: {}'.format(self.new_idx),
			  'Epoch: {:04d}'.format(epoch + 1),
		      'loss: {:.4f}'.format(loss_total.item()),
		      'pred loss: {:.4f}'.format(loss_pred.item()),
		      'graph loss: {:.4f}'.format(loss_graph_dist.item()))
            print('Output: {}\n'.format(output[self.new_idx].data),
		      'Output nondiff: {}\n'.format(output_actual[self.new_idx].data),
		      'orig pred: {}, new pred: {}, new pred nondiff: {}'.format(self.y_pred_orig, y_pred_new, y_pred_new_actual))
            print(" ")
            cf_stats = []
            if y_pred_new_actual != self.y_pred_orig:
                cf_stats = [self.node_idx.item(), self.new_idx.item(),
                            cf_adj.detach().numpy(), self.sub_adj.detach().numpy(),
                            self.y_pred_orig.item(), y_pred_new.item(),
                            y_pred_new_actual.item(), self.sub_labels[self.new_idx].numpy(),
                            self.sub_adj.shape[0], loss_total.item(),
                            loss_pred.item(), loss_graph_dist.item()]
            if new_example != [] and loss_total < best_loss:
                best_cf_example.append(new_example)
                best_loss = loss_total
                num_cf_examples += 1
        print("{} CF examples for node_idx = {}".format(num_cf_examples, self.node_idx))
        print(" ")
        return(best_cf_example)

        # for i in range(self.epochs):
        #     cf_optimizer.zero_grad()

        #     h = x if self.node_mask is None else x * self.node_mask.sigmoid()
        #     y_hat, y = model(h, edge_index, **kwargs), target

        #     if index is not None:
        #         y_hat, y = y_hat[index], y[index]

        #     loss = self._loss(y_hat, y)

        #     loss.backward()
        #     optimizer.step()

        #     # In the first iteration, we collect the nodes and edges that are
        #     # involved into making the prediction. These are all the nodes and
        #     # edges with gradient != 0 (without regularization applied).
        #     if i == 0 and self.node_mask is not None:
        #         self.hard_node_mask = self.node_mask.grad != 0.0
        #     if i == 0 and self.edge_mask is not None:
        #         self.hard_edge_mask = self.edge_mask.grad != 0.0

    def _initialize_masks(self, x: Tensor, edge_index: Tensor):
        node_mask_type = self.explainer_config.node_mask_type
        edge_mask_type = self.explainer_config.edge_mask_type

        device = x.device
        (N, F), E = x.size(), edge_index.size(1)

        std = 0.1
        if node_mask_type is None:
            self.node_mask = None
        elif node_mask_type == MaskType.object:
            self.node_mask = Parameter(torch.randn(N, 1, device=device) * std)
        elif node_mask_type == MaskType.attributes:
            self.node_mask = Parameter(torch.randn(N, F, device=device) * std)
        elif node_mask_type == MaskType.common_attributes:
            self.node_mask = Parameter(torch.randn(1, F, device=device) * std)
        else:
            assert False

        if edge_mask_type is None:
            self.edge_mask = None
        elif edge_mask_type == MaskType.object:
            std = torch.nn.init.calculate_gain('relu') * sqrt(2.0 / (2 * N))
            self.edge_mask = Parameter(torch.randn(E, device=device) * std)
        else:
            assert False

    # def _loss(self, y_hat: Tensor, y: Tensor) -> Tensor:
    #     if self.model_config.mode == ModelMode.binary_classification:
    #         loss = self._loss_binary_classification(y_hat, y)
    #     elif self.model_config.mode == ModelMode.multiclass_classification:
    #         loss = self._loss_multiclass_classification(y_hat, y)
    #     elif self.model_config.mode == ModelMode.regression:
    #         loss = self._loss_regression(y_hat, y)
    #     else:
    #         assert False

    #     if self.hard_edge_mask is not None:
    #         assert self.edge_mask is not None
    #         m = self.edge_mask[self.hard_edge_mask].sigmoid()
    #         edge_reduce = getattr(torch, self.coeffs['edge_reduction'])
    #         loss = loss + self.coeffs['edge_size'] * edge_reduce(m)
    #         ent = -m * torch.log(m + self.coeffs['EPS']) - (
    #             1 - m) * torch.log(1 - m + self.coeffs['EPS'])
    #         loss = loss + self.coeffs['edge_ent'] * ent.mean()

    #     if self.hard_node_mask is not None:
    #         assert self.node_mask is not None
    #         m = self.node_mask[self.hard_node_mask].sigmoid()
    #         node_reduce = getattr(torch, self.coeffs['node_feat_reduction'])
    #         loss = loss + self.coeffs['node_feat_size'] * node_reduce(m)
    #         ent = -m * torch.log(m + self.coeffs['EPS']) - (
    #             1 - m) * torch.log(1 - m + self.coeffs['EPS'])
    #         loss = loss + self.coeffs['node_feat_ent'] * ent.mean()

    #     return loss
    def _loss(self, output, y_pred_orig, y_pred_new_actual):
        pred_same = (y_pred_new_actual == y_pred_orig).float()

		# Need dim >=2 for F.nll_loss to work
        output = output.unsqueeze(0)
        y_pred_orig = y_pred_orig.unsqueeze(0)

        if self.edge_additions:
            cf_adj = self.P
        else:
            cf_adj = self.P * self.adj
        cf_adj.requires_grad = True  # Need to change this otherwise loss_graph_dist has no gradient

		# Want negative in front to maximize loss instead of minimizing it to find CFs
        loss_pred = - F.nll_loss(output, y_pred_orig)
        loss_graph_dist = sum(sum(abs(cf_adj - self.adj))) / 2      # Number of edges changed (symmetrical)

        # Zero-out loss_pred with pred_same if prediction flips
        loss_total = pred_same * loss_pred + self.beta * loss_graph_dist
        return loss_total, loss_pred, loss_graph_dist, cf_adj

    def _clean_model(self, model):
        clear_masks(model)
        self.node_mask = self.hard_node_mask = None
        self.edge_mask = self.hard_edge_mask = None