from abc import abstractmethod
from dataclasses import dataclass
from typing import Dict, Iterator, List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import PeftModel
from torch.func import functional_call, grad, vmap
from torch.nn import CrossEntropyLoss
from transformers import WhisperForConditionalGeneration


@dataclass
class GradBase:
    lr: float
    loss: nn.Module
    optimizer_unlearn: torch.optim
    optimizer_retain: torch.optim

    def __post_init__(self):
        assert self.lr > 0, "Learning rate must be greater than 0"

    @staticmethod
    def _extract_gradients(params: Iterator):
        grads = []
        for param in params:
            grads.append(param.grad.clone() if param.grad is not None else None)

        return grads

    def _compute_per_sample_grads_unefficient(
        self,
        model: nn.Module,
        optimizer: torch.optim,
        data: torch.Tensor,
        labels: torch.Tensor,
        original_model: nn.Module,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        losses = []

        def compute_grad(sample, target):
            sample = sample.unsqueeze(0)  # prepend batch dimension for processing
            target = target.unsqueeze(0)

            prediction = model(sample)

            if isinstance(self.loss, CrossEntropyLoss):
                loss = self.loss(prediction, target)

            elif isinstance(self.loss, DistillKL):
                assert original_model != None, "The original model is not defined"
                with torch.no_grad():
                    target_logits = original_model(sample)

                loss = self.loss(prediction, target_logits)

            losses.append(loss)
            return torch.autograd.grad(loss, list(model.parameters()))

        per_sample_grads = [
            compute_grad(data[i], labels[i]) for i in range(data.shape[0])
        ]
        per_sample_grads = zip(*per_sample_grads)
        per_sample_grads = [torch.stack(shards) for shards in per_sample_grads]

        if isinstance(optimizer, AdamUpdateDirection):
            per_sample_grads = optimizer.step(grad=per_sample_grads)

        return per_sample_grads, torch.stack(losses)

    def _compute_per_sample_grads(
        self,
        model: nn.Module,
        optimizer: torch.optim,
        data: torch.Tensor,
        labels: torch.Tensor,
        original_model: nn.Module,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        # todo: check what happens if a layer does not have grad
        model.zero_grad()
        if optimizer is not None:
            optimizer.zero_grad()
        params = {k: v.detach() for k, v in model.named_parameters() if v.requires_grad}
        buffers = {k: v.detach() for k, v in model.named_buffers() if v.requires_grad}

        if original_model is not None:
            original_params = {
                k: v.detach()
                for k, v in original_model.named_parameters()
                if v.requires_grad
            }
            original_buffers = {
                k: v.detach()
                for k, v in original_model.named_buffers()
                if v.requires_grad
            }

        def compute_loss(
            params: Dict[str, torch.Tensor],
            buffers: Dict[str, torch.Tensor],
            sample: torch.Tensor,
            target: torch.Tensor,
        ) -> torch.Tensor:
            target = target.unsqueeze(0)
            batch = sample.unsqueeze(0)
            if isinstance(model, (WhisperForConditionalGeneration, PeftModel)):
                predictions = functional_call(model, (params, buffers), (batch, target))
                # logits = predictions.logits
                #
                # # Calculate loss
                # shift_logits = logits[:, :-1, :].contiguous()
                # shift_labels = target[:, 1:].contiguous()
                # loss = self.loss(shift_logits.view(-1, logits.size(-1)), shift_labels.view(-1))
                loss = predictions.loss
            else:
                predictions = functional_call(model, (params, buffers), (batch,))
                # todo: we can inherit from whisper and modify forward

                if isinstance(self.loss, CrossEntropyLoss):
                    loss = self.loss(predictions, target)

                elif isinstance(self.loss, DistillKL):
                    assert original_model != None, "The original model is not defined"
                    with torch.no_grad():
                        target_logits = functional_call(
                            original_model,
                            (original_params, original_buffers),
                            (batch,),
                        )
                    loss = self.loss(predictions, target_logits)

                else:
                    raise ValueError(f"Invalid loss function {self.loss}")

            return loss

        # Define a function to compute both loss and gradients
        def compute_loss_and_grad(
            params: Dict[str, torch.Tensor],
            buffers: Dict[str, torch.Tensor],
            sample: torch.Tensor,
            target: torch.Tensor,
        ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
            loss = compute_loss(params, buffers, sample, target)
            grads = grad(compute_loss)(params, buffers, sample, target)

            return loss, grads

        ft_compute_sample_grad = vmap(compute_loss_and_grad, in_dims=(None, None, 0, 0))
        per_sample_loss, per_sample_grads = ft_compute_sample_grad(
            params, buffers, data, labels
        )

        if isinstance(optimizer, AdamUpdateDirection):
            per_sample_grads = optimizer.step(grad=list(per_sample_grads.values()))

        if isinstance(per_sample_grads, dict):
            per_sample_grads = list(per_sample_grads.values())

        return per_sample_grads, per_sample_loss

    def _compute_grads(
        self,
        model: nn.Module,
        optimizer: torch.optim,
        data: torch.Tensor,
        labels: torch.Tensor,
        grad_mode: str,
        original_model: nn.Module,
    ) -> Tuple[Union[Dict[str, torch.Tensor], List[torch.Tensor]], torch.Tensor]:
        if grad_mode == "mean":
            return self._compute_agg_grads_and_loss(
                model, optimizer, data, labels, original_model
            )
        else:  # self.retain_grad_mode == "per_sample"
            per_sample_grads, per_sample_loss = self._compute_per_sample_grads(
                model, optimizer, data, labels, original_model
            )
            return per_sample_grads, per_sample_loss.mean()

    def _compute_loss_and_grads(
        self,
        model: nn.Module,
        unlearn_data: torch.Tensor,
        unlearn_labels: torch.Tensor,
        optimizer_unlearn: torch.optim,
        retain_data: torch.Tensor,
        retain_labels: torch.Tensor,
        optimizer_retain: torch.optim,
        retain_grad_mode: str,
        original_model: nn.Module,
    ) -> Tuple[
        List[torch.Tensor],
        Union[Dict[str, torch.Tensor], List[torch.Tensor]],
        torch.Tensor,
        torch.Tensor,
    ]:
        unlearn_grads, unlearn_loss = self._compute_grads(
            model,
            optimizer_unlearn,
            unlearn_data,
            unlearn_labels,
            grad_mode="mean",
            original_model=original_model,
        )
        retain_grads, retain_loss = self._compute_grads(
            model,
            optimizer_retain,
            retain_data,
            retain_labels,
            retain_grad_mode,
            original_model,
        )
        return unlearn_grads, retain_grads, unlearn_loss, retain_loss

    def _compute_agg_grads_and_loss(
        self,
        model: nn.Module,
        optimizer: torch.optim,
        data_input: torch.Tensor,
        labels: torch.Tensor,
        original_model: nn.Module,
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        model.zero_grad()
        if optimizer is not None:
            optimizer.zero_grad()

        if False:# isinstance(model, (WhisperForConditionalGeneration, PeftModel)):
            outputs = model(data_input, labels=labels)
            # logits = outputs.logits
            #
            # # Calculate loss
            # shift_logits = logits[:, :-1, :].contiguous()
            # shift_labels = labels[:, 1:].contiguous()
            # loss = self.loss(shift_logits.view(-1, logits.size(-1)), shift_labels.view(-1))
            loss = outputs.loss

        else:
            logits = model(data_input)

            if isinstance(self.loss, CrossEntropyLoss):
                loss = self.loss(logits, labels)

            elif isinstance(self.loss, DistillKL):
                assert original_model != None, "The original model is not defined"
                with torch.no_grad():
                    target_logits = original_model(data_input)
                loss = self.loss(logits, target_logits)

        loss.backward()
        if optimizer is None:
            grads = self._extract_gradients(model.parameters())
        elif isinstance(optimizer, AdamUpdateDirection):
            grads = optimizer.step()
        else:
            raise ValueError(
                f"Given optimizer {type(self.optimizer_unlearn)} is not supported"
            )

        return grads, loss

    def _update_model_params(
        self, params: Iterator, grads: List[torch.Tensor], mode: str = "decent"
    ) -> None:
        if mode == "decent":
            sign = -1
        else:  # mode == "ascent"
            sign = 1

        for p, g in zip((p for p in params if p.requires_grad), grads):
            # if p.requires_grad:
            if g is not torch.nan:
                assert p.shape == g.shape, "Shape mismatch model param and gradient"
                p.data += sign * self.lr * g

    @staticmethod
    def _check_for_nan_gradients(unlearn_grads, retain_grads):
        for g_u, g_r in zip(unlearn_grads, retain_grads):
            if torch.isnan(g_u).any() or torch.isnan(g_r).any():
                raise ValueError("The gradients exploaded !")

    @abstractmethod
    def __call__(self, model, unlearn_data, unlearn_labels, retain_data, retain_labels):
        pass


class NegGrad(GradBase):
    def __init__(
        self, lr, loss, optimizer_unlearn, optimizer_retain, update_mode: str, **kwargs
    ):
        assert update_mode in ("both", "accent")
        self.update_mode = update_mode
        self.retain_grad_mode = "mean"
        super().__init__(lr, loss, optimizer_unlearn, optimizer_retain)

    def __call__(
        self,
        model: nn.Module,
        unlearn_data: torch.Tensor,
        unlearn_labels: torch.Tensor,
        retain_data: torch.Tensor,
        retain_labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        (
            unlearn_grads,
            retain_grads,
            unlearn_loss,
            retain_loss,
        ) = self._compute_loss_and_grads(
            model,
            unlearn_data,
            unlearn_labels,
            self.optimizer_unlearn,
            retain_data,
            retain_labels,
            self.optimizer_retain,
            self.retain_grad_mode,
            original_model=None,
        )

        self._update_model_params(model.parameters(), unlearn_grads, mode="accent")
        if self.update_mode == "both":
            self._update_model_params(model.parameters(), retain_grads, mode="decent")
        return unlearn_loss, retain_loss


class OrthogonalGrad(GradBase):
    def __init__(
        self,
        lr,
        loss,
        optimizer_unlearn,
        optimizer_retain,
        retain_grad_mode,
        update_mode,
        original_model,
        grad_mask,
        alpha,
        **kwargs,
    ):
        assert update_mode in ("both", "accent")
        self.alpha = alpha
        self.update_mode = update_mode
        self.retain_grad_mode = retain_grad_mode
        self.original_model = original_model
        self.grad_mask = grad_mask
        assert self.retain_grad_mode in [
            "mean",
            "per_sample",
        ], "Invalid retain grad mode"
        super().__init__(lr, loss, optimizer_unlearn, optimizer_retain)

    @staticmethod
    def _project_orthogonal(unlearn_grads, retain_grads):
        # Start with v_proj as the original vector v
        unlearn_grads_proj = unlearn_grads.clone()
        retain_grads_proj = retain_grads.clone()

        # Iterate through each vector in g
        for g_i in retain_grads_proj:
            projection = (
                torch.dot(unlearn_grads_proj, g_i) / (g_i.norm() ** 2) * g_i
            )  # calc projection of v_proj onto g_i
            unlearn_grads_proj -= projection  # Subtract the projection

        return unlearn_grads_proj

    def _per_sample_projection(self, unlearn_grads, retain_grads):
        orig_shape = unlearn_grads.shape
        retain_grads = (
            retain_grads.flatten(1)
            if self.retain_grad_mode == "per_sample"
            else retain_grads.flatten().unsqueeze(0)
        )
        unlearn_grads = unlearn_grads.flatten()
        q, r = torch.linalg.qr(retain_grads.T)  # QR decomposition
        orthogonal_unlearn_grads = self._project_orthogonal(unlearn_grads, q.T)
        return orthogonal_unlearn_grads.reshape(orig_shape)

    def __call__(
        self,
        model: nn.Module,
        unlearn_data: torch.Tensor,
        unlearn_labels: torch.Tensor,
        retain_data: torch.Tensor,
        retain_labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        (
            unlearn_grads,
            retain_grads,
            unlearn_loss,
            retain_loss,
        ) = self._compute_loss_and_grads(
            model,
            unlearn_data,
            unlearn_labels,
            self.optimizer_unlearn,
            retain_data,
            retain_labels,
            self.optimizer_retain,
            self.retain_grad_mode,
            self.original_model,
        )

        if self.grad_mask is not None:
            for g_u, g_r, g_m in zip(unlearn_grads, retain_grads, self.grad_mask):
                g_u *= g_m
                g_r *= ~g_m

        # since we are using vmap to compute per sample grads, retain_grads is a dict
        unlearn_orthogonal_grads = []
        self._check_for_nan_gradients(unlearn_grads, retain_grads)

        for g_u, g_r in zip(unlearn_grads, retain_grads):
            unlearn_orthogonal_grads.append(self._per_sample_projection(g_u, g_r))

        if self.alpha == 0:
            self._update_model_params(
                model.parameters(), unlearn_orthogonal_grads, mode="accent"
            )

        else:
            retain_averaged_grads = [
                g.mean(dim=0) for g in retain_grads
            ]  # default reduction is mean
            combined_grads = []
            for g_r, g_u in zip(retain_averaged_grads, unlearn_orthogonal_grads):
                combined_grads.append(self.alpha * g_r - ((1 - self.alpha) * g_u))

            self._update_model_params(model.parameters(), combined_grads, mode="decent")

        return unlearn_loss, retain_loss


GRAD_METHODS = dict(neg_grad=NegGrad, ortho_grad=OrthogonalGrad)


import torch
from torch.optim.optimizer import _use_grad_for_differentiable


class AdamUpdateDirection(torch.optim.Adam):
    @_use_grad_for_differentiable
    def step(self, closure=None, grad=None):
        if grad is not None:
            return self._step_with_grads(grad)
        else:
            return self._step_without_grads()

    def _step_with_grads(self, grads):
        layers = []
        for group in self.param_groups:
            params = [p for p in group["params"] if p.requires_grad]
            for p, grad in zip(params, grads):
                if grad is None:
                    continue
                update_direction = []
                for sample_idx in range(grads[0].shape[0]):
                    update_direction.append(
                        self._get_update_direction(p, group, grad[sample_idx])
                    )

                layers.append(torch.stack(update_direction))

        return layers

    def _step_without_grads(self):
        update_direction = []
        for param_group in self.param_groups:
            for param in param_group["params"]:
                if param.grad is None:
                    continue

                # Clone the parameter and gradient to avoid in-place modification
                grad = param.grad.clone()

                update_direction.append(
                    self._get_update_direction(param, param_group, grad)
                )

        return update_direction

    def _get_update_direction(self, param, param_group, grad):
        # Adam optimizer's state
        state = self.state[param]

        # State initialization if not done yet
        if len(state) == 0:
            state["step"] = 0
            state["exp_avg"] = torch.zeros_like(param.data)  # first moment (m)
            state["exp_avg_sq"] = torch.zeros_like(param.data)  # second moment (v)

        # Retrieve the state variables
        exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
        beta1, beta2 = param_group["betas"]
        eps = param_group["eps"]

        # Update step
        state["step"] += 1
        step = state["step"]

        # Update biased first moment estimate (m)
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

        # Update biased second raw moment estimate (v)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        # Compute bias-corrected first and second moment estimates
        bias_correction1 = 1 - beta1**step
        bias_correction2 = 1 - beta2**step

        # Corrected first moment
        corrected_exp_avg = exp_avg / bias_correction1

        # Corrected second moment
        corrected_exp_avg_sq = exp_avg_sq / bias_correction2

        # Compute the update direction (step size)
        denom = corrected_exp_avg_sq.sqrt().add_(eps)

        u_direction = corrected_exp_avg / denom

        return u_direction
    

import argparse
import copy
import logging
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torch.utils.data import random_split
from torchvision import datasets, transforms


class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t, reduction="batchmean"):
        p_s = F.log_softmax(y_s, dim=1)  # F.log_softmax(y_s / self.T, dim=1)
        p_t = F.softmax(y_t, dim=1)  # F.softmax(y_t / self.T, dim=1)
        loss = F.kl_div(p_s, p_t, reduction=reduction)  # * (self.T**2) / y_s.shape[0]
        return loss


def AUS(a_t, a_or, a_f, unlearn_type):
    a_or /= 100
    a_t /= 100
    a_f /= 100
    if unlearn_type == "random":
        return (1 - (a_or - a_t)) / (1 + abs(a_f - a_t))
    return (1 - (a_or - a_t)) / (1 + abs(a_f))


def UIS(O_t, U_t, U_u, unlearn_type):  # Unlearning Impact Score (UIS)
    if unlearn_type == "random":
        return (abs(O_t - U_t) / O_t + abs(O_t - U_u) / O_t) / 2
    return (abs(O_t - U_t) / O_t + abs(U_u) / O_t) / 2


def load_dataset(dataset):
    if not os.path.exists("./data"):
        os.mkdir("./data")

    if dataset == "mnist":
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        train_set = torchvision.datasets.MNIST(
            root="./data", train=True, download=True, transform=transform
        )
        test_set = torchvision.datasets.MNIST(
            root="./data", train=False, download=True, transform=transform
        )
    elif dataset == "cifar10":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        train_set = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform
        )
        test_set = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform
        )
    elif dataset == "cifar100":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)
                ),
            ]
        )
        train_set = torchvision.datasets.CIFAR100(
            root="./data", train=True, download=True, transform=transform
        )
        test_set = torchvision.datasets.CIFAR100(
            root="./data", train=False, download=True, transform=transform
        )
    elif dataset == "svhn":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        train_set = torchvision.datasets.SVHN(
            root="./data", split="train", download=True, transform=transform
        )
        test_set = torchvision.datasets.SVHN(
            root="./data", split="test", download=True, transform=transform
        )
    else:
        raise ValueError("Undefined Dataset.")

    return train_set, test_set


def load_data(dataset, batch_size, seed=42):
    train_set, test_set = load_dataset(dataset)

    torch.manual_seed(seed)

    val_size = int(len(train_set) * 0.2)
    train_set, val_set = random_split(train_set, [len(train_set) - val_size, val_size])

    trainloader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=2
    )
    valloader = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, shuffle=True, num_workers=2
    )
    testloader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=2
    )

    return trainloader, valloader, testloader


def load_unlearn_data(
    dataset, unlearn_type, class_idx_to_remove, unlearn_ratio, retain_ratio
):
    train_set, test_set = load_dataset(dataset)

    # torch.manual_seed(0)
    if unlearn_type == "random":
        if 1 - unlearn_ratio - retain_ratio != 0:
            res_set, unl_set, _ = random_split(
                train_set,
                [retain_ratio, unlearn_ratio, 1 - retain_ratio - unlearn_ratio],
            )
        else:
            res_set, unl_set = random_split(train_set, [retain_ratio, unlearn_ratio])

        # res_set, unl_set = random_split(train_set, [1 - unlearn_ratio, unlearn_ratio])
    elif unlearn_type == "class":
        if class_idx_to_remove > len(train_set.classes) - 1 and class_idx_to_remove < 0:
            raise ValueError(
                "Class index must be between 0 and the total number of targets in the dataset."
            )

        res_set = copy.deepcopy(train_set)
        unl_set = copy.deepcopy(train_set)

        res_indicies = torch.as_tensor(train_set.targets) != class_idx_to_remove
        res_set.data = res_set.data[res_indicies]
        res_set.targets = torch.as_tensor(train_set.targets)[res_indicies].tolist()

        unl_indicies = torch.as_tensor(train_set.targets) == class_idx_to_remove
        unl_set.data = unl_set.data[unl_indicies]
        unl_set.targets = torch.as_tensor(train_set.targets)[unl_indicies].tolist()
    else:
        raise ValueError("Unsupported unlearn type")

    # unlearnloader = torch.utils.data.DataLoader(unl_set, batch_size=len(unl_set), shuffle=False, num_workers=2)
    # residualloader = torch.utils.data.DataLoader(res_set, batch_size=len(res_set), shuffle=False, num_workers=2)

    return unl_set, res_set, test_set


def load_train_data(dataset, batch_size, seed=42):
    torch.manual_seed(seed)

    val_size = int(len(dataset) * 0.2)
    train_set, val_set = random_split(dataset, [len(dataset) - val_size, val_size])

    trainloader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=2
    )
    valloader = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, shuffle=True, num_workers=2
    )

    return trainloader, valloader


def params_to_vec(parameters, grad=False):
    vec = []
    for param in parameters:
        if grad:
            vec.append(param.grad.view(1, -1))
        else:
            vec.append(param.data.view(1, -1))
    return torch.cat(vec, dim=1).squeeze()


def vec_to_params(vec, parameters):
    param = []
    for p in parameters:
        size = p.view(1, -1).size(1)
        param.append(vec[:size].view(p.size()))
        vec = vec[size:]
    return param


def batch_grads_to_vec(parameters):
    vec = []
    for param in parameters:
        # vec.append(param.view(1, -1))
        vec.append(param.reshape(1, -1))
    return torch.cat(vec, dim=1).squeeze()


def batch_vec_to_grads(vec, parameters):
    grads = []
    for param in parameters:
        size = param.view(1, -1).size(1)
        grads.append(vec[:size].view(param.size()))
        vec = vec[size:]
    return grads


def set_seed(seed):
    """for reproducibility
    :param seed:
    :return:
    """
    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def str2bool(v):
    if isinstance(v, bool):
        return v
    v = v.strip().lower()
    if v in ("yes", "true", "t", "y", "1"):
        return True
    elif v in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def set_logger():
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )