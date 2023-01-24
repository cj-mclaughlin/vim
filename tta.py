import torch
from functorch import hessian, vmap, grad
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import numpy as np
import primefac

from functools import reduce

DEVICE = "cuda"
# BATCH_SIZE_TTA = 100


def factors(n):    
    return set(reduce(list.__add__, 
                ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))

def get_batch_size(n_samples, max_batch=128):
    f = np.array(list(factors(n_samples)))
    f = f[f < max_batch]
    return int(max(f))

def get_loss(loss, num_classes=1000):
    if loss == "ce":
        # cross entropy with pseudo labels
        criterion = torch.nn.CrossEntropyLoss().to(DEVICE)
        loss_func = lambda logits: criterion(logits, logits.detach().argmax(1))
    elif loss == "kl":
        # KL of uniform and outputs - from GradNorm paper
        logsoftmax = torch.nn.LogSoftmax(dim=-1).to(DEVICE)
        targets = torch.ones((1, num_classes)).to(DEVICE)
        loss_func = lambda logits: torch.mean(torch.sum(-targets * logsoftmax(logits), dim=-1))
    elif loss == "entropy":
        # entropy - i.e. from domain adaptation, like MEMO or TENT
        softmax = torch.nn.Softmax(dim=-1).to(DEVICE)
        logsoftmax = torch.nn.LogSoftmax(dim=-1).to(DEVICE)
        loss_func = lambda logits: (-1 * (softmax(logits)*logsoftmax(logits)).sum())
    elif loss == "energy":
        # energy
        loss_func = lambda logits: -torch.logsumexp(logits, dim=-1)
    return loss_func

def tta_predict(feat, w, b, grad=0):
    logits = torch.matmul(feat.reshape(1,-1), (w - grad).T) + b
    return logits

def objective_fn(feat, w, b, grad=0, loss="ce", proximal_loss=0):
    logits = torch.matmul(feat.reshape(1,-1), (w - grad).T) + b
    loss_fn = get_loss(loss)
    proximal_cost = proximal_loss * torch.sum(torch.square(grad))
    loss = loss_fn(logits) + proximal_cost
    return loss

def compute_tta_scores(x, w, b, loss, proximal_loss=0, learning_rate=1e-4, num_steps=3):
    data = torch.Tensor(x)
    dataset = TensorDataset(data)
    B = get_batch_size(x.shape[0])
    dataloader = DataLoader(dataset, batch_size=B)
    w, b = torch.Tensor(w).to(DEVICE), torch.Tensor(b).to(DEVICE)
    logits = {f"logits_step{i}": [] for i in range(1, num_steps+1)}
    grads = {f"grads_step{i}": [] for i in range(1, num_steps+1)}
    # for each sample, record:
    # 1) the logits at each step of tta
    # 2) the norm of the gradient at each step of tta
    def objective(feat, w, b, grad):
        return objective_fn(feat, w, b, grad, loss=loss, proximal_loss=proximal_loss)
    for feat in tqdm(dataloader):
        current_grad = torch.zeros((B, w.shape[0], w.shape[1])).cuda()
        feat = feat[0].to(DEVICE)
        for step in range(1, num_steps+1):
            grads_tta_parallel = vmap(grad(objective, argnums=1), in_dims=(0, None, None, 0))
            g = grads_tta_parallel(feat, w, b, learning_rate*current_grad)
            grad_norm = torch.norm(g.reshape(B, -1), dim=1)
            grads[f"grads_step{step}"].append(grad_norm.cpu().numpy())
            current_grad += g
            logits_tta_parallel = vmap(tta_predict, in_dims=(0, None, None, 0))
            l = logits_tta_parallel(feat, w, b, learning_rate*current_grad)
            logits[f"logits_step{step}"].append(l.squeeze().cpu().numpy())
    for key in logits.keys():
        logits[key] = np.concatenate(logits[key], axis=0)
    for key in grads.keys():
        grads[key] = np.concatenate(grads[key], axis=0)
    return logits, grads
