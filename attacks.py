import torch
import torch.nn as nn

def fgsm_attack(model, X, y, epsilon):
    X_adv = X.clone().detach()
    X_adv.requires_grad = True
    outputs = model(X_adv)
    loss = nn.CrossEntropyLoss()(outputs, y)
    model.zero_grad()
    loss.backward()
    perturbation = epsilon * X_adv.grad.data.sign()
    X_adv = X_adv + perturbation
    return X_adv.detach()

def pgd_attack(model, X, y, epsilon, alpha=0.01, num_iter=40):
    X_adv = X.clone().detach() + torch.empty_like(X).uniform_(-epsilon, epsilon)
    for _ in range(num_iter):
        X_adv.requires_grad = True
        outputs = model(X_adv)
        loss = nn.CrossEntropyLoss()(outputs, y)
        model.zero_grad()
        loss.backward()
        with torch.no_grad():
            X_adv = X_adv + alpha * X_adv.grad.sign()
            eta = torch.clamp(X_adv - X, min=-epsilon, max=epsilon)
            X_adv = torch.clamp(X + eta, min=X.min(), max=X.max())
    return X_adv.detach()

def bim_attack(model, X, y, epsilon, alpha=0.005, num_iter=10):
    X_adv = X.clone().detach()
    for _ in range(num_iter):
        X_adv.requires_grad = True
        outputs = model(X_adv)
        loss = nn.CrossEntropyLoss()(outputs, y)
        model.zero_grad()
        loss.backward()
        with torch.no_grad():
            X_adv = X_adv + alpha * X_adv.grad.sign()
            eta = torch.clamp(X_adv - X, min=-epsilon, max=epsilon)
            X_adv = torch.clamp(X + eta, min=X.min(), max=X.max())
    return X_adv.detach()

def transfer_attack(surrogate_model, X, y, epsilon):
    return fgsm_attack(surrogate_model, X, y, epsilon)

def random_attack(model, X, y, epsilon, num_trials):
    best_X_adv = X.clone().detach()
    best_acc = 1.0
    for _ in range(num_trials):
        noise = torch.empty_like(X).uniform_(-epsilon, epsilon)
        X_candidate = torch.clamp(X + noise, min=X.min(), max=X.max())
        outputs = model(X_candidate)
        _, predicted = torch.max(outputs, 1)
        acc_candidate = (predicted == y).float().mean().item()
        if acc_candidate < best_acc:
            best_acc = acc_candidate
            best_X_adv = X_candidate
    return best_X_adv.detach()

def square_attack(model, X, y, epsilon=0.3, iters=100):
    X_adv = X.clone().detach()
    for _ in range(iters):
        mask = torch.randint(0, 2, X_adv.shape).float()
        perturbation = (torch.rand_like(X_adv) - 0.5) * 2 * epsilon
        X_adv = X_adv + mask * perturbation
    return X_adv

def fgsm_attack_def(model, X, y, epsilon=0.1):
    X_adv = X.clone().detach()
    X_adv.requires_grad = True
    outputs = model(X_adv)
    loss = nn.CrossEntropyLoss()(outputs, y)
    model.zero_grad()
    loss.backward()
    with torch.no_grad():
        X_adv += epsilon * X_adv.grad.sign()
    return X_adv.detach()