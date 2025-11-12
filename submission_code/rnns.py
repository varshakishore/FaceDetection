import torch
from torch.optim import LBFGS


def step_lbfgs(adv_image, image, target, model, lr, max_iter, epsilon, criterion=torch.nn.BCEWithLogitsLoss(reduction='sum'), hamming=False):
    adv_image.requires_grad = True
    optimizer = LBFGS([adv_image], lr=lr, max_iter=max_iter)

    def closure():
        outputs = model(adv_image)
        loss = criterion(outputs.view(-1)[:target.numel()], target.view(-1))

        optimizer.zero_grad()
        loss.backward()
        return loss

    optimizer.step(closure)
    delta = torch.clamp(adv_image - image, min=-epsilon, max=epsilon)
    return torch.clamp(image + delta, min=0, max=1).detach()
