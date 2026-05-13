import torch
import torch.nn as nn
import timm

SUPPORTED_MODELS = ["efficientnet", "convnext", "dinov2"]


def build_model(model_key: str, cfg: dict, num_classes: int = 6) -> nn.Module:
    """
    Build a pretrained model and replace its classifier head.
    model_key: one of 'efficientnet', 'convnext', 'dinov2'
    """
    model_cfg = cfg["models"][model_key]
    timm_name = model_cfg["timm_name"]
    pretrained = model_cfg.get("pretrained", True)

    model = timm.create_model(timm_name, pretrained=pretrained, num_classes=num_classes)
    return model


def freeze_backbone(model: nn.Module):
    """Freeze everything except the classifier head."""
    # timm models expose get_classifier() for the head
    head = model.get_classifier()
    head_params = set(id(p) for p in head.parameters())

    for param in model.parameters():
        if id(param) not in head_params:
            param.requires_grad = False


def unfreeze_all(model: nn.Module):
    for param in model.parameters():
        param.requires_grad = True


def get_optimizer(model: nn.Module, phase: int, cfg: dict):
    """
    phase=1: head-only LR
    phase=2: low LR for full model (backbone gets lower LR than head)
    """
    training = cfg["training"]
    wd = training.get("weight_decay", 0.01)

    if phase == 1:
        params = filter(lambda p: p.requires_grad, model.parameters())
        return torch.optim.AdamW(params, lr=training["head_lr"], weight_decay=wd)
    else:
        head = model.get_classifier()
        head_ids = set(id(p) for p in head.parameters())
        backbone_params = [p for p in model.parameters() if id(p) not in head_ids]
        head_params = list(head.parameters())
        return torch.optim.AdamW([
            {"params": backbone_params, "lr": training["unfreeze_lr"]},
            {"params": head_params,     "lr": training["head_lr"]},
        ], weight_decay=wd)


def get_scheduler(optimizer, cfg: dict, steps_per_epoch: int):
    training = cfg["training"]
    total_epochs = training["epochs"]
    warmup_epochs = training.get("warmup_epochs", 2)

    # Linear warmup then cosine annealing (total steps)
    total_steps = total_epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
        import math
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing: float = 0.1, num_classes: int = 6):
        super().__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes

    def forward(self, logits, targets):
        log_probs = nn.functional.log_softmax(logits, dim=-1)

        if isinstance(targets, tuple):
            # mixup: targets = (labels_a, labels_b, lam)
            labels_a, labels_b, lam = targets
            nll_a = -log_probs.gather(dim=-1, index=labels_a.unsqueeze(1)).squeeze(1)
            nll_b = -log_probs.gather(dim=-1, index=labels_b.unsqueeze(1)).squeeze(1)
            nll = lam * nll_a + (1 - lam) * nll_b
        else:
            nll = -log_probs.gather(dim=-1, index=targets.unsqueeze(1)).squeeze(1)

        smooth_loss = -log_probs.mean(dim=-1)
        loss = (1 - self.smoothing) * nll + self.smoothing * smooth_loss
        return loss.mean()
