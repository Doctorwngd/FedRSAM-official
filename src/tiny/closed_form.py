import torch


@torch.no_grad()
def finish_delta(grad, direction, variance, rho=.1, lambda_z=.001, alpha_lr=.5):
    if lambda_z <= 0:
        raise ValueError('closed-form lambda_z must be positive')
    axis_grad = torch.dot(grad, direction)
    q = grad - axis_grad * direction
    std = (variance + 1e-12).sqrt()
    z = ((variance + 1e-12) * q / (2 * lambda_z)).clamp(-std, std)
    z_perp = z - torch.dot(z, direction) * direction
    alpha = .1 + alpha_lr * (.2 * axis_grad)
    main = alpha.square() * direction
    raw = main + z_perp
    delta = raw * (rho / (raw.norm() + 1e-12))
    diagnostics = torch.stack((alpha, z_perp.norm(),
        z_perp.norm() / (main.norm() + z_perp.norm() + 1e-12),
        delta.norm(), (z.abs() >= std).float().mean()))
    return delta, diagnostics


@torch.no_grad()
def build_cone(applied_update, updates):
    direction = applied_update / (applied_update.norm() + 1e-12)
    variance = torch.stack(updates).var(0, unbiased=False)
    return direction, variance
