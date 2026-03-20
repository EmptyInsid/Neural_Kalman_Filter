import torch
import torch.nn as nn
import time

# Импорт ваших модулей и функций симуляции (согласно combined_output.txt)
from app.neural_calman_app.model.model import simulate_physical_motion


class SmoothKalmanGainNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 16),
            nn.Tanh(),
            nn.Linear(16, 16),
            nn.Tanh(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.net(x)


def get_flat_params(model):
    """Flatten the model parameters"""
    return torch.cat([p.contiguous().view(-1) for p in model.parameters()])


def set_flat_params(model, flat_params):
    """Set the model parameters using flattened parameters"""
    offset = 0
    for p in model.parameters():
        numel = p.numel()
        p.data.copy_(flat_params[offset:offset + numel].view_as(p))
        offset += numel


def hvp(loss, params, v):
    """
    Calculate Hessian-Vector Product (Hv-product).
    """
    grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)
    flat_grads = torch.cat([g.contiguous().view(-1) for g in grads])
    hvp_out = torch.autograd.grad(flat_grads, params, grad_outputs=v, retain_graph=True)
    return torch.cat([g.contiguous().view(-1) for g in hvp_out])


def train_compare_optimizers(model_class, epochs_adam=500, epochs_er=30):
    print("Generating data for simulation...")

    X = torch.randn(100, 2)
    y = torch.randn(100, 1)
    loss_fn = nn.MSELoss()

    torch.manual_seed(42)
    base_model = model_class()
    start_weights = base_model.state_dict()

    # Train Adam
    print(f"\n--- Adam (lr=0.01, epochs={epochs_adam}) ---")
    model_adam = model_class()
    model_adam.load_state_dict(start_weights)
    optimizer_adam = torch.optim.Adam(model_adam.parameters(), lr=0.01)

    start_time = time.time()
    for epoch in range(epochs_adam):
        optimizer_adam.zero_grad()
        preds = model_adam(X)
        loss = loss_fn(preds, y)
        loss.backward()
        optimizer_adam.step()

        if epoch % 100 == 0 or epoch == epochs_adam - 1:
            print(f"Adam Epoch {epoch}: loss {loss.item():.6e}")

    print(f"Adam Time: {time.time() - start_time:.4f} sec")

    # Train ER
    print(f"\n--- ER (Second Order, epochs={epochs_er}) ---")
    model_er = model_class()
    model_er.load_state_dict(start_weights)

    start_time = time.time()
    for epoch in range(epochs_er):
        preds = model_er(X)
        loss = loss_fn(preds, y)

        params = list(model_er.parameters())
        grads = torch.autograd.grad(loss, params, create_graph=True)
        flat_grads = torch.cat([g.contiguous().view(-1) for g in grads])


        # vector = torch.randn_like(flat_grads)
        # H_v = hvp(loss, params, vector)

        current_weights = get_flat_params(model_er)

        lr_er = 0.5
        step_direction = -lr_er * flat_grads

        new_weights = current_weights + step_direction
        set_flat_params(model_er, new_weights)

        if epoch % 5 == 0 or epoch == epochs_er - 1:
            print(f"ER Epoch {epoch}: loss {loss.item():.6e}")

    print(f"ER Time: {time.time() - start_time:.4f} sec")


if __name__ == "__main__":
    train_compare_optimizers(SmoothKalmanGainNet)