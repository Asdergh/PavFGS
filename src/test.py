import torch as th
import torch.nn as nn
from torch.optim import Adam
import matplotlib.pyplot as plt
plt.style.use("dark_background")



test_model = nn.Sequential(
    nn.Linear(10, 10),
    nn.GELU(),
    nn.Dropout(0.45)
)
mu = nn.Parameter(th.rand(1, ).requires_grad_(True))
sigma = nn.Parameter(th.rand(1, ).requires_grad_(True))

def fn(x: th.Tensor, mu, sigma) -> th.Tensor:
    return sigma * x + mu

print(test_model.state_dict())
print(mu)
print(sigma)
optim = Adam([
    {"params": [mu], "lr": 0.01, "name": "mu"},
    {"params": [sigma], "lr": 0.01, "name": "sigma"},
    {
        "params": test_model.parameters(),
        "lr": 0.01,
        "name": "test_model"
    }
 ], lr=0.01)
a = th.normal(0, 1, (1, 10))
b = th.normal(0, 1, (1, 10))

losses = []
EPOCHS = 100
for _ in range(EPOCHS):

    optim.zero_grad()
    x = test_model(a)
    x = fn(x, mu, sigma)

    loss = th.mean(b - x)
    loss.backward()
    optim.step()
    losses.append(loss.item())


_, axis = plt.subplots()
axis.plot(losses, color="red")
plt.show()
print(test_model.state_dict())
print(mu)
print(sigma)

    