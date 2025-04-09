import torch
from torch import nn, optim
from torch.nn.functional import normalize
from torch.func import functional_call

class NeuralMemory(nn.Module):

    def __init__(self, emb_dim = 16, n_layers = 2, hidden_dim = 32, alpha = 0.999, eta = 0.60, theta = 0.05):
        super().__init__()

        # Define the layers of the network
        self.layers = None
        if n_layers == 1:
            self.layers = nn.ModuleList([nn.Linear(emb_dim, emb_dim)])
        else:
            self.layers = nn.ModuleList([])
            self.layers.append(nn.Sequential(
                nn.Linear(emb_dim, hidden_dim),
                nn.SiLU()
            ))
            for k in range(n_layers - 2):
                self.layers.append(nn.Sequential(
                    nn.Linear(emb_dim, hidden_dim),
                    nn.SiLU()
                ))
            self.layers.append(nn.Sequential(
                nn.Linear(hidden_dim, emb_dim)
            ))

        # Mapping to keys
        self.K = nn.Linear(emb_dim, emb_dim, bias = False)

        # Mapping to values
        self.V = nn.Linear(emb_dim, emb_dim, bias = False)

        torch.nn.init.xavier_uniform_(self.K.weight)
        torch.nn.init.xavier_uniform_(self.V.weight)

        self.alpha = alpha
        self.eta = eta
        self.theta = theta

        self.silu = nn.SiLU()
        self.surprise = {}

    def retrieve(self, x):

        return functional_call(self, dict(self.named_parameters()), x)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x

    def update(self, x):

        z = x.detach()

        # Evaluate the corresponding keys and values
        keys = normalize(self.silu(self.K(z)))
        vals = self.silu(self.V(z))

        # Propagate the keys through the model
        for layer in self.layers:
            keys = layer(keys)

        # Calculate the loss || M(keys) - vals ||_2 ^2
        loss = ((keys - vals) ** 2).mean(axis=0).sum()

        # Compute gradients of aux loss w.r.t. NMM's parameters
        grads = torch.autograd.grad(loss, self.parameters())

        # Update the surprise dictionary and the parameters of the network
        updated_params = {}

        for (name, param), grad in zip(self.named_parameters(), grads):
            if self.surprise.get(name, None) is None:
                self.surprise[name] = torch.zeros_like(grad)
            self.surprise[name] = self.surprise[name] * self.eta - self.theta * grad
            updated_params[name] = self.alpha * param.data + self.surprise[name] if not name[0] in ['K', 'V'] else param.data
            param.data = updated_params[name]

        return loss.item(), updated_params