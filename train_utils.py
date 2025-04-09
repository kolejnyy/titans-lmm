import torch
from tqdm import tqdm
from copy import deepcopy
from torch import nn, optim

def train_and_test(model, optimizer, train_loader, valid_loader, test_loader, n_epochs=20, print_loss=False):

    # Initialize the optimizer and loss function
    mse_loss = nn.MSELoss()

    optim_all = optim.Adam(model.parameters())

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    best_model = None
    best_mse = 1e9

    print("Training:")

    for epoch in tqdm(range(n_epochs)):

        train_loss = 0

        # Train the model for a full epoch
        for x, y in (train_loader):
            optim_all.zero_grad()
            out = model(x.float().to(device))[:,:-1].to(device)
            labels = y[:,1:].float().to(device)
            loss = mse_loss(out, labels.float())
            loss.backward(retain_graph=True)
            optimizer.step()

            train_loss += loss.item()

        total_loss = 0
        sample_num = 0

        # Validate the model
        for x, y in (valid_loader):
            out = model(x.float().to(device))[:,15:-1].to(device)
            labels = y[:,16:].float().to(device)
            loss = mse_loss(out, labels.float())

            total_loss += x.shape[0] * loss.item()
            sample_num += x.shape[0]

        if total_loss/sample_num < best_mse:
            best_model = deepcopy(model)
            best_mse = total_loss/sample_num

        if print_loss:
            print(f"Validation loss in Epoch {epoch}: {total_loss/sample_num}")


    print("\n Testing the best model:")
    test_loss = 0
    test_num = 0

    for x, y in (test_loader):
        out = model(x.float().to(device))[:,15:-1].to(device)
        labels = y[:,16:].float().to(device)
        loss = mse_loss(out, labels.float())

        test_loss += x.shape[0] * loss.item()
        test_num += x.shape[0]

    print(f"Test MSE: {test_loss/test_num}\n")

    return best_model