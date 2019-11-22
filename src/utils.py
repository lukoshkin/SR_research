import torch
from tqdm import trange

def train(
        net, pde, optimizer, loss_history, 
        num_batches, batch_size=5120):
    for _ in trange(num_batches, desc='Training'):
        optimizer.zero_grad()

        batch = pde.sampleBatch(batch_size)
        batch.requires_grad_(True)

        loss = pde.computeLoss(batch, net)
        loss_history.append(loss.item())
        loss.backward()

        optimizer.step()


# Evaluation metrics
max_error = lambda u_exact, u_approx: torch.norm(
        u_exact - u_approx, p=float('inf'))
avg_error = lambda u_exact, u_approx: torch.mean(
        (u_exact - u_approx)**2)**.5
