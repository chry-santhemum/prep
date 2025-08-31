# %%
from tqdm.auto import tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision import transforms
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])


mnist_ds = load_dataset("ylecun/mnist")

def preprocess_item(item: dict):
    img = item["image"]
    img_tensor = transform(img)
    return {
        "data": img_tensor
    }


mnist_train_ds = mnist_ds["train"].map(preprocess_item, remove_columns="image", num_proc=8)
mnist_test_ds = mnist_ds["test"].map(preprocess_item, remove_columns="image", num_proc=8)

# Set format to torch for proper tensor handling
mnist_train_ds.set_format(type='torch', columns=['data', 'label'])
mnist_test_ds.set_format(type='torch', columns=['data', 'label'])

train_dl = DataLoader(mnist_train_ds, batch_size=64, shuffle=True)
test_dl = DataLoader(mnist_test_ds, batch_size=1024, shuffle=True)

# %%

from typing import OrderedDict, Callable
from jaxtyping import Float

import torch 
from torch import Tensor, nn
from torch.nn import Module
from torch.optim import AdamW
from torch.autograd import Function


class Quantize(Function):

    @staticmethod
    def forward(ctx, input: Tensor, num_levels):
        max_val = num_levels - 1
        scale = input.abs().max().item()
        out = torch.round(input / scale * max_val)
        out = out * scale / max_val
        return out

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None  # None for num_levels gradient

quantize = Quantize.apply

class QuantizeModule(Module):
    def __init__(self, num_levels):
        super().__init__()
        self.num_levels = num_levels
    
    def forward(self, x):
        return quantize(x, self.num_levels)


class QuantizedCNN(Module):
    def __init__(self, num_levels=256):
        super().__init__()

        self.model = nn.Sequential(
            OrderedDict([
                ("conv_1", nn.Conv2d(1, 16, kernel_size=3, padding=1)),
                ("relu_1", nn.ReLU()),
                ("pool_1", nn.MaxPool2d(kernel_size=2, stride=2)),
                ("quantize_1", QuantizeModule(num_levels)),
                ("conv_2", nn.Conv2d(16, 32, kernel_size=3, padding=1)),
                ("relu_2", nn.ReLU()),
                ("pool_2", nn.MaxPool2d(kernel_size=2, stride=2)),
                ("quantize_2", QuantizeModule(num_levels)),
                ("flatten", nn.Flatten()),
                ("ff_1", nn.Linear(32 * 7 * 7, 128)),
                ("relu_3", nn.ReLU()),
                ("quantize_3", QuantizeModule(num_levels)),
                ("ff_2", nn.Linear(128, 10)),
            ])
        )

    def forward(self, x: Float[Tensor, "1 h w"]):
        return self.model(x)

# %%
from functools import partial
from optimizer import cosine_anneal_scheduler

default_device = "cuda" if torch.cuda.is_available() else "mps"

def train(
    model, 
    train_dataloader: DataLoader, 
    test_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer, 
    scheduler: Callable, 
    steps: int|None=None, 
    val_steps: int=100,
    device=default_device,
):
    model.train()
    loss_fn = torch.nn.CrossEntropyLoss()

    for batch_idx, data in enumerate(train_dataloader):
        optimizer.zero_grad()
        step_lr = scheduler(batch_idx)
        for p_group in optimizer.param_groups:
            p_group["lr"] = step_lr

        inputs = data["data"].to(device)
        labels = data["label"].to(device)

        preds = model(inputs)
        loss = loss_fn(preds, labels)
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx}/{len(train_dataloader)} | Loss: {loss.item():.4f}")
        
        if batch_idx % val_steps == 0:
            total_loss = 0
            accurate = 0
            with torch.no_grad():
                for test_data in test_dataloader:
                    inputs = test_data["data"].to(device)
                    labels = test_data["label"].to(device)

                    preds = model(inputs)
                    pred_labels = preds.argmax(dim=-1)
                    loss = loss_fn(preds, labels)
                    total_loss += loss.item()
                    accurate += (pred_labels == labels).sum().item()
            
            print(f"Validation Loss: {total_loss:.4f} | Validation Accuracy: {accurate/10000}")


        if (steps is not None) and (batch_idx == steps):
            break


model = QuantizedCNN(num_levels=16).to(default_device)
optimizer = AdamW(model.parameters())
scheduler = partial(cosine_anneal_scheduler, lr_max=1e-3, lr_min=1e-6, T_warmup=10, T_anneal=len(train_dl))
train(model, train_dl, test_dl, optimizer, scheduler)

# %%
