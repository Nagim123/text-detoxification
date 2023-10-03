import tqdm as tqdm

def train_one_epoch(model, loader, epoch, optimizer, loss_fn, device):
    loop = tqdm(
        enumerate(loader, 1),
        total=len(loader),
        desc=f"Epoch {epoch}: train",
        leave=True,
    )
    model.train()
    train_loss = 0.0
    total = 0
    for i, batch in loop:
        optimizer.zero_grad()
        texts, labels = batch
        texts, labels = texts.to(device), labels.to(device)
        outputs = model(texts).to(device)
        
        
        loss = loss_fn(outputs.permute((0, 2, 1)), labels)
        total += labels.size(1)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        
        loop.set_postfix({"loss": train_loss/total})

def val_one_epoch():
    pass