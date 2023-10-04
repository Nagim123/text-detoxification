import torch
import tqdm as tqdm

def train_one_epoch(model, loader, epoch, loss_fn, optimizer, device):
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
        
        
        loss = loss_fn(outputs, labels)
        total += labels.size(1)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        
        loop.set_postfix({"loss": train_loss/total})

def val_one_epoch(model, loader, epoch, loss_fn, optimizer, device):
    loop = tqdm(
        enumerate(loader, 1),
        total=len(loader),
        desc=f"Epoch {epoch}: val",
        leave=True,
    )
    val_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_predicted = []
    with torch.no_grad():
        model.eval()  # evaluation mode
        for i, batch in loop:
            texts, labels = batch
            texts, labels = texts.to(device), labels.to(device)
            
            outputs = model(texts).to(device)
            loss = loss_fn(outputs, labels)
            
            #_, predicted = torch.max(outputs, 2)
            #total += labels.size(1)
            #correct += (predicted == labels).sum().item()
            #all_labels.append(labels)
            #all_predicted.append(predicted)
            
            val_loss += loss.item()
            loop.set_postfix({"loss": val_loss/total})
        