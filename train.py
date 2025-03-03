import torch
import matplotlib.pyplot as plt

def train_model(model, train_loader, criterion, optimizer, num_epochs):
    all_epoch_losses = []
    for epoch in range(num_epochs):
        model.train()
        epoch_losses = []
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {avg_loss:.7f}")
        all_epoch_losses.append(avg_loss)
        plt.figure(figsize=(8, 6))
        plt.hist(epoch_losses, bins=20, color='skyblue', edgecolor='black')
        plt.title(f"Epoch {epoch + 1} Loss Distribution")
        plt.xlabel("Loss")
        plt.ylabel("Frequency")
        plt.savefig(f"loss_hist_epoch_{epoch+1}.png", bbox_inches="tight", dpi=300)
        plt.close()
    return all_epoch_losses

def train_surrogate(model, train_loader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
        print(f"Surrogate Epoch {epoch + 1}/{num_epochs}")

def adversarial_training(model, train_loader, criterion, optimizer, num_epochs, attack_fn):
    for epoch in range(num_epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            X_batch_adv = attack_fn(model, X_batch, y_batch, epsilon=0.1)
            X_combined = torch.cat([X_batch, X_batch_adv], dim=0)
            y_combined = torch.cat([y_batch, y_batch], dim=0)
            outputs = model(X_combined)
            loss = criterion(outputs, y_combined)
            loss.backward()
            optimizer.step()
        print(f"Adversarial Training Epoch {epoch + 1}/{num_epochs}")