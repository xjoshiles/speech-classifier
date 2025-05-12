import copy
import torch
from torch import nn
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from config import USE_GPU, EPOCHS, LEARNING_RATE


def train_model(model,
                train_loader,
                val_loader=None,
                loss_fn=None,
                val_loss_fn=None,
                domain_loss_fn=None,
                return_loss_history=False,
                early_stopping_patience=5,
                min_delta=0.0,
                domain_weight=1.0):
    """
    Trains a PyTorch model with optional Domain-Adversarial Training (DAT).
    
    Args:
        model (nn.Module): Model to be trained.
        train_loader (DataLoader): Training data.
        val_loader (DataLoader, optional): Validation data.
        loss_fn (callable): Main classification loss.
        val_loss_fn (callable): Validation loss.
        domain_loss_fn (callable, optional): Domain classification loss (for DAT).
        return_loss_history (bool): Whether to return loss curves.
        early_stopping_patience (int): Max epochs to wait for val improvement.
        min_delta (float): Minimum change in val loss to count as improvement.
        domain_weight (float): Relative weight of domain loss in DAT.
    
    Returns:
        nn.Module or (model, train_loss_history, val_loss_history)
    """
    assert loss_fn is not None, "Classification loss function must be provided"
    if val_loader:
        assert val_loss_fn is not None, "Validation loss function must be provided"
    
    device = torch.device("cuda" if USE_GPU else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = GradScaler() if USE_GPU else None
    
    if domain_loss_fn is None:
        if hasattr(train_loader.dataset, "compute_domain_weights"):
            domain_weights = train_loader.dataset.compute_domain_weights().to(device)
            
            # Check if weights are all (approximately) equal
            if torch.allclose(domain_weights, domain_weights[0].expand_as(domain_weights), rtol=0.01):
                domain_loss_fn = nn.CrossEntropyLoss()
            else:
                domain_loss_fn = nn.CrossEntropyLoss(weight=domain_weights)
        else:
            domain_loss_fn = nn.CrossEntropyLoss()
    
    train_loss_history = []
    val_loss_history = []
    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0
    
    print(f"\n[INFO] Training {model.__class__.__name__} on device: {device}")
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            # Move inputs and labels to device
            X = batch["input"].to(device)
            y_cls = batch["label"].to(device)
            y_domain = batch["domain"].to(device)
            attn_mask = batch.get("attention_mask")
            if attn_mask is not None:
                attn_mask = attn_mask.to(device)
            
            optimizer.zero_grad()
            
            def forward_and_loss():
                # Forward pass with or without attention mask
                output = (
                    model(X, attention_mask=attn_mask)
                    if hasattr(model, "requires_attention_mask") and model.requires_attention_mask
                    else model(X)
                )
                
                # If model returns both class and domain logits (DAT)
                if isinstance(output, tuple):
                    class_logits, domain_logits = output
                    loss = loss_fn(class_logits, y_cls)
                    if domain_loss_fn:
                        domain_loss = domain_loss_fn(domain_logits, y_domain)
                        loss += domain_weight * domain_loss
                    return loss
                else:
                    return loss_fn(output, y_cls)
            
            # Backpropagation
            if USE_GPU:
                # Use mixed precision with AMP on CUDA (leverages Tensor cores)
                with autocast(device_type='cuda'):
                    loss = forward_and_loss()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard precision training on CPU
                loss = forward_and_loss()
                loss.backward()
                optimizer.step()
            
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_loader)
        train_loss_history.append(avg_train_loss)
        
        # === Validation ===
        if val_loader:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    X = batch["input"].to(device)
                    y_val = batch["label"].to(device)
                    attn_mask = batch.get("attention_mask")
                    if attn_mask is not None:
                        attn_mask = attn_mask.to(device)
                    
                    output = (
                        model(X, attention_mask=attn_mask)
                        if hasattr(model, "requires_attention_mask") and model.requires_attention_mask
                        else model(X)
                    )
                    
                    # For DAT models, only use classification logits
                    if isinstance(output, tuple):
                        output = output[0]
                    
                    loss = val_loss_fn(output, y_val)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            val_loss_history.append(avg_val_loss)
            
            print(f"📉 Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f} | Val Loss = {avg_val_loss:.4f}")
            
            # Early stopping
            if best_val_loss - avg_val_loss > min_delta:
                best_val_loss = avg_val_loss
                best_model_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"⚠️  No val improvement. Patience: {patience_counter}/{early_stopping_patience}")
                if patience_counter >= early_stopping_patience:
                    print(f"🛑 Early stopping triggered at epoch {epoch+1}")
                    break
        else:
            print(f"📉 Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}")
        
        torch.cuda.empty_cache()
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"✅ Best model restored (lowest val loss = {best_val_loss:.4f})")
    
    return (model, train_loss_history, val_loss_history) if return_loss_history else model

