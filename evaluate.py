import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score


def evaluate_model(model, dataloader, device,
                   return_preds=False,
                   return_probs=False,
                   loss_fn=None):
    """
    Evaluates a model on a validation or test set.
    
    Args:
        model (nn.Module): The trained model to evaluate.
        dataloader (DataLoader): The evaluation data loader.
        device (torch.device): Device to run evaluation on.
        return_preds (bool): If True, return true and predicted labels.
        return_probs (bool): If True, return class probabilities for ROC curve.
        loss_fn (callable, optional): Loss function (defaults to CrossEntropyLoss).
    
    Returns:
        tuple:
            avg_loss (float)
            accuracy (float)
            (optional) y_true, y_pred
            (optional) y_probs (probabilities for class 1)
    """
    model.eval()
    model.to(device)
    
    y_true, y_pred, y_probs = [], [], []
    total_loss = 0.0
    
    if loss_fn is None:
        loss_fn = torch.nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch in dataloader:
            # === Unpack input ===
            X = batch["input"].to(device)
            y = batch["label"].to(device)
            attn_mask = batch.get("attention_mask")
            if attn_mask is not None:
                attn_mask = attn_mask.to(device)
            
            # === Forward pass ===
            outputs = (model(X, attention_mask=attn_mask)
                       if getattr(model, "requires_attention_mask", False)
                       else model(X))
            
            # === Handle domain-adversarial outputs ===
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            
            # === Loss ===
            total_loss += loss_fn(outputs, y).item()
            
            if return_probs:
                probs = F.softmax(outputs, dim=1)
                y_probs.extend(probs[:, 1].cpu().numpy())
                preds = probs.argmax(dim=1)
            else:
                preds = outputs.argmax(dim=1)
            
            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(y_true, y_pred)
    
    results = [avg_loss, acc]
    if return_preds:
        results.extend([y_true, y_pred])
    if return_probs:
        results.append(y_probs)
    
    return tuple(results)
