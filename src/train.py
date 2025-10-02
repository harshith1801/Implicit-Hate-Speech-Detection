import torch
import torch.nn as nn
from tqdm.auto import tqdm
from sklearn.metrics import classification_report

def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler):
    """
    Performs a single training epoch and returns the average loss and accuracy.
    """
    model = model.train()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0
    
    progress_bar = tqdm(data_loader, desc="Training Epoch", leave=False)
    
    for batch in progress_bar:
        optimizer.zero_grad()
        
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels = batch["labels"].to(device)

        # Forward pass to get model outputs (logits)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # Calculate loss
        loss = loss_fn(outputs, labels)
        
        # Calculate accuracy
        _, preds = torch.max(outputs, dim=1)
        correct_predictions += torch.sum(preds == labels)
        total_samples += labels.size(0)
        
        total_loss += loss.item()
        
        # Backward pass and optimization
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        progress_bar.set_postfix({'loss': loss.item()})
        
    avg_loss = total_loss / len(data_loader)
    accuracy = correct_predictions.double() / total_samples
    
    return avg_loss, accuracy.item()

def eval_model(model, data_loader, loss_fn, device):
    """
    Evaluates the model on a validation/test set and returns loss, accuracy, and a classification report.
    """
    model = model.eval()
    
    total_loss = 0
    correct_predictions = 0
    total_samples = 0
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["labels"].to(device)

            # Get model predictions
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            
            # Calculate loss
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()

            _, preds = torch.max(outputs, dim=1)
            correct_predictions += torch.sum(preds == labels)
            total_samples += labels.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    avg_loss = total_loss / len(data_loader)
    accuracy = correct_predictions.double() / total_samples
    
    # Generate and return the classification report as a dictionary
    report = classification_report(all_labels, all_preds, zero_division=0, output_dict=True)
    
    return avg_loss, accuracy.item(), report

