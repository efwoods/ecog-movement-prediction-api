def predict_and_export(model, data_loader, device, output_file_path):
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    predictions = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    
    # Save as .mat file for visualization
    savemat(output_file_path, {
        "predictions":predictions,
        "targets": targets
    })
    print("Saved predictions to ecog_predictions.mat")

    return predictions, targets