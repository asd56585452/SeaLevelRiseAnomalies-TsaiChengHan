### Summary Description of the Model

**Model Name**: DynamicCNN-based Anomaly Detection for 12 Locations

**Purpose**: The model is designed to predict anomalies in data across 12 predefined locations. Each location is evaluated as either anomalous (1) or normal (0). It accepts 2D numerical input data, preprocesses it to handle missing values, and outputs a binary classification for each location.

**Key Components**:
1. **DynamicCNN Architecture**:
   - A Convolutional Neural Network (CNN) with three convolutional layers and a fully connected layer.
   - Dropout layers and ReLU activations are used to improve generalization.
   - A max-pooling operation reduces dimensionality after each convolutional layer.
   - Outputs predictions for 12 locations.

2. **Preprocessing**:
   - Input data is cleaned by replacing NaN values with 0.
   - Data is reshaped to fit the expected input size for the CNN.

3. **Backend Details**:
   - The model is implemented using PyTorch.
   - It supports GPU acceleration when available.

4. **Pretrained Model**:
   - 5 pretrained model (`{1-5}best_sla_cnn_model.pth`) is loaded at initialization.
   - Those use 5-fold cross-validation method.

---

### README: Backend Execution Guide

```markdown
# Backend Execution Guide for DynamicCNN Anomaly Detection

## Prerequisites
Ensure the following dependencies are installed in your Python environment:
- Python 3.7+
- PyTorch (compatible with your hardware, including GPU if available)
- NumPy
- Pandas

Install missing dependencies using pip:
```bash
pip install torch numpy pandas
```

## Files Overview
- `model.py`: Contains the implementation of the DynamicCNN architecture and the prediction logic.
- `{1-5}best_sla_cnn_model.pth`: The heighest F1-score Pretrained model weights for test dataset during training. Ensure this file is located in the same directory as `model.py`.
- `my_sla.csv`: Used to store the results of the model's predictions on the training dataset.

## Running the Model

1. **Initialization**:
   Import the `Model` class from `model.py` and create an instance:
   ```python
   from model import Model
   model = Model()
   ```

2. **Input Preparation**:
   - Input data should be a 4D or 3D NumPy array (batch_size,1,100,160) or (1,100,160).
   - Ensure dimensions align with the model's expected input size.
   - Replace missing values (e.g., NaN) if needed.

3. **Prediction**:
   Use the `predict` method to get anomaly predictions:
   ```python
   input_data = np.random.random((1,100, 160))  # Example input
   predictions = model.predict(input_data)
   print(predictions)  # Outputs a binary array of shape (12,)
   ```

4. **Output**:
   The model outputs a binary array where:
   - `1` represents an anomaly.
   - `0` represents normal data.

## Notes
- **GPU Acceleration**: If a GPU is available, the model will automatically utilize it.
- **Handling Missing Data**: The model internally replaces NaN values with 0 during preprocessing.

## Troubleshooting
- Ensure the `{1-5}sla_cnn_model.pth` file is correctly placed and not corrupted.
- Verify that the input data has the correct dimensions and type (NumPy array).
- Check for compatibility between PyTorch and your hardware setup.

## Contact
For further assistance, contact the model's developer or refer to the project's documentation.
蔡承翰:[a565854525658545256585452@gmail.com](a565854525658545256585452@gmail.com)
