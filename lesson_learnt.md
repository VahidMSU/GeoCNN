# **CNN-Transformer Spatio-Temporal Model Development**

This repository documents the development, optimization, and key insights from building a CNN-Transformer architecture for spatio-temporal data modeling. The process spanned nine versions, culminating in a highly optimized design suitable for geospatial predictions.

---
## **Key Learnings and Insights**

### **1. Learning Rate and Optimization**
- **Insight**: Learning rates below \( 1e-4 \) lead to underfitting. Optimal learning rates are essential for convergence.
- **Discussion**: Smaller learning rates can stabilize training but may hinder convergence. We have tried several learning scheduler to achieve better model training. We found the CosinAnnealing schedualer with cycles, hard restart and warmup period was the most effective one. The minimum learning rate that results in convergance was found to be 5e-5. 
convergence. We have tried several learning scheduler to achieve better model training. We found the CosinAnnealing schedualer with cycles, hard restart and warmup period was the most effective one. The minimum learning rate that results in convergance was found to be 5e-5. 

---

### **2. Activation Functions**
- **Insight**: Initially considered essential, activation functions for the final convolutional layer were found to be optional but occasionally helpful.
- **Discussion**: Sigmoid activation, while useful for bounded distributions, may not always be necessary if the data is pre-scaled. Properly trained models can learn the distribution inherently.

---

### **3. GPU and Parallel Processing**
- **Insight**: A100 GPUs processed the same model 3x slower than L40S GPUs. Parallel processing aided memory management but didn’t reduce training time significantly.
- **Discussion**: Optimization of EarlyStopping parameters, accumulation steps, sequence length, and batch size could further improve GPU utilization and speed. However, using mix percision could significantly reduce memory usage (almost halves the memory usage). Adjusting batch size and accumulation steps are essential to fit a model and its data efficiently into one or more GPU devices. 

---

### **4. Handling No-Values**
- **Insight**: Replacing no-values with small negative values (e.g., -0.01) and assigning higher loss weights to these areas significantly improved the model’s ability to learn.
- **Discussion**: Unlike common practices of excluding no-values, this approach preserves spatial integrity and allows training on sparse datasets without degradation.

---

### **5. Model Architecture Development**
- **Insight**: Nine versions of the CNN-Transformer (V1–V9) were developed and refined by each version to achieve an optimized architecture for robust and efficent spatial-temporal modeling capabilities. V8 emerged as the most efficient version:
  - **Key Features**: Deformable convolutions, sub-pixel convolutions, Squeeze-Excitation blocks.
  - **Challenge**: Separating static and dynamic channels increased parameters and degraded predictions.
- **Discussion**: Static features like DEM and soil data must be integrated into temporal modeling to capture their evolving interactions with dynamic inputs.

---

### **6. Loss Function Innovations**
- **Insight**: A custom **Spatio-Temporal Loss Function** incorporating:
  - **Quantile Weighted MSE (QWMSE)** for balanced error distribution.
  - **Boundary Masking** to improve edge predictions.
  - **Seasonal Weighting** to emphasize extreme ET levels.
- **Discussion**: Domain-specific loss functions ensure alignment with real-world phenomena and enable nuanced model training.

---

### **7. Batch Size and Processing**
- **Insight**: Smaller batch sizes (64x64) provided better stability and spatial coverage. Early stopping after seven epochs without improvement enhanced efficiency.
- **Discussion**: Smaller batches increase granularity but require careful tuning of gradient accumulation for computational efficiency.

---

### **8. Boundary Artifacts**
- **Insight**: Boundary-specific loss components and padding followed by cropping improved edge predictions and reduced inconsistencies.
- **Discussion**: Geometric priors could further refine edge handling in large-scale geospatial models.

---

### **9. Validation Stability**
- **Insight**: Fixing validation batch orders while shuffling only during training stabilized validation loss across epochs.
- **Discussion**: This ensures reliable performance metrics and aligns with standard best practices in validation protocols.

---

## **The Final Model: CNNTransformer V8**
**Key Features**:
- **Encoder**: Deformable convolutions, sub-pixel convolutions, and Squeeze-Excitation blocks.
- **Decoder**: Optimized for high-quality spatial reconstruction with edge-preserving filters.
- **Transformer**: Fourier positional encoding and multi-head attention for temporal dynamics.
- **Efficiency**: Streamlined design reduces memory usage and accelerates convergence.

---

## **Future Directions**
- **No-Values Handling**: Explore enhanced labeling and replacement strategies for sparse data.
- **Static-Dynamic Interactions**: Develop hybrid architectures to reduce parameter count while preserving temporal interactions.
- **Hardware Optimization**: Tailor designs for different GPU architectures to improve scalability.

This project showcases the iterative refinement process required for state-of-the-art spatio-temporal modeling, laying the foundation for robust geospatial applications.

