# Vision System Software
The GeoCNN is a deep learning framework for hydrological modeling and prediction, leveraging two key datasets: **SWATCentral.h5** and **HydroGeoDataset.h5**. It provides advanced tools for deep learning applications tailored to hydrological simulations and data analysis.

## Model Performance Visualization
![Target vs Predicted Results](CNNTransformerRegressor_v8_AdaBelief_SpatioTemporalLoss_CosineAnnealingHardRestarts_lr0.0001_wd0.0001_w11_w20.1_nh8_nl6_fe4_embs1024_bs36_bw64_dropout0.3__target_vs_predicted.gif)

### Key Components

1. **Data Pipeline**  
   A fully integrated data pipeline capable of:
   - **Processing, Loading, Reloading, and Deloading:** Handles large-scale hydrological data efficiently.
   - **Queue Management:** Maintains a continuous flow of data during training, ensuring smooth operation.

2. **Deep Learning Models**  
   A robust suite of Vision System deep learning architectures:
   - **Inception-LSTM:** Combines spatial and temporal modeling.
   - **CNN-Transformers:**  
     1. Separate static and dynamic data processing.  
     2. Mixed static and dynamic data integration.  
     3. Enhanced versions with layer normalization across blocks.
   - **Fully Transformer-Based Models:** Two versions with different ViTBlock and handling spatial attentions.

3. **Extensible Configuration and Registry**  
   A flexible system for managing:
   - Data class configurations and parameter combinations.
   - Selection and setup of learning rate schedulers, optimization algorithms, deep learning models, early stopping criteria, and more.

4. **Specialized Loss Function**  
   A custom loss function designed to evaluate both spatial and temporal components of model predictions:
   - **Spatial Components:**
     - Boundary Loss
     - Overall Loss
     - Torrential Loss
     - No-Value Loss
     - Outlier Loss
   - **Temporal Components:**
     - Seasonal Losses (Winter, Fall, Spring, Summer)
5. **Inference**
    A seperate class desinged for inferencing, predicting usin
