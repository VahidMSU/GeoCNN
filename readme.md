# Vision System Software
The Vision System is a comprehensive framework for hydrological modeling and prediction, leveraging two key datasets: **SWATCentral.h5** and **HydroGeoDataset.h5**. It provides advanced tools for deep learning applications tailored to hydrological simulations and data analysis.

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

### Clean Modularization  
The entire system is modularized for scalability, and ease of maintenance. The module has one controller, one pipline, trainer, prediction, inference, losses, registry, helpers, utils. 


classDiagram
    class CNNTransformerModel {
        +encoder: Encoder
        +transformer: TransformerLayer
        +decoder: Decoder
        +custom_loss_function()
        +train()
        +validate()
    }
    class Encoder {
        +deformable_convolutions()
        +sub_pixel_convolutions()
        +squeeze_excitation_blocks()
    }
    class TransformerLayer {
        +fourier_positional_encoding()
        +multi_head_attention()
    }
    class Decoder {
        +edge_preserving_filters()
        +spatial_reconstruction()
    }
    class CustomLossFunction {
        +quantile_weighted_mse()
        +boundary_masking()
        +seasonal_weighting()
    }
    CNNTransformerModel --> Encoder
    CNNTransformerModel --> TransformerLayer
    CNNTransformerModel --> Decoder
    CNNTransformerModel --> CustomLossFunction