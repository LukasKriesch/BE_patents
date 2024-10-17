This repository contains the code used for creating a [dataset of bioeconomy-related inventions](https://osf.io/kj7fw/).
The repository contains three files:

  `train_pat.py`: Contains the code for fine-tuning an SBERT-based model using [SetFit](https://github.com/huggingface/setfit).
  
  `topic_modeling_be_pats.py`: Provides the code for generating a topic model with the  the [BERTopic-Framework](https://github.com/MaartenGr/BERTopic).
  
  `technical_validation.py`: Includes the code for comparing different pretrained SBERT models and evaluating them against keyword-based approaches for identifying bioeconomy-related inventions. This also includes code for a stratified validation of the classification results by CPC section.

The data folder contains the training data used in `train_pat.py`. The validation folder in the data folder contains the data used for the stratified validation of the classification results by CPC section.

