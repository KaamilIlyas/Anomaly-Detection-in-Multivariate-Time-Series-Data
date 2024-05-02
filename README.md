# Anomaly Detection with Contrastive Learning and Transformer-based Autoencoder

This repository contains code for anomaly detection in multivariate time series data using contrastive learning and a Transformer-based Autoencoder. The approach integrates data augmentation with geometric distribution masks, a Transformer-based Autoencoder architecture, and contrastive loss to achieve superior performance in anomaly detection.

## Dataset
The dataset used for this project is provided in CSV format and consists of three files:
- `train.csv`: Contains training data with columns timestamp_(min), feature_0, ..., feature_24.
- `test.csv`: Contains test data with the same columns as `train.csv`.
- `test_label.csv`: Contains labels for test data, indicating anomalies with value 1 and normal data with value 0.

## Requirements
- Python 3.x
- PyTorch
- pandas
- NumPy
- scikit-learn

## Usage
1. Install the required dependencies using `pip install -r requirements.txt`.
2. Place the dataset files (`train.csv`, `test.csv`, `test_label.csv`) in the project directory.
3. Run the `main.py` script to train the contrastive model and detect anomalies.

## File Descriptions
- `main.py`: Main script to load data, train the contrastive model, and detect anomalies.
- `README.md`: Detailed information about the project, including setup instructions and usage guidelines.

## References
- D. Kingma, J. Ba. "Adam: A Method for Stochastic Optimization." arXiv:1412.6980.
- H. Hotelling. "Analysis of a Complex of Statistical Variables into Principal Components." Journal of Educational Psychology, 24(6), 417–441.
- E. W. Dijkstra. "A note on two problems in connexion with graphs." Numerische Mathematik, 1(1), 269–271.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
