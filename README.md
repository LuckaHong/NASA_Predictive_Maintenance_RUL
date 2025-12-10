# Predictive Maintenance: RUL Prediction for Turbofan Engines (NASA C-MAPSS)

This project was developed as part of the Machine Learning course at ESILV. The objective is to develop a solution capable of predicting the Remaining Useful Life (RUL) of aircraft turbofan engines using simulated sensor data provided by NASA.

## Project Context

Predictive maintenance is a critical and important issue in the aerospace industry, in order to reduce unplanned maintenance costs and increase safety. By predicting failures, airlines can transition from corrective maintenance, which is when we repair an object when it is already broken, to predictive maintenance, which is when we repair just before failure.

The goal is to predict the number of flight cycles remaining before engine failure (Y) based on a time series of sensor measurements (X).

## Dataset

We use the FD001 dataset from the C-MAPSS challenge.
- Source: NASA Ames Prognostics Data Repository.
- Content: Multivariate time series of 100 engines in a run-to-failure setting.
- Features: 3 operational settings and 21 sensors (Temperature, Pressure, Fan Speed, etc.).

## Methodology and Pipeline

The project follows a structured Data Science pipeline detailed in the technical report.

1. Data Preparation and Feature Engineering
- Feature Selection: Removal of sensors with null or low variance (Sensors 1, 5, 6, 10, 16, 18, 19) based on correlation analysis.
- Target Engineering (Piecewise RUL): Calculation of RUL with a clipping threshold of 125 cycles. This models the fact that the engine is healthy in its early life stages and degradation is not immediately observable.
- Scaling: Application of MinMaxScaler to normalize sensor data between 0 and 1.
- Windowing: Transformation of data into time sequences (sliding window of 30 cycles) to feed the LSTM model.

2. Modeling
We implemented and compared four different approaches:
- Linear Regression: Used as a baseline model.
- Random Forest Regressor: An ensemble method using Bagging.
- Support Vector Regressor (SVR): Using the RBF kernel to handle non-linearity.
- Long Short-Term Memory (LSTM): A Deep Learning model specialized for time-series data.

## Results and Performance

The models were evaluated on the independent test set (test_FD001.txt). All results below integrate the RUL clipping strategy, where the treshold is 125 cycles.

Model: Linear Regression
- RMSE: 21.75 cycles
- MAE: 17.67 cycles
- R2: 0.7212
- NASA Score: 40847.55

Model: Random Forest Regressor
- RMSE: 19.02 cycles
- MAE: 13.77 cycles
- R2: 0.7869
- NASA Score: 64019.49

Model: Support Vector Regressor (SVR)
- RMSE: 19.52 cycles
- MAE: 12.75 cycles
- R2: 0.7753
- NASA Score: 109704.54

Model: LSTM (Deep Learning)
- RMSE: 15.38 cycles
- MAE: 11.31 cycles
- R2: 0.8630
- NASA Score: 413.49

In conclusion, The LSTM model obviously outperforms the other approaches byachieving the lowest error, with a RMSE 15.38, and the best NASA Score by far, NASA Score: 413.49, thus demonstrating its ability to capture temporal degradation patterns.

## Installation and Usage

To run the notebook, you need Python 3.8+ and the following libraries:

pip install pandas numpy scikit-learn matplotlib seaborn tensorflow keras

Project Structure:
- Projet ML TURBOFAN AVEC LSTM.ipynb: Main notebook containing code for exploration, cleaning, and modeling.
- data/: train_FD001.txt, test_FD001.txt, and RUL_FD001.txt.
- Report_Machine_learning_NASA.pdf: Detailed technical report of the project.

## Authors

The project was realized by Lucka Hong, Nicolas Huynh and Léo Hassenforder at Ecole Superieure d'Ingenieurs Leonard de Vinci (ESILV) in year 2025.

Based on the work of A. Saxena, K. Goebel, D. Simon, and N. Eklund, Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation, in the Proceedings of the 1st International Conference on Prognostics and Health Management (PHM08), Denver CO, Oct 2008.