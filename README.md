# Passive-Sensing-using-smart-phone-sensors-Studentlife-dataset-
This project investigates the feasibility of predicting stress levels using passive smartphone sensing data from the StudentLife dataset. Behavioral features related to mobility, physical activity, sleep patterns, social interactions, and Bluetooth proximity were extracted from raw smartphone sensor data and aggregated on a daily basis.

To account for individual behavioral differences, personalized baseline features were generated using rolling historical statistics and deviation-based normalization. Temporal patterns were captured using MiniRocket, a state-of-the-art time-series feature extraction method, while XGBoost was used for stress classification.

The proposed approach combines passive sensing, personalized behavioral modeling, and machine learning to identify stress-related behavioral changes without requiring continuous user input. Experimental results demonstrate that incorporating personal baseline features improves model generalization and stress detection performance by focusing on deviations from an individual's normal routine rather than absolute behavioral values.
