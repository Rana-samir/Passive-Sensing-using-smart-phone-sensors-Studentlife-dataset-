from typing import Dict
import numpy as np
import pandas as pd
import joblib
import os
from app.utils.daily_merge import cast_datetime


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "pipeline.pkl")


class MoodPredictor:
    def __init__(self):
        self.model_path = MODEL_PATH
        self._minirocket = None
        self._selected_indices = None
        self._model = None
        self.FEATURE_NAMES = [
            'created_at',
            'activity_stationary_ratio', 'activity_walking_ratio', 'activity_running_ratio', 'activity_unknown_ratio',
            'conversation_count', 'total_conversation_minutes', 'avg_conversation_minutes',
            'avg_bt_level', 'num_bt_devices',
            'dist_total_km', 'dist_mean_km', 'dist_std_km', 'lat_std', 'lon_std', 'home_points',
            'sleep_duration', 'sleep_quality',
            'sleep_duration_z', 'dist_total_km_z', 'conversation_count_z', 'total_conversation_minutes_z', 'num_bt_devices_z',
            'activity_stationary_ratio_z', 'activity_walking_ratio_z', 'activity_running_ratio_z',
        ]
        self.WINDOW_SIZE = 10
        self._THRESHOLD = 0.35
    
    def load_model(self):
        if self._model is None:
            package = joblib.load(self.model_path)
            self._minirocket = package["minirocket"]
            self._selected_indices = package["selected_indices"]
            self._model = package["model"]
    
    def predict(self, user_daily_data: pd.DataFrame) -> Dict[str, int | float]:
        # 1. Load model (once)
        self.load_model()

        # 2. Filter and Reorder columns
        user_daily_data = user_daily_data[self.FEATURE_NAMES]

        # 3. Sort by date
        cast_datetime(user_daily_data, 'created_at')
        user_daily_data = user_daily_data.sort_values('created_at')

        # 4. Identify feature columns
        exclude_cols = ['created_at']
        feature_cols = [col for col in user_daily_data.columns if col not in exclude_cols]
        features = user_daily_data[feature_cols].values.T  # shape: (num_features, num_days)
        n_days = features.shape[1]

        # 5. Create sliding windows
        windows_X = []
        for day in range(self.WINDOW_SIZE, n_days):
            window = features[:, day - self.WINDOW_SIZE:day]
            windows_X.append(window)

        X_windows = np.stack(windows_X)                # (num_windows, num_features, window_size)
        X_windows = np.transpose(X_windows, (0, 2, 1)) # (num_windows, window_size, num_features)

        # 6. Transform and select features
        X_mr = self._minirocket.transform(X_windows)

        X_mr = np.asarray(X_mr)

        X_selected = X_mr[:, self._selected_indices]

        # 7. Predict probabilities and apply threshold
        probs = self._model.predict_proba(X_selected)[:, 1]         # stress confidence
        predictions = (probs >= self._THRESHOLD).astype(int)        # binary prediction

        return {
            'is_stressed': int(predictions[0]),
            'confidence': float(probs[0]),
        }


predictor = MoodPredictor()