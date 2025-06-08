from ml_reimbursement_engine import EnhancedMLReimbursementEngine
from sklearn.tree import DecisionTreeRegressor
import pandas as pd

# Load data
X, y = EnhancedMLReimbursementEngine.load_training_data()
engine = EnhancedMLReimbursementEngine(model_type='dt')

# Train unconstrained Decision Tree
engine.model = DecisionTreeRegressor(max_depth=None, min_samples_leaf=1, random_state=42)
X_features = engine._create_features(X)
from sklearn.preprocessing import RobustScaler
engine.scaler = RobustScaler()
X_scaled = engine.scaler.fit_transform(X_features)
engine.model.fit(X_scaled, y)
engine.is_trained = True
engine.save_model('reimbursement_model.joblib')
print('Unconstrained DecisionTreeRegressor trained and saved.')
