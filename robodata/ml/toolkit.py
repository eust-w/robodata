"""Machine Learning integration for Robodata.

Provides ML model integration, training utilities, and
inference capabilities for robotics datasets.
"""

from __future__ import annotations

import pickle
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from robodata.core.logging_system import get_logger
from robodata.core.models import DatasetInfo, Episode, Frame


class ModelType(Enum):
    """Types of ML models."""
    BEHAVIOR_CLONING = "behavior_cloning"
    INVERSE_DYNAMICS = "inverse_dynamics"
    FORWARD_DYNAMICS = "forward_dynamics"
    REWARD_MODEL = "reward_model"
    VALUE_FUNCTION = "value_function"
    POLICY = "policy"
    Q_FUNCTION = "q_function"


class TrainingMode(Enum):
    """Training modes."""
    SUPERVISED = "supervised"
    REINFORCEMENT = "reinforcement"
    SELF_SUPERVISED = "self_supervised"
    SEMI_SUPERVISED = "semi_supervised"


@dataclass
class ModelConfig:
    """Configuration for ML models."""
    model_type: ModelType
    architecture: str = "mlp"
    hidden_dims: List[int] = field(default_factory=lambda: [256, 256])
    activation: str = "relu"
    dropout: float = 0.1
    learning_rate: float = 1e-3
    batch_size: int = 64
    epochs: int = 100
    validation_split: float = 0.2
    early_stopping: bool = True
    patience: int = 10


@dataclass
class TrainingResult:
    """Result of model training."""
    model: Any
    training_history: Dict[str, List[float]]
    validation_metrics: Dict[str, float]
    training_time: float
    model_path: Optional[Path] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PredictionResult:
    """Result of model prediction."""
    predictions: np.ndarray
    confidence: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class MLModel(ABC):
    """Abstract base class for ML models."""
    
    @abstractmethod
    def train(self, episodes: List[Episode], config: ModelConfig) -> TrainingResult:
        """Train the model."""
        pass
    
    @abstractmethod
    def predict(self, frames: List[Frame]) -> PredictionResult:
        """Make predictions."""
        pass
    
    @abstractmethod
    def save(self, path: Path) -> None:
        """Save the model."""
        pass
    
    @abstractmethod
    def load(self, path: Path) -> None:
        """Load the model."""
        pass
    
    @property
    @abstractmethod
    def model_type(self) -> ModelType:
        """Model type."""
        pass


class BehaviorCloningModel(MLModel):
    """Behavior cloning model for imitation learning."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.model = None
        self.scaler = None
        self._is_trained = False
    
    @property
    def model_type(self) -> ModelType:
        return ModelType.BEHAVIOR_CLONING
    
    def train(self, episodes: List[Episode], config: ModelConfig) -> TrainingResult:
        """Train behavior cloning model."""
        self.logger.info(f"Training behavior cloning model with {len(episodes)} episodes")
        start_time = time.time()
        
        # Prepare training data
        X_train, y_train, X_val, y_val = self._prepare_data(episodes, config.validation_split)
        
        # Build model
        self.model = self._build_model(config, X_train.shape[1], y_train.shape[1])
        
        # Train model
        training_history = self._train_model(
            self.model, X_train, y_train, X_val, y_val, config
        )
        
        # Evaluate
        validation_metrics = self._evaluate_model(self.model, X_val, y_val)
        
        training_time = time.time() - start_time
        self._is_trained = True
        
        self.logger.info(f"Training completed in {training_time:.2f}s")
        
        return TrainingResult(
            model=self.model,
            training_history=training_history,
            validation_metrics=validation_metrics,
            training_time=training_time,
            metadata={"model_type": self.model_type.value, "config": config}
        )
    
    def predict(self, frames: List[Frame]) -> PredictionResult:
        """Predict actions from states."""
        if not self._is_trained:
            raise ValueError("Model must be trained before prediction")
        
        # Extract states
        states = []
        for frame in frames:
            if frame.state is not None:
                states.append(np.array(frame.state).flatten())
        
        if not states:
            raise ValueError("No valid states found in frames")
        
        X = np.array(states)
        
        # Scale features
        if self.scaler:
            X = self.scaler.transform(X)
        
        # Predict
        predictions = self.model.predict(X)
        
        return PredictionResult(
            predictions=predictions,
            metadata={"model_type": self.model_type.value}
        )
    
    def save(self, path: Path) -> None:
        """Save the model."""
        path.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "is_trained": self._is_trained,
            "model_type": self.model_type.value
        }
        
        with open(path / "model.pkl", "wb") as f:
            pickle.dump(model_data, f)
        
        self.logger.info(f"Model saved to {path}")
    
    def load(self, path: Path) -> None:
        """Load the model."""
        with open(path / "model.pkl", "rb") as f:
            model_data = pickle.load(f)
        
        self.model = model_data["model"]
        self.scaler = model_data["scaler"]
        self._is_trained = model_data["is_trained"]
        
        self.logger.info(f"Model loaded from {path}")
    
    def _prepare_data(self, episodes: List[Episode], validation_split: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare training and validation data."""
        states = []
        actions = []
        
        for episode in episodes:
            for frame in episode.load_frames():
                if frame.state is not None and frame.action is not None:
                    states.append(np.array(frame.state).flatten())
                    actions.append(np.array(frame.action).flatten())
        
        if not states or not actions:
            raise ValueError("No valid state-action pairs found")
        
        X = np.array(states)
        y = np.array(actions)
        
        # Split data
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Normalize features
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)
        
        return X_train, y_train, X_val, y_val
    
    def _build_model(self, config: ModelConfig, input_dim: int, output_dim: int) -> Any:
        """Build the neural network model."""
        try:
            import tensorflow as tf
            from tensorflow import keras
            
            # Build MLP
            model = keras.Sequential()
            model.add(keras.layers.Input(shape=(input_dim,)))
            
            for hidden_dim in config.hidden_dims:
                model.add(keras.layers.Dense(hidden_dim, activation=config.activation))
                if config.dropout > 0:
                    model.add(keras.layers.Dropout(config.dropout))
            
            model.add(keras.layers.Dense(output_dim))
            
            # Compile
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=config.learning_rate),
                loss="mse",
                metrics=["mae"]
            )
            
            return model
        
        except ImportError:
            # Fallback to sklearn
            from sklearn.neural_network import MLPRegressor
            
            model = MLPRegressor(
                hidden_layer_sizes=config.hidden_dims,
                activation=config.activation,
                learning_rate_init=config.learning_rate,
                max_iter=config.epochs,
                random_state=42
            )
            
            return model
    
    def _train_model(self, model: Any, X_train: np.ndarray, y_train: np.ndarray, 
                    X_val: np.ndarray, y_val: np.ndarray, config: ModelConfig) -> Dict[str, List[float]]:
        """Train the model."""
        try:
            # TensorFlow/Keras training
            import tensorflow as tf
            
            callbacks = []
            if config.early_stopping:
                early_stop = tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=config.patience,
                    restore_best_weights=True
                )
                callbacks.append(early_stop)
            
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                batch_size=config.batch_size,
                epochs=config.epochs,
                callbacks=callbacks,
                verbose=0
            )
            
            return {
                "loss": history.history["loss"],
                "val_loss": history.history["val_loss"],
                "mae": history.history["mae"],
                "val_mae": history.history["val_mae"]
            }
        
        except ImportError:
            # Scikit-learn training
            model.fit(X_train, y_train)
            
            # Simulate training history
            train_losses = []
            val_losses = []
            
            for epoch in range(config.epochs):
                train_pred = model.predict(X_train)
                val_pred = model.predict(X_val)
                
                train_loss = np.mean((train_pred - y_train) ** 2)
                val_loss = np.mean((val_pred - y_val) ** 2)
                
                train_losses.append(train_loss)
                val_losses.append(val_loss)
            
            return {
                "loss": train_losses,
                "val_loss": val_losses,
                "mae": train_losses,  # Placeholder
                "val_mae": val_losses   # Placeholder
            }
    
    def _evaluate_model(self, model: Any, X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, float]:
        """Evaluate the model."""
        predictions = model.predict(X_val)
        
        mse = np.mean((predictions - y_val) ** 2)
        mae = np.mean(np.abs(predictions - y_val))
        
        # R^2 score
        ss_res = np.sum((y_val - predictions) ** 2)
        ss_tot = np.sum((y_val - np.mean(y_val)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        return {
            "mse": mse,
            "mae": mae,
            "r2": r2
        }


class RewardModel(MLModel):
    """Reward model for reinforcement learning."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.model = None
        self._is_trained = False
    
    @property
    def model_type(self) -> ModelType:
        return ModelType.REWARD_MODEL
    
    def train(self, episodes: List[Episode], config: ModelConfig) -> TrainingResult:
        """Train reward model."""
        self.logger.info(f"Training reward model with {len(episodes)} episodes")
        start_time = time.time()
        
        # Prepare training data (state-action-reward tuples)
        X_train, y_train, X_val, y_val = self._prepare_reward_data(episodes, config.validation_split)
        
        # Build model
        self.model = self._build_reward_model(config, X_train.shape[1])
        
        # Train model
        training_history = self._train_model(
            self.model, X_train, y_train, X_val, y_val, config
        )
        
        # Evaluate
        validation_metrics = self._evaluate_model(self.model, X_val, y_val)
        
        training_time = time.time() - start_time
        self._is_trained = True
        
        return TrainingResult(
            model=self.model,
            training_history=training_history,
            validation_metrics=validation_metrics,
            training_time=training_time,
            metadata={"model_type": self.model_type.value}
        )
    
    def predict(self, frames: List[Frame]) -> PredictionResult:
        """Predict rewards."""
        if not self._is_trained:
            raise ValueError("Model must be trained before prediction")
        
        # Extract state-action pairs
        features = []
        for frame in frames:
            if frame.state is not None and frame.action is not None:
                feature = np.concatenate([
                    np.array(frame.state).flatten(),
                    np.array(frame.action).flatten()
                ])
                features.append(feature)
        
        if not features:
            raise ValueError("No valid state-action pairs found")
        
        X = np.array(features)
        predictions = self.model.predict(X)
        
        return PredictionResult(
            predictions=predictions,
            metadata={"model_type": self.model_type.value}
        )
    
    def save(self, path: Path) -> None:
        """Save the model."""
        path.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            "model": self.model,
            "is_trained": self._is_trained,
            "model_type": self.model_type.value
        }
        
        with open(path / "reward_model.pkl", "wb") as f:
            pickle.dump(model_data, f)
    
    def load(self, path: Path) -> None:
        """Load the model."""
        with open(path / "reward_model.pkl", "rb") as f:
            model_data = pickle.load(f)
        
        self.model = model_data["model"]
        self._is_trained = model_data["is_trained"]
    
    def _prepare_reward_data(self, episodes: List[Episode], validation_split: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare reward training data."""
        features = []
        rewards = []
        
        for episode in episodes:
            for frame in episode.load_frames():
                if (frame.state is not None and 
                    frame.action is not None and 
                    hasattr(frame, 'reward') and 
                    frame.reward is not None):
                    
                    feature = np.concatenate([
                        np.array(frame.state).flatten(),
                        np.array(frame.action).flatten()
                    ])
                    features.append(feature)
                    rewards.append(frame.reward)
        
        if not features:
            raise ValueError("No valid reward data found")
        
        X = np.array(features)
        y = np.array(rewards)
        
        # Split data
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        return X_train, y_train, X_val, y_val
    
    def _build_reward_model(self, config: ModelConfig, input_dim: int) -> Any:
        """Build reward model."""
        try:
            import tensorflow as tf
            from tensorflow import keras
            
            model = keras.Sequential([
                keras.layers.Input(shape=(input_dim,)),
                keras.layers.Dense(128, activation='relu'),
                keras.layers.Dropout(config.dropout),
                keras.layers.Dense(64, activation='relu'),
                keras.layers.Dropout(config.dropout),
                keras.layers.Dense(1)  # Single reward output
            ])
            
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=config.learning_rate),
                loss="mse",
                metrics=["mae"]
            )
            
            return model
        
        except ImportError:
            from sklearn.ensemble import RandomForestRegressor
            
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            
            return model
    
    def _train_model(self, model: Any, X_train: np.ndarray, y_train: np.ndarray,
                    X_val: np.ndarray, y_val: np.ndarray, config: ModelConfig) -> Dict[str, List[float]]:
        """Train reward model."""
        try:
            import tensorflow as tf
            
            callbacks = []
            if config.early_stopping:
                early_stop = tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=config.patience,
                    restore_best_weights=True
                )
                callbacks.append(early_stop)
            
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                batch_size=config.batch_size,
                epochs=config.epochs,
                callbacks=callbacks,
                verbose=0
            )
            
            return {
                "loss": history.history["loss"],
                "val_loss": history.history["val_loss"]
            }
        
        except ImportError:
            model.fit(X_train, y_train)
            
            # Simulate training history
            train_losses = []
            val_losses = []
            
            for epoch in range(config.epochs):
                train_pred = model.predict(X_train)
                val_pred = model.predict(X_val)
                
                train_loss = np.mean((train_pred - y_train) ** 2)
                val_loss = np.mean((val_pred - y_val) ** 2)
                
                train_losses.append(train_loss)
                val_losses.append(val_loss)
            
            return {"loss": train_losses, "val_loss": val_losses}
    
    def _evaluate_model(self, model: Any, X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, float]:
        """Evaluate reward model."""
        predictions = model.predict(X_val)
        
        mse = np.mean((predictions - y_val) ** 2)
        mae = np.mean(np.abs(predictions - y_val))
        
        return {"mse": mse, "mae": mae}


class MLToolkit:
    """Machine learning toolkit for robotics datasets."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.models: Dict[str, MLModel] = {}
    
    def create_model(self, model_type: ModelType, model_name: str) -> MLModel:
        """Create a new model."""
        if model_type == ModelType.BEHAVIOR_CLONING:
            model = BehaviorCloningModel()
        elif model_type == ModelType.REWARD_MODEL:
            model = RewardModel()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        self.models[model_name] = model
        self.logger.info(f"Created {model_type.value} model: {model_name}")
        
        return model
    
    def train_model(
        self, 
        model_name: str, 
        episodes: List[Episode], 
        config: ModelConfig
    ) -> TrainingResult:
        """Train a model."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        result = model.train(episodes, config)
        
        self.logger.info(f"Model {model_name} training completed")
        return result
    
    def predict(self, model_name: str, frames: List[Frame]) -> PredictionResult:
        """Make predictions with a model."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        return model.predict(frames)
    
    def save_model(self, model_name: str, path: Path) -> None:
        """Save a model."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        model.save(path)
    
    def load_model(self, model_name: str, path: Path) -> None:
        """Load a model."""
        if model_name not in self.models:
            # Create model based on saved data
            model_data_path = path / "model.pkl" if (path / "model.pkl").exists() else path / "reward_model.pkl"
            
            with open(model_data_path, "rb") as f:
                model_data = pickle.load(f)
            
            model_type_str = model_data["model_type"]
            model_type = ModelType(model_type_str)
            
            model = self.create_model(model_type, model_name)
        
        model = self.models[model_name]
        model.load(path)
    
    def evaluate_model(self, model_name: str, episodes: List[Episode]) -> Dict[str, float]:
        """Evaluate a model on test data."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        
        # Prepare test data
        test_frames = []
        for episode in episodes:
            test_frames.extend(list(episode.load_frames()))
        
        # Make predictions
        predictions = model.predict(test_frames)
        
        # Simple evaluation metrics
        return {
            "num_predictions": len(predictions.predictions),
            "prediction_mean": np.mean(predictions.predictions),
            "prediction_std": np.std(predictions.predictions)
        }
    
    def list_models(self) -> List[str]:
        """List all available models."""
        return list(self.models.keys())
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about a model."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        
        return {
            "name": model_name,
            "type": model.model_type.value,
            "is_trained": hasattr(model, '_is_trained') and model._is_trained
        }


# Convenience functions
def train_behavior_cloning(
    episodes: List[Episode],
    model_config: Optional[ModelConfig] = None,
    model_name: str = "bc_model"
) -> TrainingResult:
    """Train a behavior cloning model."""
    toolkit = MLToolkit()
    
    if model_config is None:
        model_config = ModelConfig(model_type=ModelType.BEHAVIOR_CLONING)
    
    model = toolkit.create_model(ModelType.BEHAVIOR_CLONING, model_name)
    return toolkit.train_model(model_name, episodes, model_config)


def train_reward_model(
    episodes: List[Episode],
    model_config: Optional[ModelConfig] = None,
    model_name: str = "reward_model"
) -> TrainingResult:
    """Train a reward model."""
    toolkit = MLToolkit()
    
    if model_config is None:
        model_config = ModelConfig(model_type=ModelType.REWARD_MODEL)
    
    model = toolkit.create_model(ModelType.REWARD_MODEL, model_name)
    return toolkit.train_model(model_name, episodes, model_config)
