import os
import pickle
import glob
from pathlib import Path

class ModelLoader:
    def __init__(self, models_dir="models"):
        """
        Initialize the ModelLoader with a specified models directory.
        
        Args:
            models_dir (str): Path to the directory containing model files
        """
        self.models_dir = Path(models_dir)
        self.loaded_models = {}
        
    def load_all_models(self):
        """
        Load all .pkl files from the models directory.
        
        Returns:
            dict: Dictionary with model names as keys and loaded models as values
        """
        if not self.models_dir.exists():
            print(f"Models directory '{self.models_dir}' does not exist!")
            return {}
        
        # Find all .pkl files in the models directory
        model_files = glob.glob(str(self.models_dir / "*.pkl"))
        
        if not model_files:
            print(f"No .pkl files found in '{self.models_dir}'")
            return {}
        
        print(f"Found {len(model_files)} model files:")
        
        for model_path in model_files:
            model_name = Path(model_path).stem  # Get filename without extension
            try:
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                self.loaded_models[model_name] = model
                print(f"✓ Loaded: {model_name}")
            except Exception as e:
                print(f"✗ Failed to load {model_name}: {str(e)}")
        
        return self.loaded_models
    
    def load_specific_model(self, model_name):
        """
        Load a specific model by name.
        
        Args:
            model_name (str): Name of the model file (with or without .pkl extension)
            
        Returns:
            object: Loaded model or None if failed
        """
        # Remove .pkl extension if provided
        if model_name.endswith('.pkl'):
            model_name = model_name[:-4]
        
        model_path = self.models_dir / f"{model_name}.pkl"
        
        if not model_path.exists():
            print(f"Model file '{model_path}' does not exist!")
            return None
        
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            self.loaded_models[model_name] = model
            print(f"✓ Loaded: {model_name}")
            return model
        except Exception as e:
            print(f"✗ Failed to load {model_name}: {str(e)}")
            return None
    
    def get_model(self, model_name):
        """
        Get a loaded model by name.
        
        Args:
            model_name (str): Name of the model
            
        Returns:
            object: The loaded model or None if not found
        """
        return self.loaded_models.get(model_name)
    
    def list_loaded_models(self):
        """
        Print all currently loaded models.
        """
        if not self.loaded_models:
            print("No models currently loaded.")
        else:
            print("Loaded models:")
            for model_name in self.loaded_models.keys():
                print(f"  - {model_name}")
    
    def get_available_models(self):
        """
        Get list of all available model files in the directory.
        
        Returns:
            list: List of available model file names (without .pkl extension)
        """
        if not self.models_dir.exists():
            return []
        
        model_files = glob.glob(str(self.models_dir / "*.pkl"))
        return [Path(f).stem for f in model_files]

# Example usage
if __name__ == "__main__":
    # Initialize the model loader
    loader = ModelLoader("models")  # Change path if needed
    
    # Load all models
    models = loader.load_all_models()
    
    # List loaded models
    loader.list_loaded_models()
    
    # Access a specific model (replace with your actual model name)
    # nr_ahr_model = loader.get_model("NR-AhR_model")
    # if nr_ahr_model:
    #     print(f"NR-AhR model loaded successfully!")
    #     # Use the model for predictions, etc.
    
    # Load a specific model if needed
    # specific_model = loader.load_specific_model("SR-ARE_model")
    
    # Get list of available models
    available_models = loader.get_available_models()
    print(f"\nAvailable models: {available_models}")
