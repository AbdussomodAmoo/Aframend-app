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

        loaded_count = 0
        for model_path in model_files:
            model_name = Path(model_path).stem  # Get filename without extension
            try:
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                self.loaded_models[model_name] = model
                print(f"✓ Loaded: {model_name}")
                st.write(f"✅ Successfully loaded: {model_name}")
                loaded_count += 1
            except Exception as e:
                print(f"✗ Failed to load {model_name}: {str(e)}")
                st.error(error_msg)
                # Continue loading other models even if one fails
                continue
                
        st.write(f"📊 Successfully loaded {loaded_count}/{len(model_files)} models")
        
        return self.loaded_models.copy() # return a copy to avoid reference issues
    
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
            st.success(f"✅ Loaded: {model_name}")
            return model
        except Exception as e:
            error_msg = f"✗ Failed to load {model_name}: {str(e)}"
            print(error_msg)
            st.error(error_msg)
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
            st.warning("No models currently loaded.")
        else:
            print("Loaded models:")
            st.write("📋 Currently loaded models:")
            for model_name in self.loaded_models.keys():
                print(f"  - {model_name}")
                st.write(f"  • {model_name}")
    
    def get_available_models(self):
        """
        Get list of all available model files in the directory.
        
        Returns:
            list: List of available model file names (without .pkl extension)
        """
        if not self.models_dir.exists():
            return []
        
        available = [Path(f).stem for f in model_files]
        
        # Debug output
        st.write(f"🔍 Available model files: {available}")
        
        return available
    def verify_models_exist(self):
        """
        Verify that model files exist and can be opened.
        
        Returns:
            dict: Status of each model file
        """
        model_status = {}
        model_files = glob.glob(str(self.models_dir / "*.pkl"))
        
        for model_path in model_files:
            model_name = Path(model_path).stem
            try:
                # Try to open the file and check if it's a valid pickle
                with open(model_path, 'rb') as f:
                    # Just peek at the file, don't load the full model
                    pickle.load(f)
                model_status[model_name] = "✅ Valid"
            except Exception as e:
                model_status[model_name] = f"❌ Error: {str(e)}"
        
        return model_status
        
# Example usage and testing function
def test_model_loader():
    """Test function to verify model loading works"""
    st.write("🧪 **Testing Model Loader**")
    
    # Initialize the model loader
    loader = ModelLoader("models")
    
    # Check available models
    available = loader.get_available_models()
    st.write(f"Available models: {available}")
    
    # Verify model files
    status = loader.verify_models_exist()
    st.write("Model file verification:")
    for name, stat in status.items():
        st.write(f"  {name}: {stat}")
    
    # Load all models
    models = loader.load_all_models()
    
    if models:
        st.success(f"🎉 Successfully loaded {len(models)} models!")
        st.write("Loaded model names:", list(models.keys()))
        
        # Test a specific model if available
        if models:
            first_model_name = list(models.keys())[0]
            first_model = models[first_model_name]
            st.write(f"Sample model '{first_model_name}' type: {type(first_model)}")
    else:
        st.error("❌ No models were loaded successfully")
    
    return models

if __name__ == "__main__":
    # Run the test
    test_model_loader()
