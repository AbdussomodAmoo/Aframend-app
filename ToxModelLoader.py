import os
import pickle
import glob
from pathlib import Path
import streamlit as st

class ToxicityModelLoader:
    def __init__(self, models_dir="models"):
        """
        Initialize the ModelLoader for toxicity prediction models.
        
        Args:
            models_dir (str): Path to the directory containing model files
        """
        self.models_dir = Path(models_dir)
        self.loaded_models = {}
        self.metadata = None
        # Debug: Show what we're looking for
        st.info(f"🔍 Initialized ToxicityModelLoader")
        st.info(f"📁 Expected models directory: {self.models_dir}")
        st.info(f"📁 Current working directory: {Path.cwd()}")
        
        # Expected endpoint names from your metadata
        self.expected_endpoints = [
            'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 
            'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 
            'SR-HSE', 'SR-MMP', 'SR-p53'
        ]
        
    def load_metadata(self, metadata_path="metadata.pkl"):
        """
        Load the metadata file containing endpoint information.
        
        Args:
            metadata_path (str): Path to the metadata file
            
        Returns:
            dict: Loaded metadata or None if failed
        """
        metadata_file = Path(metadata_path)
        
        if not metadata_file.exists():
            st.error(f"❌ Metadata file '{metadata_path}' does not exist!")
            return None
            
        try:
            with open(metadata_file, 'rb') as f:
                self.metadata = pickle.load(f)
            st.success(f"✅ Successfully loaded metadata from {metadata_path}")
            return self.metadata
        except Exception as e:
            error_msg = f"❌ Failed to load metadata: {str(e)}"
            st.error(error_msg)
            return None
        
        def load_all_models(self):
            """
            Load all toxicity model .pkl files from the models directory.
            
            Returns:
                dict: Dictionary with model names as keys and loaded models as values
            """
            # Try multiple possible locations for model files
            possible_patterns = [
                str(self.models_dir / "*.pkl"),  # Original: models/*.pkl
                "*.pkl",                         # Current directory: *.pkl
                str(Path("models") / "*.pkl"),   # Explicit models path
                "models/*.pkl"                   # Simple models path
            ]
            
            model_files = []
            for pattern in possible_patterns:
                found_files = glob.glob(pattern)
                if found_files:
                    model_files = found_files
                    st.info(f"📁 Found models using pattern: {pattern}")
                    break
            
            if not model_files:
                st.error(f"❌ No .pkl files found in any of these locations:")
                for pattern in possible_patterns:
                    st.text(f"  • {pattern}")
                return {}
        
        # Filter out metadata.pkl if it's included
        model_files = [f for f in model_files if not f.endswith('metadata.pkl')]

        
        st.info(f"🔍 Found {len(model_files)} model files")

        loaded_count = 0
        failed_models = []
        
        # Create a progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, model_path in enumerate(model_files):
            model_name = Path(model_path).stem  # Get filename without extension
            
            # Update progress
            progress = (i + 1) / len(model_files)
            progress_bar.progress(progress)
            status_text.text(f"Loading: {model_name}")
            
            try:
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                self.loaded_models[model_name] = model
                st.success(f"✅ Successfully loaded: {model_name}")
                loaded_count += 1
            except Exception as e:
                error_msg = f"❌ Failed to load {model_name}: {str(e)}"
                st.error(error_msg)
                failed_models.append((model_name, str(e)))
                continue
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        # Summary
        if loaded_count > 0:
            st.success(f"🎉 Successfully loaded {loaded_count}/{len(model_files)} models")
        
        if failed_models:
            st.error("❌ Failed to load the following models:")
            for model_name, error in failed_models:
                st.text(f"  • {model_name}: {error}")
        
        return self.loaded_models.copy()
        
    def load_specific_model(self, model_name):
        # Remove .pkl extension if provided
        if model_name.endswith('.pkl'):
            model_name = model_name[:-4]
        
        # Try multiple possible locations, similar to metadata loading
        possible_paths = [
            self.models_dir / f"{model_name}.pkl",  # Original: models/model_name.pkl
            Path(f"{model_name}.pkl"),              # Current directory: model_name.pkl
            Path("models") / f"{model_name}.pkl",   # Explicit: models/model_name.pkl
        ]
        
        model_path = None
        for path in possible_paths:
            if path.exists():
                model_path = path
                break
        
        if model_path is None:
            st.error(f"❌ Model file '{model_name}.pkl' not found in any location!")
            return None
        
        # Actually load the model
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            self.loaded_models[model_name] = model
            st.success(f"✅ Successfully loaded: {model_name}")
            return model
        except Exception as e:
            error_msg = f"❌ Failed to load {model_name}: {str(e)}"
            st.error(error_msg)
            return None
    
    def load_model_and_metadata(self, model_name="NR-AR", metadata_path="metadata.pkl"):
        """
        Load a specific model and metadata (similar to your original function).
        
        Args:
            model_name (str): Name of the model to load
            metadata_path (str): Path to metadata file
            
        Returns:
            tuple: (model, metadata) or (None, None) if failed
        """
        # Load metadata if not already loaded
        if self.metadata is None:
            self.metadata = self.load_metadata(metadata_path)
            if self.metadata is None:
                return None, None
        
        # Load the specific model
        model = self.load_specific_model(model_name)
        if model is None:
            return None, None
        
        return model, self.metadata
    
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
        Display all currently loaded models in Streamlit.
        """
        if not self.loaded_models:
            st.warning("⚠️ No models currently loaded.")
        else:
            st.write("📋 **Currently loaded models:**")
            for model_name in self.loaded_models.keys():
                # Check if it's an expected endpoint
                if any(endpoint in model_name for endpoint in self.expected_endpoints):
                    st.write(f"  ✅ {model_name}")
                else:
                    st.write(f"  ❓ {model_name}")
    
    def get_available_models(self):
        """
        Get list of all available model files in the directory.
        
        Returns:
            list: List of available model file names (without .pkl extension)
        """
        if not self.models_dir.exists():
            return []
        
        model_files = glob.glob(str(self.models_dir / "*.pkl"))
        available = [Path(f).stem for f in model_files]
        
        return available
    
    def verify_models_exist(self):
        """
        Verify that model files exist and can be opened.
        
        Returns:
            dict: Status of each model file
        """
        model_status = {}
        model_files = glob.glob(str(self.models_dir / "*.pkl"))
        
        if not model_files:
            st.warning("⚠️ No model files found to verify")
            return model_status
        
        for model_path in model_files:
            model_name = Path(model_path).stem
            try:
                # Try to open the file and check if it's a valid pickle
                with open(model_path, 'rb') as f:
                    pickle.load(f)
                model_status[model_name] = "✅ Valid"
            except Exception as e:
                model_status[model_name] = f"❌ Error: {str(e)}"
        
        return model_status
    
    def display_model_info(self):
        """
        Display information about loaded models and metadata.
        """
        if self.metadata:
            st.write("🔬 **Toxicity Prediction Endpoints:**")
            
            endpoint_names = self.metadata.get('endpoint_names', {})
            training_stats = self.metadata.get('training_stats', {})
            
            for endpoint, description in endpoint_names.items():
                col1, col2, col3 = st.columns([2, 3, 2])
                
                with col1:
                    # Check if model is loaded
                    model_loaded = any(endpoint in name for name in self.loaded_models.keys())
                    status = "✅" if model_loaded else "❌"
                    st.write(f"{status} {endpoint}")
                
                with col2:
                    st.write(description)
                
                with col3:
                    if endpoint in training_stats:
                        stats = training_stats[endpoint]
                        st.write(f"Samples: {stats.get('n_samples', 'N/A')}")
        
        # Display loaded models count
        if self.loaded_models:
            st.info(f"📊 {len(self.loaded_models)} models currently loaded")

# Streamlit app integration function
def create_model_loader_app():
    """
    Create a Streamlit app for testing the model loader.
    """
    st.title("🧪 Toxicity Model Loader")
    st.write("Load and manage toxicity prediction models")
    
    # Initialize the model loader
    if 'model_loader' not in st.session_state:
        st.session_state.model_loader = ToxicityModelLoader("models")
    
    loader = st.session_state.model_loader
    
    # Sidebar controls
    st.sidebar.header("🎛️ Controls")
    
    if st.sidebar.button("🔄 Load All Models"):
        with st.spinner("Loading all models..."):
            models = loader.load_all_models()
        
        if models:
            st.balloons()
    
    if st.sidebar.button("📊 Load Metadata"):
        with st.spinner("Loading metadata..."):
            metadata = loader.load_metadata()
    
    if st.sidebar.button("🔍 Verify Models"):
        status = loader.verify_models_exist()
        if status:
            st.write("**Model Verification Results:**")
            for name, stat in status.items():
                st.write(f"  {name}: {stat}")
        else:
            st.warning("No models found to verify")
    
    # Main content
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Model Status")
        loader.list_loaded_models()
    
    with col2:
        st.subheader("📈 Model Information")
        loader.display_model_info()
    
    # Available models
    available = loader.get_available_models()
    if available:
        st.subheader("📁 Available Model Files")
        st.write(available)

# Test function
def test_model_loader():
    """Test function to verify model loading works"""
    st.write("🧪 **Testing Model Loader**")
    
    # Initialize the model loader
    loader = ToxicityModelLoader("models")
    
    # Check available models
    available = loader.get_available_models()
    st.write(f"Available models: {available}")
    
    # Verify model files
    status = loader.verify_models_exist()
    if status:
        st.write("**Model file verification:**")
        for name, stat in status.items():
            st.write(f"  {name}: {stat}")
    
    # Load metadata
    metadata = loader.load_metadata()
    
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
    # For running as a standalone Streamlit app
    if 'streamlit' in globals():
        create_model_loader_app()
    else:
        # For testing without Streamlit
        print("Testing model loader without Streamlit...")
