#!/usr/bin/env python3
"""
African Phytochemical Toxicity Prediction App
A Streamlit application for researchers to analyze compound toxicity
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path
import io
import base64
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    RDKIT_AVAILABLE = True
except ImportError:
    st.warning("RDKit not available. Some features may be limited.")
    RDKIT_AVAILABLE = False
import warnings
import pickle
import gzip
warnings.filterwarnings('ignore')

# Page configuration - moved to top
st.set_page_config(
    page_title="African Phytochemical Toxicity Predictor",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
TOX21_ENDPOINTS = [
    'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase',
    'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE',
    'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'
]

ENDPOINT_NAMES = {
    'NR-AR': 'Androgen Receptor Disruption',
    'NR-AR-LBD': 'Androgen Receptor Binding',
    'NR-AhR': 'Aryl Hydrocarbon Receptor',
    'NR-Aromatase': 'Aromatase Inhibition',
    'NR-ER': 'Estrogen Receptor Disruption',
    'NR-ER-LBD': 'Estrogen Receptor Binding',
    'NR-PPAR-gamma': 'PPAR-gamma Activation',
    'SR-ARE': 'Antioxidant Response',
    'SR-ATAD5': 'DNA Damage Response',
    'SR-HSE': 'Heat Shock Response',
    'SR-MMP': 'Mitochondrial Toxicity',
    'SR-p53': 'p53 Tumor Suppressor'
}

def extract_molecular_features(smiles):
    """
    Extract molecular features that work well for toxicity prediction
    (Same function as in the pipeline)
    """
    if not RDKIT_AVAILABLE:
        return [0] * 12  # Return zeros for invalid SMILES
        
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return [0] * 12  # Return zeros for invalid SMILES

    try:
        features = [
            Descriptors.MolWt(mol),                    # Molecular weight
            Descriptors.MolLogP(mol),                  # Lipophilicity
            Descriptors.TPSA(mol),                     # Topological polar surface area
            Descriptors.NumHDonors(mol),               # Hydrogen bond donors
            Descriptors.NumHAcceptors(mol),            # Hydrogen bond acceptors
            Descriptors.NumRotatableBonds(mol),        # Rotatable bonds
            Descriptors.HeavyAtomCount(mol),           # Heavy atom count
            Descriptors.NumAromaticRings(mol),         # Aromatic rings
            Descriptors.FractionCSP3(mol),             # Fraction sp3 carbons
            Descriptors.BertzCT(mol),                  # Molecular complexity
            Descriptors.NumSaturatedRings(mol),        # Saturated rings
            Descriptors.NumAliphaticRings(mol)         # Aliphatic rings
        ]
        return features
    except:
        return [0] * 12

class StreamlitToxicityPredictor:
    """Streamlit wrapper for the toxicity predictor"""
    
    def __init__(self):
        self.is_loaded = False
        self.models = {}
        self.feature_names = []
        self.training_stats = {}
    
    def load_models_for_streamlit(self, model_path=None):
        """Load pre-trained models with Streamlit error handling"""
        try:
            # Try multiple possible locations for the model file
            possible_paths = [
                'african_phytochemical_toxicity_models.pkl.gz',  # Same directory
                'models/african_phytochemical_toxicity_models.pkl.gz',  # Models subdirectory  
                'african_phytochemical_toxicity_models.pkl',  # Uncompressed version
                'models/african_phytochemical_toxicity_models.pkl',  # Models subdirectory uncompressed
            ]
            
            if model_path:
                possible_paths.insert(0, model_path)
            
            st.write(f"🔍 Looking for model files in: {os.getcwd()}")
            st.write(f"📋 Available files: {os.listdir('.')}")
            if os.path.exists('models/'):
                st.write(f"📋 Files in models/: {os.listdir('models/')}")
            
            for path in possible_paths:
                if os.path.exists(path):
                    st.write(f"✅ Found model file: {path}")
                    try:
                        if path.endswith('.gz'):
                            with gzip.open(path, 'rb') as f:
                                model_data = pickle.load(f)
                        else:
                            with open(path, 'rb') as f:
                                model_data = joblib.load(f)
                        
                        # Debug: Check what we loaded
                        st.write(f"🔍 Loaded data type: {type(model_data)}")
                        
                        if isinstance(model_data, dict):
                            st.write(f"🔑 Dictionary keys: {list(model_data.keys())}")
                            
                            # Extract the components properly
                            self.models = model_data.get('models', {})
                            self.feature_names = model_data.get('feature_names', [])
                            self.training_stats = model_data.get('training_stats', {})
                            
                            st.write(f"📊 Number of endpoint models: {len(self.models)}")
                            st.write(f"🎯 Available endpoints: {list(self.models.keys())}")
                            
                            if len(self.models) > 0:
                                self.is_loaded = True
                                st.success(f"✅ Successfully loaded {len(self.models)} toxicity models!")
                                return True
                            else:
                                st.error("❌ No models found in the model data")
                                return False
                                
                        else:
                            # If it's the predictor object itself
                            if hasattr(model_data, 'models') and hasattr(model_data, 'predict_single_compound'):
                                self.models = model_data.models
                                self.feature_names = getattr(model_data, 'feature_names', [])
                                self.training_stats = getattr(model_data, 'training_stats', {})
                                self.is_loaded = True
                                st.success(f"✅ Successfully loaded predictor object!")
                                return True
                            else:
                                st.warning(f"🤔 Unexpected data structure: {type(model_data)}")
                                return False
                        
                    except Exception as e:
                        st.error(f"💥 Error loading {path}: {str(e)}")
                        continue
            
            st.warning("❌ Model file not found in any expected location. Running in demo mode.")
            return False
            
        except Exception as e:
            st.warning(f"❌ Error loading models: {str(e)}. Running in demo mode.")
            return False
    
    def predict_single_compound(self, smiles):
        """
        Predict toxicity for a single compound (matching the original function)
        """
        if not self.is_loaded or not self.models:
            # Return mock prediction for demo
            return self._mock_prediction(smiles)
        
        # Extract features
        features = extract_molecular_features(smiles)
        features_array = np.array(features).reshape(1, -1)

        results = {'smiles': smiles, 'predictions': {}}

        for endpoint in self.models:
            try:
                model = self.models[endpoint]
                pred_proba = model.predict_proba(features_array)[0]

                # Handle cases where model only learned one class
                if len(pred_proba) > 1:
                    toxic_prob = pred_proba[1]  # Probability of toxic class
                else:
                    toxic_prob = pred_proba[0]

                results['predictions'][endpoint] = {
                    'endpoint_name': ENDPOINT_NAMES.get(endpoint, endpoint),
                    'toxic_probability': round(toxic_prob, 3),
                    'prediction': 'Toxic' if toxic_prob > 0.5 else 'Non-toxic',
                    'risk_level': 'High' if toxic_prob > 0.7 else 'Medium' if toxic_prob > 0.3 else 'Low'
                }
            except Exception as e:
                st.error(f"Error predicting for endpoint {endpoint}: {str(e)}")
                continue

        # Overall risk assessment
        if results['predictions']:
            high_risk_count = sum(1 for p in results['predictions'].values() if p['risk_level'] == 'High')
            results['overall_risk'] = 'High' if high_risk_count >= 2 else 'Medium' if high_risk_count >= 1 else 'Low'
        else:
            results['overall_risk'] = 'Unknown'

        return results
    
    def _mock_prediction(self, smiles):
        """Generate mock prediction for demo mode"""
        import random
        random.seed(hash(smiles) % 1000)  # Consistent results for same SMILES
        
        predictions = {}
        for endpoint in TOX21_ENDPOINTS:
            prob = random.uniform(0.1, 0.9)
            predictions[endpoint] = {
                'endpoint_name': ENDPOINT_NAMES[endpoint],
                'toxic_probability': round(prob, 3),
                'prediction': 'Toxic' if prob > 0.5 else 'Non-toxic',
                'risk_level': 'High' if prob > 0.7 else 'Medium' if prob > 0.3 else 'Low'
            }
        
        high_risk_count = sum(1 for p in predictions.values() if p['risk_level'] == 'High')
        overall_risk = 'High' if high_risk_count >= 2 else 'Medium' if high_risk_count >= 1 else 'Low'
        
        return {
            'smiles': smiles,
            'predictions': predictions,
            'overall_risk': overall_risk
        }
    
    def predict_batch_for_streamlit(self, smiles_list):
        """Batch prediction with progress bar"""
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total = min(len(smiles_list), 100)  # Limit to 100 compounds
        
        for i, smiles in enumerate(smiles_list[:total]):
            status_text.text(f'Processing compound {i+1} of {total}...')
            result = self.predict_single_compound(smiles)
            results.append(result)
            progress_bar.progress((i + 1) / total)
        
        status_text.text('Analysis complete!')
        return results

# Initialize predictor
@st.cache_resource
def load_predictor():
    predictor = StreamlitToxicityPredictor()
    predictor.load_models_for_streamlit()
    return predictor

def create_download_link(df, filename):
    """Create a download link for DataFrame"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download Results as CSV</a>'
    return href

def display_single_prediction(result):
    """Display results for single compound prediction"""
    if not result:
        st.error("❌ Could not process the compound")
        return
    
    # Overall risk
    risk_color = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}
    st.subheader(f"🎯 Overall Risk: {risk_color[result['overall_risk']]} {result['overall_risk']}")
    
    # Individual predictions
    st.subheader("📊 Individual Endpoint Analysis")
    
    # Create DataFrame for better display
    pred_data = []
    for endpoint, data in result['predictions'].items():
        pred_data.append({
            'Endpoint': data['endpoint_name'],
            'Prediction': data['prediction'],
            'Probability': f"{data['toxic_probability']:.3f}",
            'Risk Level': data['risk_level']
        })
    
    pred_df = pd.DataFrame(pred_data)
    
    # Style the dataframe
    def style_risk_level(val):
        if val == 'High':
            return 'background-color: #ffcccc'
        elif val == 'Medium':
            return 'background-color: #fff2cc'
        else:
            return 'background-color: #ccffcc'
    
    styled_df = pred_df.style.applymap(style_risk_level, subset=['Risk Level'])
    st.dataframe(styled_df, use_container_width=True)

def home_page():
    """Home page content"""
    st.title("🌿 African Phytochemical Toxicity Predictor")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## Welcome to the African Phytochemical Toxicity Analysis Platform
        
        This application provides advanced toxicity prediction capabilities specifically designed 
        for African phytochemicals and natural compounds. Our models are trained on the Tox21 
        dataset and optimized for natural product analysis.
        
        ### 🎯 Key Features:
        - **Single Compound Analysis**: Test individual compounds using SMILES notation
        - **Batch Processing**: Upload CSV files for bulk toxicity analysis
        - **Comprehensive Endpoints**: Analysis across 12 critical toxicity endpoints
        - **Plant Information**: Explore African medicinal plants database
        - **Research-Grade Results**: Probability scores and risk assessments
        
        ### 🧪 Toxicity Endpoints Analyzed:
        - **Nuclear Receptors**: AR, ER, AhR, Aromatase, PPAR-gamma
        - **Stress Response**: Antioxidant, Heat Shock, DNA Damage
        - **Cellular Toxicity**: Mitochondrial, p53 pathway
        
        ### 🚀 Getting Started:
        1. Navigate to **Toxicity Analysis** for compound testing
        2. Use **Plant Info** to explore African medicinal plants
        3. Check the **About** page for methodology details
        """)
    
    with col2:
        st.info("""
        **Demo Mode Notice**
        
        If models are not loaded, the app will run in demo mode with mock predictions for testing purposes.
        
        To use real predictions, ensure the model file `african_phytochemical_toxicity_models.pkl` is available.
        """)
        
        predictor = load_predictor()
        if predictor.is_loaded:
            st.success("✅ Real Models Loaded")
            st.metric("Loaded Endpoints", len(predictor.models))
        else:
            st.warning("⚠️ Running in Demo Mode")

def about_page():
    """About page content"""
    st.title("📚 About the Platform")
    
    tab1, tab2, tab3 = st.tabs(["Methodology", "Model Details", "References"])
    
    with tab1:
        st.header("🔬 Methodology")
        st.markdown("""
        ### Machine Learning Approach
        Our toxicity prediction models are built using Random Forest classifiers trained on the 
        comprehensive Tox21 dataset. The models analyze molecular descriptors to predict potential 
        toxicity across multiple biological endpoints.
        
        ### Feature Extraction
        We extract 12 key molecular descriptors:
        - **Molecular Weight**: Overall size of the molecule
        - **LogP**: Lipophilicity measure
        - **TPSA**: Topological Polar Surface Area
        - **Hydrogen Bonding**: Donors and acceptors
        - **Structural Features**: Rotatable bonds, rings, complexity
        
        ### Risk Assessment
        - **Low Risk**: Probability < 0.3
        - **Medium Risk**: Probability 0.3-0.7
        - **High Risk**: Probability > 0.7
        """)
    
    with tab2:
        st.header("🤖 Model Performance")
        predictor = load_predictor()
        if predictor.is_loaded and predictor.training_stats:
            st.subheader("Training Statistics")
            for endpoint, stats in predictor.training_stats.items():
                with st.expander(f"{ENDPOINT_NAMES.get(endpoint, endpoint)}"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Samples", stats.get('n_samples', 'N/A'))
                    with col2:
                        st.metric("Positive Cases", stats.get('n_positive', 'N/A'))
                    with col3:
                        st.metric("Positive Rate", f"{stats.get('positive_rate', 0):.2%}")
        else:
            st.info("Model performance statistics will be displayed when models are loaded.")
    
    with tab3:
        st.header("📖 References")
        st.markdown("""
        ### Key Publications
        1. **Tox21 Data Challenge**: Advancing computational toxicology
        2. **African Natural Products**: Traditional medicine and modern drug discovery
        3. **Machine Learning in Toxicology**: Predictive modeling approaches
        
        ### Datasets Used
        - **Tox21**: EPA/NIH/FDA collaboration dataset
        - **AfroDB**: African natural products database
        - **COCONUT**: Natural products database
        
        ### Acknowledgments
        This work builds upon the contributions of the global toxicology and 
        natural products research communities.
        """)

def plant_info_page():
    """Plant information page"""
    st.title("🌱 African Medicinal Plants Database")
    
    st.markdown("""
    ### Explore Traditional African Medicine
    
    This section provides information about African medicinal plants and their phytochemical compounds.
    Our database contains information about traditional uses, chemical compounds, and safety profiles.
    """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🔍 Search by Plant Name")
        plant_name = st.text_input("Enter plant name (e.g., Vernonia amygdalina):")
        
        if plant_name:
            st.info(f"Searching for compounds from: {plant_name}")
            st.markdown("*Note: This feature requires connection to the COCONUT database*")
    
    with col2:
        st.subheader("📊 Popular African Medicinal Plants")
        popular_plants = [
            "Vernonia amygdalina",
            "Aloe vera",
            "Catharanthus roseus",
            "Artemisia afra",
            "Kanna (Sceletium tortuosum)",
            "African potato (Hypoxis hemerocallidea)"
        ]
        
        for plant in popular_plants:
            if st.button(f"🌿 {plant}", key=plant):
                st.info(f"Loading information for {plant}...")
    
    st.subheader("💡 Traditional Uses & Modern Research")
    st.markdown("""
    - **Vernonia amygdalina**: Antimalarial, antidiabetic properties
    - **Aloe vera**: Wound healing, anti-inflammatory effects  
    - **Catharanthus roseus**: Anticancer compounds (vincristine, vinblastine)
    - **Artemisia afra**: Traditional fever remedy, antimalarial research
    """)

def toxicity_analysis_page():
    """Main toxicity analysis page"""
    st.title("🧪 Toxicity Analysis")
    
    # Check if models are loaded
    predictor = load_predictor()
    if not predictor.is_loaded:
        st.warning("⚠️ Models not loaded. Running in demo mode with mock predictions.")
        st.info("To use real predictions, ensure the model file is available in the application directory.")
    else:
        st.success(f"✅ Models loaded successfully! {len(predictor.models)} endpoints available.")
    
    tab1, tab2 = st.tabs(["🔬 Single Compound", "📁 Batch Analysis (CSV)"])
    
    with tab1:
        st.header("🔬 Single Compound Analysis")
        st.markdown("Enter a SMILES string to analyze toxicity for a single compound.")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            smiles_input = st.text_input(
                "Enter SMILES string:",
                placeholder="e.g., CCO (ethanol)",
                help="Enter the SMILES notation of your compound"
            )
        
        with col2:
            st.markdown("### Example SMILES:")
            example_smiles = {
                "Caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
                "Aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
                "Ethanol": "CCO"
            }
            
            for name, smi in example_smiles.items():
                if st.button(f"{name}", key=f"example_{name}"):
                    smiles_input = smi
                    st.rerun()
        
        if smiles_input and st.button("🔬 Analyze Compound", key="single_analyze"):
            with st.spinner("Analyzing compound..."):
                result = predictor.predict_single_compound(smiles_input)
                
                if result:
                    st.success("✅ Analysis complete!")
                    display_single_prediction(result)
                    
                    # Show molecular descriptors if RDKit is available
                    if RDKIT_AVAILABLE:
                        with st.expander("🧮 Molecular Descriptors"):
                            features = extract_molecular_features(smiles_input)
                            feature_names = [
                                'Molecular Weight', 'LogP', 'TPSA', 'H-Bond Donors', 
                                'H-Bond Acceptors', 'Rotatable Bonds', 'Heavy Atoms',
                                'Aromatic Rings', 'Fraction SP3', 'Complexity',
                                'Saturated Rings', 'Aliphatic Rings'
                            ]
                            
                            desc_df = pd.DataFrame({
                                'Descriptor': feature_names,
                                'Value': [f"{f:.2f}" if f != int(f) else str(int(f)) for f in features]
                            })
                            st.dataframe(desc_df, use_container_width=True)
                else:
                    st.error("❌ Could not analyze the compound. Please check the SMILES string.")
    
    with tab2:
        st.header("📊 Batch Toxicity Analysis")
        st.markdown("Upload a CSV file with SMILES strings to analyze multiple compounds at once.")
        
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv", key="batch_upload")
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.write("📋 Uploaded Data Preview:")
                st.dataframe(df.head())
                
                # SMILES column selection
                smiles_columns = [col for col in df.columns if 'smiles' in col.lower()]
                if not smiles_columns:
                    smiles_columns = df.columns.tolist()
                
                smiles_col = st.selectbox("Select SMILES column:", smiles_columns)
                
                if st.button("🚀 Run Batch Analysis", key="batch_analyze"):
                    with st.spinner("Analyzing compounds... This may take a while."):
                        smiles_list = df[smiles_col].dropna().tolist()
                        
                        if len(smiles_list) > 100:
                            st.warning("⚠️ Large dataset detected. Processing first 100 compounds.")
                            smiles_list = smiles_list[:100]
                        
                        results = predictor.predict_batch_for_streamlit(smiles_list)
                        
                        if results:
                            # Process results for display
                            result_data = []
                            for i, result in enumerate(results):
                                row = {
                                    'Index': i,
                                    'SMILES': result['smiles'],
                                    'Overall_Risk': result['overall_risk']
                                }
                                
                                # Add individual endpoint predictions
                                for endpoint, data in result['predictions'].items():
                                    row[f"{endpoint}_Risk"] = data['risk_level']
                                    row[f"{endpoint}_Probability"] = data['toxic_probability']
                                
                                result_data.append(row)
                            
                            result_df = pd.DataFrame(result_data)
                            
                            st.success(f"✅ Analysis complete! Processed {len(results)} compounds.")
                            
                            # Display summary
                            risk_summary = result_df['Overall_Risk'].value_counts()
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("🟢 Low Risk", risk_summary.get('Low', 0))
                            with col2:
                                st.metric("🟡 Medium Risk", risk_summary.get('Medium', 0))
                            with col3:
                                st.metric("🔴 High Risk", risk_summary.get('High', 0))
                            
                            # Display results
                            st.subheader("📊 Results")
                            st.dataframe(result_df)
                            
                            # Download link
                            st.markdown(create_download_link(result_df, "toxicity_results.csv"), 
                                      unsafe_allow_html=True)
                        
                        else:
                            st.error("❌ Failed to process compounds")
                            
            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")

def main():
    """Main application function"""
    # Sidebar navigation
    st.sidebar.title("🧪 Navigation")
    
    pages = {
        "🏠 Home": home_page,
        "🧪 Toxicity Analysis": toxicity_analysis_page,
        "🌱 Plant Info": plant_info_page,
        "📚 About": about_page
    }
    
    selected_page = st.sidebar.radio("Go to:", list(pages.keys()))
    
    # Display selected page
    pages[selected_page]()
    
    # Sidebar info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 System Status")
    
    predictor = load_predictor()
    if predictor.is_loaded:
        st.sidebar.success("✅ Models Loaded")
        st.sidebar.info(f"🎯 {len(predictor.models)} endpoints available")
        
        # Show available endpoints
        with st.sidebar.expander("Available Endpoints"):
            for endpoint in predictor.models.keys():
                st.sidebar.text(f"• {ENDPOINT_NAMES.get(endpoint, endpoint)}")
    else:
        st.sidebar.warning("⚠️ Demo Mode")
        st.sidebar.info("Models not found - using mock predictions")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("*Built for African phytochemical research*")

if __name__ == "__main__":
    main()
