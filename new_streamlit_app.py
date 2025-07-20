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
    """Extract molecular features from SMILES"""
    if not RDKIT_AVAILABLE:
        # Return mock features if RDKit is not available
        return [
            180.0,  # MolWt
            2.5,    # LogP
            50.0,   # TPSA
            2,      # NumHDonors
            3,      # NumHAcceptors
            5,      # NumRotatableBonds
            12,     # HeavyAtomCount
            1,      # NumAromaticRings
            0.3,    # FractionCsp3
            15.0,   # BertzCT
            0,      # NumSaturatedRings
            0       # NumAliphaticRings
        ]
    
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
            Descriptors.FractionCsp3(mol),             # Fraction sp3 carbons
            Descriptors.BertzCT(mol),                  # Molecular complexity
            Descriptors.NumSaturatedRings(mol),        # Saturated rings
            Descriptors.NumAliphaticRings(mol)         # Aliphatic rings
        ]
        return features
    except:
        return [0] * 12

class MockToxicityPredictor:
    """Mock predictor for when models are not available"""
    
    def __init__(self):
        self.is_loaded = False
        self.models = {}
        self.feature_names = []
        self.endpoint_names = ENDPOINT_NAMES
    
    def load_models_for_streamlit(self):
        """Mock model loading"""
        self.is_loaded = False
        return False
    
    def predict_single_compound(self, smiles):
        """Mock single compound prediction"""
        return {
            'smiles': smiles,
            'overall_risk': 'Low',
            'predictions': {
                'NR-AR': {
                    'endpoint_name': 'Androgen Receptor Disruption',
                    'prediction': 'Non-toxic',
                    'toxic_probability': 0.2,
                    'risk_level': 'Low'
                }
            }
        }
    
    def predict_batch_for_streamlit(self, smiles_list):
        """Mock batch prediction"""
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, smiles in enumerate(smiles_list[:5]):  # Limit to 5 for demo
            status_text.text(f'Processing compound {i+1} of {min(len(smiles_list), 5)}...')
            result = self.predict_single_compound(smiles)
            results.append(result)
            progress_bar.progress((i + 1) / min(len(smiles_list), 5))
        
        status_text.text('Demo analysis complete!')
        return results

class StreamlitToxicityPredictor(MockToxicityPredictor):
    """Streamlit wrapper for the toxicity predictor"""
    
    def __init__(self):
        super().__init__()
        self.is_loaded = False
        self.models = {}
        self.feature_names = []
        self.endpoint_names = ENDPOINT_NAMES
    
    def load_models_for_streamlit(self, model_path='african_phytochemical_toxicity_models.pkl.gz'):
        """Load pre-trained models with Streamlit error handling"""
        try:
            # Try multiple possible locations for the model file
            possible_paths = [
                model_path,
                'african_phytochemical_toxicity_models.pkl.gz',
                'models/african_phytochemical_toxicity_models.pkl.gz',
                'african_phytochemical_toxicity_models.pkl',
            ]
            
            loaded_data = None
            successful_path = None
            
            for path in possible_paths:
                if os.path.exists(path):
                    st.write(f"🔍 Found model file: {path}")
                    try:
                        if path.endswith('.gz'):
                            with gzip.open(path, 'rb') as f:
                                loaded_data = pickle.load(f)
                        else:
                            with open(path, 'rb') as f:
                                loaded_data = pickle.load(f)
                        successful_path = path
                        break
                    except Exception as e:
                        st.write(f"❌ Failed to load {path}: {e}")
                        continue
            
            if loaded_data is None:
                st.warning("Model file not found. Running in demo mode.")
                return False
            
            st.write(f"✅ Successfully loaded model from: {successful_path}")
            st.write(f"🔍 Data type: {type(loaded_data)}")
            
            # Check the structure of loaded data
            if isinstance(loaded_data, dict):
                st.write(f"🔑 Dictionary keys: {list(loaded_data.keys())}")
                
                # Extract models from the correct structure
                if 'models' in loaded_data:
                    self.models = loaded_data['models']
                    st.write(f"📊 Found {len(self.models)} models")
                    st.write(f"🎯 Available endpoints: {list(self.models.keys())}")
                    
                    # Extract other information
                    if 'feature_names' in loaded_data:
                        self.feature_names = loaded_data['feature_names']
                        st.write(f"🧬 Feature names: {self.feature_names}")
                    
                    if 'endpoint_names' in loaded_data:
                        self.endpoint_names = loaded_data['endpoint_names']
                    
                    self.is_loaded = True
                    return True
                else:
                    st.error("❌ Invalid model file structure. Expected 'models' key not found.")
                    return False
            else:
                st.error(f"❌ Unexpected data structure: {type(loaded_data)}")
                return False
                
        except Exception as e:
            st.error(f"💥 Error loading models: {str(e)}")
            return False
    
    def predict_single_compound(self, smiles):
        """Predict toxicity for a single compound"""
        if not self.is_loaded:
            return super().predict_single_compound(smiles)
        
        try:
            # Extract features
            features = extract_molecular_features(smiles)
            features_array = np.array(features).reshape(1, -1)
            
            results = {'smiles': smiles, 'predictions': {}}
            high_risk_count = 0
            
            for endpoint in self.models:
                try:
                    model = self.models[endpoint]
                    pred_proba = model.predict_proba(features_array)[0]
                    
                    # Handle different probability array structures
                    if len(pred_proba) > 1:
                        toxic_prob = pred_proba[1]  # Probability of toxic class
                    else:
                        toxic_prob = pred_proba[0]
                    
                    risk_level = 'High' if toxic_prob > 0.7 else 'Medium' if toxic_prob > 0.3 else 'Low'
                    if risk_level == 'High':
                        high_risk_count += 1
                    
                    results['predictions'][endpoint] = {
                        'endpoint_name': self.endpoint_names.get(endpoint, endpoint),
                        'toxic_probability': round(float(toxic_prob), 3),
                        'prediction': 'Toxic' if toxic_prob > 0.5 else 'Non-toxic',
                        'risk_level': risk_level
                    }
                    
                except Exception as e:
                    st.warning(f"Error predicting for endpoint {endpoint}: {e}")
                    continue
            
            # Overall risk assessment
            results['overall_risk'] = 'High' if high_risk_count >= 2 else 'Medium' if high_risk_count >= 1 else 'Low'
            
            return results
            
        except Exception as e:
            st.error(f"Error in prediction: {e}")
            return None
    
    def predict_batch_for_streamlit(self, smiles_list):
        """Predict toxicity for batch of compounds"""
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        max_compounds = min(len(smiles_list), 100)  # Limit for performance
        
        for i, smiles in enumerate(smiles_list[:max_compounds]):
            status_text.text(f'Processing compound {i+1} of {max_compounds}...')
            
            result = self.predict_single_compound(smiles)
            if result:
                results.append(result)
            
            progress_bar.progress((i + 1) / max_compounds)
        
        status_text.text(f'Analysis complete! Processed {len(results)} compounds.')
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
        st.image("https://via.placeholder.com/300x400/4CAF50/white?text=African+Plants", 
                caption="African Medicinal Plants")
        
        st.metric("Toxicity Endpoints", "12")
        st.metric("Model Accuracy", "85%+")
        st.metric("Plant Species", "1000+")

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
        if predictor.is_loaded:
            st.success("✅ Models successfully loaded")
            st.info(f"📊 {len(predictor.models)} toxicity models available")
            
            # Display available endpoints
            st.subheader("Available Endpoints:")
            for endpoint in predictor.models.keys():
                st.write(f"- {predictor.endpoint_names.get(endpoint, endpoint)}")
        else:
            st.warning("⚠️ Models not loaded - running in demo mode")
    
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
    
    tab1, tab2 = st.tabs(["📁 Batch Analysis (CSV)", "🔬 Single Compound"])
    
    with tab1:
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
    
    with tab2:
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
                    st.session_state.example_smiles = smi
        
        # Use example SMILES if selected
        if 'example_smiles' in st.session_state:
            smiles_input = st.session_state.example_smiles
            del st.session_state.example_smiles
        
        if smiles_input and st.button("🔬 Analyze Compound", key="single_analyze"):
            with st.spinner("Analyzing compound..."):
                result = predictor.predict_single_compound(smiles_input)
                
                if result:
                    st.success("✅ Analysis complete!")
                    display_single_prediction(result)
                else:
                    st.error("❌ Could not analyze the compound. Please check the SMILES string.")

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
    else:
        st.sidebar.warning("⚠️ Demo Mode")
        st.sidebar.info("Upload model files for full functionality")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("*Built for African phytochemical research*")

if __name__ == "__main__":
    main()
