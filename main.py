import streamlit as st
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Fragments
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect, GetMACCSKeysFingerprint
from typing import List, Dict, Optional, Union
import joblib
import warnings
import os
import requests
from datetime import datetime
from groq import Groq
warnings.filterwarnings('ignore')

# Set your Groq API key here (replace with your actual key)
os.environ['GROQ_API_KEY'] = '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'

# ===== TOXICITY PREDICTION CONSTANTS =====
TOX21_ENDPOINTS = [
    'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
    'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'
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

# ===== COMMON GROQ CLIENT =====
def get_groq_api_key(): 
    """Securely retrieve Groq API key"""
    api_key = os.getenv('GROQ_API_KEY')
    
    if not api_key or api_key == 'your_groq_api_key_here':
        api_key = st.sidebar.text_input(
            "Enter Groq API Key (optional for AI analysis):",
            type="password",
            help="Your API key will not be stored. Leave blank to skip AI analysis."
        )
    
    return api_key

class GroqClient:
    def __init__(self, api_key: str):
        self.client = Groq(api_key=api_key)
    
    def generate_response(self, system_prompt: str, user_prompt: str, model: str = "llama3-8b-8192") -> str:
        """Generate conversational response using Groq"""
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model=model,
                max_tokens=1500,
                temperature=0.7
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            return f"Error generating response: {str(e)}"

# ===== BIOACTIVITY PREDICTION FUNCTIONS =====
def featurize(smiles):
    """
    Extract molecular features exactly as used in training.
    This matches the featurize function from your training code.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    try:
        return {
            'MolWt': Descriptors.MolWt(mol),
            'TPSA': Descriptors.TPSA(mol),
            'NumHDonors': Descriptors.NumHDonors(mol),
            'NumHAcceptors': Descriptors.NumHAcceptors(mol),
            'NumRotatableBonds': Descriptors.NumRotatableBonds(mol),
            'LogP': Descriptors.MolLogP(mol),
            'NumRings': rdMolDescriptors.CalcNumRings(mol),
            'HeavyAtomCount': Descriptors.HeavyAtomCount(mol),
            'FractionCSP3': rdMolDescriptors.CalcFractionCSP3(mol),
            'NumAromaticRings': Descriptors.NumAromaticRings(mol),
            'NumAliphaticRings': Descriptors.NumAliphaticRings(mol),
            'NumValenceElectrons': Descriptors.NumValenceElectrons(mol),
            'BertzCT': Descriptors.BertzCT(mol),
            'NumHeteroatoms': Descriptors.NumHeteroatoms(mol)
        }
    except Exception as e:
        return None

def predict_ic50(smiles_list: List[str], model, scaler_X) -> pd.DataFrame:
    """
    Make IC50 predictions for a list of SMILES using the trained regression model.
    """
    results = []
    
    for smiles in smiles_list:
        # Validate SMILES
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            results.append({
                'SMILES': smiles,
                'Valid': False,
                'Predicted_IC50_Scaled': None,
                'Activity_Level': 'Invalid SMILES',
                'Molecular_Properties': 'N/A'
            })
            continue
        
        # Extract features
        features_dict = featurize(smiles)
        if features_dict is None:
            results.append({
                'SMILES': smiles,
                'Valid': False,
                'Predicted_IC50_Scaled': None,
                'Activity_Level': 'Feature extraction failed',
                'Molecular_Properties': 'N/A'
            })
            continue
        
        try:
            # Convert features to array in the same order as training
            feature_names = ['MolWt', 'TPSA', 'NumHDonors', 'NumHAcceptors', 'NumRotatableBonds',
                           'LogP', 'NumRings', 'HeavyAtomCount', 'FractionCSP3', 'NumAromaticRings',
                           'NumAliphaticRings', 'NumValenceElectrons', 'BertzCT', 'NumHeteroatoms']
            
            features_array = np.array([features_dict[name] for name in feature_names]).reshape(1, -1)
            
            # Scale features using the same scaler from training
            features_scaled = scaler_X.transform(features_array)
            
            # Make prediction
            ic50_scaled = model.predict(features_scaled)[0]
            
            # Determine activity level based on scaled IC50 prediction
            if ic50_scaled < -0.5:
                activity_level = "Highly Active"
            elif ic50_scaled < 0:
                activity_level = "Active"
            elif ic50_scaled < 0.5:
                activity_level = "Moderately Active"
            else:
                activity_level = "Low Activity"
            
            # Format molecular properties for display
            mol_props = f"MW: {features_dict['MolWt']:.1f}, LogP: {features_dict['LogP']:.2f}, TPSA: {features_dict['TPSA']:.1f}"
            
            results.append({
                'SMILES': smiles,
                'Valid': True,
                'Predicted_IC50_Scaled': round(float(ic50_scaled), 3),
                'Activity_Level': activity_level,
                'Molecular_Properties': mol_props
            })
            
        except Exception as e:
            results.append({
                'SMILES': smiles,
                'Valid': False,
                'Predicted_IC50_Scaled': None,
                'Activity_Level': f'Prediction failed: {str(e)}',
                'Molecular_Properties': 'N/A'
            })
    
    return pd.DataFrame(results)

def load_bioactivity_model_and_scaler_from_github(model_url: str, scaler_url: str = None):
    """Load bioactivity model and scaler from GitHub repository"""
    try:
        # Load model
        if 'raw.githubusercontent.com' in model_url:
            response = requests.get(model_url)
            if response.status_code == 200:
                with open('temp_model.joblib', 'wb') as f:
                    f.write(response.content)
                model = joblib.load('temp_model.joblib')
                os.remove('temp_model.joblib')
            else:
                raise Exception(f"Failed to download model: {response.status_code}")
        else:
            raise Exception("Please provide a direct raw GitHub URL to the model file")
        
        # Load scaler if URL provided
        scaler = None
        if scaler_url and 'raw.githubusercontent.com' in scaler_url:
            response = requests.get(scaler_url)
            if response.status_code == 200:
                with open('temp_scaler.joblib', 'wb') as f:
                    f.write(response.content)
                scaler = joblib.load('temp_scaler.joblib')
                os.remove('temp_scaler.joblib')
        
        return model, scaler
    except Exception as e:
        raise Exception(f"Error loading from GitHub: {str(e)}")

def create_default_scaler():
    """Create a default StandardScaler for when scaler is not available"""
    from sklearn.preprocessing import StandardScaler
    st.warning("⚠️ No scaler provided. Using identity transformation (no scaling).")
    
    class IdentityScaler:
        def transform(self, X):
            return X
        def fit_transform(self, X):
            return X
    
    return IdentityScaler()

# ===== TOXICITY PREDICTION FUNCTIONS =====
def extract_molecular_features(mol: Chem.Mol,
                             include_fingerprints: bool = True,
                             morgan_radius: int = 2,
                             morgan_bits: int = 2048,
                             include_fragments: bool = True) -> Dict[str, Union[float, int]]:
    """
    Extract comprehensive molecular features optimized for toxicity prediction.
    """
    if mol is None:
        return {}

    features = {}

    try:
        # === BASIC MOLECULAR PROPERTIES ===
        features['mol_weight'] = Descriptors.MolWt(mol)
        features['mol_logp'] = Descriptors.MolLogP(mol)
        features['tpsa'] = Descriptors.TPSA(mol)
        features['labute_asa'] = Descriptors.LabuteASA(mol)

        # === HYDROGEN BONDING ===
        features['num_hbd'] = Descriptors.NumHDonors(mol)
        features['num_hba'] = Descriptors.NumHAcceptors(mol)
        features['max_partial_charge'] = Descriptors.MaxPartialCharge(mol)
        features['min_partial_charge'] = Descriptors.MinPartialCharge(mol)
        features['max_abs_partial_charge'] = Descriptors.MaxAbsPartialCharge(mol)

        # === STRUCTURAL FEATURES ===
        features['num_rotatable_bonds'] = Descriptors.NumRotatableBonds(mol)
        features['heavy_atom_count'] = Descriptors.HeavyAtomCount(mol)
        features['num_aromatic_rings'] = Descriptors.NumAromaticRings(mol)
        features['num_saturated_rings'] = Descriptors.NumSaturatedRings(mol)
        features['num_aliphatic_rings'] = Descriptors.NumAliphaticRings(mol)
        features['ring_count'] = Descriptors.RingCount(mol)
        features['fraction_csp3'] = Descriptors.FractionCSP3(mol)
        features['num_heteroatoms'] = Descriptors.NumHeteroatoms(mol)

        # === COMPLEXITY MEASURES ===
        features['bertz_ct'] = Descriptors.BertzCT(mol)
        features['hall_kier_alpha'] = Descriptors.HallKierAlpha(mol)
        features['kappa1'] = Descriptors.Kappa1(mol)
        features['kappa2'] = Descriptors.Kappa2(mol)
        features['kappa3'] = Descriptors.Kappa3(mol)

        # === DRUG-LIKENESS ===
        features['qed'] = Descriptors.qed(mol)

        # === VSA DESCRIPTORS (selected for toxicity relevance) ===
        features['vsa_estate4'] = Descriptors.VSA_EState4(mol)
        features['vsa_estate9'] = Descriptors.VSA_EState9(mol)
        features['slogp_vsa4'] = Descriptors.SlogP_VSA4(mol)
        features['slogp_vsa6'] = Descriptors.SlogP_VSA6(mol)
        features['smr_vsa5'] = Descriptors.SMR_VSA5(mol)
        features['smr_vsa7'] = Descriptors.SMR_VSA7(mol)

        # === BALABAN J INDEX ===
        features['balaban_j'] = Descriptors.BalabanJ(mol)

        # === FRAGMENT COUNTS (toxicity-relevant) ===
        if include_fragments:
            features['fr_phenol'] = Fragments.fr_phenol(mol)
            features['fr_benzene'] = Fragments.fr_benzene(mol)
            features['fr_halogen'] = Fragments.fr_halogen(mol)
            features['fr_ar_n'] = Fragments.fr_Ar_N(mol)
            features['fr_al_coo'] = Fragments.fr_Al_COO(mol)
            features['fr_alkyl_halide'] = Fragments.fr_alkyl_halide(mol)
            features['fr_amide'] = Fragments.fr_amide(mol)
            features['fr_aniline'] = Fragments.fr_aniline(mol)
            features['fr_nitro'] = Fragments.fr_nitro(mol)
            features['fr_sulfide'] = Fragments.fr_sulfide(mol)
            features['fr_ester'] = Fragments.fr_ester(mol)
            features['fr_ether'] = Fragments.fr_ether(mol)

        # === FINGERPRINTS ===
        if include_fingerprints:
            # Morgan fingerprints
            morgan_fp = GetMorganFingerprintAsBitVect(mol, radius=morgan_radius, nBits=morgan_bits)
            for i, bit in enumerate(morgan_fp):
                features[f'morgan_{i}'] = int(bit)

            # MACCS keys
            maccs_fp = GetMACCSKeysFingerprint(mol)
            for i, bit in enumerate(maccs_fp):
                features[f'maccs_{i}'] = int(bit)

    except Exception as e:
        st.warning(f"Error calculating features for molecule: {str(e)}")
        return {}

    return features

def smiles_to_features(smiles: str) -> np.ndarray:
    """Convert SMILES string to feature array"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    features_dict = extract_molecular_features(mol)
    if not features_dict:
        return None
    
    # Convert to numpy array (values only, in consistent order)
    return np.array(list(features_dict.values()))

def load_toxicity_models():
    """Load all 12 endpoint models"""
    models = {}
    missing_models = []
    
    for endpoint in TOX21_ENDPOINTS:
        try:
            models[endpoint] = joblib.load(f'{endpoint}.pkl')
        except FileNotFoundError:
            missing_models.append(endpoint)
        except Exception as e:
            st.error(f"Error loading {endpoint} model: {str(e)}")
            missing_models.append(endpoint)
    
    if missing_models:
        st.warning(f"⚠️ Missing models: {', '.join(missing_models)}")
    
    if models:
        st.success(f"✅ Successfully loaded {len(models)} toxicity models!")
    
    return models

def predict_toxicity_multi_endpoint(smiles_list: List[str], models: Dict, selected_endpoints: List[str]) -> pd.DataFrame:
    """Make toxicity predictions for multiple endpoints"""
    results = []
    
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        
        # Initialize result row
        result_row = {
            'SMILES': smiles,
            'Valid': mol is not None
        }
        
        if mol is None:
            # Add None values for all endpoints
            for endpoint in selected_endpoints:
                result_row.update({
                    f'{endpoint}_Probability': None,
                    f'{endpoint}_Prediction': 'Invalid SMILES',
                    f'{endpoint}_Risk_Level': 'N/A'
                })
            results.append(result_row)
            continue
        
        # Extract features
        features = smiles_to_features(smiles)
        if features is None:
            # Add failed values for all endpoints
            for endpoint in selected_endpoints:
                result_row.update({
                    f'{endpoint}_Probability': None,
                    f'{endpoint}_Prediction': 'Feature extraction failed',
                    f'{endpoint}_Risk_Level': 'N/A'
                })
            results.append(result_row)
            continue
        
        # Make predictions for each selected endpoint
        features_reshaped = features.reshape(1, -1)
        
        for endpoint in selected_endpoints:
            if endpoint in models:
                try:
                    pred_proba = models[endpoint].predict_proba(features_reshaped)[0]
                    
                    # Handle single class case
                    if len(pred_proba) == 1:
                        toxic_prob = pred_proba[0]
                    else:
                        toxic_prob = pred_proba[1]  # Probability of toxic class
                    
                    prediction = 'Toxic' if toxic_prob > 0.5 else 'Non-toxic'
                    risk_level = 'High' if toxic_prob > 0.7 else 'Medium' if toxic_prob > 0.3 else 'Low'
                    
                    result_row.update({
                        f'{endpoint}_Probability': round(toxic_prob, 3),
                        f'{endpoint}_Prediction': prediction,
                        f'{endpoint}_Risk_Level': risk_level
                    })
                    
                except Exception as e:
                    result_row.update({
                        f'{endpoint}_Probability': None,
                        f'{endpoint}_Prediction': f'Prediction failed: {str(e)}',
                        f'{endpoint}_Risk_Level': 'N/A'
                    })
            else:
                result_row.update({
                    f'{endpoint}_Probability': None,
                    f'{endpoint}_Prediction': 'Model not available',
                    f'{endpoint}_Risk_Level': 'N/A'
                })
        
        results.append(result_row)
    
    return pd.DataFrame(results)

def add_endpoint_selection(available_models):
    """Add endpoint selection interface"""
    st.header("🎯 Select Toxicity Endpoints")
    
    # Show available models
    st.info(f"**Available Models ({len(available_models)}):** {', '.join(available_models.keys())}")
    
    # Option to select all or individual endpoints
    select_all = st.checkbox("Select All Available Endpoints", value=True)
    
    if select_all:
        selected_endpoints = list(available_models.keys())
    else:
        st.write("**Choose specific endpoints:**")
        selected_endpoints = []
        
        # Group endpoints by category
        available_nr = [ep for ep in available_models.keys() if ep.startswith('NR-')]
        available_sr = [ep for ep in available_models.keys() if ep.startswith('SR-')]
        
        if available_nr or available_sr:
            col1, col2 = st.columns(2)
            
            with col1:
                if available_nr:
                    st.write("**Nuclear Receptor (NR) Endpoints:**")
                    for endpoint in available_nr:
                        if st.checkbox(f"{endpoint}: {ENDPOINT_NAMES[endpoint]}", key=endpoint):
                            selected_endpoints.append(endpoint)
            
            with col2:
                if available_sr:
                    st.write("**Stress Response (SR) Endpoints:**")
                    for endpoint in available_sr:
                        if st.checkbox(f"{endpoint}: {ENDPOINT_NAMES[endpoint]}", key=endpoint):
                            selected_endpoints.append(endpoint)
    
    return selected_endpoints

def display_multi_endpoint_results(results_df: pd.DataFrame, selected_endpoints: List[str]):
    """Display results for multiple endpoints"""
    
    # Overall summary
    st.subheader("📊 Overall Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Compounds", len(results_df))
    with col2:
        valid_count = results_df['Valid'].sum()
        st.metric("Valid Predictions", valid_count)
    with col3:
        st.metric("Endpoints Tested", len(selected_endpoints))
    
    # Endpoint-specific summaries
    st.subheader("🎯 Endpoint-Specific Results")
    
    # Create metrics for each endpoint
    if len(selected_endpoints) <= 6:
        # Show all endpoints in columns
        endpoint_cols = st.columns(len(selected_endpoints))
        
        for i, endpoint in enumerate(selected_endpoints):
            with endpoint_cols[i]:
                prob_col = f'{endpoint}_Probability'
                if prob_col in results_df.columns:
                    valid_probs = results_df[prob_col].dropna()
                    if len(valid_probs) > 0:
                        toxic_count = len(valid_probs[valid_probs > 0.5])
                        st.metric(
                            f"{endpoint}",
                            f"{toxic_count}/{len(valid_probs)}",
                            help=f"{ENDPOINT_NAMES[endpoint]} - Toxic predictions"
                        )
    else:
        # Show in multiple rows for many endpoints
        for i in range(0, len(selected_endpoints), 4):
            cols = st.columns(4)
            batch = selected_endpoints[i:i+4]
            
            for j, endpoint in enumerate(batch):
                with cols[j]:
                    prob_col = f'{endpoint}_Probability'
                    if prob_col in results_df.columns:
                        valid_probs = results_df[prob_col].dropna()
                        if len(valid_probs) > 0:
                            toxic_count = len(valid_probs[valid_probs > 0.5])
                            st.metric(
                                f"{endpoint}",
                                f"{toxic_count}/{len(valid_probs)}",
                                help=f"{ENDPOINT_NAMES[endpoint]} - Toxic predictions"
                            )
    
    # Detailed results table
    st.subheader("📋 Detailed Results")
    
    # Create tabs for each endpoint
    if len(selected_endpoints) <= 4:
        # Show all in tabs
        tabs = st.tabs([f"{ep}" for ep in selected_endpoints])
        
        for i, endpoint in enumerate(selected_endpoints):
            with tabs[i]:
                endpoint_cols = ['SMILES', f'{endpoint}_Probability', f'{endpoint}_Prediction', f'{endpoint}_Risk_Level']
                available_cols = ['SMILES'] + [col for col in endpoint_cols[1:] if col in results_df.columns]
                endpoint_df = results_df[available_cols].copy()
                
                # Rename columns for display
                if f'{endpoint}_Probability' in endpoint_df.columns:
                    endpoint_df = endpoint_df.rename(columns={
                        f'{endpoint}_Probability': 'Probability',
                        f'{endpoint}_Prediction': 'Prediction',
                        f'{endpoint}_Risk_Level': 'Risk Level'
                    })
                
                st.dataframe(endpoint_df, use_container_width=True)
                st.caption(f"**{ENDPOINT_NAMES[endpoint]}**")
    else:
        # Use selectbox for many endpoints
        selected_endpoint_view = st.selectbox(
            "Select endpoint to view:",
            selected_endpoints,
            format_func=lambda x: f"{x}: {ENDPOINT_NAMES[x]}"
        )
        
        endpoint_cols = ['SMILES', f'{selected_endpoint_view}_Probability', 
                        f'{selected_endpoint_view}_Prediction', f'{selected_endpoint_view}_Risk_Level']
        available_cols = ['SMILES'] + [col for col in endpoint_cols[1:] if col in results_df.columns]
        endpoint_df = results_df[available_cols].copy()
        
        # Rename columns for display
        if f'{selected_endpoint_view}_Probability' in endpoint_df.columns:
            endpoint_df = endpoint_df.rename(columns={
                f'{selected_endpoint_view}_Probability': 'Probability',
                f'{selected_endpoint_view}_Prediction': 'Prediction',
                f'{selected_endpoint_view}_Risk_Level': 'Risk Level'
            })
        
        st.dataframe(endpoint_df, use_container_width=True)
        st.caption(f"**{ENDPOINT_NAMES[selected_endpoint_view]}**")

def create_comprehensive_summary(results_df: pd.DataFrame, selected_endpoints: List[str]):
    """Create a comprehensive toxicity summary across all endpoints"""
    
    st.subheader("🔍 Comprehensive Toxicity Analysis")
    
    # Calculate overall risk scores
    summary_data = []
    
    for idx, row in results_df.iterrows():
        if not row['Valid']:
            continue
            
        compound_summary = {
            'SMILES': row['SMILES'],
            'Toxic_Endpoints': 0,
            'High_Risk_Endpoints': 0,
            'Avg_Probability': 0,
            'Max_Risk_Endpoint': 'None',
            'Max_Probability': 0
        }
        
        probabilities = []
        max_prob = 0
        max_endpoint = 'None'
        
        for endpoint in selected_endpoints:
            prob_col = f'{endpoint}_Probability'
            pred_col = f'{endpoint}_Prediction'
            risk_col = f'{endpoint}_Risk_Level'
            
            if prob_col in row and pd.notna(row[prob_col]):
                prob = row[prob_col]
                probabilities.append(prob)
                
                if prob > max_prob:
                    max_prob = prob
                    max_endpoint = endpoint
                
                if pred_col in row and row[pred_col] == 'Toxic':
                    compound_summary['Toxic_Endpoints'] += 1
                    
                if risk_col in row and row[risk_col] == 'High':
                    compound_summary['High_Risk_Endpoints'] += 1
        
        if probabilities:
            compound_summary['Avg_Probability'] = round(np.mean(probabilities), 3)
            compound_summary['Max_Probability'] = round(max_prob, 3)
            compound_summary['Max_Risk_Endpoint'] = max_endpoint
        
        summary_data.append(compound_summary)
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        
        # Sort by average probability (highest risk first)
        summary_df = summary_df.sort_values('Avg_Probability', ascending=False)
        
        st.dataframe(summary_df, use_container_width=True)
        
        # Download comprehensive results
        full_csv = results_df.to_csv(index=False)
        summary_csv = summary_df.to_csv(index=False)
        
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="📥 Download Full Results",
                data=full_csv,
                file_name="multi_endpoint_toxicity_predictions.csv",
                mime="text/csv"
            )
        
        with col2:
            st.download_button(
                label="📥 Download Summary",
                data=summary_csv,
                file_name="toxicity_summary.csv",
                mime="text/csv"
            )

# ===== AI ANALYSIS SYSTEM PROMPTS =====
TOXICOLOGY_EXPERT_SYSTEM_PROMPT = """
You are Dr. Sarah Chen, a senior toxicologist with 15+ years of experience in computational toxicology and drug safety assessment. You specialize in explaining complex toxicological findings to both scientific and non-scientific audiences.

YOUR ROLE:
- Translate technical toxicity predictions into clear, accessible explanations
- Provide context about health implications and risk levels
- Offer practical recommendations based on findings
- Maintain scientific accuracy while being conversational and approachable

KNOWLEDGE CONTEXT:
- Multiple endpoints tested including NR (Nuclear Receptor) and SR (Stress Response) pathways
- Risk levels: High (>70% probability), Medium (30-70%), Low (<30%)
- Predictions are computational estimates, not definitive clinical assessments

COMMUNICATION STYLE:
- Professional yet approachable
- Use analogies when helpful
- Acknowledge uncertainties appropriately
- Provide actionable insights when possible
- Always include appropriate disclaimers about limitations
"""

IC50_EXPERT_SYSTEM_PROMPT = """
You are Dr. Alex Chen, a computational medicinal chemist with 15+ years of experience in drug discovery and IC50 prediction. You specialize in explaining IC50 predictions and their implications for drug development.

YOUR ROLE:
- Interpret IC50 predictions and their biological significance
- Explain structure-activity relationships
- Provide insights for lead optimization
- Maintain scientific accuracy while being accessible

KNOWLEDGE CONTEXT:
- IC50 values represent the concentration needed to inhibit 50% of biological activity
- Lower IC50 = higher potency (more active compound)
- Your model predicts scaled IC50 values from multiple targets:
  * Dihydrofolate reductase
  * Phosphodiesterase 5A  
  * Voltage-gated T-type calcium channel alpha-1H subunit
  * Tyrosine-protein kinase ABL
  * Epidermal growth factor receptor erbB1
  * Maltase-glucoamylase
- Activity levels: Highly Active (scaled IC50 < -0.5), Active (-0.5 to 0), Moderately Active (0 to 0.5), Low Activity (>0.5)
- Molecular descriptors used: MW, TPSA, H-bond donors/acceptors, rotatable bonds, LogP, rings, etc.

COMMUNICATION STYLE:
- Professional yet accessible
- Use drug discovery terminology appropriately
- Provide actionable medicinal chemistry insights
- Acknowledge computational prediction limitations
"""

# ===== AI ANALYSIS PROMPT TEMPLATES =====
PROMPT_TEMPLATES = {
    "multi_endpoint_analysis": """
ANALYSIS REQUEST: Multi-Endpoint Toxicity Assessment

DATASET SUMMARY:
- Total Compounds Analyzed: {total_compounds}
- Valid Predictions: {valid_predictions}
- Endpoints Tested: {endpoints_tested}

ENDPOINT BREAKDOWN:
{endpoint_summaries}

COMPREHENSIVE SUMMARY:
{comprehensive_summary}

Please provide:
1. Overall risk assessment across all endpoints
2. Most concerning endpoints and compounds
3. Cross-endpoint patterns and correlations
4. Priority recommendations for risk management
5. Limitations and next steps

Focus on multi-endpoint toxicity implications and integrated risk assessment.
""",

    "single_compound_toxicity": """
ANALYSIS REQUEST: Single Compound Toxicity Assessment

COMPOUND DATA:
- SMILES: {smiles}
- Compound Name: {compound_name}
- Toxicity Probability: {probability}%
- Prediction: {prediction}
- Risk Level: {risk_level}

Please provide a comprehensive but accessible explanation covering:
1. What this compound is (if identifiable from SMILES)
2. The toxicity prediction and what it means
3. Health implications of the risk level
4. Recommendations for handling/exposure
5. Limitations of this computational assessment

Keep the explanation conversational but scientifically accurate.
""",

    "single_compound_ic50": """
ANALYSIS REQUEST: Single Compound IC50 Prediction Analysis

COMPOUND DATA:
- SMILES: {smiles}
- Predicted IC50 (scaled): {ic50_scaled}
- Activity Level: {activity_level}
- Key Properties: {properties}

MOLECULAR DESCRIPTORS:
- Molecular Weight: {mol_weight}
- LogP (Lipophilicity): {logp}
- TPSA: {tpsa}
- H-bond Donors: {hbd}
- H-bond Acceptors: {hba}
- Rotatable Bonds: {rotatable_bonds}
- Aromatic Rings: {aromatic_rings}
- Heavy Atom Count: {heavy_atoms}
- Fraction CSP3: {fraction_csp3}
- Bertz Complexity: {bertz_ct}

Please provide:
1. IC50 prediction interpretation and biological significance
2. **Physicochemical Analysis**: How do the molecular descriptors (MW, LogP, TPSA, etc.) explain the predicted activity level?
3. **Drug-likeness Assessment**: Evaluate Lipinski's Rule of Five and other drug-likeness criteria
4. **SAR Insights**: Which molecular features are likely driving the bioactivity?
5. **Optimization Strategy**: Based on the descriptors, what modifications could improve activity?
6. **Target Binding Rationale**: How do these properties relate to binding to the protein targets?

Keep scientifically accurate but Focus on connecting molecular features to bioactivity prediction.
""",

    "multiple_compounds_toxicity": """
ANALYSIS REQUEST: Multiple Compounds Toxicity Assessment

DATASET SUMMARY:
- Total Compounds Analyzed: {total_compounds}
- Valid Predictions: {valid_predictions}
- Predicted Toxic: {toxic_count}
- High Risk Compounds: {high_risk_count}

DETAILED RESULTS:
{compounds_data}

Please provide:
1. Overall assessment of the compound set
2. Key patterns or concerns identified
3. Prioritization of compounds by risk
4. General recommendations for the dataset
5. Notable findings or outliers

Focus on actionable insights and risk prioritization.
""",

    "multiple_compounds_ic50": """
ANALYSIS REQUEST: Compound Library IC50 Screening Analysis

DATASET SUMMARY:
- Total Compounds: {total_compounds}
- Valid Predictions: {valid_predictions}
- Highly Active: {highly_active_count}
- Active: {active_count}
- Average Scaled IC50: {avg_ic50:.3f}

TOP COMPOUNDS:
{top_compounds_data}

Please provide:
1. Overall library screening assessment
2. Hit identification and prioritization
3. Structure-activity patterns observed
4. Lead optimization recommendations
5. Next steps for medicinal chemistry teams

Focus on actionable insights for drug discovery.
"""
}

# ===== AI ANALYSIS HELPER FUNCTIONS =====
def format_single_ic50_data(row: Dict, features_dict: Dict) -> Dict:
    """Format single IC50 compound data for LLM input"""
    return {
        "smiles": row.get('SMILES', 'Unknown'),
        "ic50_scaled": row.get('Predicted_IC50_Scaled', 0),
        "activity_level": row.get('Activity_Level', 'Unknown'),
        "properties": row.get('Molecular_Properties', 'N/A'),
        "mol_weight": features_dict.get('MolWt', 'N/A'),
        "logp": features_dict.get('LogP', 'N/A'),
        "tpsa": features_dict.get('TPSA', 'N/A'),
        "hbd": features_dict.get('NumHDonors', 'N/A'),
        "hba": features_dict.get('NumHAcceptors', 'N/A'),
        "rotatable_bonds": features_dict.get('NumRotatableBonds', 'N/A'),
        "aromatic_rings": features_dict.get('NumAromaticRings', 'N/A'),
        "heavy_atoms": features_dict.get('HeavyAtomCount', 'N/A'),
        "fraction_csp3": features_dict.get('FractionCSP3', 'N/A'),
        "bertz_ct": features_dict.get('BertzCT', 'N/A')
    }

def format_multiple_ic50_data(results_df) -> Dict:
    """Format multiple IC50 compounds data for LLM input"""
    valid_results = results_df[results_df['Valid'] == True]
    
    # Summary statistics
    highly_active = len(valid_results[valid_results['Activity_Level'] == 'Highly Active'])
    active = len(valid_results[valid_results['Activity_Level'] == 'Active'])
    avg_ic50 = valid_results['Predicted_IC50_Scaled'].mean()
    
    summary = {
        "total_compounds": len(results_df),
        "valid_predictions": len(valid_results),
        "highly_active_count": highly_active,
        "active_count": active,
        "avg_ic50": avg_ic50
    }
    
    # Top compounds (most active - lowest IC50)
    top_compounds = valid_results.nsmallest(5, 'Predicted_IC50_Scaled')
    compounds_data = []
    for idx, row in top_compounds.iterrows():
        features = featurize(row['SMILES'])
        compounds_data.append(
            f"- {row['SMILES']}: IC50 {row['Predicted_IC50_Scaled']:.3f} ({row['Activity_Level']})\n"
            f"  MW: {features.get('MolWt', 'N/A'):.1f}, LogP: {features.get('LogP', 'N/A'):.2f}, "
            f"TPSA: {features.get('TPSA', 'N/A'):.1f}, HBD: {features.get('NumHDonors', 'N/A')}, "
            f"HBA: {features.get('NumHAcceptors', 'N/A')}"
        )
    
    summary["top_compounds_data"] = "\n".join(compounds_data)
    
    return summary

def generate_ic50_analysis(results_df, client: GroqClient, analysis_type: str = "multiple") -> str:
    """Generate AI analysis for IC50 predictions"""
    if analysis_type == "single" and len(results_df) == 1:
        row = results_df.iloc[0].to_dict()
        features_dict = featurize(row['SMILES'])
        data = format_single_ic50_data(row, features_dict)
        prompt = PROMPT_TEMPLATES["single_compound_ic50"].format(**data)
    else:
        data = format_multiple_ic50_data(results_df)
        prompt = PROMPT_TEMPLATES["multiple_compounds_ic50"].format(**data)
    
    return client.generate_response(IC50_EXPERT_SYSTEM_PROMPT, prompt)

# ===== COMMON INPUT FUNCTIONS =====
def get_smiles_input(key_prefix=""):
    """Common function to get SMILES input - used by both pages"""
    input_method = st.radio(
        "Choose input method:",
        ["Single SMILES", "Multiple SMILES (text area)", "Upload CSV file"],
        key=f"{key_prefix}_input_method"
    )
    
    smiles_list = []
    
    if input_method == "Single SMILES":
        smiles_input = st.text_input(
            "Enter SMILES string:",
            placeholder="e.g., CC(=O)OC1=CC=CC=C1C(=O)O (aspirin)",
            help="Enter a valid SMILES notation",
            key=f"{key_prefix}_single_smiles"
        )
        if smiles_input:
            smiles_list = [smiles_input.strip()]
    
    elif input_method == "Multiple SMILES (text area)":
        smiles_text = st.text_area(
            "Enter SMILES (one per line):",
            placeholder="CC(=O)OC1=CC=CC=C1C(=O)O\nCN1C=NC2=C1C(=O)N(C(=O)N2C)C\nCC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
            height=200,
            help="Enter multiple SMILES, one per line",
            key=f"{key_prefix}_multi_smiles"
        )
        if smiles_text:
            smiles_list = [s.strip() for s in smiles_text.split('\n') if s.strip()]
    
    else:  # CSV upload
        uploaded_file = st.file_uploader(
            "Upload CSV file with SMILES column:",
            type=['csv'],
            help="CSV file should contain a column named 'canonical_smiles', 'SMILES', or 'smiles'",
            key=f"{key_prefix}_csv_upload"
        )
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                st.write("**Preview of uploaded data:**")
                st.dataframe(df.head())
                
                # Find SMILES column
                smiles_col = None
                for col in df.columns:
                    if col.lower() in ['canonical_smiles', 'smiles', 'smile']:
                        smiles_col = col
                        break
                
                if smiles_col:
                    smiles_list = df[smiles_col].dropna().tolist()
                    st.success(f"Found {len(smiles_list)} SMILES in column '{smiles_col}'")
                else:
                    st.error("No SMILES column found. Please ensure your CSV has a column named 'canonical_smiles' or 'SMILES'.")
            
            except Exception as e:
                st.error(f"Error reading CSV file: {str(e)}")
    
    return smiles_list

# ===== BIOACTIVITY PREDICTION PAGE =====
def bioactivity_page():
    st.title("🧬 IC50 Bioactivity Predictor")
    st.markdown("**Predict IC50 values for chemical compounds using Random Forest regression**")
    st.markdown("*Trained on multiple protein targets: DHFR, PDE5A, T-type calcium channel, ABL kinase, EGFR, Maltase-glucoamylase*")
    
    # Model loading section
    st.header("📊 Model Loading")
    
    model_source = st.radio(
        "Choose model source:",
        ["Load from local file", "Load from GitHub URL"],
        key="bioactivity_model_source"
    )
    
    model = None
    scaler_X = None
    
    if model_source == "Load from local file":
        # Try to load local model
        try:
            model = joblib.load('bioactivity_model.joblib')
            st.success("✅ Model loaded successfully from local file!")
            
            # Try to load scaler
            try:
                from sklearn.preprocessing import StandardScaler
                # If you saved the scaler during training, load it here
                # For now, we'll create a warning about missing scaler
                scaler_X = create_default_scaler()
            except:
                scaler_X = create_default_scaler()
                
        except FileNotFoundError:
            st.error("❌ Model file 'bioactivity_model.joblib' not found in current directory.")
            st.info("💡 Please upload your model file or provide a GitHub URL.")
        except Exception as e:
            st.error(f"❌ Error loading local model: {str(e)}")
    
    else:  # GitHub URL
        model_url = st.text_input(
            "Enter GitHub raw URL to your model file:",
            placeholder="https://raw.githubusercontent.com/username/repo/main/bioactivity_model.joblib",
            help="Make sure to use the 'raw' GitHub URL",
            key="bioactivity_model_url"
        )
        
        scaler_url = st.text_input(
            "Enter GitHub raw URL to your scaler file (optional):",
            placeholder="https://raw.githubusercontent.com/username/repo/main/scaler_X.joblib",
            help="If you saved the StandardScaler during training",
            key="bioactivity_scaler_url"
        )
        
        if model_url:
            try:
                with st.spinner("Loading model from GitHub..."):
                    model, scaler_from_github = load_bioactivity_model_and_scaler_from_github(model_url, scaler_url)
                    scaler_X = scaler_from_github if scaler_from_github else create_default_scaler()
                st.success("✅ Model loaded successfully from GitHub!")
            except Exception as e:
                st.error(f"❌ {str(e)}")
    
    if model is None:
        st.warning("Please load a model to continue with predictions.")
        st.stop()
    
    # Input methods
    st.header("📝 Input SMILES")
    smiles_list = get_smiles_input("bioactivity")
    
    # Make predictions
    if smiles_list:
        st.header("🔮 IC50 Predictions")
        
        with st.spinner("Making IC50 predictions..."):
            results_df = predict_ic50(smiles_list, model, scaler_X)
        
        # Display results
        st.subheader("📊 Results Summary")
        
        valid_results = results_df[results_df['Valid'] == True]
        if len(valid_results) > 0:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Compounds", len(results_df))
            with col2:
                st.metric("Valid Predictions", len(valid_results))
            with col3:
                highly_active = len(valid_results[valid_results['Activity_Level'] == 'Highly Active'])
                st.metric("Highly Active", highly_active)
            with col4:
                avg_ic50 = valid_results['Predicted_IC50_Scaled'].mean()
                st.metric("Avg IC50 (scaled)", f"{avg_ic50:.3f}")

        # Detailed results table
        st.subheader("📋 Detailed Results")
        
        # Color code the results based on activity level
        def color_activity(val):
            if val == 'Highly Active':
                return 'background-color: #00ff00'
            elif val == 'Active':
                return 'background-color: #90ee90'
            elif val == 'Moderately Active':
                return 'background-color: #ffff99'
            elif val == 'Low Activity':
                return 'background-color: #ffcccb'
            else:
                return ''
        
        styled_df = results_df.style.applymap(color_activity, subset=['Activity_Level'])
        st.dataframe(styled_df, use_container_width=True)
        
        # Download results
        csv = results_df.to_csv(index=False)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv,
            file_name=f"ic50_predictions_{timestamp}.csv",
            mime="text/csv"
        )
        
        # Visualization
        if len(valid_results) > 0:
            st.subheader("📈 Results Visualization")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Activity level distribution
                activity_counts = valid_results['Activity_Level'].value_counts()
                st.bar_chart(activity_counts)
                st.caption("Activity Level Distribution")
            
            with col2:
                # IC50 distribution
                ic50_data = valid_results['Predicted_IC50_Scaled'].values
                fig_data = pd.DataFrame({'Scaled IC50': ic50_data})
                st.bar_chart(fig_data['Scaled IC50'])
                st.caption("IC50 Value Distribution")
            
            # Top compounds table
            st.subheader("🏆 Most Active Compounds (Lowest IC50)")
            top_compounds = valid_results.nsmallest(10, 'Predicted_IC50_Scaled')
            st.dataframe(top_compounds[['SMILES', 'Predicted_IC50_Scaled', 'Activity_Level', 'Molecular_Properties']])

        # Add AI Analysis section
        if len(valid_results) > 0:
            add_bioactivity_ai_analysis_section(results_df)

def add_bioactivity_ai_analysis_section(results_df):
    """Add AI analysis section for bioactivity predictions"""
    st.header("🤖 AI IC50 Analysis")
    
    # Get API key
    api_key = get_groq_api_key()
    
    if not api_key:
        st.info("💡 Add your Groq API key in the sidebar to enable AI-powered analysis and insights!")
        return
    
    # Initialize client
    try:
        client = GroqClient(api_key)
        
        # Analysis type selection
        analysis_options = ["Compound Overview", "Single Compound Deep Dive"]
        analysis_type = st.selectbox("Choose analysis type:", analysis_options, key="bioactivity_analysis_type")
        
        if analysis_type == "Single Compound Deep Dive":
            valid_results = results_df[results_df['Valid'] == True]
            if len(valid_results) == 0:
                st.error("No valid compounds to analyze")
                return
            
            # Let user select a compound
            selected_idx = st.selectbox(
                "Select compound for detailed analysis:",
                range(len(valid_results)),
                format_func=lambda x: f"{valid_results.iloc[x]['SMILES'][:30]}... (IC50: {valid_results.iloc[x]['Predicted_IC50_Scaled']:.3f})",
                key="bioactivity_compound_select"
            )
        
        if st.button("🧬 Generate AI Analysis", type="primary", key="bioactivity_generate_analysis"):
            with st.spinner("Generating expert IC50 analysis..."):
                valid_results = results_df[results_df['Valid'] == True]
                
                if len(valid_results) == 0:
                    st.error("No valid compounds to analyze")
                    return
                
                if analysis_type == "Single Compound Deep Dive":
                    # Create single compound dataframe
                    single_compound_df = valid_results.iloc[[selected_idx]].copy()
                    analysis = generate_ic50_analysis(single_compound_df, client, "single")
                else:
                    analysis = generate_ic50_analysis(results_df, client, "multiple")
            
            # Display analysis
            st.subheader("🧬 Expert Analysis")
            st.markdown(analysis)
            
            # Save analysis option
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            analysis_filename = f"ic50_analysis_{timestamp}.txt"
            
            st.download_button(
                label="📄 Download Analysis Report",
                data=analysis,
                file_name=analysis_filename,
                mime="text/plain",
                key="bioactivity_download_analysis"
            )
    
    except Exception as e:
        st.error(f"Error initializing AI analysis: {str(e)}")
        st.error("Please check your Groq API key and try again.")

# ===== TOXICITY PREDICTION PAGE =====
def toxicity_page():
    st.title("🧪 Multi-Endpoint Toxicity Predictor")
    st.markdown("**Predict toxicity across multiple biological endpoints using TOX21 models**")
    
    # Load all models
    models = load_toxicity_models()
    
    if not models:
        st.error("❌ No models could be loaded. Please ensure model files (.pkl) are available.")
        st.stop()
    
    # Input methods
    st.header("📝 Input SMILES")
    smiles_list = get_smiles_input("toxicity")
    
    # Add endpoint selection
    selected_endpoints = add_endpoint_selection(models)
    
    if not selected_endpoints:
        st.warning("Please select at least one endpoint to proceed.")
        return
    
    # Make predictions
    if smiles_list and selected_endpoints:
        st.header("🔮 Multi-Endpoint Predictions")
        
        with st.spinner("Making predictions across selected endpoints..."):
            results_df = predict_toxicity_multi_endpoint(smiles_list, models, selected_endpoints)
        
        # Display multi-endpoint results
        display_multi_endpoint_results(results_df, selected_endpoints)
        
        # Comprehensive summary
        create_comprehensive_summary(results_df, selected_endpoints)
        
        # LLM analysis with multi-endpoint support
        add_toxicity_llm_analysis_to_ui(results_df, selected_endpoints)

def add_toxicity_llm_analysis_to_ui(results_df, selected_endpoints):
    """Add LLM analysis section to toxicity page"""
    
    # Get API key
    api_key = get_groq_api_key()
    
    if not api_key:
        st.warning("Please provide your Groq AI API key to generate conversational analysis.")
        return
    
    # Initialize client
    client = GroqClient(api_key)
    
    st.header("🤖 AI Toxicologist Analysis")
    
    # Analysis options
    analysis_options = ["Multi-Endpoint Overview", "Single Compound Deep Dive"]
    analysis_type = st.selectbox("Choose analysis type:", analysis_options, key="toxicity_analysis_type")
    
    if analysis_type == "Single Compound Deep Dive":
        valid_results = results_df[results_df['Valid'] == True]
        if len(valid_results) == 0:
            st.error("No valid compounds to analyze")
            return
        
        # Let user select a compound
        selected_idx = st.selectbox(
            "Select compound for detailed analysis:",
            range(len(valid_results)),
            format_func=lambda x: f"{valid_results.iloc[x]['SMILES'][:30]}...",
            key="toxicity_compound_select"
        )
    
    if st.button("🧬 Generate AI Analysis", type="primary", key="toxicity_generate_analysis"):
        with st.spinner("Generating expert toxicity analysis..."):
            valid_results = results_df[results_df['Valid'] == True]
            
            if len(valid_results) == 0:
                st.error("No valid compounds to analyze")
                return
            
            if analysis_type == "Single Compound Deep Dive":
                # Find first available endpoint for analysis
                selected_compound = valid_results.iloc[selected_idx].to_dict()
                first_endpoint = None
                for endpoint in selected_endpoints:
                    if f'{endpoint}_Prediction' in selected_compound:
                        first_endpoint = endpoint
                        break
                
                if first_endpoint:
                    # Reformat for single compound analysis
                    single_data = {
                        'smiles': selected_compound['SMILES'],
                        'compound_name': 'Unknown compound',
                        'probability': selected_compound.get(f'{first_endpoint}_Probability', 0) * 100,
                        'prediction': selected_compound.get(f'{first_endpoint}_Prediction', 'Unknown'),
                        'risk_level': selected_compound.get(f'{first_endpoint}_Risk_Level', 'Unknown')
                    }
                    prompt = PROMPT_TEMPLATES["single_compound_toxicity"].format(**single_data)
                    analysis = client.generate_response(TOXICOLOGY_EXPERT_SYSTEM_PROMPT, prompt)
                else:
                    st.error("No valid endpoint data for selected compound")
                    return
                
            else:  # Multi-Endpoint Overview
                # Create multi-endpoint summary
                endpoint_summaries = []
                
                for endpoint in selected_endpoints:
                    prob_col = f'{endpoint}_Probability'
                    if prob_col in results_df.columns:
                        valid_probs = results_df[prob_col].dropna()
                        if len(valid_probs) > 0:
                            toxic_count = len(valid_probs[valid_probs > 0.5])
                            endpoint_summaries.append(f"{endpoint}: {toxic_count}/{len(valid_probs)} toxic ({ENDPOINT_NAMES[endpoint]})")
                
                # Use the multi-endpoint template
                multi_data = {
                    "total_compounds": len(results_df),
                    "valid_predictions": len(valid_results),
                    "endpoints_tested": len(selected_endpoints),
                    "endpoint_summaries": "\n".join(endpoint_summaries),
                    "comprehensive_summary": f"Comprehensive analysis across {len(selected_endpoints)} biological endpoints"
                }
                
                prompt = PROMPT_TEMPLATES["multi_endpoint_analysis"].format(**multi_data)
                analysis = client.generate_response(TOXICOLOGY_EXPERT_SYSTEM_PROMPT, prompt)
        
        # Display analysis
        st.subheader("🧬 Expert Analysis")
        st.markdown(analysis)
        
        # Save analysis option
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        analysis_filename = f"multi_endpoint_analysis_{timestamp}.txt"
        
        st.download_button(
            label="📄 Download Analysis Report",
            data=analysis,
            file_name=analysis_filename,
            mime="text/plain",
            key="toxicity_download_analysis"
        )

# ===== MAIN APP WITH PAGE NAVIGATION =====
def main():
    st.set_page_config(
        page_title="Chemical Analysis Platform", 
        page_icon="⚗️", 
        layout="wide"
    )
    
    # Sidebar for navigation
    with st.sidebar:
        st.title("⚗️ Chemical Analysis Platform")
        
        # Page selection
        page = st.selectbox(
            "Select Analysis Type:",
            ["🧪 Toxicity Prediction", "🧬 Bioactivity Prediction (IC50)"],
            index=0
        )
        
        st.markdown("---")
        
        # Common sidebar information
        st.header("ℹ️ Platform Information")
        st.markdown("""
        **Available Analyses:**
        - **Toxicity**: Multi-endpoint TOX21 predictions
        - **Bioactivity**: IC50 predictions for drug discovery
        
        **Input Formats:**
        - Single SMILES string
        - Multiple SMILES (text area)
        - CSV file upload
        
        **AI Features:**
        - Expert analysis with Groq API
        - Conversational insights
        - Downloadable reports
        """)
        
        st.header("🧪 Example SMILES")
        st.markdown("""
        - **Aspirin**: `CC(=O)OC1=CC=CC=C1C(=O)O`
        - **Caffeine**: `CN1C=NC2=C1C(=O)N(C(=O)N2C)C`
        - **Ethanol**: `CCO`
        - **Benzene**: `C1=CC=CC=C1`
        - **Ibuprofen**: `CC(C)CC1=CC=C(C=C1)C(C)C(=O)O`
        """)
    
    # Main content based on page selection
    if page == "🧪 Toxicity Prediction":
        toxicity_page()
    elif page == "🧬 Bioactivity Prediction (IC50)":
        bioactivity_page()

if __name__ == "__main__":
    main()
