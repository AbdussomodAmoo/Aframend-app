import streamlit as st
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Fragments, Draw
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect, GetMACCSKeysFingerprint
from typing import List, Dict, Optional, Union
import joblib
import warnings
import os
import json
import requests
from typing import Dict, List, Optional
import streamlit as st
from datetime import datetime
from groq import Groq
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import base64
warnings.filterwarnings('ignore')

os.environ['GROQ_API_KEY'] = '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'

# Constants
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

def predict_toxicity(smiles_list: List[str], model) -> pd.DataFrame:
    """Make toxicity predictions for a list of SMILES"""
    results = []
    
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            results.append({
                'SMILES': smiles,
                'Valid': False,
                'Toxic_Probability': None,
                'Prediction': 'Invalid SMILES',
                'Risk_Level': 'N/A'
            })
            continue
        
        # Extract features
        features = smiles_to_features(smiles)
        if features is None:
            results.append({
                'SMILES': smiles,
                'Valid': False,
                'Toxic_Probability': None,
                'Prediction': 'Feature extraction failed',
                'Risk_Level': 'N/A'
            })
            continue
        
        # Make prediction
        try:
            features_reshaped = features.reshape(1, -1)
            pred_proba = model.predict_proba(features_reshaped)[0]
            
            # Handle single class case
            if len(pred_proba) == 1:
                toxic_prob = pred_proba[0]
            else:
                toxic_prob = pred_proba[1]  # Probability of toxic class
            
            prediction = 'Toxic' if toxic_prob > 0.5 else 'Non-toxic'
            risk_level = 'High' if toxic_prob > 0.7 else 'Medium' if toxic_prob > 0.3 else 'Low'
            
            results.append({
                'SMILES': smiles,
                'Valid': True,
                'Toxic_Probability': round(toxic_prob, 3),
                'Prediction': prediction,
                'Risk_Level': risk_level
            })
            
        except Exception as e:
            results.append({
                'SMILES': smiles,
                'Valid': False,
                'Toxic_Probability': None,
                'Prediction': f'Prediction failed: {str(e)}',
                'Risk_Level': 'N/A'
            })
    
    return pd.DataFrame(results)

def predict_bioactivity(smiles_list: List[str], model) -> pd.DataFrame:
    """Make bioactivity predictions for a list of SMILES (regression model)"""
    results = []
    
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            results.append({
                'SMILES': smiles,
                'Valid': False,
                'Predicted_Activity': None,
                'Activity_Level': 'Invalid SMILES',
                'IC50_Estimate': 'N/A'
            })
            continue
        
        # Extract features
        features = smiles_to_features(smiles)
        if features is None:
            results.append({
                'SMILES': smiles,
                'Valid': False,
                'Predicted_Activity': None,
                'Activity_Level': 'Feature extraction failed',
                'IC50_Estimate': 'N/A'
            })
            continue
        
        # Make prediction
        try:
            features_reshaped = features.reshape(1, -1)
            predicted_activity = model.predict(features_reshaped)[0]
            
            # Convert log activity to IC50 estimate (assuming log10(IC50) in nM)
            ic50_estimate = 10**predicted_activity  # Convert back from log scale
            
            # Classify activity level
            if predicted_activity < 5:  # IC50 < 100 µM (potent)
                activity_level = 'Highly Active'
            elif predicted_activity < 6:  # IC50 < 1 mM (moderate)
                activity_level = 'Moderately Active'
            elif predicted_activity < 7:  # IC50 < 10 mM (weak)
                activity_level = 'Weakly Active'
            else:
                activity_level = 'Inactive'
            
            results.append({
                'SMILES': smiles,
                'Valid': True,
                'Predicted_Activity': round(predicted_activity, 3),
                'Activity_Level': activity_level,
                'IC50_Estimate': f"{ic50_estimate:.2e} nM"
            })
            
        except Exception as e:
            results.append({
                'SMILES': smiles,
                'Valid': False,
                'Predicted_Activity': None,
                'Activity_Level': f'Prediction failed: {str(e)}',
                'IC50_Estimate': 'N/A'
            })
    
    return pd.DataFrame(results)

def calculate_drug_likeness_properties(smiles: str) -> Dict:
    """Calculate drug-likeness properties (Lipinski's Rule of Five, etc.)"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}
    
    properties = {
        'Molecular_Weight': Descriptors.MolWt(mol),
        'LogP': Descriptors.MolLogP(mol),
        'HBD': Descriptors.NumHDonors(mol),
        'HBA': Descriptors.NumHAcceptors(mol),
        'TPSA': Descriptors.TPSA(mol),
        'Rotatable_Bonds': Descriptors.NumRotatableBonds(mol),
        'QED': Descriptors.qed(mol)
    }
    
    # Lipinski's Rule of Five violations
    violations = 0
    if properties['Molecular_Weight'] > 500: violations += 1
    if properties['LogP'] > 5: violations += 1
    if properties['HBD'] > 5: violations += 1
    if properties['HBA'] > 10: violations += 1
    
    properties['Lipinski_Violations'] = violations
    properties['Drug_Like'] = 'Yes' if violations <= 1 else 'No'
    
    return properties

def get_groq_api_key(): 
    """Securely retrieve Groq API key"""
    # Option 1: Environment variable
    api_key = os.getenv('GROQ_API_KEY')
    
    # Option 2: Streamlit secrets
    if not api_key:
        try:
            api_key = st.secrets["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"]
            st.success("✅ API key loaded from secrets!")
        except Exception as e:
            st.warning(f"Could not load from secrets: {e}")
            pass
    
    # Option 3: User input
    if not api_key:
        api_key = st.sidebar.text_input(
            "Enter Groq API Key:",
            type="password",
            help="Your API key will not be stored"
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

# SYSTEM PROMPTS
TOXICOLOGY_EXPERT_SYSTEM_PROMPT = """
You are Dr. Sarah Chen, a senior toxicologist with 15+ years of experience in computational toxicology and drug safety assessment. You specialize in explaining complex toxicological findings to both scientific and non-scientific audiences.

YOUR ROLE:
- Translate technical toxicity predictions into clear, accessible explanations
- Provide context about health implications and risk levels
- Offer practical recommendations based on findings
- Maintain scientific accuracy while being conversational and approachable

KNOWLEDGE CONTEXT:
- NR-AR endpoint: Androgen Receptor disruption affects hormone balance, potentially causing reproductive issues, developmental problems, and endocrine disruption
- Risk levels: High (>70% probability), Medium (30-70%), Low (<30%)
- Predictions are computational estimates, not definitive clinical assessments
"""

BIOACTIVITY_EXPERT_SYSTEM_PROMPT = """
You are Dr. Michael Rodriguez, a medicinal chemist and drug discovery expert with 20+ years of experience in pharmaceutical research. You specialize in structure-activity relationships, lead optimization, and bioactivity prediction.

YOUR ROLE:
- Interpret bioactivity predictions and IC50 estimates
- Explain structure-activity relationships and drug-likeness
- Provide medicinal chemistry insights and optimization suggestions
- Guide compound prioritization for drug discovery

KNOWLEDGE CONTEXT:
- Bioactivity predictions are log-transformed values (typically log10 IC50 in nM)
- Activity levels: Highly Active (<100 µM), Moderately Active (100 µM-1 mM), Weakly Active (1-10 mM), Inactive (>10 mM)
- Drug-likeness follows Lipinski's Rule of Five and other ADMET principles
- Predictions are computational estimates for early-stage screening
"""

PROMPT_TEMPLATES = {
    "bioactivity_analysis": """
ANALYSIS REQUEST: Bioactivity Prediction Analysis

COMPOUND DATA:
- SMILES: {smiles}
- Predicted Activity (log scale): {predicted_activity}
- Activity Level: {activity_level}
- IC50 Estimate: {ic50_estimate}
- Drug-likeness: {drug_like}

MOLECULAR PROPERTIES:
- Molecular Weight: {mol_weight:.2f}
- LogP: {logp:.2f}
- H-bond Donors: {hbd}
- H-bond Acceptors: {hba}
- TPSA: {tpsa:.2f}
- Lipinski Violations: {lipinski_violations}

Please provide:
1. Interpretation of the bioactivity prediction
2. Assessment of drug-likeness and developability
3. Structure-activity insights (if identifiable from SMILES)
4. Recommendations for lead optimization
5. Next steps in drug discovery workflow

Focus on actionable medicinal chemistry insights.
""",

    "compound_screening": """
ANALYSIS REQUEST: Compound Library Screening Results

SCREENING SUMMARY:
- Total Compounds: {total_compounds}
- Valid Predictions: {valid_predictions}
- Highly Active: {highly_active}
- Drug-like Compounds: {drug_like_count}

TOP COMPOUNDS:
{top_compounds}

Please provide:
1. Overall assessment of the compound library
2. Hit identification and prioritization
3. Structure-activity patterns observed
4. Drug-likeness analysis
5. Recommendations for follow-up studies

Emphasize compound prioritization and next steps.
"""
}

def generate_bioactivity_analysis(smiles: str, prediction_data: Dict, drug_props: Dict, client: GroqClient) -> str:
    """Generate bioactivity analysis for a single compound"""
    data = {
        "smiles": smiles,
        "predicted_activity": prediction_data['Predicted_Activity'],
        "activity_level": prediction_data['Activity_Level'],
        "ic50_estimate": prediction_data['IC50_Estimate'],
        "drug_like": drug_props.get('Drug_Like', 'Unknown'),
        "mol_weight": drug_props.get('Molecular_Weight', 0),
        "logp": drug_props.get('LogP', 0),
        "hbd": drug_props.get('HBD', 0),
        "hba": drug_props.get('HBA', 0),
        "tpsa": drug_props.get('TPSA', 0),
        "lipinski_violations": drug_props.get('Lipinski_Violations', 0)
    }
    
    prompt = PROMPT_TEMPLATES["bioactivity_analysis"].format(**data)
    return client.generate_response(BIOACTIVITY_EXPERT_SYSTEM_PROMPT, prompt)

def generate_screening_analysis(results_df, client: GroqClient) -> str:
    """Generate compound screening analysis"""
    valid_results = results_df[results_df['Valid'] == True]
    
    # Get drug-likeness data for valid compounds
    drug_like_count = 0
    for _, row in valid_results.iterrows():
        drug_props = calculate_drug_likeness_properties(row['SMILES'])
        if drug_props.get('Drug_Like') == 'Yes':
            drug_like_count += 1
    
    # Top compounds (highly active and drug-like)
    top_compounds = []
    for idx, row in valid_results.head(5).iterrows():
        drug_props = calculate_drug_likeness_properties(row['SMILES'])
        top_compounds.append(
            f"Compound {idx}: {row['SMILES']} - "
            f"{row['Activity_Level']} ({row['IC50_Estimate']}, "
            f"Drug-like: {drug_props.get('Drug_Like', 'Unknown')})"
        )
    
    data = {
        "total_compounds": len(results_df),
        "valid_predictions": len(valid_results),
        "highly_active": len(valid_results[valid_results['Activity_Level'] == 'Highly Active']),
        "drug_like_count": drug_like_count,
        "top_compounds": "\n".join(top_compounds)
    }
    
    prompt = PROMPT_TEMPLATES["compound_screening"].format(**data)
    return client.generate_response(BIOACTIVITY_EXPERT_SYSTEM_PROMPT, prompt)

def render_molecule_image(smiles: str, size=(300, 300)):
    """Render molecule structure from SMILES"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        img = Draw.MolToImage(mol, size=size)
        return img
    except:
        return None

def toxicity_page():
    """Toxicity prediction page"""
    st.title("🧪 NR-AR Toxicity Predictor")
    st.markdown("**Predict Androgen Receptor Disruption for chemical compounds**")
    
    # Load model
    try:
        model = joblib.load('NR-AR.pkl')
        st.success("✅ NR-AR model loaded successfully!")
        st.info(f"**Target Endpoint:** {ENDPOINT_NAMES['NR-AR']}")
    except FileNotFoundError:
        st.error("❌ Model file 'NR-AR.pkl' not found. Please ensure the file is in the same directory.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()
    
    # Input methods
    st.header("📝 Input SMILES")
    
    input_method = st.radio(
        "Choose input method:",
        ["Single SMILES", "Multiple SMILES (text area)", "Upload CSV file"]
    )
    
    smiles_list = []
    
    if input_method == "Single SMILES":
        smiles_input = st.text_input(
            "Enter SMILES string:",
            placeholder="e.g., CCO (ethanol)",
            help="Enter a valid SMILES notation"
        )
        if smiles_input:
            smiles_list = [smiles_input.strip()]
    
    elif input_method == "Multiple SMILES (text area)":
        smiles_text = st.text_area(
            "Enter SMILES (one per line):",
            placeholder="CCO\nC1=CC=CC=C1\nCC(=O)O",
            height=200,
            help="Enter multiple SMILES, one per line"
        )
        if smiles_text:
            smiles_list = [s.strip() for s in smiles_text.split('\n') if s.strip()]
    
    else:  # CSV upload
        uploaded_file = st.file_uploader(
            "Upload CSV file with SMILES column:",
            type=['csv'],
            help="CSV file should contain a column named 'SMILES' or 'smiles'"
        )
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                st.write("**Preview of uploaded data:**")
                st.dataframe(df.head())
                
                # Find SMILES column
                smiles_col = None
                for col in df.columns:
                    if col.lower() in ['smiles', 'smile', 'canonical_smiles']:
                        smiles_col = col
                        break
                
                if smiles_col:
                    smiles_list = df[smiles_col].dropna().tolist()
                    st.success(f"Found {len(smiles_list)} SMILES in column '{smiles_col}'")
                else:
                    st.error("No SMILES column found. Please ensure your CSV has a column named 'SMILES'")
            
            except Exception as e:
                st.error(f"Error reading CSV file: {str(e)}")
    
    # Make predictions
    if smiles_list:
        st.header("🔮 Predictions")
        
        with st.spinner("Making predictions..."):
            results_df = predict_toxicity(smiles_list, model)
        
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
                toxic_count = len(valid_results[valid_results['Prediction'] == 'Toxic'])
                st.metric("Predicted Toxic", toxic_count)
            with col4:
                high_risk_count = len(valid_results[valid_results['Risk_Level'] == 'High'])
                st.metric("High Risk", high_risk_count)

        # Detailed results table
        st.subheader("📋 Detailed Results")
        
        # Color code the results
        def color_prediction(val):
            if val == 'Toxic':
                return 'background-color: #ffcccc'
            elif val == 'Non-toxic':
                return 'background-color: #ccffcc'
            else:
                return ''
        
        def color_risk(val):
            if val == 'High':
                return 'background-color: #ff9999'
            elif val == 'Medium':
                return 'background-color: #ffff99'
            elif val == 'Low':
                return 'background-color: #ccffcc'
            else:
                return ''
        
        styled_df = results_df.style.applymap(color_prediction, subset=['Prediction']) \
                                    .applymap(color_risk, subset=['Risk_Level'])
        
        st.dataframe(styled_df, use_container_width=True)
        
        # Download results
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv,
            file_name="toxicity_predictions.csv",
            mime="text/csv"
        )
        
        # Visualization
        if len(valid_results) > 0:
            st.subheader("📈 Results Visualization")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Prediction distribution
                pred_counts = valid_results['Prediction'].value_counts()
                fig1 = px.bar(x=pred_counts.index, y=pred_counts.values,
                             title="Prediction Distribution",
                             labels={'x': 'Prediction', 'y': 'Count'})
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                # Risk level distribution
                risk_counts = valid_results['Risk_Level'].value_counts()
                fig2 = px.pie(values=risk_counts.values, names=risk_counts.index,
                             title="Risk Level Distribution")
                st.plotly_chart(fig2, use_container_width=True)

        # AI Analysis
        add_llm_toxicity_analysis_to_ui(results_df)

def bioactivity_page():
    """Bioactivity prediction page"""
    st.title("🎯 Bioactivity Predictor & Compound Screening")
    st.markdown("**Predict compound bioactivity and screen chemical libraries for drug discovery**")
    
    # Load bioactivity model
    try:
        model = joblib.load('Bioactivity.pkl')  # Your bioactivity model
        st.success("✅ Bioactivity model loaded successfully!")
        st.info("**Model Performance:** R² = 0.51, MAPE = 0.68% (Excellent accuracy)")
    except FileNotFoundError:
        st.error("❌ Model file 'Bioactivity.pkl' not found. Please ensure the file is in the same directory.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()
    
    # Tabs for different functionalities
    tab1, tab2, tab3 = st.tabs(["🔬 Single Compound", "📊 Batch Screening", "🧬 Structure Viewer"])
    
    with tab1:
        st.header("Single Compound Analysis")
        
        smiles_input = st.text_input(
            "Enter SMILES string:",
            placeholder="e.g., CC(=O)OC1=CC=CC=C1C(=O)O (aspirin)",
            help="Enter a valid SMILES notation for bioactivity prediction"
        )
        
        if smiles_input and st.button("Predict Bioactivity", type="primary"):
            with st.spinner("Analyzing compound..."):
                # Make prediction
                results_df = predict_bioactivity([smiles_input], model)
                result = results_df.iloc[0]
                
                if result['Valid']:
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        # Display results
                        st.subheader("🎯 Bioactivity Results")
                        
                        # Metrics
                        met_col1, met_col2, met_col3 = st.columns(3)
                        with met_col1:
                            st.metric("Activity Level", result['Activity_Level'])
                        with met_col2:
                            st.metric("IC50 Estimate", result['IC50_Estimate'])
                        with met_col3:
                            activity_score = result['Predicted_Activity']
                            st.metric("Activity Score", f"{activity_score:.3f}")
                        
                        # Drug-likeness analysis
                        drug_props = calculate_drug_likeness_properties(smiles_input)
                        
                        st.subheader("💊 Drug-likeness Assessment")
                        
                        if drug_props:
                            prop_col1, prop_col2, prop_col3, prop_col4 = st.columns(4)
                            
                            with prop_col1:
                                st.metric("Molecular Weight", f"{drug_props['Molecular_Weight']:.1f} Da")
                                st.metric("LogP", f"{drug_props['LogP']:.2f}")
                            
                            with prop_col2:
                                st.metric("H-bond Donors", drug_props['HBD'])
                                st.metric("H-bond Acceptors", drug_props['HBA'])
                            
                            with prop_col3:
                                st.metric("TPSA", f"{drug_props['TPSA']:.1f} Ų")
                                st.metric("Rotatable Bonds", drug_props['Rotatable_Bonds'])
                            
                            with prop_col4:
                                st.metric("QED Score", f"{drug_props['QED']:.3f}")
                                violations = drug_props['Lipinski_Violations']
                                st.metric("Lipinski Violations", violations)
                            
                            # Drug-likeness indicator
                            if drug_props['Drug_Like'] == 'Yes':
                                st.success("✅ **Drug-like compound** (≤1 Lipinski violation)")
                            else:
                                st.warning("⚠️ **Poor drug-likeness** (>1 Lipinski violation)")
                    
                    with col2:
                        # Molecule structure
                        st.subheader("🧪 Structure")
                        img = render_molecule_image(smiles_input)
                        if img:
                            st.image(img, caption="Molecular Structure", use_column_width=True)
                        else:
                            st.error("Could not render molecule structure")
                    
                    # AI Analysis
                    st.subheader("🤖 AI Medicinal Chemist Analysis")
                    api_key = get_groq_api_key()
                    
                    if api_key and st.button("Generate Expert Analysis"):
                        client = GroqClient(api_key)
                        with st.spinner("Generating medicinal chemistry insights..."):
                            analysis = generate_bioactivity_analysis(
                                smiles_input, result.to_dict(), drug_props, client
                            )
                            st.markdown(analysis)
                
                else:
                    st.error(f"❌ {result['Activity_Level']}")
    
    with tab2:
        st.header("Batch Compound Screening")
        
        # Input methods for batch screening
        input_method = st.radio(
            "Choose input method:",
            ["Multiple SMILES (text area)", "Upload CSV file"],
            key="batch_input"
        )
        
        smiles_list = []
        
        if input_method == "Multiple SMILES (text area)":
            smiles_text = st.text_area(
                "Enter SMILES (one per line):",
                placeholder="CC(=O)OC1=CC=CC=C1C(=O)O\nCCO\nC1=CC=CC=C1",
                height=200,
                help="Enter multiple SMILES for batch screening"
            )
            if smiles_text:
                smiles_list = [s.strip() for s in smiles_text.split('\n') if s.strip()]
        
        else:  # CSV upload
            uploaded_file = st.file_uploader(
                "Upload CSV file with SMILES column:",
                type=['csv'],
                help="CSV file should contain a column named 'SMILES'",
                key="batch_upload"
            )
            if uploaded_file:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.write("**Preview of uploaded data:**")
                    st.dataframe(df.head())
                    
                    # Find SMILES column
                    smiles_col = None
                    for col in df.columns:
                        if col.lower() in ['smiles', 'smile', 'canonical_smiles']:
                            smiles_col = col
                            break
                    
                    if smiles_col:
                        smiles_list = df[smiles_col].dropna().tolist()
                        st.success(f"Found {len(smiles_list)} SMILES in column '{smiles_col}'")
                    else:
                        st.error("No SMILES column found. Please ensure your CSV has a column named 'SMILES'")
                
                except Exception as e:
                    st.error(f"Error reading CSV file: {str(e)}")
        
        # Run batch screening
        if smiles_list and st.button("🚀 Run Compound Screening", type="primary"):
            with st.spinner(f"Screening {len(smiles_list)} compounds..."):
                # Make predictions
                results_df = predict_bioactivity(smiles_list, model)
                
                # Add drug-likeness properties
                drug_like_data = []
                for _, row in results_df.iterrows():
                    if row['Valid']:
                        props = calculate_drug_likeness_properties(row['SMILES'])
                        drug_like_data.append(props.get('Drug_Like', 'Unknown'))
                    else:
                        drug_like_data.append('N/A')
                
                results_df['Drug_Like'] = drug_like_data
            
            # Display screening results
            st.header("🔬 Screening Results")
            
            valid_results = results_df[results_df['Valid'] == True]
            
            if len(valid_results) > 0:
                # Summary metrics
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    st.metric("Total Compounds", len(results_df))
                with col2:
                    st.metric("Valid Predictions", len(valid_results))
                with col3:
                    highly_active = len(valid_results[valid_results['Activity_Level'] == 'Highly Active'])
                    st.metric("Highly Active", highly_active)
                with col4:
                    drug_like_count = len(valid_results[valid_results['Drug_Like'] == 'Yes'])
                    st.metric("Drug-like", drug_like_count)
                with col5:
                    hits = len(valid_results[
                        (valid_results['Activity_Level'] == 'Highly Active') & 
                        (valid_results['Drug_Like'] == 'Yes')
                    ])
                    st.metric("Quality Hits", hits)
                
                # Results table with filtering
                st.subheader("📋 Detailed Results")
                
                # Filters
                filter_col1, filter_col2, filter_col3 = st.columns(3)
                
                with filter_col1:
                    activity_filter = st.selectbox(
                        "Filter by Activity Level:",
                        ["All"] + list(valid_results['Activity_Level'].unique())
                    )
                
                with filter_col2:
                    drug_like_filter = st.selectbox(
                        "Filter by Drug-likeness:",
                        ["All", "Yes", "No"]
                    )
                
                with filter_col3:
                    top_n = st.slider("Show top N compounds:", 10, len(valid_results), 50)
                
                # Apply filters
                filtered_results = valid_results.copy()
                
                if activity_filter != "All":
                    filtered_results = filtered_results[filtered_results['Activity_Level'] == activity_filter]
                
                if drug_like_filter != "All":
                    filtered_results = filtered_results[filtered_results['Drug_Like'] == drug_like_filter]
                
                # Sort by predicted activity (most active first)
                filtered_results = filtered_results.sort_values('Predicted_Activity').head(top_n)
                
                # Color coding function
                def color_activity(val):
                    if val == 'Highly Active':
                        return 'background-color: #90EE90'
                    elif val == 'Moderately Active':
                        return 'background-color: #FFE4B5'
                    elif val == 'Weakly Active':
                        return 'background-color: #FFC0CB'
                    else:
                        return 'background-color: #F0F0F0'
                
                def color_drug_like(val):
                    if val == 'Yes':
                        return 'background-color: #90EE90'
                    elif val == 'No':
                        return 'background-color: #FFB6C1'
                    else:
                        return ''
                
                styled_df = filtered_results.style.applymap(color_activity, subset=['Activity_Level']) \
                                                  .applymap(color_drug_like, subset=['Drug_Like'])
                
                st.dataframe(styled_df, use_container_width=True)
                
                # Download results
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Screening Results",
                    data=csv,
                    file_name=f"bioactivity_screening_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                
                # Visualizations
                st.subheader("📊 Screening Analysis")
                
                viz_col1, viz_col2 = st.columns(2)
                
                with viz_col1:
                    # Activity distribution
                    activity_counts = valid_results['Activity_Level'].value_counts()
                    fig1 = px.bar(
                        x=activity_counts.index, 
                        y=activity_counts.values,
                        title="Activity Level Distribution",
                        labels={'x': 'Activity Level', 'y': 'Count'},
                        color=activity_counts.values,
                        color_continuous_scale='Viridis'
                    )
                    st.plotly_chart(fig1, use_container_width=True)
                
                with viz_col2:
                    # Drug-likeness vs Activity scatter plot
                    scatter_data = valid_results.copy()
                    scatter_data['Drug_Like_Numeric'] = scatter_data['Drug_Like'].map({'Yes': 1, 'No': 0})
                    
                    fig2 = px.scatter(
                        scatter_data,
                        x='Predicted_Activity',
                        y='Drug_Like_Numeric',
                        color='Activity_Level',
                        title="Activity vs Drug-likeness",
                        labels={
                            'Predicted_Activity': 'Predicted Activity (log scale)',
                            'Drug_Like_Numeric': 'Drug-like (0=No, 1=Yes)'
                        },
                        hover_data=['SMILES', 'IC50_Estimate']
                    )
                    st.plotly_chart(fig2, use_container_width=True)
                
                # Activity distribution histogram
                fig3 = px.histogram(
                    valid_results,
                    x='Predicted_Activity',
                    bins=30,
                    title="Predicted Activity Distribution",
                    labels={'Predicted_Activity': 'Predicted Activity (log scale)', 'count': 'Number of Compounds'}
                )
                st.plotly_chart(fig3, use_container_width=True)
                
                # AI Screening Analysis
                st.subheader("🤖 AI Screening Analysis")
                api_key = get_groq_api_key()
                
                if api_key and st.button("Generate Screening Analysis"):
                    client = GroqClient(api_key)
                    with st.spinner("Generating compound screening insights..."):
                        analysis = generate_screening_analysis(results_df, client)
                        st.markdown(analysis)
                        
                        # Save analysis option
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        analysis_filename = f"screening_analysis_{timestamp}.txt"
                        
                        st.download_button(
                            label="📄 Download Analysis Report",
                            data=analysis,
                            file_name=analysis_filename,
                            mime="text/plain"
                        )
            
            else:
                st.warning("No valid compounds found for screening.")
    
    with tab3:
        st.header("Structure Viewer & Comparison")
        
        # Multiple compound structure viewer
        st.subheader("🧬 Compare Compound Structures")
        
        compare_smiles = st.text_area(
            "Enter SMILES for comparison (one per line):",
            placeholder="CC(=O)OC1=CC=CC=C1C(=O)O\nCCO\nC1=CC=CC=C1",
            height=150,
            help="Enter 2-6 SMILES for visual comparison"
        )
        
        if compare_smiles:
            smiles_list = [s.strip() for s in compare_smiles.split('\n') if s.strip()]
            
            if len(smiles_list) > 6:
                st.warning("Please enter maximum 6 compounds for comparison")
                smiles_list = smiles_list[:6]
            
            if len(smiles_list) >= 2:
                st.subheader("Structure Comparison")
                
                # Create columns for structures
                cols = st.columns(min(len(smiles_list), 3))
                
                for i, smiles in enumerate(smiles_list):
                    col_idx = i % 3
                    with cols[col_idx]:
                        st.write(f"**Compound {i+1}**")
                        st.code(smiles, language=None)
                        
                        img = render_molecule_image(smiles, size=(250, 250))
                        if img:
                            st.image(img, caption=f"Structure {i+1}", use_column_width=True)
                        else:
                            st.error(f"Could not render structure {i+1}")
                        
                        # Quick properties
                        props = calculate_drug_likeness_properties(smiles)
                        if props:
                            st.write(f"MW: {props['Molecular_Weight']:.1f}")
                            st.write(f"LogP: {props['LogP']:.2f}")
                            st.write(f"Drug-like: {props['Drug_Like']}")
                
                # Predict activities for comparison
                if st.button("Compare Bioactivities"):
                    with st.spinner("Predicting activities..."):
                        comparison_results = predict_bioactivity(smiles_list, model)
                        
                        st.subheader("Activity Comparison")
                        
                        # Create comparison table
                        comparison_data = []
                        for i, (_, row) in enumerate(comparison_results.iterrows()):
                            if row['Valid']:
                                props = calculate_drug_likeness_properties(row['SMILES'])
                                comparison_data.append({
                                    'Compound': f"Compound {i+1}",
                                    'SMILES': row['SMILES'],
                                    'Activity_Level': row['Activity_Level'],
                                    'IC50_Estimate': row['IC50_Estimate'],
                                    'Predicted_Activity': row['Predicted_Activity'],
                                    'Drug_Like': props.get('Drug_Like', 'Unknown'),
                                    'Molecular_Weight': props.get('Molecular_Weight', 0),
                                    'LogP': props.get('LogP', 0)
                                })
                        
                        if comparison_data:
                            comparison_df = pd.DataFrame(comparison_data)
                            st.dataframe(comparison_df, use_container_width=True)
                            
                            # Comparison chart
                            fig = px.bar(
                                comparison_df,
                                x='Compound',
                                y='Predicted_Activity',
                                color='Activity_Level',
                                title="Activity Comparison",
                                labels={'Predicted_Activity': 'Predicted Activity (log scale)'}
                            )
                            st.plotly_chart(fig, use_container_width=True)

def add_llm_toxicity_analysis_to_ui(results_df):
    """Add LLM analysis section for toxicity results"""
    
    # Get API key
    api_key = get_groq_api_key()
    
    if not api_key:
        st.warning("Please provide your Groq AI API key to generate conversational analysis.")
        return
    
    # Initialize client
    client = GroqClient(api_key)
    
    st.header("🤖 AI Toxicologist Analysis")
    
    # Analysis options
    analysis_type = st.selectbox(
        "Choose analysis type:",
        ["Multiple Compounds Overview", "Risk Assessment Summary", "Single Compound Deep Dive", "Compare Selected Compounds"]
    )
    
    if st.button("Generate AI Analysis", type="primary"):
        with st.spinner("Generating expert analysis..."):
            
            if analysis_type == "Multiple Compounds Overview":
                analysis = generate_multiple_compounds_analysis(results_df, client)
                
            elif analysis_type == "Risk Assessment Summary":
                analysis = generate_risk_assessment(results_df, client)
                
            elif analysis_type == "Single Compound Deep Dive":
                # Let user select a compound
                valid_results = results_df[results_df['Valid'] == True]
                if len(valid_results) == 0:
                    st.error("No valid compounds to analyze")
                    return
                
                selected_idx = st.selectbox(
                    "Select compound for detailed analysis:",
                    range(len(valid_results)),
                    format_func=lambda x: f"{valid_results.iloc[x]['SMILES'][:20]}... ({valid_results.iloc[x]['Prediction']})"
                )
                
                selected_compound = valid_results.iloc[selected_idx].to_dict()
                analysis = generate_single_compound_analysis(selected_compound, client)
                
            elif analysis_type == "Compare Selected Compounds":
                # Let user select multiple compounds
                valid_results = results_df[results_df['Valid'] == True]
                if len(valid_results) < 2:
                    st.error("Need at least 2 valid compounds for comparison")
                    return
                
                selected_indices = st.multiselect(
                    "Select compounds to compare (2-5):",
                    range(len(valid_results)),
                    format_func=lambda x: f"{valid_results.iloc[x]['SMILES'][:20]}... ({valid_results.iloc[x]['Prediction']})",
                    max_selections=5
                )
                
                if len(selected_indices) < 2:
                    st.warning("Please select at least 2 compounds")
                    return
                
                selected_compounds = [valid_results.iloc[i].to_dict() for i in selected_indices]
                analysis = generate_comparative_analysis(selected_compounds, client)
        
        # Display analysis
        st.subheader("🧬 Expert Analysis")
        st.markdown(analysis)
        
        # Save analysis option
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        analysis_filename = f"toxicity_analysis_{timestamp}.txt"
        
        st.download_button(
            label="📄 Download Analysis Report",
            data=analysis,
            file_name=analysis_filename,
            mime="text/plain"
        )

# Helper functions for toxicity analysis (keeping your existing ones)
def format_single_compound_data(row: Dict) -> Dict:
    """Format single compound data for LLM input"""
    return {
        "smiles": row.get('SMILES', 'Unknown'),
        "compound_name": row.get('Compound_Name', 'Unknown compound'),
        "probability": round(row.get('Toxic_Probability', 0) * 100, 1),
        "prediction": row.get('Prediction', 'Unknown'),
        "risk_level": row.get('Risk_Level', 'Unknown')
    }

def format_multiple_compounds_data(results_df) -> Dict:
    """Format multiple compounds data for LLM input"""
    valid_results = results_df[results_df['Valid'] == True]
    
    # Summary statistics
    summary = {
        "total_compounds": len(results_df),
        "valid_predictions": len(valid_results),
        "toxic_count": len(valid_results[valid_results['Prediction'] == 'Toxic']),
        "high_risk_count": len(valid_results[valid_results['Risk_Level'] == 'High'])
    }
    
    # Detailed compound data (limit to top 10 for brevity)
    compounds_data = []
    for idx, row in valid_results.head(10).iterrows():
        compounds_data.append(
            f"Compound {idx}: {row['SMILES']} - "
            f"{row['Prediction']} ({row['Toxic_Probability']:.1%} probability, "
            f"{row['Risk_Level']} risk)"
        )
    
    summary["compounds_data"] = "\n".join(compounds_data)
    if len(valid_results) > 10:
        summary["compounds_data"] += f"\n... and {len(valid_results) - 10} more compounds"
    
    return summary

def format_risk_assessment_data(results_df) -> Dict:
    """Format risk assessment data for LLM input"""
    valid_results = results_df[results_df['Valid'] == True]
    
    # Risk distribution
    risk_counts = valid_results['Risk_Level'].value_counts()
    risk_distribution = "\n".join([f"- {level}: {count} compounds" 
                                 for level, count in risk_counts.items()])
    
    # High-risk compounds
    high_risk = valid_results[valid_results['Risk_Level'] == 'High']
    high_risk_compounds = []
    for idx, row in high_risk.iterrows():
        high_risk_compounds.append(
            f"- {row['SMILES']}: {row['Toxic_Probability']:.1%} probability"
        )
    
    return {
        "risk_distribution": risk_distribution,
        "high_risk_compounds": "\n".join(high_risk_compounds[:5])  # Top 5
    }

def format_comparative_data(compounds_list: List[Dict]) -> Dict:
    """Format comparative analysis data for LLM input"""
    comparison_data = []
    for i, compound in enumerate(compounds_list):
        comparison_data.append(
            f"Compound {i+1}: {compound['SMILES']}\n"
            f"  - Prediction: {compound['Prediction']}\n"
            f"  - Probability: {compound['Toxic_Probability']:.1%}\n"
            f"  - Risk Level: {compound['Risk_Level']}\n"
        )
    
    return {"comparison_data": "\n".join(comparison_data)}

def generate_single_compound_analysis(row: Dict, client: GroqClient) -> str:
    """Generate conversational analysis for a single compound"""
    data = format_single_compound_data(row)
    
    prompt = f"""
ANALYSIS REQUEST: Single Compound Toxicity Assessment

COMPOUND DATA:
- SMILES: {data['smiles']}
- Compound Name: {data['compound_name']}
- Toxicity Probability: {data['probability']}%
- Prediction: {data['prediction']}
- Risk Level: {data['risk_level']}

Please provide a comprehensive but accessible explanation covering:
1. What this compound is (if identifiable from SMILES)
2. The toxicity prediction and what it means
3. Health implications of the risk level
4. Recommendations for handling/exposure
5. Limitations of this computational assessment

Keep the explanation conversational but scientifically accurate.
"""
    
    return client.generate_response(TOXICOLOGY_EXPERT_SYSTEM_PROMPT, prompt)

def generate_multiple_compounds_analysis(results_df, client: GroqClient) -> str:
    """Generate conversational analysis for multiple compounds"""
    data = format_multiple_compounds_data(results_df)
    
    prompt = f"""
ANALYSIS REQUEST: Multiple Compounds Toxicity Assessment

DATASET SUMMARY:
- Total Compounds Analyzed: {data['total_compounds']}
- Valid Predictions: {data['valid_predictions']}
- Predicted Toxic: {data['toxic_count']}
- High Risk Compounds: {data['high_risk_count']}

DETAILED RESULTS:
{data['compounds_data']}

Please provide:
1. Overall assessment of the compound set
2. Key patterns or concerns identified
3. Prioritization of compounds by risk
4. General recommendations for the dataset
5. Notable findings or outliers

Focus on actionable insights and risk prioritization.
"""
    
    return client.generate_response(TOXICOLOGY_EXPERT_SYSTEM_PROMPT, prompt)

def generate_risk_assessment(results_df, client: GroqClient) -> str:
    """Generate risk assessment summary"""
    data = format_risk_assessment_data(results_df)
    
    prompt = f"""
ANALYSIS REQUEST: Risk Assessment Summary

RISK DISTRIBUTION:
{data['risk_distribution']}

HIGH-RISK COMPOUNDS:
{data['high_risk_compounds']}

Please provide:
1. Risk assessment overview
2. Immediate concerns and priorities
3. Risk management recommendations
4. Monitoring suggestions
5. Next steps for further evaluation

Emphasize practical risk management strategies.
"""
    
    return client.generate_response(TOXICOLOGY_EXPERT_SYSTEM_PROMPT, prompt)

def generate_comparative_analysis(compounds_list: List[Dict], client: GroqClient) -> str:
    """Generate comparative analysis between compounds"""
    data = format_comparative_data(compounds_list)
    
    prompt = f"""
ANALYSIS REQUEST: Comparative Compound Analysis

COMPOUNDS FOR COMPARISON:
{data['comparison_data']}

Please provide:
1. Side-by-side comparison of toxicity profiles
2. Relative risk ranking
3. Structural or chemical factors influencing toxicity
4. Recommendations for compound selection/prioritization
5. Suggested alternatives if high-risk compounds are identified

Focus on helping with decision-making between options.
"""
    
    return client.generate_response(TOXICOLOGY_EXPERT_SYSTEM_PROMPT, prompt)

# Main App
def main():
    st.set_page_config(
        page_title="Drug Discovery Platform", 
        page_icon="🧬", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Sidebar navigation
    with st.sidebar:
        st.title("🧬 Drug Discovery Platform")
        st.markdown("---")
        
        page = st.radio(
            "Select Analysis Type:",
            ["🧪 Toxicity Prediction", "🎯 Bioactivity & Screening"],
            index=0
        )
        
        st.markdown("---")
        st.markdown("""
        ### 📊 Platform Features
        
        **Toxicity Analysis:**
        - NR-AR endpoint prediction
        - Risk assessment
        - AI expert analysis
        
        **Bioactivity Screening:**
        - IC50 prediction (R² = 0.51)
        - Drug-likeness assessment  
        - Compound library screening
        - Structure visualization
        
        ### 💡 Model Performance
        - **Toxicity Model**: Classification
        - **Bioactivity Model**: Regression (MAPE = 0.68%)
        """)
    
    # Route to appropriate page
    if page == "🧪 Toxicity Prediction":
        toxicity_page()
    else:
        bioactivity_page()

if __name__ == "__main__":
    main()
