import streamlit as st
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Fragments
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
warnings.filterwarnings('ignore')

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

# SECURE API KEY MANAGEMENT
def get_qualcomm_api_key():
    """Securely retrieve Qualcomm AI API key"""
    # Option 1: Environment variable (recommended for production)
    api_key = os.getenv('QUALCOMM_AI_API_KEY')
    
    # Option 2: Streamlit secrets (recommended for Streamlit deployment)
    if not api_key and hasattr(st, 'secrets'):
        try:
            api_key = st.secrets["qualcomm"]["api_key"]
        except:
            pass
    
    # Option 3: User input (for development/testing)
    if not api_key:
        api_key = st.sidebar.text_input(
            "Enter Qualcomm AI API Key:",
            type="password",
            help="Your API key will not be stored"
        )
    
    return api_key         


class QualcommAIClient:
    def __init__(self, api_key: str, base_url: str = None):
        self.api_key = api_key
        # Update this URL based on Qualcomm's actual endpoint
        self.base_url = base_url or "https://api.qualcomm-ai.com/v1"  # Placeholder
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def generate_response(self, system_prompt: str, user_prompt: str, model: str = "qualcomm-llm") -> str:
        """Generate conversational response using Qualcomm AI"""
        try:
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "max_tokens": 1500,
                "temperature": 0.7
            }
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                return f"API Error: {response.status_code} - {response.text}"
                
        except Exception as e:
            return f"Error generating response: {str(e)}"


# SYSTEM PROMPTS AND TEMPLATES


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

COMMUNICATION STYLE:
- Professional yet approachable
- Use analogies when helpful
- Acknowledge uncertainties appropriately
- Provide actionable insights when possible
- Always include appropriate disclaimers about limitations
"""

PROMPT_TEMPLATES = {
    "single_compound": """
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

    "multiple_compounds": """
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

    "risk_assessment": """
ANALYSIS REQUEST: Risk Assessment Summary

RISK DISTRIBUTION:
{risk_distribution}

HIGH-RISK COMPOUNDS:
{high_risk_compounds}

Please provide:
1. Risk assessment overview
2. Immediate concerns and priorities
3. Risk management recommendations
4. Monitoring suggestions
5. Next steps for further evaluation

Emphasize practical risk management strategies.
""",

    "comparative_analysis": """
ANALYSIS REQUEST: Comparative Compound Analysis

COMPOUNDS FOR COMPARISON:
{comparison_data}

Please provide:
1. Side-by-side comparison of toxicity profiles
2. Relative risk ranking
3. Structural or chemical factors influencing toxicity
4. Recommendations for compound selection/prioritization
5. Suggested alternatives if high-risk compounds are identified

Focus on helping with decision-making between options.
"""
}

# DATA FORMATTING FUNCTIONS
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

# MAIN LLM INTEGRATION FUNCTIONS
def generate_single_compound_analysis(row: Dict, client: QualcommAIClient) -> str:
    """Generate conversational analysis for a single compound"""
    data = format_single_compound_data(row)
    prompt = PROMPT_TEMPLATES["single_compound"].format(**data)
    
    return client.generate_response(TOXICOLOGY_EXPERT_SYSTEM_PROMPT, prompt)

def generate_multiple_compounds_analysis(results_df, client: QualcommAIClient) -> str:
    """Generate conversational analysis for multiple compounds"""
    data = format_multiple_compounds_data(results_df)
    prompt = PROMPT_TEMPLATES["multiple_compounds"].format(**data)
    
    return client.generate_response(TOXICOLOGY_EXPERT_SYSTEM_PROMPT, prompt)

def generate_risk_assessment(results_df, client: QualcommAIClient) -> str:
    """Generate risk assessment summary"""
    data = format_risk_assessment_data(results_df)
    prompt = PROMPT_TEMPLATES["risk_assessment"].format(**data)
    
    return client.generate_response(TOXICOLOGY_EXPERT_SYSTEM_PROMPT, prompt)

def generate_comparative_analysis(compounds_list: List[Dict], client: QualcommAIClient) -> str:
    """Generate comparative analysis between compounds"""
    data = format_comparative_data(compounds_list)
    prompt = PROMPT_TEMPLATES["comparative_analysis"].format(**data)
    
    return client.generate_response(TOXICOLOGY_EXPERT_SYSTEM_PROMPT, prompt)

# Streamlit UI Integration
def add_llm_analysis_to_ui(results_df):
    """Add LLM analysis section to your Streamlit UI"""
    
    # Get API key
    api_key = get_qualcomm_api_key()
    
    if not api_key:
        st.warning("Please provide your Qualcomm AI API key to generate conversational analysis.")
        return
    
    # Initialize client
    client = QualcommAIClient(api_key)
    
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



# Streamlit App
def main():
    st.set_page_config(page_title="Toxicity Predictor", page_icon="🧪", layout="wide")
    
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
                st.bar_chart(pred_counts)
                st.caption("Prediction Distribution")
            
            with col2:
                # Risk level distribution
                risk_counts = valid_results['Risk_Level'].value_counts()
                st.bar_chart(risk_counts)
                st.caption("Risk Level Distribution")
            if len(valid_results) > 0:
            add_llm_analysis_to_ui(results_df)
    # Information sidebar
    with st.sidebar:
        st.header("ℹ️ Information")
        st.markdown("""
        **Model Details:**
        - Endpoint: NR-AR (Androgen Receptor Disruption)
        - Algorithm: Random Forest Classifier
        - Features: Molecular descriptors + fingerprints
        
        **Risk Levels:**
        - 🔴 **High**: Probability > 0.7
        - 🟡 **Medium**: Probability 0.3-0.7
        - 🟢 **Low**: Probability < 0.3
        
        **SMILES Examples:**
        - Ethanol: `CCO`
        - Benzene: `C1=CC=CC=C1`
        - Aspirin: `CC(=O)OC1=CC=CC=C1C(=O)O`
        """)

if __name__ == "__main__":
    main()
