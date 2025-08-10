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
from groq import Groq
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

# ===== MODIFIED: NEW MODEL LOADING FUNCTION =====
def load_models():
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
        st.success(f"✅ Successfully loaded {len(models)} models!")
    
    return models

# ===== MODIFIED: NEW MULTI-ENDPOINT PREDICTION FUNCTION =====
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

# ===== NEW: ENDPOINT SELECTION FUNCTION =====
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

# ===== NEW: MULTI-ENDPOINT RESULTS DISPLAY =====
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

# ===== NEW: COMPREHENSIVE SUMMARY FUNCTION =====
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

# ===== KEEP ALL YOUR EXISTING LLM FUNCTIONS UNCHANGED =====
def get_groq_api_key(): 
    """Securely retrieve Groq API key"""
    # Option 1: Environment variable
    api_key = os.getenv('GROQ_API_KEY')
    
    # Option 2: Streamlit secrets
    if not api_key:
        try:
            api_key = st.secrets["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"]   #["groq"]["api_key"]  
            st.success("✅ API key loaded from secrets!")
        except Exception as e:
            st.warning(f"Could not load from secrets: {e}")
            pass
    
    # Option 3: User input
    if not api_key:
        api_key = st.sidebar.text_input(
            "Enter Groq API Key:",  # Update label
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
                model=model,  # Available: llama3-8b-8192, llama3-70b-4096, mixtral-8x7b-32768
                max_tokens=1500,
                temperature=0.7
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            return f"Error generating response: {str(e)}"     

# ===== KEEP ALL YOUR EXISTING LLM SYSTEM PROMPTS AND TEMPLATES =====
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

# ===== KEEP ALL YOUR EXISTING PROMPT TEMPLATES BUT ADD THIS NEW ONE =====
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

# ===== KEEP ALL YOUR EXISTING LLM FORMATTING FUNCTIONS =====
def format_single_compound_data(row: Dict) -> Dict:
    """Format single compound data for LLM input"""
    return {
        "smiles": row.get('SMILES', 'Unknown'),
        "compound_name": row.get('Compound_Name', 'Unknown compound'),
        "probability": round(row.get('Toxic_Probability', 0) * 100, 1),
        "prediction": row.get('Prediction', 'Unknown'),
        "risk_level": row.get('Risk_Level', 'Unknown')
    }

# ===== MODIFIED: UPDATE LLM FORMATTING FOR MULTI-ENDPOINT =====
def format_multiple_compounds_data(results_df) -> Dict:
    """Format multiple compounds data for LLM input - updated for multi-endpoint"""
    valid_results = results_df[results_df['Valid'] == True]
    
    # Count toxic predictions across all endpoints
    total_toxic = 0
    total_high_risk = 0
    
    # Count endpoint-specific results
    for col in results_df.columns:
        if col.endswith('_Prediction'):
            total_toxic += len(valid_results[valid_results[col] == 'Toxic'])
        elif col.endswith('_Risk_Level'):
            total_high_risk += len(valid_results[valid_results[col] == 'High'])
    
    # Summary statistics
    summary = {
        "total_compounds": len(results_df),
        "valid_predictions": len(valid_results),
        "toxic_count": total_toxic,
        "high_risk_count": total_high_risk
    }
    
    # Detailed compound data (limit to top 10 for brevity)
    compounds_data = []
    for idx, row in valid_results.head(10).iterrows():
        # Find the first probability column for basic info
        prob_cols = [col for col in row.index if col.endswith('_Probability') and pd.notna(row[col])]
        if prob_cols:
            first_prob = row[prob_cols[0]]
            compounds_data.append(
                f"Compound {idx}: {row['SMILES']} - "
                f"Multi-endpoint analysis ({first_prob:.1%} example probability)"
            )
    
    summary["compounds_data"] = "\n".join(compounds_data)
    if len(valid_results) > 10:
        summary["compounds_data"] += f"\n... and {len(valid_results) - 10} more compounds"
    
    return summary

# ===== KEEP ALL OTHER EXISTING LLM FUNCTIONS UNCHANGED =====
def format_risk_assessment_data(results_df) -> Dict:
    """Format risk assessment data for LLM input"""
    valid_results = results_df[results_df['Valid'] == True]
    
    # Risk distribution across all endpoints
    all_risks = []
    for col in results_df.columns:
        if col.endswith('_Risk_Level'):
            all_risks.extend(valid_results[col].dropna().tolist())
    
    if all_risks:
        risk_counts = pd.Series(all_risks).value_counts()
        risk_distribution = "\n".join([f"- {level}: {count} predictions" 
                                     for level, count in risk_counts.items()])
    else:
        risk_distribution = "No valid risk assessments available"
    
    # High-risk compounds
    high_risk_compounds = []
    for idx, row in valid_results.iterrows():
        high_risk_endpoints = []
        for col in row.index:
            if col.endswith('_Risk_Level') and row[col] == 'High':
                endpoint = col.replace('_Risk_Level', '')
                high_risk_endpoints.append(endpoint)
        
        if high_risk_endpoints:
            high_risk_compounds.append(
                f"- {row['SMILES']}: High risk in {', '.join(high_risk_endpoints[:3])}"
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
            f"  - Prediction: {compound.get('Prediction', 'N/A')}\n"
            f"  - Probability: {compound.get('Toxic_Probability', 0):.1%}\n"
            f"  - Risk Level: {compound.get('Risk_Level', 'N/A')}\n"
        )
    
    return {"comparison_data": "\n".join(comparison_data)}

# ===== KEEP ALL EXISTING LLM GENERATION FUNCTIONS =====
def generate_single_compound_analysis(row: Dict, client: GroqClient) -> str:
    """Generate conversational analysis for a single compound"""
    data = format_single_compound_data(row)
    prompt = PROMPT_TEMPLATES["single_compound"].format(**data)
    
    return client.generate_response(TOXICOLOGY_EXPERT_SYSTEM_PROMPT, prompt)

def generate_multiple_compounds_analysis(results_df, client: GroqClient) -> str:
    """Generate conversational analysis for multiple compounds"""
    data = format_multiple_compounds_data(results_df)
    prompt = PROMPT_TEMPLATES["multiple_compounds"].format(**data)
    
    return client.generate_response(TOXICOLOGY_EXPERT_SYSTEM_PROMPT, prompt)

def generate_risk_assessment(results_df, client: GroqClient) -> str:
    """Generate risk assessment summary"""
    data = format_risk_assessment_data(results_df)
    prompt = PROMPT_TEMPLATES["risk_assessment"].format(**data)
    
    return client.generate_response(TOXICOLOGY_EXPERT_SYSTEM_PROMPT, prompt)

def generate_comparative_analysis(compounds_list: List[Dict], client: GroqClient) -> str:
    """Generate comparative analysis between compounds"""
    data = format_comparative_data(compounds_list)
    prompt = PROMPT_TEMPLATES["comparative_analysis"].format(**data)
    
    return client.generate_response(TOXICOLOGY_EXPERT_SYSTEM_PROMPT, prompt)

# ===== MODIFIED: UPDATE LLM UI INTEGRATION FOR MULTI-ENDPOINT =====
def add_llm_analysis_to_ui(results_df, selected_endpoints):
    """Add LLM analysis section to your Streamlit UI - updated for multi-endpoint"""
    
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
        ["Multi-Endpoint Overview", "Risk Assessment Summary", "Single Compound Deep Dive", "Compare Selected Compounds"]
    )
    
    if st.button("Generate AI Analysis", type="primary"):
        with st.spinner("Generating expert analysis..."):
            
            if analysis_type == "Multi-Endpoint Overview":
                # Create multi-endpoint summary
                valid_results = results_df[results_df['Valid'] == True]
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
                    format_func=lambda x: f"{valid_results.iloc[x]['SMILES'][:20]}..."
                )
                
                selected_compound = valid_results.iloc[selected_idx].to_dict()
                
                # For multi-endpoint, we'll analyze the first available endpoint
                first_endpoint = None
                for endpoint in selected_endpoints:
                    if f'{endpoint}_Prediction' in selected_compound:
                        first_endpoint = endpoint
                        break
                
                if first_endpoint:
                    # Reformat for single compound analysis
                    single_data = {
                        'SMILES': selected_compound['SMILES'],
                        'Compound_Name': 'Unknown compound',
                        'Toxic_Probability': selected_compound.get(f'{first_endpoint}_Probability', 0),
                        'Prediction': selected_compound.get(f'{first_endpoint}_Prediction', 'Unknown'),
                        'Risk_Level': selected_compound.get(f'{first_endpoint}_Risk_Level', 'Unknown')
                    }
                    analysis = generate_single_compound_analysis(single_data, client)
                else:
                    st.error("No valid endpoint data for selected compound")
                    return
                
            elif analysis_type == "Compare Selected Compounds":
                # Let user select multiple compounds
                valid_results = results_df[results_df['Valid'] == True]
                if len(valid_results) < 2:
                    st.error("Need at least 2 valid compounds for comparison")
                    return
                
                selected_indices = st.multiselect(
                    "Select compounds to compare (2-5):",
                    range(len(valid_results)),
                    format_func=lambda x: f"{valid_results.iloc[x]['SMILES'][:20]}...",
                    max_selections=5
                )
                
                if len(selected_indices) < 2:
                    st.warning("Please select at least 2 compounds")
                    return
                
                # Prepare comparison data
                selected_compounds = []
                for i in selected_indices:
                    compound = valid_results.iloc[i].to_dict()
                    
                    # Find first available endpoint for comparison
                    first_endpoint = None
                    for endpoint in selected_endpoints:
                        if f'{endpoint}_Prediction' in compound:
                            first_endpoint = endpoint
                            break
                    
                    if first_endpoint:
                        comp_data = {
                            'SMILES': compound['SMILES'],
                            'Toxic_Probability': compound.get(f'{first_endpoint}_Probability', 0),
                            'Prediction': compound.get(f'{first_endpoint}_Prediction', 'Unknown'),
                            'Risk_Level': compound.get(f'{first_endpoint}_Risk_Level', 'Unknown')
                        }
                        selected_compounds.append(comp_data)
                
                if selected_compounds:
                    analysis = generate_comparative_analysis(selected_compounds, client)
                else:
                    st.error("No valid data for comparison")
                    return
        
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
            mime="text/plain"
        )

# ===== COMPLETELY REWRITTEN: MAIN FUNCTION =====
def main():
    st.set_page_config(page_title="Multi-Endpoint Toxicity Predictor", page_icon="🧪", layout="wide")
    
    st.title("🧪 Multi-Endpoint Toxicity Predictor")
    st.markdown("**Predict toxicity across multiple biological endpoints using TOX21 models**")
    
    # ===== MODIFIED: Load all models instead of single model =====
    models = load_models()
    
    if not models:
        st.error("❌ No models could be loaded. Please ensure model files (.pkl) are available.")
        st.stop()
    
    # ===== KEEP EXISTING: Input methods section (UNCHANGED) =====
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
    
    # ===== NEW: Add endpoint selection =====
    selected_endpoints = add_endpoint_selection(models)
    
    if not selected_endpoints:
        st.warning("Please select at least one endpoint to proceed.")
        return
    
    # ===== MODIFIED: Make predictions only if we have SMILES and endpoints =====
    if smiles_list and selected_endpoints:
        st.header("🔮 Multi-Endpoint Predictions")
        
        with st.spinner("Making predictions across selected endpoints..."):
            results_df = predict_toxicity_multi_endpoint(smiles_list, models, selected_endpoints)
        
        # ===== NEW: Display multi-endpoint results =====
        display_multi_endpoint_results(results_df, selected_endpoints)
        
        # ===== NEW: Comprehensive summary =====
        create_comprehensive_summary(results_df, selected_endpoints)
        
        # ===== MODIFIED: LLM analysis with multi-endpoint support =====
        add_llm_analysis_to_ui(results_df, selected_endpoints)
    
    # ===== MODIFIED: Information sidebar =====
    with st.sidebar:
        st.header("ℹ️ Information")
        st.markdown(f"""
        **Model Details:**
        - Available Models: {len(models)}
        - Endpoints: {', '.join(models.keys())}
        - Algorithm: Random Forest Classifier
        - Features: Molecular descriptors + fingerprints
        
        **Risk Levels:**
        - 🔴 **High**: Probability > 0.7
        - 🟡 **Medium**: Probability 0.3-0.7
        - 🟢 **Low**: Probability < 0.3
        
        **Endpoint Categories:**
        - **NR-**: Nuclear Receptor pathways
        - **SR-**: Stress Response pathways
        
        **SMILES Examples:**
        - Ethanol: `CCO`
        - Benzene: `C1=CC=CC=C1`
        - Aspirin: `CC(=O)OC1=CC=CC=C1C(=O)O`
        """)

if __name__ == "__main__":
    main()
