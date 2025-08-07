import streamlit as st
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors
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
            # Since your model predicts scaled values, we'll categorize based on the scaled prediction
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

# SYSTEM PROMPTS FOR IC50 ANALYSIS
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

IC50_PROMPT_TEMPLATES = {
    "single_compound": """
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

    "multiple_compounds": """
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

def format_single_ic50_data(row: Dict) -> Dict:
    """Format single compound data for LLM input"""
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
    """Format multiple compounds data for LLM input"""
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
        data = format_single_ic50_data(row)
        prompt = IC50_PROMPT_TEMPLATES["single_compound"].format(**data)
    else:
        data = format_multiple_ic50_data(results_df)
        prompt = IC50_PROMPT_TEMPLATES["multiple_compounds"].format(**data)
    
    return client.generate_response(IC50_EXPERT_SYSTEM_PROMPT, prompt)

def add_ai_analysis_section(results_df):
    """Add AI analysis section to the Streamlit UI"""
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
        analysis_type = st.selectbox("Choose analysis type:", analysis_options)
        
        if analysis_type == "Single Compound Deep Dive":
            valid_results = results_df[results_df['Valid'] == True]
            if len(valid_results) == 0:
                st.error("No valid compounds to analyze")
                return
            
            # Let user select a compound
            selected_idx = st.selectbox(
                "Select compound for detailed analysis:",
                range(len(valid_results)),
                format_func=lambda x: f"{valid_results.iloc[x]['SMILES'][:30]}... (IC50: {valid_results.iloc[x]['Predicted_IC50_Scaled']:.3f})"
            )
        
        if st.button("🧬 Generate AI Analysis", type="primary"):
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
                mime="text/plain"
            )
    
    except Exception as e:
        st.error(f"Error initializing AI analysis: {str(e)}")
        st.error("Please check your Groq API key and try again.")

def load_model_and_scaler_from_github(model_url: str, scaler_url: str = None):
    """Load model and scaler from GitHub repository"""
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

# Streamlit App
def main():
    st.set_page_config(page_title="IC50 Bioactivity Predictor", page_icon="🧬", layout="wide")
    
    st.title("🧬 IC50 Bioactivity Predictor")
    st.markdown("**Predict IC50 values for chemical compounds using Random Forest regression**")
    st.markdown("*Trained on multiple protein targets: DHFR, PDE5A, T-type calcium channel, ABL kinase, EGFR, Maltase-glucoamylase*")
    
    # Model loading section
    st.header("📊 Model Loading")
    
    model_source = st.radio(
        "Choose model source:",
        ["Load from local file", "Load from GitHub URL"]
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
            help="Make sure to use the 'raw' GitHub URL"
        )
        
        scaler_url = st.text_input(
            "Enter GitHub raw URL to your scaler file (optional):",
            placeholder="https://raw.githubusercontent.com/username/repo/main/scaler_X.joblib",
            help="If you saved the StandardScaler during training"
        )
        
        if model_url:
            try:
                with st.spinner("Loading model from GitHub..."):
                    model, scaler_from_github = load_model_and_scaler_from_github(model_url, scaler_url)
                    scaler_X = scaler_from_github if scaler_from_github else create_default_scaler()
                st.success("✅ Model loaded successfully from GitHub!")
            except Exception as e:
                st.error(f"❌ {str(e)}")
    
    if model is None:
        st.warning("Please load a model to continue with predictions.")
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
            placeholder="e.g., CC(=O)OC1=CC=CC=C1C(=O)O (aspirin)",
            help="Enter a valid SMILES notation"
        )
        if smiles_input:
            smiles_list = [smiles_input.strip()]
    
    elif input_method == "Multiple SMILES (text area)":
        smiles_text = st.text_area(
            "Enter SMILES (one per line):",
            placeholder="CC(=O)OC1=CC=CC=C1C(=O)O\nCN1C=NC2=C1C(=O)N(C(=O)N2C)C\nCC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
            height=200,
            help="Enter multiple SMILES, one per line"
        )
        if smiles_text:
            smiles_list = [s.strip() for s in smiles_text.split('\n') if s.strip()]
    
    else:  # CSV upload
        uploaded_file = st.file_uploader(
            "Upload CSV file with SMILES column:",
            type=['csv'],
            help="CSV file should contain a column named 'canonical_smiles', 'SMILES', or 'smiles'"
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
            add_ai_analysis_section(results_df)
    
    # Information sidebar
    with st.sidebar:
        st.header("ℹ️ Model Information")
        st.markdown("""
        **Algorithm:** Random Forest Regressor
        **Output:** Scaled IC50 values
        **Training Targets:**
        - Dihydrofolate reductase
        - Phosphodiesterase 5A
        - T-type calcium channel α-1H
        - Tyrosine-protein kinase ABL
        - EGFR erbB1
        - Maltase-glucoamylase
        
        **Molecular Descriptors Used:**
        - Molecular Weight (MolWt)
        - Topological Polar Surface Area (TPSA)
        - H-bond Donors & Acceptors
        - Rotatable Bonds
        - LogP (Lipophilicity)
        - Ring Counts
        - Heavy Atom Count
        - Fraction CSP3
        - Bertz Complexity Index
        - Heteroatom Count
        """)
        
        st.header("📊 Activity Levels")
        st.markdown("""
        **Activity Classification:**
        - 🟢 **Highly Active**: IC50 < -0.5
        - 🟡 **Active**: IC50 -0.5 to 0
        - 🟠 **Moderately Active**: IC50 0 to 0.5
        - 🔴 **Low Activity**: IC50 > 0.5
        
        *Lower scaled IC50 = Higher potency*
        """)
        
        st.header("🧪 Example SMILES")
        st.markdown("""
        - **Aspirin**: `CC(=O)OC1=CC=CC=C1C(=O)O`
        - **Caffeine**: `CN1C=NC2=C1C(=O)N(C(=O)N2C)C`
        - **Ibuprofen**: `CC(C)CC1=CC=C(C=C1)C(C)C(=O)O`
        - **Methotrexate**: `CN(CC1=CN=C2C(=N1)C(=NC(=N2)N)N)C3=CC=C(C=C3)C(=O)NC(CCC(=O)O)C(=O)O`
        """)

if __name__ == "__main__":
    main()
