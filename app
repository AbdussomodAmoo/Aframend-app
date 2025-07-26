import streamlit as st
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Fragments
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect, GetMACCSKeysFingerprint
from typing import List, Dict, Optional, Union
import joblib
import warnings
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
