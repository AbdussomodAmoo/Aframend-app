# mobile_app.py - Main entry point for mobile app
# Place this file in your project root directory

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.scrollview import ScrollView
import os
import pickle
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Fragments, rdMolDescriptors
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect, GetMACCSKeysFingerprint

class ChemPredictorApp(App):
    def __init__(self):
        super().__init__()
        self.toxicity_models = {}
        self.bioactivity_model = None
        self.load_models()
        
        # Endpoint names
        self.endpoint_names = {
            'NR-AR': 'Androgen Receptor',
            'NR-AR-LBD': 'Androgen Receptor Binding', 
            'NR-AhR': 'Aryl Hydrocarbon Receptor',
            'NR-Aromatase': 'Aromatase Inhibition',
            'NR-ER': 'Estrogen Receptor',
            'NR-ER-LBD': 'Estrogen Receptor Binding',
            'NR-PPAR-gamma': 'PPAR-gamma Activation',
            'SR-ARE': 'Antioxidant Response',
            'SR-ATAD5': 'DNA Damage Response', 
            'SR-HSE': 'Heat Shock Response',
            'SR-MMP': 'Mitochondrial Toxicity',
            'SR-p53': 'p53 Tumor Suppressor'
        }

    def load_models(self):
        """Load all available model files from models/ directory"""
        tox_endpoints = [
            'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
            'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'
        ]
        
        models_loaded = 0
        model_dir = 'models'
        
        # Load toxicity models from models/ directory
        for endpoint in tox_endpoints:
            try:
                model_path = os.path.join(model_dir, f'{endpoint}.pkl')
                with open(model_path, 'rb') as f:
                    self.toxicity_models[endpoint] = pickle.load(f)
                models_loaded += 1
            except FileNotFoundError:
                print(f"Model {endpoint}.pkl not found in {model_dir}/")
            except Exception as e:
                print(f"Error loading {endpoint}: {e}")
        
        # Load bioactivity model from models/ directory
        try:
            bio_model_path = os.path.join(model_dir, 'bioactivity_model.joblib')
            with open(bio_model_path, 'rb') as f:
                import joblib
                self.bioactivity_model = joblib.load(f)
            print("Bioactivity model loaded")
        except:
            print("Bioactivity model not found in models/")
        
        print(f"Loaded {models_loaded} toxicity models")

    def extract_toxicity_features(self, mol):
        """Extract molecular features for toxicity prediction"""
        if mol is None:
            return None
        
        features = {}
        try:
            # Basic properties (matching your original code)
            features['mol_weight'] = Descriptors.MolWt(mol)
            features['mol_logp'] = Descriptors.MolLogP(mol)
            features['tpsa'] = Descriptors.TPSA(mol)
            features['labute_asa'] = Descriptors.LabuteASA(mol)
            features['num_hbd'] = Descriptors.NumHDonors(mol)
            features['num_hba'] = Descriptors.NumHAcceptors(mol)
            features['max_partial_charge'] = Descriptors.MaxPartialCharge(mol)
            features['min_partial_charge'] = Descriptors.MinPartialCharge(mol)
            features['max_abs_partial_charge'] = Descriptors.MaxAbsPartialCharge(mol)
            features['num_rotatable_bonds'] = Descriptors.NumRotatableBonds(mol)
            features['heavy_atom_count'] = Descriptors.HeavyAtomCount(mol)
            features['num_aromatic_rings'] = Descriptors.NumAromaticRings(mol)
            features['num_saturated_rings'] = Descriptors.NumSaturatedRings(mol)
            features['num_aliphatic_rings'] = Descriptors.NumAliphaticRings(mol)
            features['ring_count'] = Descriptors.RingCount(mol)
            features['fraction_csp3'] = Descriptors.FractionCSP3(mol)
            features['num_heteroatoms'] = Descriptors.NumHeteroatoms(mol)
            features['bertz_ct'] = Descriptors.BertzCT(mol)
            features['hall_kier_alpha'] = Descriptors.HallKierAlpha(mol)
            features['kappa1'] = Descriptors.Kappa1(mol)
            features['kappa2'] = Descriptors.Kappa2(mol)
            features['kappa3'] = Descriptors.Kappa3(mol)
            features['qed'] = Descriptors.qed(mol)
            
            # VSA descriptors
            features['vsa_estate4'] = Descriptors.VSA_EState4(mol)
            features['vsa_estate9'] = Descriptors.VSA_EState9(mol)
            features['slogp_vsa4'] = Descriptors.SlogP_VSA4(mol)
            features['slogp_vsa6'] = Descriptors.SlogP_VSA6(mol)
            features['smr_vsa5'] = Descriptors.SMR_VSA5(mol)
            features['smr_vsa7'] = Descriptors.SMR_VSA7(mol)
            features['balaban_j'] = Descriptors.BalabanJ(mol)
            
            # Fragment counts
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
            
            # Morgan fingerprints
            morgan_fp = GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
            for i, bit in enumerate(morgan_fp):
                features[f'morgan_{i}'] = int(bit)
            
            # MACCS keys
            maccs_fp = GetMACCSKeysFingerprint(mol)
            for i, bit in enumerate(maccs_fp):
                features[f'maccs_{i}'] = int(bit)
                
        except Exception as e:
            print(f"Error extracting features: {e}")
            return None
        
        return np.array(list(features.values()))

    def extract_bioactivity_features(self, mol):
        """Extract features for IC50 prediction"""
        if mol is None:
            return None
        
        try:
            features = {
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
            
            feature_names = ['MolWt', 'TPSA', 'NumHDonors', 'NumHAcceptors', 'NumRotatableBonds',
                           'LogP', 'NumRings', 'HeavyAtomCount', 'FractionCSP3', 'NumAromaticRings',
                           'NumAliphaticRings', 'NumValenceElectrons', 'BertzCT', 'NumHeteroatoms']
            
            return np.array([features[name] for name in feature_names]).reshape(1, -1)
            
        except Exception as e:
            print(f"Error extracting bioactivity features: {e}")
            return None

    def predict_compound(self, smiles_string):
        """Make predictions for a compound"""
        mol = Chem.MolFromSmiles(smiles_string)
        if mol is None:
            return "❌ Invalid SMILES string"
        
        results = []
        results.append(f"🧪 Analysis for: {smiles_string}")
        results.append(f"📊 Molecular Weight: {Descriptors.MolWt(mol):.1f}")
        results.append(f"🔬 LogP: {Descriptors.MolLogP(mol):.2f}")
        results.append("")
        
        # Toxicity predictions
        tox_features = self.extract_toxicity_features(mol)
        if tox_features is not None and self.toxicity_models:
            results.append("🧪 TOXICITY PREDICTIONS:")
            features_reshaped = tox_features.reshape(1, -1)
            
            toxic_count = 0
            total_models = len(self.toxicity_models)
            
            for endpoint, model in self.toxicity_models.items():
                try:
                    pred_proba = model.predict_proba(features_reshaped)[0]
                    toxic_prob = pred_proba[1] if len(pred_proba) > 1 else pred_proba[0]
                    
                    prediction = "Toxic" if toxic_prob > 0.5 else "Safe"
                    risk = "High" if toxic_prob > 0.7 else "Medium" if toxic_prob > 0.3 else "Low"
                    
                    if toxic_prob > 0.5:
                        toxic_count += 1
                    
                    results.append(f"  {endpoint}: {prediction} ({toxic_prob:.3f}) - {risk} risk")
                    
                except Exception as e:
                    results.append(f"  {endpoint}: Error - {str(e)}")
            
            results.append(f"\n📊 Overall: {toxic_count}/{total_models} endpoints predicted toxic")
        
        # Bioactivity prediction
        if self.bioactivity_model:
            bio_features = self.extract_bioactivity_features(mol)
            if bio_features is not None:
                try:
                    ic50_scaled = self.bioactivity_model.predict(bio_features)[0]
                    
                    if ic50_scaled < -0.5:
                        activity = "Highly Active"
                    elif ic50_scaled < 0:
                        activity = "Active"
                    elif ic50_scaled < 0.5:
                        activity = "Moderately Active"
                    else:
                        activity = "Low Activity"
                    
                    results.append(f"\n🧬 BIOACTIVITY PREDICTION:")
                    results.append(f"  IC50 (scaled): {ic50_scaled:.3f}")
                    results.append(f"  Activity Level: {activity}")
                    
                except Exception as e:
                    results.append(f"\n🧬 Bioactivity: Error - {str(e)}")
        
        return "\n".join(results)

    def build(self):
        """Build the mobile app interface"""
        layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
        
        # Title
        title = Label(text='🧪 ChemPredictor Mobile', size_hint_y=0.1, 
                     font_size='20sp', halign='center')
        layout.add_widget(title)
        
        # SMILES input
        self.smiles_input = TextInput(hint_text='Enter SMILES string (e.g., CCO for ethanol)',
                                     size_hint_y=0.1, multiline=False)
        layout.add_widget(self.smiles_input)
        
        # Predict button
        predict_btn = Button(text='🔮 Predict Toxicity & Bioactivity', 
                           size_hint_y=0.1, on_press=self.on_predict)
        layout.add_widget(predict_btn)
        
        # Results area (scrollable)
        scroll = ScrollView(size_hint_y=0.7)
        self.results_label = Label(text='Enter a SMILES string and tap Predict!',
                                  text_size=(None, None), halign='left', valign='top')
        scroll.add_widget(self.results_label)
        layout.add_widget(scroll)
        
        return layout

    def on_predict(self, instance):
        """Handle predict button press"""
        smiles = self.smiles_input.text.strip()
        if not smiles:
            self.results_label.text = "Please enter a SMILES string"
            return
        
        # Make prediction
        results = self.predict_compound(smiles)
        
        # Update results display
        self.results_label.text = results
        self.results_label.text_size = (400, None)  # Enable text wrapping

if __name__ == '__main__':
    ChemPredictorApp().run()
