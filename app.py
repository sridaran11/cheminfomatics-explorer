import io
import requests
import streamlit as st
import streamlit.components.v1 as components
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, Descriptors, rdMolDescriptors, Lipinski
from rdkit.Chem.rdchem import BondType

def _png_bytes(mol):
    from rdkit.Chem import Draw
    image = Draw.MolToImage(mol, size=(1000, 700))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()

st.set_page_config(page_title="Cheminformatics Explorer", page_icon="🧪", layout="wide")

st.title("🧪 Cheminformatics Explorer")
st.caption("Explore 2D and 3D molecular structures, formulae, bonds, SMILES, and stereochemistry.")
st.info("Educational tool: generated 3D conformations are estimates, not experimental structures or a full conformational analysis.")

with st.sidebar:
    st.header("Molecule input")
    input_mode = st.radio("Input type", ["Molecule name", "SMILES"])
    default_value = "aspirin" if input_mode == "Molecule name" else "CC(=O)OC1=CC=CC=C1C(=O)O"
    query = st.text_input(
        "Enter molecule name" if input_mode == "Molecule name" else "Enter SMILES",
        value=default_value,
        help="Examples: aspirin, caffeine, ethanol, benzene" if input_mode == "Molecule name" else "Examples: CCO (ethanol), c1ccccc1 (benzene)"
    )
    include_h = st.checkbox("Show explicit hydrogens in 3D", value=True)
    make_3d = st.button("Analyze molecule", type="primary", use_container_width=True)
    st.markdown("---")
    st.caption("Molecule-name lookup uses the PubChem public service and requires internet access.")

if "last_result" not in st.session_state:
    st.session_state.last_result = None

def fetch_smiles_from_pubchem(name: str) -> str:
    url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{}/property/IsomericSMILES,CanonicalSMILES/JSON".format(requests.utils.quote(name.strip(), safe=""))
    response = requests.get(url, timeout=15)
    if response.status_code != 200:
        raise ValueError("PubChem could not find that name. Try a more specific name or enter SMILES directly.")
    data = response.json()
    props = data.get("PropertyTable", {}).get("Properties", [])
    if not props:
        raise ValueError("No matching compound was returned by PubChem.")
    return props[0].get("SMILES") or props[0].get("ConnectivitySMILES") or props[0].get("CanonicalSMILES")

def build_molecule(text: str, mode: str):
    if not text.strip():
        raise ValueError("Enter a molecule name or SMILES string.")
    resolved_smiles = fetch_smiles_from_pubchem(text) if mode == "Molecule name" else text.strip()
    mol = Chem.MolFromSmiles(resolved_smiles)
    if mol is None:
        raise ValueError("This input could not be parsed as a valid molecule. Check the name or SMILES.")
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    return mol, resolved_smiles

def count_bonds(mol):
    single = double = triple = aromatic = other = 0
    sigma = pi = 0
    # Aromatic systems are kekulized on a copy to estimate sigma/pi contributions.
    kekule = Chem.Mol(mol)
    try:
        Chem.Kekulize(kekule, clearAromaticFlags=True)
        for bond in kekule.GetBonds():
            bt = bond.GetBondType()
            if bt == BondType.SINGLE:
                single += 1
                sigma += 1
            elif bt == BondType.DOUBLE:
                double += 1
                sigma += 1
                pi += 1
            elif bt == BondType.TRIPLE:
                triple += 1
                sigma += 1
                pi += 2
            else:
                other += 1
                # Aromatic bonds in a successful Kekulé representation should be resolved.
                if bt == BondType.AROMATIC:
                    aromatic += 1
                    sigma += 1
                    pi += 0.5
    except Exception:
        for bond in mol.GetBonds():
            bt = bond.GetBondType()
            if bt == BondType.SINGLE:
                single += 1; sigma += 1
            elif bt == BondType.DOUBLE:
                double += 1; sigma += 1; pi += 1
            elif bt == BondType.TRIPLE:
                triple += 1; sigma += 1; pi += 2
            elif bt == BondType.AROMATIC:
                aromatic += 1; sigma += 1
    return {"Single": single, "Double": double, "Triple": triple, "Aromatic (unresolved)": aromatic,
            "Other": other, "Sigma bonds (estimated)": sigma, "Pi bonds (estimated)": pi}

def make_3d_block(mol, explicit_h=True):
    m3d = Chem.AddHs(mol) if explicit_h else Chem.Mol(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 2026
    status = AllChem.EmbedMolecule(m3d, params)
    if status != 0:
        raise ValueError("RDKit could not generate a 3D conformer for this molecule.")
    try:
        if AllChem.MMFFHasAllMoleculeParams(m3d):
            AllChem.MMFFOptimizeMolecule(m3d, maxIters=250)
        elif AllChem.UFFHasAllMoleculeParams(m3d):
            AllChem.UFFOptimizeMolecule(m3d, maxIters=250)
    except Exception:
        pass
    return Chem.MolToMolBlock(m3d), m3d

if make_3d:
    try:
        with st.spinner("Resolving molecule and calculating properties..."):
            mol, resolved_smiles = build_molecule(query, input_mode)
            mol_2d = Chem.Mol(mol)
            AllChem.Compute2DCoords(mol_2d)
            molblock_3d, mol_3d = make_3d_block(mol, include_h)
            st.session_state.last_result = {
                "mol": mol, "mol_2d": mol_2d, "resolved_smiles": resolved_smiles,
                "molblock_3d": molblock_3d, "mol_3d": mol_3d, "query": query, "mode": input_mode
            }
    except requests.RequestException:
        st.error("Could not reach PubChem. Check your internet connection, or choose SMILES input.")
    except Exception as exc:
        st.error(str(exc))

result = st.session_state.last_result

if result:
    mol = result["mol"]
    st.subheader(f"Results for: {result['query']}")
    st.caption(f"Input type: {result['mode']} · Structure parsed successfully")

    formula = rdMolDescriptors.CalcMolFormula(mol)
    mw = Descriptors.MolWt(mol)
    exact_mw = Descriptors.ExactMolWt(mol)
    atoms = mol.GetNumAtoms()
    heavy_atoms = mol.GetNumHeavyAtoms()
    rings = rdMolDescriptors.CalcNumRings(mol)
    rotatable = Lipinski.NumRotatableBonds(mol)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Molecular formula", formula)
    c2.metric("Molecular weight", f"{mw:.3f} g/mol")
    c3.metric("Heavy atoms", heavy_atoms)
    c4.metric("Rings", rings)
    c5.metric("Rotatable bonds", rotatable)

    tab2d, tab3d, tab_bonds, tab_stereo, tab_export = st.tabs(
        ["🧬 2D Structure", "🧊 3D Structure", "🔗 Bond Analysis", "🧭 Configuration", "⬇️ Export"]
    )

    with tab2d:
        left, right = st.columns([1.1, 1])
        with left:
            st.image(Draw.MolToImage(result["mol_2d"], size=(650, 450)), caption="2D molecular depiction", use_container_width=True)
        with right:
            st.markdown("**SMILES**")
            st.code(Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True), language="text")
            st.markdown("**Connectivity-only SMILES**")
            st.code(Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False), language="text")
            st.markdown("**Atom summary**")
            atom_rows = []
            for atom in mol.GetAtoms():
                atom_rows.append({
                    "Atom index": atom.GetIdx(),
                    "Element": atom.GetSymbol(),
                    "Formal charge": atom.GetFormalCharge(),
                    "Degree": atom.GetDegree(),
                    "Aromatic": atom.GetIsAromatic()
                })
            st.dataframe(atom_rows, use_container_width=True, hide_index=True)

    with tab3d:
        st.markdown("Rotate with the mouse; scroll to zoom. The displayed conformer is computationally generated.")
        escaped_block = result["molblock_3d"].replace("\\", "\\\\").replace("`", "\\`")
        html = f"""
        <div id="viewer" style="width:100%;height:520px;position:relative;border:1px solid #ddd;border-radius:10px;"></div>
        <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
        <script>
        const element = document.getElementById('viewer');
        const viewer = $3Dmol.createViewer(element, {{backgroundColor: 'white'}});
        const molData = `{escaped_block}`;
        viewer.addModel(molData, 'sdf');
        viewer.setStyle({{}}, {{stick: {{radius: 0.16}}, sphere: {{scale: 0.23}}}});
        viewer.zoomTo();
        viewer.render();
        window.addEventListener('resize', () => viewer.resize());
        </script>
        """
        components.html(html, height=540, scrolling=False)
        st.download_button(
            "Download generated 3D structure (.mol)",
            data=result["molblock_3d"],
            file_name="molecule_3d.mol",
            mime="chemical/x-mdl-mol"
        )

    with tab_bonds:
        counts = count_bonds(mol)
        st.subheader("Bond counts")
        st.dataframe([{"Bond/property": k, "Count": v} for k, v in counts.items()], hide_index=True, use_container_width=True)
        st.warning("Sigma/pi counts are estimates from a Kekulé representation. Aromatic bonding is delocalized, so individual aromatic bonds do not have fixed localized single/double character.")
        bond_rows = []
        for bond in mol.GetBonds():
            bond_rows.append({
                "Bond index": bond.GetIdx(),
                "Atom 1": f"{bond.GetBeginAtomIdx()} ({bond.GetBeginAtom().GetSymbol()})",
                "Atom 2": f"{bond.GetEndAtomIdx()} ({bond.GetEndAtom().GetSymbol()})",
                "Bond type": str(bond.GetBondType()),
                "Aromatic": bond.GetIsAromatic(),
                "In ring": bond.IsInRing()
            })
        st.subheader("Individual bonds")
        st.dataframe(bond_rows, hide_index=True, use_container_width=True)

    with tab_stereo:
        st.subheader("Stereochemical information")
        stereo_atoms = []
        for atom in mol.GetAtoms():
            if atom.HasProp("_CIPCode"):
                stereo_atoms.append({
                    "Atom index": atom.GetIdx(),
                    "Element": atom.GetSymbol(),
                    "CIP label": atom.GetProp("_CIPCode"),
                    "Chiral tag": str(atom.GetChiralTag())
                })
        stereo_bonds = []
        for bond in mol.GetBonds():
            stereo = str(bond.GetStereo())
            if stereo not in ("STEREONONE", "STEREOANY"):
                stereo_bonds.append({
                    "Bond index": bond.GetIdx(),
                    "Atoms": f"{bond.GetBeginAtomIdx()}–{bond.GetEndAtomIdx()}",
                    "Stereo": stereo
                })
        if stereo_atoms:
            st.markdown("**Tetrahedral stereocentres (CIP labels)**")
            st.dataframe(stereo_atoms, hide_index=True, use_container_width=True)
        else:
            st.write("No assigned R/S tetrahedral stereocentres were found in this input.")
        if stereo_bonds:
            st.markdown("**Stereochemical double bonds**")
            st.dataframe(stereo_bonds, hide_index=True, use_container_width=True)
        else:
            st.write("No assigned E/Z double-bond stereochemistry was found in this input.")
        st.caption("A blank result does not prove that a molecule has no possible stereoisomers; the input must specify stereochemistry for it to be reported.")

    with tab_export:
        st.subheader("Download your results")
        st.download_button(
            "Download 2D structure image (.png)",
            data=_png_bytes(result["mol_2d"]),
            file_name="molecule_2d.png",
            mime="image/png"
        )
        st.download_button(
            "Download structure (.sdf)",
            data=Chem.MolToMolBlock(result["mol"]),
            file_name="molecule.sdf",
            mime="chemical/x-mdl-sdfile"
        )
        report = (
            f"Query: {result['query']}\nInput mode: {result['mode']}\n"
            f"Formula: {formula}\nMolecular weight: {mw:.5f} g/mol\nExact molecular weight: {exact_mw:.5f} g/mol\n"
            f"Atoms (excluding implicit H): {atoms}\nHeavy atoms: {heavy_atoms}\nRings: {rings}\n"
            f"Rotatable bonds: {rotatable}\nCanonical isomeric SMILES: {Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)}\n"
            f"Bond analysis: {counts}\n"
        )
        st.download_button("Download property report (.txt)", data=report, file_name="molecule_report.txt", mime="text/plain")

else:
    st.markdown("### Try a sample")
    st.write("Choose **Molecule name** and enter `caffeine`, or choose **SMILES** and enter `CCO` for ethanol.")
    st.markdown("### How it works")
    st.write("RDKit parses the structure and calculates properties. For molecule-name searches, PubChem supplies a matching SMILES string. RDKit then generates a 2D depiction and an estimated 3D conformer.")
