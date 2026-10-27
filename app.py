import io, math, requests
import streamlit as st
import streamlit.components.v1 as components
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, Descriptors, rdMolDescriptors, Lipinski
from rdkit.Chem.rdchem import BondType

st.set_page_config(page_title="Cheminformatics Explorer", page_icon="🧪", layout="wide")

def png_bytes(m):
    b=io.BytesIO(); Draw.MolToImage(m,size=(1000,700)).save(b,format="PNG"); return b.getvalue()

def lookup_name(name):
    url="https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"+requests.utils.quote(name.strip(),safe="")+"/property/IsomericSMILES,ConnectivitySMILES/JSON"
    r=requests.get(url,timeout=20)
    if r.status_code!=200: raise ValueError("PubChem could not find that name. Try another name or enter SMILES.")
    props=r.json().get("PropertyTable",{}).get("Properties",[])
    if not props: raise ValueError("PubChem returned no structure.")
    return props[0].get("SMILES") or props[0].get("ConnectivitySMILES")

def parse(text, mode):
    if not text.strip(): raise ValueError("Enter a molecule name or SMILES.")
    smi=lookup_name(text) if mode=="Molecule name" else text.strip()
    m=Chem.MolFromSmiles(smi)
    if m is None: raise ValueError("Invalid molecule input. Check the name or SMILES.")
    Chem.AssignStereochemistry(m,cleanIt=True,force=True)
    return m,smi

def build3d(m, add_h=True):
    x=Chem.AddHs(Chem.Mol(m)) if add_h else Chem.Mol(m)
    p=AllChem.ETKDGv3(); p.randomSeed=2026
    if AllChem.EmbedMolecule(x,p)!=0: raise ValueError("Could not generate a 3D conformer.")
    if AllChem.MMFFHasAllMoleculeParams(x): AllChem.MMFFOptimizeMolecule(x,maxIters=300)
    elif AllChem.UFFHasAllMoleculeParams(x): AllChem.UFFOptimizeMolecule(x,maxIters=300)
    return x

def bond_analysis(m):
    x=Chem.AddHs(Chem.Mol(m))
    try: Chem.Kekulize(x,clearAromaticFlags=True)
    except Exception: pass
    counts={"Single":0,"Double":0,"Triple":0,"Aromatic/unlocalized":0}
    sigma=pi=0; rows=[]
    for b in x.GetBonds():
        t=b.GetBondType()
        if t==BondType.SINGLE: counts["Single"]+=1; sigma+=1; pc=0
        elif t==BondType.DOUBLE: counts["Double"]+=1; sigma+=1; pi+=1; pc=1
        elif t==BondType.TRIPLE: counts["Triple"]+=1; sigma+=1; pi+=2; pc=2
        elif t==BondType.AROMATIC: counts["Aromatic/unlocalized"]+=1; sigma+=1; pc="unlocalized"
        else: pc=0
        rows.append({"Atom 1":f'{b.GetBeginAtomIdx()} ({b.GetBeginAtom().GetSymbol()})',"Atom 2":f'{b.GetEndAtomIdx()} ({b.GetEndAtom().GetSymbol()})',"Bond type":str(t),"Aromatic":b.GetIsAromatic(),"Pi contribution":pc})
    return counts,sigma,pi,rows

def geometry(m):
    conf=m.GetConformer(); bonds=[]; angles=[]
    for b in m.GetBonds():
        i,j=b.GetBeginAtomIdx(),b.GetEndAtomIdx(); p=conf.GetAtomPosition(i); q=conf.GetAtomPosition(j)
        d=math.sqrt((p.x-q.x)**2+(p.y-q.y)**2+(p.z-q.z)**2)
        bonds.append({"Atom 1":f"{i} ({m.GetAtomWithIdx(i).GetSymbol()})","Atom 2":f"{j} ({m.GetAtomWithIdx(j).GetSymbol()})","Bond type":str(b.GetBondType()),"Length (Å)":round(d,3)})
    for c in m.GetAtoms():
        k=c.GetIdx(); ns=[a.GetIdx() for a in c.GetNeighbors()]; pc=conf.GetAtomPosition(k)
        for ii in range(len(ns)):
            for jj in range(ii+1,len(ns)):
                i,j=ns[ii],ns[jj]; p=conf.GetAtomPosition(i); q=conf.GetAtomPosition(j)
                v=(p.x-pc.x,p.y-pc.y,p.z-pc.z); w=(q.x-pc.x,q.y-pc.y,q.z-pc.z)
                dot=sum(a*b for a,b in zip(v,w)); nv=math.sqrt(sum(a*a for a in v)); nw=math.sqrt(sum(a*a for a in w))
                if nv and nw:
                    ang=math.degrees(math.acos(max(-1,min(1,dot/(nv*nw)))))
                    angles.append({"Atom 1":f"{i} ({m.GetAtomWithIdx(i).GetSymbol()})","Central atom":f"{k} ({c.GetSymbol()})","Atom 3":f"{j} ({m.GetAtomWithIdx(j).GetSymbol()})","Angle (°)":round(ang,2)})
    return bonds,angles

def stereochemistry(m):
    Chem.AssignStereochemistry(m,cleanIt=True,force=True)
    centers=[]
    for i,label in Chem.FindMolChiralCenters(m,includeUnassigned=True,includeCIP=True,useLegacyImplementation=False):
        a=m.GetAtomWithIdx(i); centers.append({"Atom index":i,"Element":a.GetSymbol(),"Configuration":label,"Chiral tag":str(a.GetChiralTag())})
    bonds=[]
    for b in m.GetBonds():
        if b.GetStereo()!=Chem.BondStereo.STEREONONE:
            bonds.append({"Bond index":b.GetIdx(),"Atoms":f"{b.GetBeginAtomIdx()}–{b.GetEndAtomIdx()}","Stereo":str(b.GetStereo())})
    return centers,bonds

def viewer_html(block, style, labels):
    block=block.replace("\\","\\\\").replace("`","\\`").replace("</script>","<\\/script>")
    styles={"Ball and stick":"{stick:{radius:0.18},sphere:{scale:0.24}}","Stick":"{stick:{radius:0.18}}","Sphere":"{sphere:{scale:0.32}}","Line":"{line:{linewidth:2}}"}
    label_code="viewer.addPropertyLabels('index', {}, {font:'Arial',fontSize:12,fontColor:'black',showBackground:true,backgroundColor:'white',backgroundOpacity:0.75});".format("true") if labels else ""
    sty=styles[style]
    return f"""<div id="v" style="height:500px;width:100%;border:1px solid #ccc;border-radius:10px"></div><div id="atominfo" style="padding:8px;font-family:Arial">Click an atom to inspect its index, element and coordinates.</div>
<script src="https://3Dmol.org/build/3Dmol-min.js"></script><script>
const el=document.getElementById('v'); const viewer=$3Dmol.createViewer(el,{{backgroundColor:'white'}});
viewer.addModel(`{block}`,'mol'); viewer.setStyle({{}},{sty}); {label_code}
viewer.setClickable({{}},true,function(atom,viewer){{
document.getElementById('atominfo').innerHTML='<b>Selected atom</b>: index '+atom.index+' · element '+atom.elem+' · coordinates (Å): '+Number(atom.x).toFixed(3)+', '+Number(atom.y).toFixed(3)+', '+Number(atom.z).toFixed(3);
viewer.setStyle({{}},{sty}); viewer.setStyle({{index:atom.index}},{{stick:{{radius:0.24,color:'magenta'}},sphere:{{scale:0.45,color:'magenta'}}}}); viewer.render();
}});
viewer.zoomTo(); viewer.render(); window.addEventListener('resize',()=>viewer.resize());
</script>"""

st.title("🧪 Cheminformatics Explorer")
st.caption("2D/3D structures • bonding • geometry • stereochemistry")
st.info("3D coordinates are computational estimates, not experimental measurements. Aromatic sigma/pi counts are a simplified educational representation.")
with st.sidebar:
    st.header("Molecule input")
    mode=st.radio("Input type",["Molecule name","SMILES"])
    query=st.text_input("Molecule name" if mode=="Molecule name" else "SMILES",value="ethanol" if mode=="Molecule name" else "CCO")
    add_h=st.checkbox("Include hydrogens in 3D model",True)
    st.subheader("3D viewer settings")
    style=st.selectbox("Display style",["Ball and stick","Stick","Sphere","Line"])
    labels=st.checkbox("Show atom indices",False)
    go=st.button("Analyze molecule",type="primary",use_container_width=True)
    st.caption("Name lookup uses PubChem and requires internet access.")
if "chem_result" not in st.session_state: st.session_state.chem_result=None
if go:
    try:
        with st.spinner("Calculating molecular information..."):
            mol,smi=parse(query,mode); m2=Chem.Mol(mol); AllChem.Compute2DCoords(m2)
            m3=build3d(mol,add_h); block=Chem.MolToMolBlock(m3)
            st.session_state.chem_result={"mol":mol,"m2":m2,"m3":m3,"block":block,"query":query,"mode":mode,"smi":smi}
    except requests.RequestException: st.error("Could not connect to PubChem. Try SMILES input.")
    except Exception as e: st.error(str(e))
r=st.session_state.chem_result
if r:
    m=r["mol"]; m3=r["m3"]; formula=rdMolDescriptors.CalcMolFormula(m); canon=Chem.MolToSmiles(m,canonical=True,isomericSmiles=True)
    counts,sigma,pi,bondrows=bond_analysis(m); centers,stereobonds=stereochemistry(Chem.Mol(m)); lengths,angles=geometry(m3)
    st.subheader("📋 Basic Information")
    rows=[("Input SMILES",r["smi"]),("Canonical SMILES",canon),("Molecular Formula",formula),("Atoms",m.GetNumAtoms()),("Heavy Atoms",m.GetNumHeavyAtoms()),("Heavy-Atom Bonds",m.GetNumBonds()),("Sigma (σ) Bonds",sigma),("Pi (π) Bonds",pi),("Total Bonds (σ + π)",sigma+pi)]
    left,right=st.columns(2)
    with left:
        for k,v in rows[:5]: st.markdown(f"**{k}:** {v}")
    with right:
        for k,v in rows[5:]: st.markdown(f"**{k}:** {v}")
    st.caption("Sigma/pi counts include implicit hydrogens. For ethanol (CCO): formula C₂H₆O, 3 atoms, 3 heavy atoms, 2 heavy-atom bonds, 8 sigma bonds, 0 pi bonds, total 8.")
    a,b,c,d=st.columns(4)
    a.metric("Molecular weight",f"{Descriptors.MolWt(m):.3f} g/mol"); b.metric("Exact mass",f"{Descriptors.ExactMolWt(m):.4f} Da")
    c.metric("Rings",rdMolDescriptors.CalcNumRings(m)); d.metric("Rotatable bonds",Lipinski.NumRotatableBonds(m))
    t2,t3,tb,ts,te=st.tabs(["🧬 2D Structure","🧊 3D Viewer","🔗 Bond Analysis","🧭 Stereochemistry","⬇️ Export"])
    with t2:
        c1,c2=st.columns([1,1])
        with c1: st.image(Draw.MolToImage(r["m2"],size=(700,500)),caption="2D molecular structure",use_container_width=True)
        with c2:
            st.markdown("**Canonical isomeric SMILES**"); st.code(canon)
            st.markdown("**Molecular properties**")
            st.write({"Formula":formula,"Molecular weight (g/mol)":round(Descriptors.MolWt(m),4),"TPSA (Å²)":round(rdMolDescriptors.CalcTPSA(m),3),"LogP estimate":round(Descriptors.MolLogP(m),3),"H-bond donors":rdMolDescriptors.CalcNumHBD(m),"H-bond acceptors":rdMolDescriptors.CalcNumHBA(m)})
        st.subheader("Atom table")
        st.dataframe([{"Index":a.GetIdx(),"Element":a.GetSymbol(),"Formal charge":a.GetFormalCharge(),"Degree":a.GetDegree(),"Aromatic":a.GetIsAromatic(),"In ring":a.IsInRing()} for a in m.GetAtoms()],use_container_width=True,hide_index=True)
    with t3:
        st.write("Drag to rotate, scroll to zoom, and click an atom to inspect its index, element and 3D coordinates.")
        components.html(viewer_html(r["block"],style,labels),height=550,scrolling=False)
        st.download_button("Download generated 3D structure (.mol)",r["block"],"molecule_3d.mol",mime="chemical/x-mdl-molfile")
    with tb:
        st.subheader("Bond counts")
        st.dataframe([{"Bond metric":k,"Count":v} for k,v in [*counts.items(),("Sigma bonds including H",sigma),("Pi bonds (Kekulé representation)",pi),("Total σ + π contributions",sigma+pi)]],use_container_width=True,hide_index=True)
        st.warning("Aromatic bonding is delocalized. Pi counts use localized double bonds in a Kekulé representation and are intended for introductory learning.")
        st.subheader("Individual bonds in explicit-H molecule")
        st.dataframe(bondrows,use_container_width=True,hide_index=True)
        st.subheader("Generated 3D bond lengths (Å)")
        st.dataframe(lengths,use_container_width=True,hide_index=True)
        st.subheader("Generated 3D bond angles (°)")
        st.dataframe(angles,use_container_width=True,hide_index=True)
    with ts:
        st.subheader("R/S configuration and chirality")
        st.dataframe(centers,use_container_width=True,hide_index=True) if centers else st.write("No tetrahedral stereocentres were identified in the input.")
        st.subheader("E/Z and other stereochemical bonds")
        st.dataframe(stereobonds,use_container_width=True,hide_index=True) if stereobonds else st.write("No explicitly assigned stereochemical bonds were found in the input.")
        st.info("R/S and E/Z labels depend on stereochemistry specified or assigned in the input SMILES. The generated conformer alone does not establish experimental configuration.")
        st.write(f"Generated 3D conformer: {'Yes' if m3.GetNumConformers() else 'No'} · Atoms in 3D model: {m3.GetNumAtoms()}")
    with te:
        st.download_button("Download 2D structure (.png)",png_bytes(r["m2"]),"molecule_2d.png",mime="image/png")
        st.download_button("Download structure (.sdf)",Chem.MolToMolBlock(m),"molecule.sdf",mime="chemical/x-mdl-sdfile")
        report="\n".join([f"{k}: {v}" for k,v in rows]+[f"Molecular weight: {Descriptors.MolWt(m):.5f} g/mol",f"Exact mass: {Descriptors.ExactMolWt(m):.5f} Da",f"Rings: {rdMolDescriptors.CalcNumRings(m)}",f"Rotatable bonds: {Lipinski.NumRotatableBonds(m)}",f"R/S centres: {centers}",f"E/Z bonds: {stereobonds}","Note: 3D geometry is estimated; aromatic bonding is delocalized."])
        st.download_button("Download property report (.txt)",report,"molecule_report.txt",mime="text/plain")
else:
    st.subheader("Start exploring a molecule")
    st.write("Try SMILES `CCO` for ethanol. Basic Information should report C₂H₆O, 3 atoms, 3 heavy atoms, 2 heavy-atom bonds, 8 sigma bonds and 0 pi bonds.")
