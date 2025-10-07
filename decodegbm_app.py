# decodegbm_app.py
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import io, random, textwrap, datetime
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader

# ------------------------------
# Helper: Mock AI prediction generator
# ------------------------------
def generate_mock_predictions(patient_id, img_array, omics_row):
    seed = int(datetime.datetime.utcnow().timestamp() * 1000) % (2**31 - 1)
    random.seed(seed + (hash(patient_id) & 0xffff))
    np.random.seed(seed % 100000)

    subtypes = ["Proneural", "Classical", "Mesenchymal", "Neural"]
    subtype = random.choice(subtypes)
    grade = random.choice(["II", "III", "IV"])
    density_opts = ["Low", "Medium", "High"]
    density = random.choices(density_opts, weights=[0.2,0.4,0.4])[0]
    necrosis = "Detected" if random.random() > 0.6 else "Not Detected"
    ai_confidence = round(70 + 30 * random.random(),1)

    def status_from_row(key):
        val = omics_row.get(key)
        if val is None:
            return random.choice(["Mutated","Wild Type"])
        try:
            if float(val) > 0.5:
                return "Mutated/High"
            else:
                return "Wild Type/Low"
        except:
            return str(val)

    idh1 = status_from_row("IDH1")
    tp53 = status_from_row("TP53")
    egfr = status_from_row("EGFR")
    mgmt = status_from_row("MGMT")
    rna_status = random.choice(["Upregulated","Downregulated","Normal"])
    protein_signature = random.choice(["High EGFR, Low PTEN","High PDGFRA, Low TP53","No major protein shifts"])
    clinical_summary = (
        f"The AI analyzed the tissue patch and suggests a {subtype} tumor region (Grade {grade}). "
        f"Cell density appears {density} and necrosis: {necrosis}. "
        f"Key genes: IDH1={idh1}, TP53={tp53}, EGFR={egfr}, MGMT={mgmt}. "
        f"RNA: {rna_status}, Protein: {protein_signature}. AI confidence: {ai_confidence}%."
    )
    human_friendly = {
        "Tumor_Region_Explanation": f"{subtype} tumor region detected — subtype-specific treatment paths exist.",
        "Gene_Interpretations":{
            "IDH1":"Mutation indicates better treatment response.",
            "TP53":"Normal TP53 is favorable; mutation may be aggressive.",
            "EGFR":"Targetable with specific drugs if activated.",
            "MGMT":"Methylation suggests good response to Temozolomide."
        },
        "Overall_Message":"Tumor may respond well to standard and targeted therapy. Consult oncology."
    }
    return {
        "Patient_ID": patient_id,
        "Tumor_Subtype": subtype,
        "Tumor_Grade": grade,
        "Tumor_Density": density,
        "Necrosis": necrosis,
        "AI_Confidence": ai_confidence,
        "IDH1": idh1,
        "TP53": tp53,
        "EGFR": egfr,
        "MGMT": mgmt,
        "RNA_Expression": rna_status,
        "Protein_Signature": protein_signature,
        "Clinical_Summary": clinical_summary,
        "Human_Friendly_Report": human_friendly
    }

# ------------------------------
# Helper: Heatmap overlay
# ------------------------------
def create_heatmap_overlay(img_array):
    h, w = img_array.shape[:2]
    cx, cy = w//2 + random.randint(-10,10), h//2 + random.randint(-10,10)
    xv, yv = np.meshgrid(np.arange(w), np.arange(h))
    sigma = min(h,w)/6.0
    base = np.exp(-((xv-cx)*2 + (yv-cy)*2)/(2*sigma*2))
    base = (base - base.min()) / (base.max()-base.min()+1e-9)
    base += 0.15*np.random.rand(h,w)
    heatmap = np.uint8(255*base)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(np.uint8(img_array*255),0.6,heatmap,0.4,0)
    return overlay

# ------------------------------
# Helper: Build PDF report
# ------------------------------
def build_pdf_report_bytes(report_dict, original_img, overlay_img):
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4
    margin = 40
    y = height - margin

    # Header
    c.setFont("Helvetica-Bold", 16)
    c.drawString(margin, y, "DecodeGBM — AI Multi-Omics Diagnostic Report")
    c.setFont("Helvetica", 9)
    c.drawString(width-200, y, f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    y -= 30

    # Patient ID
    c.setFont("Helvetica-Bold", 11)
    c.drawString(margin, y, f"Patient ID: {report_dict.get('Patient_ID','N/A')}")
    y -= 18

    # Tumor Quick Facts
    c.setFont("Helvetica-Bold", 10)
    c.drawString(margin, y, "Key AI Findings:")
    y -= 14
    c.setFont("Helvetica", 10)
    lines = [
        f"Tumor Subtype: {report_dict.get('Tumor_Subtype')}, Grade: {report_dict.get('Tumor_Grade')}",
        f"Cell Density: {report_dict.get('Tumor_Density')}, Necrosis: {report_dict.get('Necrosis')}",
        f"AI Confidence: {report_dict.get('AI_Confidence')} %"
    ]
    for ln in lines:
        c.drawString(margin+10,y,ln)
        y -= 14

    # Molecular Summary
    y -= 6
    c.setFont("Helvetica-Bold", 10)
    c.drawString(margin,y,"Molecular / Omics Summary:")
    y -= 14
    c.setFont("Helvetica",10)
    mol_lines = [
        f"IDH1: {report_dict.get('IDH1')}",
        f"TP53: {report_dict.get('TP53')}",
        f"EGFR: {report_dict.get('EGFR')}",
        f"MGMT: {report_dict.get('MGMT')}",
        f"RNA Expression: {report_dict.get('RNA_Expression')}",
        f"Protein Signature: {report_dict.get('Protein_Signature')}"
    ]
    for ln in mol_lines:
        c.drawString(margin+10,y,ln)
        y -= 12

    # Clinical Interpretation
    y -= 10
    c.setFont("Helvetica-Bold",10)
    c.drawString(margin,y,"Clinical Interpretation (patient-friendly):")
    y -= 14
    c.setFont("Helvetica",10)
    hf = report_dict.get("Human_Friendly_Report",{})
    overall_msg = hf.get("Overall_Message") if isinstance(hf, dict) else report_dict.get("Clinical_Summary")
    for wrapped in textwrap.wrap(overall_msg,90):
        c.drawString(margin+10,y,wrapped)
        y -= 12

    # Images
    try:
        orig_pil = original_img.copy()
        ov_pil = overlay_img.copy()
        target_w = (width-2*margin-20)/2
        target_h = target_w*orig_pil.height/orig_pil.width
        orig_buf = io.BytesIO()
        orig_pil.save(orig_buf, format="PNG")
        orig_buf.seek(0)
        ov_buf = io.BytesIO()
        ov_pil.save(ov_buf, format="PNG")
        ov_buf.seek(0)
        c.drawImage(ImageReader(orig_buf),margin,y-target_h,width=target_w,height=target_h)
        c.drawImage(ImageReader(ov_buf),margin+target_w+20,y-target_h,width=target_w,height=target_h)
        y -= target_h+10
    except: y -= 10

    # Footer suggestions
    c.setFont("Helvetica-Bold",10)
    c.drawString(margin,y,"Suggested Next Steps:")
    y -= 14
    c.setFont("Helvetica",10)
    suggestions = [
        "1. Discuss AI results with oncologist/molecular pathology team.",
        "2. Confirm key gene markers before treatment decisions.",
        "3. Evaluate Temozolomide suitability if MGMT methylation present.",
        "4. Consider clinical trials for targeted agents if applicable."
    ]
    for s in suggestions:
        for wrapped in textwrap.wrap(s,110):
            c.drawString(margin+8,y,wrapped)
            y -= 12
        y -= 4
    y -= 6
    c.setFont("Helvetica-Oblique",8)
    c.drawString(margin,y,"Note: This report is AI-assisted and informational. Clinical decisions must be made by qualified healthcare professionals.")
    c.showPage()
    c.save()
    buf.seek(0)
    return buf.read()

# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="DecodeGBM Demo", layout="wide", page_icon="🧠")
st.title("🧠 DecodeGBM — AI Multi-Omics Diagnostic Assistant (Hackathon Demo)")

st.markdown("""
*Demo Features:*  
- Upload image + multi-omics Excel → DecodeGBM provides mock AI analysis.  
- Shows heatmap overlay, omics correlations, patient-friendly interpretations.  
- Generates downloadable PDF report for doctors & patients.
""")

uploaded_image = st.sidebar.file_uploader("Upload MRI / Patch Image (PNG/JPG)", type=['png','jpg','jpeg'])
uploaded_excel = st.sidebar.file_uploader("Upload Multi-Omics Excel (columns: Patient_ID, EGFR, TP53, MGMT, IDH1…)", type=['xlsx'])

if uploaded_image and uploaded_excel:
    pil_img = Image.open(uploaded_image).convert("RGB")
    img_resized = pil_img.resize((224,224))
    img_array = np.array(img_resized)/255.0

    df = pd.read_excel(uploaded_excel)
    st.subheader("📄 Omics Data Preview")
    st.dataframe(df.head())

    chosen_index = st.number_input("Select row corresponding to this image:", min_value=1, max_value=len(df), value=1)
    row = df.iloc[chosen_index-1].to_dict()

    patient_id = row.get("Patient_ID", f"GBM-DEMO-{random.randint(100,999)}")
    mock = generate_mock_predictions(patient_id,img_array,row)

    st.markdown("## 🩺 AI Predictions (Mock)")
    c1,c2,c3 = st.columns(3)
    c1.metric("Glioma Risk (mock)", f"{round(mock['AI_Confidence']*0.9,1)} %")
    c2.metric("AI Confidence", f"{mock['AI_Confidence']} %")
    c3.metric("Tumor Grade", f"{mock['Tumor_Grade']}")

    st.markdown("### 🔎 Clinical Summary (patient-friendly)")
    st.info(mock["Clinical_Summary"])

    st.markdown("### 🔥 AI Attention Heatmap (simulated)")
    overlay = create_heatmap_overlay(img_array)
    col1,col2 = st.columns(2)
    col1.image(pil_img, caption="Original Image", use_column_width=True)
    overlay_display = overlay[:,:,::-1] if overlay.shape[2]==3 else overlay
    col2.image(overlay_display, caption="AI Attention Overlay", use_column_width=True)

    st.markdown("### 🧬 Omics Correlation (numeric columns)")
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1]>=2:
        corr = numeric_df.corr()
        fig, ax = plt.subplots(figsize=(6,4))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        st.pyplot(fig)
    else:
        st.info("Excel needs numeric omics columns (gene expression) for correlation map.")

    st.markdown("### 🔬 Molecular Interpretation (human-friendly)")
    human = mock["Human_Friendly_Report"]
    st.write("*Tumor Region Explanation:*", human["Tumor_Region_Explanation"])
    st.write("*Gene Interpretations:*")
    for g,k in human["Gene_Interpretations"].items():
        st.write(f"- *{g}*: {k}")
    st.write("*Overall message for patient:*")
    st.success(human["Overall_Message"])

    # CSV Download
    report_table = {
        "Patient_ID":[mock["Patient_ID"]],
        "Tumor_Subtype":[mock["Tumor_Subtype"]],
        "Tumor_Grade":[mock["Tumor_Grade"]],
        "AI_Confidence":[mock["AI_Confidence"]],
        "IDH1":[mock["IDH1"]],
        "TP53":[mock["TP53"]],
        "EGFR":[mock["EGFR"]],
        "MGMT":[mock["MGMT"]],
        "RNA_Expression":[mock["RNA_Expression"]],
        "Protein_Signature":[mock["Protein_Signature"]],
        "Clinical_Summary":[mock["Clinical_Summary"]]
    }
    report_df = pd.DataFrame(report_table)
    st.download_button("📥 Download CSV Summary", data=report_df.to_csv(index=False).encode('utf-8'), file_name=f"{patient_id}_DecodeGBM_Summary.csv", mime="text/csv")

    # PDF Download
    overlay_pil = Image.fromarray(np.uint8(overlay_display))
    pdf_bytes = build_pdf_report_bytes(mock, pil_img.resize((600,600)), overlay_pil.resize((600,600)))
    st.download_button("📥 Download PDF Report", data=pdf_bytes, file_name=f"{patient_id}_DecodeGBM_Report.pdf", mime="application/pdf")

else:
    st.info("Upload both MRI/patch image and multi-omics Excel to run DecodeGBM demo.")