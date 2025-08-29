import os
import tempfile
from flask import Flask, request, render_template, jsonify, session
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem
from reportlab.lib import colors
from flask import send_file

load_dotenv()

# -----------------------------
# INIT FUNCTIONS
# -----------------------------
def init_llm(api_key: str):
    if not api_key:
        raise ValueError("Google API Key is required")
    return ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key=api_key)

def init_embeddings(hf_api_token: str):
    if not hf_api_token:
        raise ValueError("HuggingFace API Token is required")
    return HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        huggingfacehub_api_token=hf_api_token
    )

# -----------------------------
# PROMPTS
# -----------------------------

SUMMARY_PROMPT_DIABETES = ChatPromptTemplate.from_messages([
    ("system", """You are a medical report summarizer for lab test PDFs. 
Focus only on **Diabetes / Prediabetes Detection** using the provided text.
Always use the **reference ranges exactly as written in the report** (do not invent ranges).
For each marker, include:
- The patient's value
- The report’s specified reference range (if available)
- An interpretation (Normal / Prediabetes / Diabetes / Mildly Elevated / Protective / Optimal)

Do NOT include page numbers. 
Do NOT write "N/A" — always provide an interpretation even if no range is given.

Provide the final output in this structure:

### Core Diabetes Markers
- HbA1c: [value] ([reference range]) → [interpretation]
- Estimated Glucose: [value] → [interpretation]
- Fasting Blood Sugar: [value] ([reference range]) → [interpretation]

### Supporting Risk Factors
- Inflammation (hs-CRP): [value] ([reference range if available]) → [interpretation]
- Lipid Profile:
    - Total Cholesterol: [value] ([range]) → [interpretation]
    - Triglycerides: [value] ([range]) → [interpretation]
    - HDL Cholesterol: [value] ([range]) → [interpretation]
    - LDL Cholesterol: [value] ([range]) → [interpretation]

### Conclusion
Give a **clear, patient-friendly conclusion**:
- State whether the patient is **Normal, Prediabetic, or Diabetic** based on HbA1c and Fasting Blood Sugar.
- Consider hs-CRP (inflammation) and Lipid Profile (cardiovascular/metabolic risk).
- Integrate all findings into one clinical picture (e.g., “Borderline Prediabetes with otherwise healthy lipids, mild inflammation”).
"""),
    ("human", "Summarize the following report:\n\n{context}")
])

SUMMARY_PROMPT_HYPERTENSION = ChatPromptTemplate.from_messages([
    ("system", """You are a medical report summarizer for lab test PDFs. 
Focus only on **Hypertension Detection** using the provided text.
Always use the **reference ranges exactly as written in the report** (do not invent ranges).

For each marker, include:
- The patient's value
- The report’s specified reference range (if available)
- An interpretation (Normal / Elevated / Slightly Low / Mildly Elevated / Protective)

Do NOT include page numbers. 
Do NOT write "N/A" — always provide an interpretation even if no range is given.

Provide the final output in this structure:

### Kidney Function (linked to Hypertension)
- Serum Creatinine: [value] ([range]) → [interpretation]
- Estimated GFR: [value] → [interpretation]
- Serum Sodium: [value] ([range]) → [interpretation]
- Serum Chloride: [value] ([range]) → [interpretation]
- Blood Urea: [value] ([range]) → [interpretation]

### Cardiovascular Risk Marker
- hs-CRP: [value] ([range]) → [interpretation]

### Lipid Profile (Heart & BP Risk Link)
- Total Cholesterol: [value] ([range]) → [interpretation]
- HDL: [value] ([range]) → [interpretation]
- LDL: [value] ([range]) → [interpretation]
- Triglycerides: [value] ([range]) → [interpretation]
- Cholesterol/HDL Ratio: [value] ([range]) → [interpretation]

### Supportive Factors
- Magnesium: [value] ([range]) → [interpretation]

### Conclusion
Give a **clear, patient-friendly conclusion**:
- Summarize kidney function (normal/abnormal, hypertension impact).  
- Summarize inflammation (hs-CRP).  
- Summarize lipid profile (risk or protective).  
- Summarize supportive electrolytes (magnesium).  
- End with an integrated statement on **hypertension risk and overall cardiovascular health**.
"""),
    ("human", "Summarize the following report:\n\n{context}")
])

SUMMARY_PROMPT_DYSLIPIDEMIA = ChatPromptTemplate.from_messages([
    ("system", """You are a medical report summarizer for lab test PDFs. 
Focus only on **Dyslipidemia / Heart Disease Risk** using the provided text.
Always use the **reference ranges exactly as written in the report** (do not invent ranges).

For each marker, include:
- The patient’s value
- The report’s specified reference range (if available)
- An interpretation (Desirable / Normal / Protective / Optimal / Elevated / Low Risk)

Do NOT include page numbers. 
Do NOT write "N/A" — always provide an interpretation even if no range is given.

Provide the final output in this structure:

### Lipid Profile (Core for Dyslipidemia Assessment)
- Total Cholesterol: [value] ([range]) → [interpretation]
- Triglycerides: [value] ([range]) → [interpretation]
- HDL: [value] ([range]) → [interpretation]
- LDL: [value] ([range]) → [interpretation]
- VLDL: [value] ([range]) → [interpretation]
- Total Cholesterol/HDL Ratio: [value] ([range]) → [interpretation]
- LDL/HDL Ratio: [value] ([range]) → [interpretation]

### Cardiovascular Risk Marker
- hs-CRP: [value] ([range]) → [interpretation]

### Liver Function (affecting lipid metabolism)
- ALT (SGPT): [value] ([range]) → [interpretation]
- AST (SGOT): [value] ([range]) → [interpretation]
- GGT: [value] ([range]) → [interpretation]
- Albumin/Globulin Ratio: [value] ([range]) → [interpretation]

### Kidney Function (Cardiovascular Link)
- Serum Creatinine: [value] ([range]) → [interpretation]
- Estimated GFR: [value] → [interpretation]
- Uric Acid: [value] ([range]) → [interpretation]

### Conclusion
Give a **clear, patient-friendly conclusion**:
- Summarize lipid profile status (normal, dyslipidemia, protective).  
- Summarize hs-CRP (inflammation/cardiovascular risk).  
- Mention liver & kidney health (impact on lipid metabolism & heart disease risk).  
- End with an integrated statement on **heart disease risk** and **preventive lifestyle care**.
"""),
    ("human", "Summarize the following report:\n\n{context}")
])

SUMMARY_PROMPT_LIVER = ChatPromptTemplate.from_messages([
    ("system", """You are a medical report summarizer for lab test PDFs. 
Focus only on **Liver Disorders** using the provided text.
Always use the **reference ranges exactly as written in the report** (do not invent ranges).

For each marker, include:
- The patient’s value
- The report’s specified reference range (if available)
- An interpretation (Normal / Low / Elevated / Suggestive of disorder)

Do NOT include page numbers. 
Do NOT write "N/A" — always provide an interpretation even if no range is given.

Provide the final output in this structure:

### Bilirubin Levels
- Total Bilirubin: [value] ([range]) → [interpretation]
- Direct Bilirubin: [value] ([range]) → [interpretation]
- Indirect Bilirubin: [value] ([range]) → [interpretation]

### Liver Enzymes (Hepatocellular Injury)
- AST (SGOT): [value] ([range]) → [interpretation]
- ALT (SGPT): [value] ([range]) → [interpretation]
- SGOT/SGPT Ratio: [value] ([range]) → [interpretation]

### Cholestasis Markers
- ALP: [value] ([range]) → [interpretation]
- GGT: [value] ([range]) → [interpretation]

### Liver Synthetic Function
- Total Protein: [value] ([range]) → [interpretation]
- Albumin: [value] ([range]) → [interpretation]
- Globulin: [value] ([range]) → [interpretation]
- Albumin/Globulin Ratio: [value] ([range]) → [interpretation]

### Supporting Findings
- Iron Studies: [value] ([range]) → [interpretation]
- Zinc: [value] ([range]) → [interpretation]

### Conclusion
Give a **clear, patient-friendly conclusion**:
- State whether bilirubin and enzymes suggest normal function or liver damage.  
- Mention synthetic function (protein/albumin status).  
- Mention supporting findings (iron, zinc).  
- End with an integrated summary (e.g., “No active liver disease, mild nutritional deficiency suspected”).  
"""),
    ("human", "Summarize the following report:\n\n{context}")
])

SUMMARY_PROMPT_KIDNEY = ChatPromptTemplate.from_messages([
    ("system", """You are a medical report summarizer for lab test PDFs. 
Focus only on **Kidney Disorders** using the provided text.
Always use the **reference ranges exactly as written in the report** (do not invent ranges).

For each marker, include:
- The patient’s value
- The report’s specified reference range (if available)
- An interpretation (Normal / Low / Elevated / Mildly Abnormal)

Do NOT include page numbers. 
Do NOT write "N/A" — always provide an interpretation even if no range is given.

Provide the final output in this structure:

### Kidney Function Tests
- Serum Creatinine: [value] ([range]) → [interpretation]
- Estimated GFR: [value] → [interpretation]
- Blood Urea: [value] ([range]) → [interpretation]
- Blood Urea Nitrogen (BUN): [value] ([range]) → [interpretation]
- BUN/Creatinine Ratio: [value] → [interpretation]
- Uric Acid: [value] ([range]) → [interpretation]

### Electrolytes
- Sodium: [value] ([range]) → [interpretation]
- Chloride: [value] ([range]) → [interpretation]
- Calcium: [value] ([range]) → [interpretation]
- Phosphorus: [value] ([range]) → [interpretation]
- Magnesium: [value] ([range]) → [interpretation]

### Iron Studies (supportive, not primary kidney marker)
- Serum Iron: [value] ([range]) → [interpretation]
- UIBC: [value] ([range]) → [interpretation]
- TIBC: [value] ([range]) → [interpretation]
- Transferrin Saturation: [value] ([range]) → [interpretation]

### Conclusion
Give a **clear, patient-friendly conclusion**:
- State whether kidney function is normal or impaired (based on Creatinine & GFR).  
- Mention electrolytes balance.  
- Mention if iron deficiency anemia is present and clarify that it’s not due to kidney disease.  
- End with an integrated summary (e.g., “Kidneys functioning normally, mild anemia likely nutritional”).  
"""),
    ("human", "Summarize the following report:\n\n{context}")
])

SUMMARY_PROMPT_THYROID = ChatPromptTemplate.from_messages([
    ("system", """You are a medical report summarizer for lab test PDFs. 
Focus only on **Thyroid Disorders** using the provided text.  
Always use the **reference ranges exactly as written in the report** (do not invent ranges).

For each marker, include:
- The patient’s value
- The report’s specified reference range (if available)
- An interpretation (Normal / Low / Elevated / Hypothyroid / Hyperthyroid / Mildly Abnormal)

Do NOT include page numbers. 
Do NOT write "N/A" — always provide an interpretation.

Provide the final output in this structure:

### Thyroid Function Tests
- TSH: [value] ([range]) → [interpretation]
- Free T4: [value] ([range]) → [interpretation]
- Free T3: [value] ([range]) → [interpretation]

### Supporting Markers
- Magnesium: [value] ([range]) → [interpretation]
- Serum Iron: [value] ([range]) → [interpretation]
- TIBC: [value] ([range]) → [interpretation]
- Zinc: [value] ([range]) → [interpretation]

### Conclusion
Give a **clear, patient-friendly conclusion**:
- State whether the patient is euthyroid (normal), hypothyroid, or hyperthyroid based on TFT.  
- Mention how supporting markers (Iron, Zinc, Magnesium) affect thyroid health.  
- End with a clinical summary (e.g., “Normal thyroid hormones but low iron — may need monitoring for hypothyroid symptoms”).  
"""),
    ("human", "Summarize the following report:\n\n{context}")
])

SUMMARY_PROMPT_ANEMIA = ChatPromptTemplate.from_messages([
    ("system", """You are a medical report summarizer for lab test PDFs. 
Focus only on **Anemia / Blood Disorders** using the provided text.  
Always use the **reference ranges exactly as written in the report** (do not invent ranges).

For each marker, include:
- The patient’s value
- The report’s specified reference range (if available)
- An interpretation (Normal / Low / High / Iron Deficiency Anemia / Hemolysis / Nutritional Anemia etc.)

Do NOT include page numbers.  
Do NOT write "N/A" — always provide an interpretation.

Provide the final output in this structure:

### Key Markers for Anemia
- HbA1c: [value] ([range]) → [interpretation, note if influenced by anemia]
- Bilirubin (Total / Direct / Indirect): [value] ([range]) → [interpretation]
- Serum Iron: [value] ([range]) → [interpretation]
- UIBC: [value] ([range]) → [interpretation]
- TIBC: [value] ([range]) → [interpretation]
- Transferrin Saturation: [value] ([range]) → [interpretation]
- Zinc: [value] ([range]) → [interpretation]
- Serum Protein: [value] ([range]) → [interpretation]
- Globulin: [value] ([range]) → [interpretation]

### Conclusion
Give a **clear, patient-friendly conclusion**:
- State whether the patient has **Iron Deficiency Anemia, Hemolytic Anemia, Pernicious Anemia, or Normal blood markers**.  
- Mention nutritional contributors (protein, zinc).  
- End with a clinical summary (e.g., “Iron deficiency anemia likely, no evidence of hemolysis”).  
"""),
    ("human", "Summarize the following report:\n\n{context}")
])

SUMMARY_PROMPT_OBESITY = ChatPromptTemplate.from_messages([
    ("system", """You are a medical report summarizer for lab test PDFs. 
Focus only on **Obesity / Metabolic Syndrome** using the provided text.  
Always use the **reference ranges exactly as written in the report** (do not invent ranges).

For each marker, include:
- The patient’s value
- The report’s specified reference range (if available)
- An interpretation (Normal / Borderline / High / At Risk)

Do NOT include page numbers.  
Do NOT write "N/A" — always provide an interpretation.

Provide the final output in this structure:

### HbA1c (Long-term blood sugar control)
- HbA1c →  [value] ([range]) → [interpretation]
- Estimated Glucose →  [value] → [interpretation]

### Fasting Blood Sugar
- Fasting Blood Sugar →  [value] ([range]) → [interpretation]
     
### hs-CRP (Inflammation Marker)
- hs-CRP →  [value] ([range]) → [interpretation]
     
### Liver Function (Fatty Liver / NAFLD Risk)
- AST (SGOT) → [value] ([range]) → [interpretation]
- ALT (SGPT) → [value] ([range]) → [interpretation]
- Total Protein → [value] ([range]) → [interpretation]
- Globulin → [value] ([range]) → [interpretation]
     
### Iron Study (Metabolic Link via Anemia / Obesity)
- Serum Iron → [value] ([range]) → [interpretation]
- UIBC → [value] ([range]) → [interpretation]
- TIBC → [value] ([range]) → [interpretation]
     
### Kidney Function (Uric Acid – Metabolic Syndrome Link)
- Uric Acid → [value] ([range]) → [interpretation]

### Lipid Profile (Key for Metabolic Syndrome)
- Total Cholesterol → [value] ([range]) → [interpretation]
- Triglycerides → [value] ([range]) → [interpretation]
- HDL → [value] ([range]) → [interpretation]
- LDL → [value] ([range]) → [interpretation]
- Chol/HDL Ratio → [value] ([range]) → [interpretation]

### Conclusion
Give a **clear, patient-friendly conclusion**:
- State whether the patient is at **risk of metabolic syndrome / obesity complications**.  
- Highlight protective factors (e.g., healthy lipid profile).  
- Provide one integrated summary like:  
“Borderline metabolic syndrome risk with early inflammation and iron imbalance, though lipids are protective.”  
"""),
    ("human", "Summarize the following report:\n\n{context}")
])

SUMMARY_PROMPT_NUTRITION = ChatPromptTemplate.from_messages([
    ("system", """You are a medical report summarizer for lab test PDFs. 
Focus only on **Nutritional Deficiencies** using the provided text.  
Always use the **reference ranges exactly as written in the report** (do not invent ranges).

For each marker, include:
- The patient’s value
- The report’s specified reference range (if available)
- An interpretation (Normal / Low / Deficient / Risk)

Do NOT include page numbers.  
Do NOT write "N/A" — always provide an interpretation.

Provide the final output in this structure:

### Key Markers for Nutritional Deficiencies
- Magnesium → [value, range, interpretation, nutritional relevance]
- Total Protein → [value] ([range]) → [interpretation]
- Albumin → [value] ([range]) → [interpretation]
- Globulin → [value] ([range]) → [interpretation]
- Serum Iron →  → [value] ([range]) → [interpretation]
- TIBC → [value] ([range]) → [interpretation]
- Transferrin Saturation → [value] ([range]) → [interpretation]
- Zinc → [value, range, interpretation, nutritional importance]

### Conclusion
Give a **clear, patient-friendly conclusion**:
- Highlight confirmed deficiencies (e.g., Iron deficiency, protein deficiency).  
- Note markers that are normal but nutritionally relevant (e.g., magnesium, zinc).  
- Provide one integrated summary like:  
“Findings suggest nutritional deficiency features, especially iron deficiency and low proteins, while magnesium and zinc remain within range.”  
"""),
    ("human", "Summarize the following report:\n\n{context}")
])

# -----------------------------
# UTILS
# -----------------------------
def extract_pages(file_path, page_numbers):
    reader = PdfReader(file_path)
    text = "\n".join(
        reader.pages[i].extract_text() for i in page_numbers if i < len(reader.pages)
    )
    return text

# -----------------------------
# FLASK APP
# -----------------------------
app = Flask(__name__)
app.secret_key = "supersecretkey" 

@app.route('/')
def index():
    return render_template('index.html')

# -----------------------------
# Upload PDF (only once)
# -----------------------------
@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    file = request.files['file']
    if not file:
        return jsonify({"error": "No file uploaded"}), 400
    
    temp_path = os.path.join(tempfile.gettempdir(), file.filename)
    file.save(temp_path)
    session['uploaded_pdf'] = temp_path

    return jsonify({"message": "PDF uploaded successfully"})

# -----------------------------
# Diabetes / Prediabetes
# -----------------------------
@app.route('/summarize_diabetes', methods=['POST'])
def summarize_diabetes():
    file_path = session.get('uploaded_pdf')
    if not file_path or not os.path.exists(file_path):
        return jsonify({"error": "No PDF uploaded. Please upload first."}), 400
    
    relevant_pages = [1, 3, 4, 12]
    text = extract_pages(file_path, relevant_pages)

    embeddings = init_embeddings(os.getenv("HF_API_TOKEN"))
    texts = text.split("\n\n")
    vectorstore = FAISS.from_texts(texts, embeddings)

    all_docs = vectorstore.similarity_search("", k=len(texts))
    context = "\n\n".join([d.page_content for d in all_docs])

    llm = init_llm(os.getenv("GOOGLE_API_KEY"))
    chain = SUMMARY_PROMPT_DIABETES | llm
    summary = chain.invoke({"context": context})

    return jsonify({"summary": summary.content})

@app.route('/summarize_hypertension', methods=['POST'])
def summarize_hypertension():
    file_path = session.get('uploaded_pdf')
    if not file_path or not os.path.exists(file_path):
        return jsonify({"error": "No PDF uploaded. Please upload first."}), 400

    relevant_pages = [3, 4, 8, 9, 12] 
    text = extract_pages(file_path, relevant_pages)

    embeddings = init_embeddings(os.getenv("HF_API_TOKEN"))
    texts = text.split("\n\n")
    vectorstore = FAISS.from_texts(texts, embeddings)

    all_docs = vectorstore.similarity_search("", k=len(texts))
    context = "\n\n".join([d.page_content for d in all_docs])

    llm = init_llm(os.getenv("GOOGLE_API_KEY"))
    chain = SUMMARY_PROMPT_HYPERTENSION | llm
    summary = chain.invoke({"context": context})

    return jsonify({"summary": summary.content.strip()})

@app.route('/summarize_dyslipidemia', methods=['POST'])
def summarize_dyslipidemia():
    file_path = session.get('uploaded_pdf')
    if not file_path or not os.path.exists(file_path):
        return jsonify({"error": "No PDF uploaded. Please upload first."}), 400

    relevant_pages = [4, 5, 6, 8, 9, 12]  
    text = extract_pages(file_path, relevant_pages)

    embeddings = init_embeddings(os.getenv("HF_API_TOKEN"))
    texts = text.split("\n\n")
    vectorstore = FAISS.from_texts(texts, embeddings)

    all_docs = vectorstore.similarity_search("", k=len(texts))
    context = "\n\n".join([d.page_content for d in all_docs])

    llm = init_llm(os.getenv("GOOGLE_API_KEY"))
    chain = SUMMARY_PROMPT_DYSLIPIDEMIA | llm
    summary = chain.invoke({"context": context})

    return jsonify({"summary": summary.content.strip()})

@app.route('/summarize_liver', methods=['POST'])
def summarize_liver():
    file_path = session.get('uploaded_pdf')
    if not file_path or not os.path.exists(file_path):
        return jsonify({"error": "No PDF uploaded. Please upload first."}), 400

    relevant_pages = [5, 7, 10, 11] 
    text = extract_pages(file_path, relevant_pages)

    embeddings = init_embeddings(os.getenv("HF_API_TOKEN"))
    texts = text.split("\n\n")
    vectorstore = FAISS.from_texts(texts, embeddings)

    all_docs = vectorstore.similarity_search("", k=len(texts))
    context = "\n\n".join([d.page_content for d in all_docs])

    llm = init_llm(os.getenv("GOOGLE_API_KEY"))
    chain = SUMMARY_PROMPT_LIVER | llm
    summary = chain.invoke({"context": context})

    return jsonify({"summary": summary.content.strip()})

@app.route('/summarize_kidney', methods=['POST'])
def summarize_kidney():
    file_path = session.get('uploaded_pdf')
    if not file_path or not os.path.exists(file_path):
        return jsonify({"error": "No PDF uploaded. Please upload first."}), 400

    relevant_pages = [3, 7, 8, 9]  
    text = extract_pages(file_path, relevant_pages)

    embeddings = init_embeddings(os.getenv("HF_API_TOKEN"))
    texts = text.split("\n\n")
    vectorstore = FAISS.from_texts(texts, embeddings)

    all_docs = vectorstore.similarity_search("", k=len(texts))
    context = "\n\n".join([d.page_content for d in all_docs])

    llm = init_llm(os.getenv("GOOGLE_API_KEY"))
    chain = SUMMARY_PROMPT_KIDNEY | llm
    summary = chain.invoke({"context": context})

    return jsonify({"summary": summary.content.strip()})

@app.route('/summarize_thyroid', methods=['POST'])
def summarize_thyroid():
    file_path = session.get('uploaded_pdf')
    if not file_path or not os.path.exists(file_path):
        return jsonify({"error": "No PDF uploaded. Please upload first."}), 400

    relevant_pages = [3, 7, 10, 22]  
    text = extract_pages(file_path, relevant_pages)

    embeddings = init_embeddings(os.getenv("HF_API_TOKEN"))
    texts = text.split("\n\n")
    vectorstore = FAISS.from_texts(texts, embeddings)

    all_docs = vectorstore.similarity_search("", k=len(texts))
    context = "\n\n".join([d.page_content for d in all_docs])

    llm = init_llm(os.getenv("GOOGLE_API_KEY"))
    chain = SUMMARY_PROMPT_THYROID | llm
    summary = chain.invoke({"context": context})

    return jsonify({"summary": summary.content.strip()})

@app.route('/summarize_anemia', methods=['POST'])
def summarize_anemia():
    file_path = session.get('uploaded_pdf')
    if not file_path or not os.path.exists(file_path):
        return jsonify({"error": "No PDF uploaded. Please upload first."}), 400

    relevant_pages = [1, 5, 6, 7, 10]  
    text = extract_pages(file_path, relevant_pages)

    embeddings = init_embeddings(os.getenv("HF_API_TOKEN"))
    texts = text.split("\n\n")
    vectorstore = FAISS.from_texts(texts, embeddings)

    all_docs = vectorstore.similarity_search("", k=len(texts))
    context = "\n\n".join([d.page_content for d in all_docs])

    llm = init_llm(os.getenv("GOOGLE_API_KEY"))
    chain = SUMMARY_PROMPT_ANEMIA | llm
    summary = chain.invoke({"context": context})

    return jsonify({"summary": summary.content.strip()})

@app.route('/summarize_obesity', methods=['POST'])
def summarize_obesity():
    file_path = session.get('uploaded_pdf')
    if not file_path or not os.path.exists(file_path):
        return jsonify({"error": "No PDF uploaded. Please upload first."}), 400

    relevant_pages = [1, 3, 4, 5, 7, 8, 12]  
    text = extract_pages(file_path, relevant_pages)

    embeddings = init_embeddings(os.getenv("HF_API_TOKEN"))
    texts = text.split("\n\n")
    vectorstore = FAISS.from_texts(texts, embeddings)

    all_docs = vectorstore.similarity_search("", k=len(texts))
    context = "\n\n".join([d.page_content for d in all_docs])

    llm = init_llm(os.getenv("GOOGLE_API_KEY"))
    chain = SUMMARY_PROMPT_OBESITY | llm
    summary = chain.invoke({"context": context})

    return jsonify({"summary": summary.content.strip()})

@app.route('/summarize_nutrition', methods=['POST'])
def summarize_nutrition():
    file_path = session.get('uploaded_pdf')
    if not file_path or not os.path.exists(file_path):
        return jsonify({"error": "No PDF uploaded. Please upload first."}), 400

    relevant_pages = [3, 5, 7, 10]  
    text = extract_pages(file_path, relevant_pages)

    embeddings = init_embeddings(os.getenv("HF_API_TOKEN"))
    texts = text.split("\n\n")
    vectorstore = FAISS.from_texts(texts, embeddings)

    all_docs = vectorstore.similarity_search("", k=len(texts))
    context = "\n\n".join([d.page_content for d in all_docs])

    llm = init_llm(os.getenv("GOOGLE_API_KEY"))
    chain = SUMMARY_PROMPT_NUTRITION | llm
    summary = chain.invoke({"context": context})

    return jsonify({"summary": summary.content.strip()})

def generate_summary_pdf(summaries, output_file):
    doc = SimpleDocTemplate(output_file, pagesize=A4)
    story = []

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Title'],
        fontSize=26,  
        textColor=colors.HexColor("#041782"), 
        alignment=1,  
        spaceAfter=20
    )

    section_style = ParagraphStyle(
        'section',
        parent=styles['Heading1'],
        textColor=colors.HexColor("#0d6efd"),
        spaceAfter=12
    )
    subsection_style = styles["Heading3"]
    body_style = styles["BodyText"]

    story.append(Paragraph("Comprehensive Lab Report Summary", title_style))
    story.append(Spacer(1, 20))

    for section, content in summaries.items():
        story.append(Paragraph(section, section_style))
        story.append(Spacer(1, 10))

        for line in content.split("\n"):
            line = line.strip()
            if not line:
                continue
            elif line.startswith("### "): 
                story.append(Paragraph(line.replace("###", "").strip(), subsection_style))
                story.append(Spacer(1, 6))
            elif line.startswith("- "):  
                story.append(ListFlowable(
                    [ListItem(Paragraph(line[2:], body_style), bulletColor=colors.black)],
                    bulletType='bullet'
                ))
            else:
                story.append(Paragraph(line, body_style))
                story.append(Spacer(1, 4))

        story.append(Spacer(1, 16))

    doc.build(story)

SUMMARY_CONFIGS = {
    "Diabetes": (SUMMARY_PROMPT_DIABETES, [1, 3, 4, 12]),
    "Hypertension": (SUMMARY_PROMPT_HYPERTENSION, [3, 4, 8, 9, 12]),
    "Dyslipidemia": (SUMMARY_PROMPT_DYSLIPIDEMIA, [4, 5, 6, 8, 9, 12]),
    "Liver": (SUMMARY_PROMPT_LIVER, [5, 7, 10, 11]),
    "Kidney": (SUMMARY_PROMPT_KIDNEY, [3, 7, 8, 9]),
    "Thyroid": (SUMMARY_PROMPT_THYROID, [3, 7, 10, 22]),
    "Anemia": (SUMMARY_PROMPT_ANEMIA, [1, 5, 6, 7, 10]),
    "Obesity": (SUMMARY_PROMPT_OBESITY, [1, 3, 4, 5, 7, 8, 12]),
    "Nutrition": (SUMMARY_PROMPT_NUTRITION, [3, 5, 7, 10]),
}

@app.route('/summarize_all', methods=['GET', 'POST'])
def summarize_all():
    file_path = session.get('uploaded_pdf')
    if not file_path or not os.path.exists(file_path):
        return jsonify({"error": "No PDF uploaded. Please upload first."}), 400

    summaries = {}
    embeddings = init_embeddings(os.getenv("HF_API_TOKEN"))
    llm = init_llm(os.getenv("GOOGLE_API_KEY"))

    for section, (prompt, pages) in SUMMARY_CONFIGS.items():
        text = extract_pages(file_path, pages)
        texts = text.split("\n\n")
        vectorstore = FAISS.from_texts(texts, embeddings)
        all_docs = vectorstore.similarity_search("", k=len(texts))
        context = "\n\n".join([d.page_content for d in all_docs])

        chain = prompt | llm
        summary = chain.invoke({"context": context})
        summaries[section] = summary.content.strip()

    temp_pdf = os.path.join(tempfile.gettempdir(), "lab_report_summary.pdf")
    generate_summary_pdf(summaries, temp_pdf)

    return send_file(temp_pdf, as_attachment=True, download_name="Lab_Report_Summary.pdf")

# -----------------------------
# ENTRY POINT
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)