import streamlit as st
import os
import io
import tempfile
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import pdfplumber
import docx
import json

st.set_page_config(page_title="Resume Analyzer", layout="wide")

# Utilities

def extract_text_from_pdf(file_bytes):
    text = []
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                text.append(page_text)
    except Exception as e:
        st.warning(f"PDF parsing warning: {e}")
    return "\n".join(text)


def extract_text_from_docx(file_bytes):
    text = []
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
            tmp.write(file_bytes)
            tmp.flush()
            doc = docx.Document(tmp.name)
            for p in doc.paragraphs:
                text.append(p.text)
    except Exception as e:
        st.warning(f"DOCX parsing warning: {e}")
    finally:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass
    return "\n".join(text)


def extract_text_from_txt(file_bytes):
    try:
        return file_bytes.decode(errors='ignore')
    except Exception:
        return ""


def safe_truncate(text, max_chars=3000):
    if not text:
        return text
    return text[:max_chars]


def get_embedding(text, client, model="text-embedding-3-small"):
    # Truncate to a safe size to avoid too-large inputs
    text = safe_truncate(text, max_chars=3000)
    resp = client.embeddings.create(model=model, input=text)
    return np.array(resp.data[0].embedding, dtype=float)


def analyze_resumes(openai_api_key, job_description, uploaded_files, use_llm_report=True, embedding_model="text-embedding-3-small", chat_model="gpt-4o-mini"):
    client = OpenAI(api_key=openai_api_key)

    # Prepare job embedding
    with st.spinner("Creating job description embedding..."):
        job_emb = get_embedding(job_description, client, model=embedding_model)

    results = []
    for f in uploaded_files:
        fname = f.name
        b = f.read()
        text = ""
        if fname.lower().endswith('.pdf'):
            text = extract_text_from_pdf(b)
        elif fname.lower().endswith('.docx'):
            text = extract_text_from_docx(b)
        elif fname.lower().endswith('.txt') or fname.lower().endswith('.md'):
            text = extract_text_from_txt(b)
        else:
            # try pdf then txt
            text = extract_text_from_pdf(b)
            if not text:
                text = extract_text_from_txt(b)

        short = (text[:500] + '...') if len(text) > 500 else text

        # embedding
        try:
            with st.spinner(f"Embedding {fname}..."):
                emb = get_embedding(text, client, model=embedding_model)
            # cosine similarity
            sim = cosine_similarity([job_emb], [emb])[0][0]
            match_pct = float(np.round(sim * 100, 2))
        except Exception as e:
            sim = 0.0
            match_pct = 0.0
            st.warning(f"Embedding failed for {fname}: {e}")

        results.append({
            'filename': fname,
            'match_pct': match_pct,
            'similarity': float(sim),
            'excerpt': short,
            'full_text': text
        })

    df = pd.DataFrame(results).sort_values(by='match_pct', ascending=False).reset_index(drop=True)

    detailed_report = {
        'job_description': job_description,
        'summary': '',
        'candidates': []
    }

    if use_llm_report and len(results) > 0:
        # Build a compact prompt that includes job description and top candidates
        top_n = min(5, len(results))
        prompt_parts = ["You are an HR assistant. Given the job description and candidate resumes, produce a concise evaluation report with a short summary, strengths/weaknesses for each candidate, and a recommended action (hire/phone-screen/reject). Return the report as JSON with keys: summary, candidates (list of {filename,match_pct,comment,recommendation}).\n\n"]
        prompt_parts.append("Job description:\n" + safe_truncate(job_description, 4000))
        prompt_parts.append("\nTop candidate resumes (truncated):\n")
        for r in results[:top_n]:
            prompt_parts.append(f"---\nFilename: {r['filename']}\nMatch: {r['match_pct']}%\nExcerpt: {safe_truncate(r['full_text'], 1200)}\n")

        prompt = "\n".join(prompt_parts)

        try:
            with st.spinner("Generating detailed LLM report..."):
                completion = client.chat.completions.create(
                    model=chat_model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that summarizes and evaluates candidate resumes against a job description."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=800,
                    temperature=0.2,
                )
                llm_text = completion.choices[0].message.content
                # Try to parse JSON out of the LLM response; if not JSON, put it into summary
                try:
                    parsed = json.loads(llm_text)
                    detailed_report['summary'] = parsed.get('summary', '')
                    detailed_report['candidates'] = parsed.get('candidates', [])
                except Exception:
                    detailed_report['summary'] = llm_text
                    # fallback: create candidate comments from the similarity
                    for _, row in df.iterrows():
                        detailed_report['candidates'].append({
                            'filename': row['filename'],
                            'match_pct': row['match_pct'],
                            'comment': f"Match score based on embedding similarity: {row['match_pct']}%.",
                            'recommendation': 'phone-screen' if row['match_pct'] > 60 else 'consider' if row['match_pct'] > 40 else 'reject'
                        })
        except Exception as e:
            st.warning(f"LLM report generation failed: {e}")
            detailed_report['summary'] = 'LLM report generation failed: ' + str(e)
            for _, row in df.iterrows():
                detailed_report['candidates'].append({
                    'filename': row['filename'],
                    'match_pct': row['match_pct'],
                    'comment': f"Match score based on embedding similarity: {row['match_pct']}%.",
                    'recommendation': 'phone-screen' if row['match_pct'] > 60 else 'consider' if row['match_pct'] > 40 else 'reject'
                })

    else:
        for _, row in df.iterrows():
            detailed_report['candidates'].append({
                'filename': row['filename'],
                'match_pct': row['match_pct'],
                'comment': f"Match score based on embedding similarity: {row['match_pct']}%.",
                'recommendation': 'phone-screen' if row['match_pct'] > 60 else 'consider' if row['match_pct'] > 40 else 'reject'
            })

    return df, detailed_report


# Streamlit UI

st.title("Resume Analyzer")
st.markdown("Upload multiple resumes and a job description. Provide your OpenAI API key at runtime to analyze resumes and get matching percentages and a downloadable report.")

with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("OpenAI API Key", type="password")
    use_llm = st.checkbox("Use OpenAI LLM for a detailed report (may consume tokens)", value=True)
    model_for_embeddings = st.selectbox("Embedding model", options=["text-embedding-3-small"], index=0)

uploaded_files = st.file_uploader("Upload resumes (PDF, DOCX, TXT). You can upload multiple.", accept_multiple_files=True)
job_description = st.text_area("Job description / role profile (paste or upload below)")
job_upload = st.file_uploader("(Optional) Upload a job description file (TXT or PDF or DOCX)", accept_multiple_files=False)

if job_upload is not None and not job_description:
    b = job_upload.read()
    if job_upload.name.lower().endswith('.pdf'):
        job_description = extract_text_from_pdf(b)
    elif job_upload.name.lower().endswith('.docx'):
        job_description = extract_text_from_docx(b)
    else:
        job_description = extract_text_from_txt(b)

if st.button("Analyze"):
    if not api_key:
        st.error("Please provide your OpenAI API key in the sidebar before running analysis.")
    elif not uploaded_files:
        st.error("Please upload at least one resume to analyze.")
    elif not job_description or job_description.strip() == "":
        st.error("Please paste or upload a job description.")
    else:
        # Run analysis
        try:
            df, report = analyze_resumes(
                api_key,
                job_description,
                uploaded_files,
                use_llm_report=use_llm,
                embedding_model=model_for_embeddings,
            )

            st.success("Analysis complete")

            # Display results
            st.subheader("Matching scores")
            st.dataframe(df[['filename', 'match_pct', 'excerpt']])

            # Allow download of CSV of scores
            csv = df[['filename', 'match_pct', 'similarity', 'excerpt']].to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV of scores", data=csv, file_name="resume_scores.csv", mime='text/csv')

            # Display LLM report
            st.subheader("Detailed report")
            if report['summary']:
                st.markdown("**Summary:**")
                st.write(report['summary'])

            st.markdown("**Candidates**")
            for c in report['candidates']:
                st.markdown(f"- **{c.get('filename', '')}** — {c.get('match_pct', '')}% — **{c.get('recommendation','')}**")
                if c.get('comment'):
                    st.write(c.get('comment'))

            # Download full report as JSON
            report_json = json.dumps(report, indent=2).encode('utf-8')
            st.download_button("Download full JSON report", data=report_json, file_name="resume_report.json", mime='application/json')

        except Exception as e:
            st.exception(e)
else:
    st.info("Fill in the fields and press Analyze to start.")
