"""
app.py
------
URL-first Streamlit demo for the fake job posting detector.

Default flow
    1. paste a public job posting URL
    2. press "Fetch & Analyze"
    3. job_extractor.py reads the page and maps it onto the model features
    4. the saved MLP (model/fake_job_mlp.joblib) predicts
    5. the prediction and the extracted details are shown

Manual entry is only offered as a fallback: it appears automatically when the
page cannot be fetched or when no job description can be extracted.

Run with:   streamlit run app.py
"""

import json

import joblib
import pandas as pd
import streamlit as st

import job_extractor as je
import preprocessing as pp

MODEL_PATH = pp.MODEL_DIR / "fake_job_mlp.joblib"
METADATA_PATH = pp.MODEL_DIR / "metadata.json"

# A real, public job posting used by the "Try a sample URL" button.
SAMPLE_URL = "https://www.python.org/jobs/8219/"

st.set_page_config(page_title="Job Scam Checker", page_icon="🕵️", layout="centered")


@st.cache_resource
def load_artifacts():
    """Load the trained pipeline and the metadata once per session."""
    if not MODEL_PATH.exists():
        return None, None
    model = joblib.load(MODEL_PATH)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8")) if METADATA_PATH.exists() else {}
    return model, metadata


# ---------------------------------------------------------------- examples ----
# Two REAL postings taken from the dataset (job_id shown in the comment), used by
# the manual fallback so it can demonstrate both possible outcomes.
EXAMPLES = {
    "Genuine posting": {  # dataset job_id = 1754, label = 0
        "employment_type": "Full-time",
        "required_experience": "Internship",
        "required_education": "Bachelor's Degree",
        "industry": "Internet",
        "job_function": "Marketing",
        "telecommuting": False,
        "has_company_logo": True,
        "has_questions": True,
        "salary_range": "0-0",
        "description": (
            "Daily Secret is seeking two Marketing Interns to join our team in Athens, Greece. "
            "This position will work closely together with our editorial, marketing, and customer "
            "success teams to execute and track marketing campaigns. Our ideal candidate is "
            "passionate about the Daily Secret brand and eager to learn the basics of digital "
            "marketing. No prior experience is required, but you must be tech savvy, a great team "
            "player, and ready to roll up your sleeves. Responsibilities: Work with editorial and "
            "marketing counterparts to implement marketing campaigns, both for user acquisition "
            "and for advertising clients. Manage Daily Secrets social media presence. Produce "
            "regular campaign reports based on analysis of proprietary data. Provide support to "
            "customer success team as needed."
        ),
        "requirements": (
            "Strong academic background. Demonstrated interest in digital marketing. Proven track "
            "record of working well on teams. Excellent interpersonal and communication skills "
            "(written and verbal). Highly adaptable and enjoys learning about new technology "
            "platforms. Must enjoy and appreciate a collaborative, fast-paced, deadline-driven "
            "work environment. Believes in our company values as much as we do."
        ),
        "company_profile": (
            "Daily Secret is one of the fastest growing digital media brands for premium audiences "
            "around the world, serving 1.8 million subscribers every day. Covering over 35 markets, "
            "its signature daily email newsletters reveal coveted bars, restaurants, shops and all "
            "around local secrets while integrating brands around passion points. Headquartered in "
            "New York, NY and Athens, Greece, Daily Secret is backed by leading media investors."
        ),
    },
    "Suspicious posting": {  # dataset job_id = 2740, label = 1
        "employment_type": "Full-time",
        "required_experience": "Entry level",
        "required_education": "Unspecified",
        "industry": "Accounting",
        "job_function": "Administrative",
        "telecommuting": False,
        "has_company_logo": False,
        "has_questions": False,
        "salary_range": "36000-50000",
        "description": (
            "Administrative Assistant provides personal administrative support and assistance to "
            "executive staff. Main Job Tasks and Responsibilities: Prepare and edit correspondence, "
            "communications, presentations and other documents. Design and maintain databases. "
            "File and retrieve documents and reference materials. Conduct research, collect and "
            "analyse data to prepare reports and documents. Manage and maintain executives' "
            "schedules, appointments and travel arrangements. Arrange and co-ordinate meetings and "
            "events. Record, transcribe and distribute minutes of meetings. Monitor, screen, respond "
            "to and distribute incoming communications. Answer and manage incoming calls. Receive "
            "and interact with incoming visitors. Liaise with internal staff at all levels. "
            "Interact with external clients. Co-ordinate project-based work. Review operating "
            "practices and implement improvements where necessary. Supervise, coach and train lower "
            "level staff."
        ),
        "requirements": (
            "Education and Experience: Usually at least 3 years experience providing support at a "
            "high level. Proficient computer skills and in-depth knowledge of relevant software "
            "such as MS Office Suite. Knowledge of standard office administrative practices and "
            "procedures. Bachelors degree is an advantage. Key Competencies: Organizational and "
            "planning skills. Communication skills. Information gathering and monitoring skills. "
            "Problem analysis and problem solving skills. Judgment and decision-making ability. "
            "Initiative. Confidentiality. Team member. Attention to detail and accuracy. "
            "Adaptability."
        ),
        "company_profile": "",
    },
}

DEFAULT_INPUTS = {
    "employment_type": "Full-time",
    "required_experience": "Mid-Senior level",
    "required_education": "Bachelor's Degree",
    "industry": "Information Technology and Services",
    "job_function": "Information Technology",
    "telecommuting": False,
    "has_company_logo": True,
    "has_questions": True,
    "salary_range": "",
    "description": "",
    "requirements": "",
    "company_profile": "",
}

FEATURE_LABELS = [
    ("employment_type", "Employment type"),
    ("required_experience", "Required experience"),
    ("required_education", "Required education"),
    ("industry", "Industry"),
    ("function", "Job function"),
    ("telecommuting", "Telecommuting"),
    ("has_company_logo", "Company logo"),
    ("has_questions", "Screening questions"),
    ("salary_range", "Salary range"),
    ("description", "Description"),
    ("requirements", "Requirements"),
    ("company_profile", "Company profile"),
]


def prefill(values: dict) -> None:
    for key, value in values.items():
        st.session_state[key] = value


# --------------------------------------------------------------- prediction ----
def predict_fields(model, fields: dict) -> dict:
    """
    Turn a dict of model features into a prediction.

    The feature frame is built by preprocessing.build_input_dataframe, so the URL
    flow and the manual flow go through exactly the same pipeline.
    """
    input_df = pp.build_input_dataframe(
        employment_type=fields.get("employment_type", "") or "",
        required_experience=fields.get("required_experience", "") or "",
        required_education=fields.get("required_education", "") or "",
        industry=fields.get("industry", "") or "",
        job_function=fields.get("function", "") or "",
        description=fields.get("description", "") or "",
        requirements=fields.get("requirements", "") or "",
        company_profile=fields.get("company_profile", "") or "",
        telecommuting=bool(fields.get("telecommuting", 0)),
        has_company_logo=bool(fields.get("has_company_logo", 0)),
        has_questions=bool(fields.get("has_questions", 0)),
        salary_range=str(fields.get("salary_range", "") or ""),
    )
    probability = float(model.predict_proba(input_df)[0][1])
    return {
        "probability": probability,
        "is_fraud": probability >= 0.5,
        "input_df": input_df,
    }


def feature_summary(input_df: pd.DataFrame) -> pd.DataFrame:
    """Small table of the numbers the network actually used."""
    row = input_df.iloc[0]
    return pd.DataFrame(
        {
            "Feature the model used": [
                "Description length (words)",
                "Requirements length (words)",
                "Company profile length (words)",
                "Salary range given",
                "Company logo",
                "Screening questions",
                "Telecommuting",
            ],
            "Value": [
                str(int(row["description_length"])),
                str(int(row["requirements_length"])),
                str(int(row["company_profile_length"])),
                "Yes" if row["has_salary_range"] else "No",
                "Yes" if row["has_company_logo"] else "No",
                "Yes" if row["has_questions"] else "No",
                "Yes" if row["telecommuting"] else "No",
            ],
        }
    )


def show_prediction(analysis: dict) -> None:
    """Render the prediction box (used by both flows)."""
    probability = analysis["probability"]
    is_fraud = analysis["is_fraud"]

    st.subheader("Result")
    if is_fraud:
        st.error("⚠️ Potentially fraudulent — verify carefully.")
    else:
        st.success("✅ Likely genuine.")

    col_a, col_b = st.columns(2)
    col_a.metric("Fraud probability", f"{probability * 100:.1f}%")
    col_b.metric("Classification", "Fraudulent" if is_fraud else "Genuine")
    st.progress(min(max(probability, 0.0), 1.0))
    st.caption(
        "The score is the output of the neural network (MLPClassifier) for the fraudulent "
        "class. A score close to 50% means the model is unsure. The model only looks at "
        "structured cues such as flags and text length — not at the meaning of the words — "
        "so a well-written scam can still be missed."
    )


# ------------------------------------------------------------------- layout ----
model, metadata = load_artifacts()

st.title("🕵️ Job Scam Checker")
st.write(
    "Paste the link of a public job posting. The app fetches the page, extracts the "
    "job details and runs the trained model to estimate whether the posting looks "
    "**genuine** or **potentially fraudulent**."
)

if model is None:
    st.error(
        "No trained model found.\n\n"
        "Run `python train.py` first — it creates `model/fake_job_mlp.joblib`."
    )
    st.stop()

# Widget defaults for the manual fallback form.
for key, value in DEFAULT_INPUTS.items():
    st.session_state.setdefault(key, value)

options = metadata.get("dropdown_options", {})
top_categories = metadata.get("top_categories", {})
known_industries = [v for v in top_categories.get("industry", []) if isinstance(v, str)]
known_functions = [v for v in top_categories.get("function", []) if isinstance(v, str)]

# ---------------------------------------------------------------- sidebar ----
with st.sidebar:
    st.header("Model information")
    dataset = metadata.get("dataset", {})
    baseline = metadata.get("baseline", {})
    st.write(f"Dataset rows: **{dataset.get('rows', '?')}**")
    st.write(f"Fraudulent postings: **{dataset.get('fraud_count', '?')}** ({dataset.get('fraud_percent', '?')}%)")
    st.write(f"Hidden layers: **{baseline.get('params', {}).get('hidden_layer_sizes', '?')}**")
    st.write(f"Test accuracy: **{baseline.get('accuracy', '?')}**")
    st.write(f"Test F1 (fraud): **{baseline.get('f1', '?')}**")

    st.divider()
    st.subheader("Sample URLs")
    if st.button("Posting that works", width="stretch"):
        st.session_state["job_url"] = SAMPLE_URL
        st.rerun()
    if st.button("Link that cannot be read", width="stretch"):
        st.session_state["job_url"] = "https://example.com/"
        st.rerun()
    st.caption("The second link has no job posting, so the manual form opens.")

    st.divider()
    st.subheader("Manual examples")
    st.caption("These fill the manual form (fallback).")
    if st.button("Genuine posting", width="stretch"):
        st.session_state["manual_mode"] = True
        prefill(EXAMPLES["Genuine posting"])
        st.rerun()
    if st.button("Suspicious posting", width="stretch"):
        st.session_state["manual_mode"] = True
        prefill(EXAMPLES["Suspicious posting"])
        st.rerun()

    st.divider()
    st.caption(
        "Student mini-project. The prediction is an estimate based on the training "
        "dataset and cannot prove that a job posting is fraudulent or genuine."
    )


# ==================================================== 1. URL (primary flow) ===
with st.container(border=True):
    st.subheader("1. Paste a job posting URL")
    url_value = st.text_input(
        "Job posting URL",
        key="job_url",
        placeholder="https://www.example.com/careers/job/12345",
        label_visibility="collapsed",
    )
    url_col, manual_col = st.columns([3, 2])
    with url_col:
        fetch_clicked = st.button("🔎 Fetch & Analyze", type="primary", width="stretch")
    with manual_col:
        manual_clicked = st.button("✍️ Enter details manually", width="stretch")
    st.caption(
        "Only public pages are read. Pages that need a login, a CAPTCHA or a paywall are "
        "never bypassed — you will be asked to paste the details instead."
    )

if manual_clicked:
    st.session_state["manual_mode"] = True

if fetch_clicked:
    with st.spinner("Fetching the page and extracting the job posting..."):
        result = je.extract_job(
            url_value, allowed_industries=known_industries, allowed_functions=known_functions
        )
    if result.ok:
        with st.spinner("Running the saved model..."):
            prediction = predict_fields(model, result.fields)
        st.session_state["analysis"] = {
            "mode": "url",
            "url": result.url,
            "site": result.site,
            "page_title": result.page_title,
            "extraction_method": result.used,
            "warnings": result.warnings,
            "fields": result.fields,
            **prediction,
        }
        st.session_state["fetch_error"] = None
        st.session_state["manual_mode"] = False
    else:
        st.session_state["analysis"] = None
        st.session_state["fetch_error"] = {"url": result.url, "message": result.message}
        st.session_state["manual_mode"] = True


# ============================================== 2. Extraction failure notice ===
fetch_error = st.session_state.get("fetch_error")
if fetch_error:
    st.warning(
        "**We could not read a job posting from that link.**\n\n"
        f"{fetch_error['message']}\n\n"
        "You can paste the job details manually below — the same model will be used."
    )


# =================================================== 3. Prediction + details ===
analysis = st.session_state.get("analysis")

if analysis:
    st.divider()
    if analysis["mode"] == "url":
        st.caption(
            f"📄 Analyzing **{analysis.get('page_title') or analysis.get('url')}**"
            + (f" · {analysis['site']}" if analysis.get("site") else "")
        )
    else:
        st.caption("📝 Analyzing the details you entered manually")

    show_prediction(analysis)

    if analysis["mode"] == "url":
        missing = analysis.get("warnings") or []
        if missing:
            with st.expander(f"ℹ️ {len(missing)} details were not found on the page (prediction still ran)"):
                st.write("These values were treated as missing by the preprocessing pipeline:")
                for item in missing:
                    st.write(f"- {item}")

        st.subheader("Extracted job details")
        st.caption(f"Source page: {analysis.get('site') or 'unknown site'} — {analysis['url']}")
        if analysis.get("extraction_method"):
            st.caption(f"Extraction method: {analysis['extraction_method']}")

        rows = []
        for key, label in FEATURE_LABELS:
            value = analysis["fields"].get(key, "")
            if key in ("description", "requirements", "company_profile"):
                text = str(value or "")
                shown = f"{len(text.split())} words" if text.strip() else "not found"
            elif key in ("telecommuting", "has_company_logo", "has_questions"):
                shown = "Yes" if value else "No"
            else:
                shown = str(value) if str(value).strip() else "not found"
            rows.append({"Field": label, "Extracted value": shown})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

        with st.expander("Read the extracted text"):
            for key, label in (
                ("description", "Job description"),
                ("requirements", "Requirements"),
                ("company_profile", "Company profile"),
            ):
                st.markdown(f"**{label}**")
                text = str(analysis["fields"].get(key) or "").strip()
                st.text(text if text else "(nothing found on the page)")

    with st.expander("What the model saw (feature summary)"):
        st.dataframe(feature_summary(analysis["input_df"]), hide_index=True, width="stretch")


# ========================================== 4. Manual entry (fallback only) ===
if st.session_state.get("manual_mode"):
    st.divider()
    st.subheader("2. Enter the job posting details manually")
    st.caption(
        "Use this when the link cannot be read. Only the job description is required; "
        "anything you leave empty is treated as a missing value, exactly like during training."
    )

    def with_current(column: str, fallback: list) -> list:
        """Options list that never loses the currently selected value."""
        listed = options.get(column) or fallback
        current = st.session_state.get(column)
        if current and current not in listed:
            listed = listed + [current]
        return listed

    def choice_list(column: str, fallback: list, current: str = "") -> list:
        """Dropdown options taken from the training data, plus an 'Unknown' entry."""
        values = top_categories.get(column) or options.get(column) or fallback
        values = [v for v in values if isinstance(v, str)]
        if current and current not in values:
            values = values + [current]
        return ["Unknown"] + values

    with st.form("manual_form"):
        col1, col2 = st.columns(2)
        with col1:
            employment_options = with_current(
                "employment_type", ["Full-time", "Part-time", "Contract", "Temporary", "Other"]
            )
            employment_type = st.selectbox(
                "Employment type", employment_options, key="employment_type"
            )
            experience_options = with_current(
                "required_experience",
                ["Entry level", "Mid-Senior level", "Associate", "Director"],
            )
            required_experience = st.selectbox(
                "Required experience", experience_options, key="required_experience"
            )
        with col2:
            education_options = with_current(
                "required_education",
                ["Bachelor's Degree", "Master's Degree", "High School or equivalent"],
            )
            required_education = st.selectbox(
                "Required education", education_options, key="required_education"
            )
            salary_range = st.text_input(
                "Salary range (optional, e.g. 50000-70000)", key="salary_range"
            )

        col3, col4 = st.columns(2)
        with col3:
            industry = st.selectbox(
                "Industry",
                choice_list(
                    "industry",
                    ["Information Technology and Services"],
                    st.session_state.get("industry", ""),
                ),
                key="industry",
            )
        with col4:
            job_function = st.selectbox(
                "Function",
                choice_list(
                    "function",
                    ["Information Technology"],
                    st.session_state.get("job_function", ""),
                ),
                key="job_function",
            )

        st.write("**Flags**")
        col5, col6, col7 = st.columns(3)
        with col5:
            has_company_logo = st.checkbox("Company logo present", key="has_company_logo")
        with col6:
            has_questions = st.checkbox("Posting has screening questions", key="has_questions")
        with col7:
            telecommuting = st.checkbox("Telecommuting / work from home", key="telecommuting")

        description = st.text_area(
            "Job description",
            key="description",
            height=170,
            placeholder="Paste the job description here...",
        )
        requirements = st.text_area(
            "Requirements",
            key="requirements",
            height=110,
            placeholder="Paste the requirements here...",
        )
        company_profile = st.text_area(
            "Company profile",
            key="company_profile",
            height=110,
            placeholder="Paste the company profile here...",
        )

        submitted = st.form_submit_button("Check this job posting", width="stretch")

    if submitted:
        if not description.strip():
            st.warning(
                "Please paste at least a short job description — the model uses its length as a feature."
            )
        else:
            fields = {
                "employment_type": employment_type,
                "required_experience": required_experience,
                "required_education": required_education,
                "industry": "" if industry == "Unknown" else industry,
                "function": "" if job_function == "Unknown" else job_function,
                "description": description,
                "requirements": requirements,
                "company_profile": company_profile,
                "telecommuting": int(telecommuting),
                "has_company_logo": int(has_company_logo),
                "has_questions": int(has_questions),
                "salary_range": salary_range,
            }
            prediction = predict_fields(model, fields)
            st.session_state["analysis"] = {
                "mode": "manual",
                "url": "",
                "site": "",
                "page_title": "",
                "extraction_method": "",
                "warnings": [],
                "fields": {**je.empty_fields(), **fields},
                **prediction,
            }
            st.session_state["fetch_error"] = None
            st.rerun()

st.divider()
st.caption(
    "**Disclaimer:** this application is a college mini-project. It gives an estimate "
    "only and cannot prove that a job posting is fraudulent. Always verify the employer "
    "independently before sharing personal information or paying any money."
)
