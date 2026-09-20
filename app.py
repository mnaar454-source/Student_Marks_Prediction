
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import joblib
import os


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Student Marks Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CONSTANTS
# ============================================================

MODEL_PATH = os.path.join(
    "models",
    "student_marks_predictor.joblib"
)

FEATURE_COLUMNS = [
    "gender",
    "race/ethnicity",
    "parental level of education",
    "lunch",
    "test preparation course",
    "reading score",
    "writing score"
]


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown('''
<style>

@import url("https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap");

html, body, [class*="css"] {
    font-family: "Inter", sans-serif;
}

.stApp {
    background: linear-gradient(
        135deg,
        #f8fafc 0%,
        #eef2ff 48%,
        #f5f3ff 100%
    );
}

.block-container {
    max-width: 1400px;
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

/* Hide Deploy */
.stDeployButton {
    display: none !important;
}

/* Hide footer */
footer {
    visibility: hidden;
}


/* ============================================================
   HERO
   ============================================================ */

.hero {
    position: relative;
    overflow: hidden;
    padding: 34px 38px;
    border-radius: 28px;
    margin-bottom: 25px;
    background: linear-gradient(
        135deg,
        #312e81 0%,
        #4f46e5 48%,
        #7c3aed 100%
    );
    color: white;
    box-shadow: 0 18px 45px rgba(79,70,229,0.25);
}

.hero:before {
    content: "";
    position: absolute;
    width: 230px;
    height: 230px;
    border-radius: 50%;
    right: -70px;
    top: -90px;
    background: rgba(255,255,255,0.12);
}

.hero:after {
    content: "";
    position: absolute;
    width: 160px;
    height: 160px;
    border-radius: 50%;
    left: 45%;
    bottom: -100px;
    background: rgba(255,255,255,0.08);
}

.hero-content {
    position: relative;
    z-index: 2;
}

.hero h1 {
    margin: 0;
    font-size: 2.55rem;
    font-weight: 800;
    letter-spacing: -1px;
}

.hero p {
    margin-top: 10px;
    margin-bottom: 0;
    font-size: 1.05rem;
    opacity: 0.92;
}


/* ============================================================
   HERO STATS
   ============================================================ */

.stats-strip {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
    margin-top: 25px;
}

.stat-box {
    padding: 14px 16px;
    border-radius: 16px;
    background: rgba(255,255,255,0.13);
    border: 1px solid rgba(255,255,255,0.18);
    backdrop-filter: blur(8px);
}

.stat-label {
    font-size: 0.72rem;
    opacity: 0.75;
    text-transform: uppercase;
    letter-spacing: 0.7px;
}

.stat-value {
    margin-top: 4px;
    font-size: 0.98rem;
    font-weight: 700;
}


/* ============================================================
   SECTION TITLES
   ============================================================ */

.section-title {
    font-size: 1.08rem;
    font-weight: 750;
    color: #1e293b;
    margin: 18px 0 14px 0;
}


/* ============================================================
   CARDS
   ============================================================ */

.section-card {
    background: rgba(255,255,255,0.88);
    border: 1px solid rgba(148,163,184,0.18);
    border-radius: 22px;
    padding: 22px;
    margin-bottom: 18px;
    box-shadow: 0 8px 25px rgba(15,23,42,0.06);
}


/* ============================================================
   RESULT CARD
   ============================================================ */

.result-card {
    padding: 30px;
    border-radius: 24px;
    text-align: center;
    background: rgba(255,255,255,0.95);
    box-shadow: 0 15px 40px rgba(15,23,42,0.08);
}

.score-label {
    color: #64748b;
    font-size: 0.95rem;
}

.score-number {
    font-size: 4.5rem;
    line-height: 1;
    font-weight: 800;
    color: #4f46e5;
    margin: 13px 0;
}

.grade-badge {
    display: inline-block;
    padding: 8px 22px;
    border-radius: 999px;
    font-weight: 800;
    font-size: 1rem;
}

.grade-a {
    background: #dcfce7;
    color: #166534;
}

.grade-b {
    background: #dbeafe;
    color: #1d4ed8;
}

.grade-c {
    background: #ccfbf1;
    color: #0f766e;
}

.grade-d {
    background: #fef3c7;
    color: #b45309;
}

.grade-f {
    background: #fee2e2;
    color: #b91c1c;
}


/* ============================================================
   STUDY TIPS
   ============================================================ */

.tip-card {
    background: #f8fafc;
    border-left: 4px solid #6366f1;
    padding: 12px 15px;
    border-radius: 12px;
    margin: 8px 0;
    color: #334155;
}


/* ============================================================
   INFO
   ============================================================ */

.info-note {
    color: #64748b;
    font-size: 0.83rem;
    margin-top: 5px;
}


/* ============================================================
   SIDEBAR
   ============================================================ */

[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #eef2ff 0%,
        #f8fafc 100%
    );
}


/* ============================================================
   BUTTONS
   ============================================================ */

.stButton > button {
    border-radius: 12px;
    font-weight: 600;
    min-height: 42px;
    border: 1px solid #cbd5e1;
}

.stButton > button[kind="primary"] {
    background: linear-gradient(
        135deg,
        #4f46e5,
        #7c3aed
    );
    color: white;
    border: none;
}


/* ============================================================
   RESPONSIVE
   ============================================================ */

@media (max-width: 800px) {

    .hero {
        padding: 25px;
    }

    .hero h1 {
        font-size: 1.9rem;
    }

    .stats-strip {
        grid-template-columns: 1fr;
    }

    .score-number {
        font-size: 3.5rem;
    }
}

</style>
''', unsafe_allow_html=True)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):
        return None

    return joblib.load(MODEL_PATH)


model_package = load_model()

if model_package is None:

    st.error(
        "❌ Model file not found. Please make sure "
        "`models/student_marks_predictor.joblib` exists."
    )

    st.stop()


pipeline = model_package["pipeline"]
target_column = model_package["target_column"]
model_name = model_package["model_name"]
task_type = model_package["task_type"]


# ============================================================
# DEFAULT STUDENT
# ============================================================

default_student = {
    "gender": "female",
    "race/ethnicity": "group B",
    "parental level of education": "bachelor's degree",
    "lunch": "standard",
    "test preparation course": "completed",
    "reading score": 70,
    "writing score": 68
}


if "student" not in st.session_state:

    st.session_state.student = default_student.copy()


def load_example(example):

    st.session_state.student = example.copy()


def reset_form():

    st.session_state.student = default_student.copy()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "<h2>🎛️ Quick Controls</h2>",
        unsafe_allow_html=True
    )

    st.caption(
        "Load a sample student profile or reset the form."
    )

    st.markdown("### 👤 Example Students")

    if st.button(
        "🏆 High Performer",
        width="stretch"
    ):

        load_example({
            "gender": "female",
            "race/ethnicity": "group E",
            "parental level of education": "master's degree",
            "lunch": "standard",
            "test preparation course": "completed",
            "reading score": 92,
            "writing score": 94
        })

        st.rerun()


    if st.button(
        "📘 Average Student",
        width="stretch"
    ):

        load_example({
            "gender": "male",
            "race/ethnicity": "group C",
            "parental level of education": "some college",
            "lunch": "standard",
            "test preparation course": "none",
            "reading score": 65,
            "writing score": 62
        })

        st.rerun()


    if st.button(
        "🆘 Needs Support",
        width="stretch"
    ):

        load_example({
            "gender": "female",
            "race/ethnicity": "group A",
            "parental level of education": "high school",
            "lunch": "free/reduced",
            "test preparation course": "none",
            "reading score": 42,
            "writing score": 38
        })

        st.rerun()


    st.divider()


    if st.button(
        "🔄 Reset Form",
        width="stretch"
    ):

        reset_form()
        st.rerun()


    st.divider()

    st.markdown("### ℹ️ Project Information")

    st.write(
        "This application predicts a student's "
        "Mathematics score using the trained "
        "machine-learning model."
    )

    st.caption("Model")
    st.write(model_name)

    st.caption("Target")
    st.write(target_column)

    st.caption("Task")
    st.write(task_type.title())


# ============================================================
# HERO SECTION
# IMPORTANT: HTML STARTS AT COLUMN 1
# ============================================================

hero_html = '''
<div class="hero">
<div class="hero-content">

<h1>🎓 Student Marks Predictor</h1>

<p>
Estimate Mathematics performance from student
demographic and academic information.
</p>

<div class="stats-strip">

<div class="stat-box">
<div class="stat-label">Model</div>
<div class="stat-value">Linear Regression</div>
</div>

<div class="stat-box">
<div class="stat-label">Target</div>
<div class="stat-value">Math Score</div>
</div>

<div class="stat-box">
<div class="stat-label">Task</div>
<div class="stat-value">Regression</div>
</div>

</div>

</div>
</div>
'''

st.markdown(
    hero_html,
    unsafe_allow_html=True
)


# ============================================================
# STEP 1: STUDENT INFORMATION
# ============================================================

st.markdown(
    "<div class='section-title'>📝 Student Information</div>",
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)


with col1:

    gender_options = [
        "female",
        "male"
    ]

    gender = st.selectbox(
        "Gender",
        gender_options,
        index=gender_options.index(
            st.session_state.student["gender"]
        ),
        help="Select the student's gender."
    )


    race_options = [
        "group A",
        "group B",
        "group C",
        "group D",
        "group E"
    ]

    race = st.selectbox(
        "Race / Ethnicity",
        race_options,
        index=race_options.index(
            st.session_state.student["race/ethnicity"]
        ),
        help="Select the student's race/ethnicity group."
    )


    education_options = [
        "some high school",
        "high school",
        "some college",
        "associate's degree",
        "bachelor's degree",
        "master's degree"
    ]

    education = st.selectbox(
        "Parental Level of Education",
        education_options,
        index=education_options.index(
            st.session_state.student[
                "parental level of education"
            ]
        ),
        help="Select the highest parental education level."
    )


with col2:

    lunch_options = [
        "standard",
        "free/reduced"
    ]

    lunch = st.selectbox(
        "Lunch",
        lunch_options,
        index=lunch_options.index(
            st.session_state.student["lunch"]
        ),
        help="Select the student's lunch category."
    )


    preparation_options = [
        "none",
        "completed"
    ]

    preparation = st.selectbox(
        "Test Preparation Course",
        preparation_options,
        index=preparation_options.index(
            st.session_state.student[
                "test preparation course"
            ]
        ),
        help="Select whether the preparation course was completed."
    )


# ============================================================
# STEP 2: ACADEMIC SCORES
# ============================================================

st.markdown(
    "<div class='section-title'>📊 Academic Scores</div>",
    unsafe_allow_html=True
)

score_col1, score_col2 = st.columns(2)


with score_col1:

    reading = st.slider(
        "📖 Reading Score",
        min_value=0,
        max_value=100,
        value=int(
            st.session_state.student["reading score"]
        ),
        step=1,
        help="Enter the student's Reading score from 0 to 100."
    )


with score_col2:

    writing = st.slider(
        "✍️ Writing Score",
        min_value=0,
        max_value=100,
        value=int(
            st.session_state.student["writing score"]
        ),
        step=1,
        help="Enter the student's Writing score from 0 to 100."
    )


reading_writing_average = (
    reading + writing
) / 2

score_difference = abs(
    reading - writing
)


st.markdown(
    "<div class='info-note'>"
    f"📌 Reading + Writing average: "
    f"<b>{reading_writing_average:.1f}</b>"
    "</div>",
    unsafe_allow_html=True
)


if score_difference > 30:

    st.warning(
        "⚠️ Reading and Writing scores differ by more than "
        "30 points. Please check the entered values."
    )


# ============================================================
# UPDATE SESSION STATE
# ============================================================

st.session_state.student = {
    "gender": gender,
    "race/ethnicity": race,
    "parental level of education": education,
    "lunch": lunch,
    "test preparation course": preparation,
    "reading score": reading,
    "writing score": writing
}


# ============================================================
# STEP 3: PREDICTION
# ============================================================

st.markdown(
    "<div class='section-title'>🚀 Generate Prediction</div>",
    unsafe_allow_html=True
)


predict_clicked = st.button(
    "🔮 Predict Mathematics Score",
    type="primary",
    width="stretch"
)


if predict_clicked:

    input_data = pd.DataFrame([
        st.session_state.student
    ])


    missing_columns = [
        column
        for column in FEATURE_COLUMNS
        if column not in input_data.columns
    ]


    if missing_columns:

        st.error(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

        st.stop()


    try:

        with st.spinner(
            "🤖 Analyzing student information..."
        ):

            prediction = pipeline.predict(
                input_data[FEATURE_COLUMNS]
            )[0]


        prediction = float(
            max(
                0,
                min(
                    100,
                    prediction
                )
            )
        )


        st.session_state.prediction = prediction


    except Exception as error:

        st.error(
            "Prediction failed. Please check the saved "
            "model and scikit-learn version."
        )

        st.exception(error)
        st.stop()


# ============================================================
# RESULTS
# ============================================================

if "prediction" in st.session_state:

    prediction = st.session_state.prediction


    # ========================================================
    # GRADE
    # ========================================================

    if prediction >= 90:

        grade = "A"
        grade_class = "grade-a"

        message = (
            "Excellent performance! Keep maintaining "
            "this strong academic consistency."
        )


    elif prediction >= 80:

        grade = "B"
        grade_class = "grade-b"

        message = (
            "Very good performance. A little more focused "
            "practice can strengthen the result further."
        )


    elif prediction >= 70:

        grade = "C"
        grade_class = "grade-c"

        message = (
            "Good progress. Consistent revision and "
            "practice can help improve the score."
        )


    elif prediction >= 60:

        grade = "D"
        grade_class = "grade-d"

        message = (
            "There is room for improvement. Focus on "
            "fundamental concepts and regular practice."
        )


    else:

        grade = "F"
        grade_class = "grade-f"

        message = (
            "Consider building a structured study routine "
            "and getting additional academic support."
        )


    # ========================================================
    # TABS
    # ========================================================

    result_tab, charts_tab, summary_tab = st.tabs([
        "🎯 Result",
        "📈 Charts",
        "👤 Student Summary"
    ])


    # ========================================================
    # RESULT TAB
    # ========================================================

    with result_tab:

        st.markdown(
            "<div class='result-card'>"
            "<div class='score-label'>"
            "Predicted Mathematics Score"
            "</div>"
            f"<div class='score-number'>{prediction:.1f}</div>"
            f"<div class='grade-badge {grade_class}'>"
            f"Grade {grade}"
            "</div>"
            f"<p>{message}</p>"
            "</div>",
            unsafe_allow_html=True
        )


        st.write("")


        metric1, metric2, metric3, metric4 = st.columns(4)


        with metric1:

            st.metric(
                "🎯 Predicted Math",
                f"{prediction:.1f}"
            )


        with metric2:

            st.metric(
                "📖 Reading",
                f"{reading}"
            )


        with metric3:

            st.metric(
                "✍️ Writing",
                f"{writing}"
            )


        with metric4:

            difference_from_average = (
                prediction -
                reading_writing_average
            )

            st.metric(
                "📊 Math vs R/W Avg",
                f"{difference_from_average:+.1f}"
            )


        # ====================================================
        # PERSONALIZED TIPS
        # ====================================================

        st.markdown(
            "<div class='section-card'>"
            "<div class='section-title'>"
            "💡 Personalized Study Tips"
            "</div>",
            unsafe_allow_html=True
        )


        tips = []


        if prediction < 70:

            tips.append(
                "📚 Practice Mathematics fundamentals regularly."
            )


        if reading < writing:

            tips.append(
                "📖 Spend extra time improving reading comprehension."
            )


        elif writing < reading:

            tips.append(
                "✍️ Practice structured writing and answer presentation."
            )


        if abs(reading - writing) <= 10:

            tips.append(
                "⚖️ Reading and Writing performance is relatively balanced."
            )


        if prediction >= 80:

            tips.append(
                "🏆 Continue revision and timed practice to maintain consistency."
            )


        if not tips:

            tips.append(
                "🎯 Maintain a consistent study schedule and review mistakes."
            )


        for tip in tips[:3]:

            st.markdown(
                f"<div class='tip-card'>{tip}</div>",
                unsafe_allow_html=True
            )


        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )


        st.caption(
            "ℹ️ This is an estimate generated by the trained "
            f"{model_name} model. It is not a guaranteed examination result."
        )


        if prediction >= 90:

            st.balloons()


    # ========================================================
    # CHARTS TAB
    # ========================================================

    with charts_tab:

        st.markdown(
            "<div class='section-card'>"
            "<div class='section-title'>"
            "📊 Score Comparison"
            "</div>",
            unsafe_allow_html=True
        )


        score_chart = go.Figure()


        score_chart.add_trace(
            go.Bar(
                x=["Reading"],
                y=[reading],
                name="Reading",
                marker_color="#6366f1",
                text=[reading],
                textposition="outside"
            )
        )


        score_chart.add_trace(
            go.Bar(
                x=["Writing"],
                y=[writing],
                name="Writing",
                marker_color="#14b8a6",
                text=[writing],
                textposition="outside"
            )
        )


        score_chart.add_trace(
            go.Bar(
                x=["Predicted Math"],
                y=[prediction],
                name="Predicted Math",
                marker_color="#f59e0b",
                text=[f"{prediction:.1f}"],
                textposition="outside"
            )
        )


        score_chart.update_layout(
            yaxis=dict(
                range=[0, 100],
                title="Score"
            ),
            xaxis_title="Subject",
            showlegend=False,
            height=430,
            margin=dict(
                l=20,
                r=20,
                t=30,
                b=20
            ),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)"
        )


        st.plotly_chart(
            score_chart,
            width="stretch"
        )


        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )


        # ====================================================
        # PERFORMANCE GAUGE
        # ====================================================

        st.markdown(
            "<div class='section-card'>"
            "<div class='section-title'>"
            "🎯 Performance Gauge"
            "</div>",
            unsafe_allow_html=True
        )


        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=prediction,
                number={
                    "suffix": "/100",
                    "font": {
                        "size": 36
                    }
                },
                gauge={
                    "axis": {
                        "range": [0, 100]
                    },
                    "bar": {
                        "color": "#4f46e5",
                        "thickness": 0.18
                    },
                    "bgcolor": "rgba(0,0,0,0)",
                    "borderwidth": 0,
                    "steps": [
                        {
                            "range": [0, 50],
                            "color": "#fee2e2"
                        },
                        {
                            "range": [50, 70],
                            "color": "#fef3c7"
                        },
                        {
                            "range": [70, 85],
                            "color": "#ccfbf1"
                        },
                        {
                            "range": [85, 100],
                            "color": "#dcfce7"
                        }
                    ]
                }
            )
        )


        gauge.update_layout(
            height=330,
            margin=dict(
                l=30,
                r=30,
                t=25,
                b=20
            ),
            paper_bgcolor="rgba(0,0,0,0)"
        )


        st.plotly_chart(
            gauge,
            width="stretch"
        )


        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )


    # ========================================================
    # STUDENT SUMMARY TAB
    # ========================================================

    with summary_tab:

        st.markdown(
            "<div class='section-card'>"
            "<div class='section-title'>"
            "👤 Student Profile"
            "</div>",
            unsafe_allow_html=True
        )


        summary_col1, summary_col2 = st.columns(2)


        with summary_col1:

            st.write(
                f"**Gender:** {gender}"
            )

            st.write(
                f"**Race / Ethnicity:** {race}"
            )

            st.write(
                f"**Parental Education:** {education}"
            )

            st.write(
                f"**Lunch:** {lunch}"
            )


        with summary_col2:

            st.write(
                f"**Test Preparation:** {preparation}"
            )

            st.write(
                f"**Reading Score:** {reading}"
            )

            st.write(
                f"**Writing Score:** {writing}"
            )

            st.write(
                f"**Predicted Math Score:** {prediction:.1f}"
            )


        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )


        # ====================================================
        # CSV REPORT
        # ====================================================

        report_data = pd.DataFrame([{

            "Gender": gender,

            "Race/Ethnicity": race,

            "Parental Level of Education":
                education,

            "Lunch": lunch,

            "Test Preparation Course":
                preparation,

            "Reading Score":
                reading,

            "Writing Score":
                writing,

            "Predicted Math Score":
                round(prediction, 2),

            "Grade":
                grade

        }])


        csv_data = report_data.to_csv(
            index=False
        )


        st.download_button(
            "📥 Download Student Report (CSV)",
            data=csv_data,
            file_name="student_prediction_report.csv",
            mime="text/csv",
            width="stretch"
        )


        # ====================================================
        # TEXT REPORT
        # ====================================================

        text_report = (
            "STUDENT MARKS PREDICTION REPORT\n"
            "================================\n\n"
            f"Gender: {gender}\n"
            f"Race/Ethnicity: {race}\n"
            f"Parental Education: {education}\n"
            f"Lunch: {lunch}\n"
            f"Test Preparation: {preparation}\n"
            f"Reading Score: {reading}\n"
            f"Writing Score: {writing}\n"
            f"Predicted Math Score: {prediction:.2f}\n"
            f"Grade: {grade}\n\n"
            f"Model: {model_name}\n"
            f"Task: {task_type}\n"
        )


        st.download_button(
            "📄 Download Summary (TXT)",
            data=text_report,
            file_name="student_prediction_summary.txt",
            mime="text/plain",
            width="stretch"
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    "<div style='text-align:center; "
    "padding:25px 0 10px 0; "
    "color:#64748b; "
    "font-size:0.82rem;'>"
    "🎓 Student Marks Predictor • "
    "Machine Learning Academic Project"
    "</div>",
    unsafe_allow_html=True
)

