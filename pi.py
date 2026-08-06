import streamlit as st
import math as mt

st.markdown("""
<style>

/* ===========================
   APP BACKGROUND
=========================== */
[data-testid="stAppViewContainer"]{
    background:#f8fafc;
}

/* ===========================
   MAIN CONTAINER
=========================== */
.main .block-container{
    background:white;
    padding:2rem;
    border-radius:20px;
    box-shadow:0 8px 25px rgba(0,0,0,.08);
    margin-top:20px;
    margin-bottom:20px;
}

/* ===========================
   MAIN PAGE TEXT
=========================== */
h1{
    color:#1e3a8a;
    text-align:center;
    font-size:48px;
    font-weight:800;
}

h2,h3,p,label{
    color:black;
}

/* ===========================
   SIDEBAR
=========================== */
[data-testid="stSidebar"]{
    background:#0f172a;
}

/* Sidebar text */
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown{
    color:white !important;
}

/* ===========================
   SELECTBOX
=========================== */

/* Closed selectbox */
.stSelectbox > div > div{
    background:white !important;
    border-radius:12px;
}

/* Selected value (⚪ Circle) */
.stSelectbox div[data-baseweb="select"]{
    color:black !important;
}

.stSelectbox div[data-baseweb="select"] *{
    color:black !important;
}

/* Dropdown menu */
div[role="listbox"]{
    background:white !important;
}

/* Dropdown options */
div[role="option"]{
    background:white !important;
    color:black !important;
}

/* Hover */
div[role="option"]:hover{
    background:#f1f5f9 !important;
    color:black !important;
}

/* Selected option */
div[role="option"][aria-selected="true"]{
    background:#dbeafe !important;
    color:black !important;
}

/* ===========================
   INPUT BOXES
=========================== */
div[data-baseweb="input"]{
    border-radius:12px;
    box-shadow:0 2px 8px rgba(0,0,0,.08);
}

/* ===========================
   BUTTONS
=========================== */
.stButton > button{
    width:100%;
    background:#2563eb;
    color:white !important;
    border:none;
    border-radius:12px;
    padding:12px;
    font-weight:bold;
}

.stButton > button:hover{
    background:#1d4ed8;
}

/* ===========================
   RESULT BOXES
=========================== */
div[data-testid="stAlert"]{
    background:#dcfce7;
    border-left:6px solid #22c55e;
    border-radius:12px;
}

div[data-testid="stAlert"] *{
    color:#166534 !important;
}

</style>
""", unsafe_allow_html=True)

pi = mt.pi  # More accurate than 22/7

st.set_page_config(page_title="Shape Calculator", page_icon="📐")

st.title("📐 Shape Calculator")

st.sidebar.title("⚙️Control Panel")
import streamlit as st

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown(
        """
        # 📐 Shape Calculator
        ### Geometry Made Easy
        
        ---
        """
    )

    option = st.selectbox(
        "🧩 Choose a Shape",
        [
            "⚪ Circle",
            "▭ Rectangle",
            "🟨 Square",
            "🔺 Triangle",
            "▱ Parallelogram",
            "⬠ Trapezoid",
            "💎 Rhombus",
        ],
    )

    st.divider()

    st.info(
        """
        **Features**
        - 📏 Area
        - 📐 Perimeter
        - ⭕ Circumference
        """
    )

    st.divider()

    st.caption("Made with ❤️ using Streamlit")

st.subheader(f"{option} Calculator")

if option == "⚪ Circle":
    radius = st.number_input("Enter radius", min_value=0.0, value=0.0, step=0.1)

    area = pi * radius ** 2
    circumference = 2 * pi * radius

    st.success(f"Area: {area:.2f}")
    st.success(f"Circumference: {circumference:.2f}")

elif option == "▭ Rectangle":
    length = st.number_input("Enter length", min_value=0.0, value=0.0, step=0.1)
    width = st.number_input("Enter width", min_value=0.0, value=0.0, step=0.1)

    area = length * width
    perimeter = 2 * (length + width)

    st.success(f"Area: {area:.2f}")
    st.success(f"Perimeter: {perimeter:.2f}")

elif option == "🟨 Square":
    side = st.number_input("Enter side length", min_value=0.0, value=0.0, step=0.1)

    area = side ** 2
    perimeter = 4 * side

    st.success(f"Area: {area:.2f}")
    st.success(f"Perimeter: {perimeter:.2f}")

elif option == "🔺 Triangle":
    base = st.number_input("Enter base", min_value=0.0, value=0.0, step=0.1)
    height = st.number_input("Enter height", min_value=0.0, value=0.0, step=0.1)
    side1 = st.number_input("Enter side 1", min_value=0.0, value=0.0, step=0.1)
    side2 = st.number_input("Enter side 2", min_value=0.0, value=0.0, step=0.1)

    area = 0.5 * base * height
    perimeter = base + side1 + side2

    st.success(f"Area: {area:.2f}")
    st.success(f"Perimeter: {perimeter:.2f}")

elif option == "▱ Parallelogram":
    base = st.number_input("Enter base", min_value=0.0, value=0.0, step=0.1)
    height = st.number_input("Enter height", min_value=0.0, value=0.0, step=0.1)
    side = st.number_input("Enter side length", min_value=0.0, value=0.0, step=0.1)

    area = base * height
    perimeter = 2 * (base + side)

    st.success(f"Area: {area:.2f}")
    st.success(f"Perimeter: {perimeter:.2f}")

elif option == "⬠ Trapezoid":
    base1 = st.number_input("Enter base 1", min_value=0.0, value=0.0, step=0.1)
    base2 = st.number_input("Enter base 2", min_value=0.0, value=0.0, step=0.1)
    height = st.number_input("Enter height", min_value=0.0, value=0.0, step=0.1)
    side1 = st.number_input("Enter side 1", min_value=0.0, value=0.0, step=0.1)
    side2 = st.number_input("Enter side 2", min_value=0.0, value=0.0, step=0.1)

    area = ((base1 + base2) / 2) * height
    perimeter = base1 + base2 + side1 + side2

    st.success(f"Area: {area:.2f}")
    st.success(f"Perimeter: {perimeter:.2f}")

elif option == "💎 Rhombus":
    diagonal1 = st.number_input("Enter diagonal 1", min_value=0.0, value=0.0, step=0.1)
    diagonal2 = st.number_input("Enter diagonal 2", min_value=0.0, value=0.0, step=0.1)
    side = st.number_input("Enter side length", min_value=0.0, value=0.0, step=0.1)

    area = (diagonal1 * diagonal2) / 2
    perimeter = 4 * side

    st.success(f"Area: {area:.2f}")
    st.success(f"Perimeter: {perimeter:.2f}")