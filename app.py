import os
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

from sklearn.model_selection import train_test_split

# =====================================================
# CONFIG PAGE
# =====================================================

st.set_page_config(
    page_title="Analisis Sentimen Mobile JKN",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# LOAD MODEL
# =====================================================

model = joblib.load("model_sentimen.pkl")

# =====================================================
# DEFAULT VARIABLE
# =====================================================

submit = False
clear = False

# =====================================================
# FILE USER
# =====================================================

USER_FILE = "users.csv"

if not os.path.exists(USER_FILE):

    pd.DataFrame(
        columns=["username", "password"]
    ).to_csv(USER_FILE, index=False)

# =====================================================
# SESSION STATE
# =====================================================

if "login_status" not in st.session_state:
    st.session_state.login_status = False

if "username" not in st.session_state:
    st.session_state.username = ""

if "riwayat" not in st.session_state:
    st.session_state.riwayat = []

# =====================================================
# FUNCTION REGISTER
# =====================================================

def register_user(username, password):

    users = pd.read_csv(
        USER_FILE,
        dtype=str
    )

    username = str(username).strip()
    password = str(password).strip()

    users["username"] = users["username"].str.strip()

    # cek username
    if username in users["username"].values:
        return False, "Username sudah digunakan!"

    new_user = pd.DataFrame({
        "username": [username],
        "password": [password]
    })

    users = pd.concat(
        [users, new_user],
        ignore_index=True
    )

    users.to_csv(
        USER_FILE,
        index=False
    )

    return True, "Register berhasil!"

# =====================================================
# FUNCTION LOGIN
# =====================================================

def login_user(username, password):

    users = pd.read_csv(
        USER_FILE,
        dtype=str
    )

    username = str(username).strip()
    password = str(password).strip()

    users["username"] = users["username"].str.strip()
    users["password"] = users["password"].str.strip()

    user = users[
        (users["username"] == username) &
        (users["password"] == password)
    ]

    return not user.empty

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>

/* =====================================================
GLOBAL
===================================================== */

html, body, [class*="css"] {
    font-family: 'Segoe UI', sans-serif;
}

/* =====================================================
BACKGROUND
===================================================== */

.stApp{
    background: linear-gradient(
        135deg,
        #f6f1eb,
        #efe6dc,
        #e7ddd1
    );
    background-attachment: fixed;
}

/* =====================================================
HIDE STREAMLIT
===================================================== */

#MainMenu{
    visibility:hidden;
}

footer{
    visibility:hidden;
}

header{
    visibility:hidden;
}

/* =====================================================
SIDEBAR
===================================================== */

section[data-testid="stSidebar"]{
    background: #b08968 !important;
    width: 290px !important;
    border-right: none !important;
}

section[data-testid="stSidebar"] > div{
    background: #b08968 !important;
}

section[data-testid="stSidebar"] *{
    color:white !important;
}

/* title sidebar */

.sidebar-title{
    color:white;
    font-size:32px;
    font-weight:bold;
    text-align:center;
    margin-top:15px;
    margin-bottom:35px;
}

/* radio title */

.stRadio > label{
    color:white !important;
    font-size:18px !important;
    font-weight:bold;
}

/* menu item */

section[data-testid="stSidebar"] div[role="radiogroup"] > label{
    background: rgba(255,255,255,0.12);
    padding:14px 18px;
    border-radius:15px;
    margin-bottom:14px;
    transition:0.3s;
    border:none;
}

/* hover menu */

section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover{
    background: rgba(255,255,255,0.22);
    transform: translateX(6px);
}

/* =====================================================
INPUT
===================================================== */

.stTextInput input{
    background-color: rgba(255,255,255,0.95);
    border-radius: 14px;
    border:none;
    color:black;
}

.stTextArea textarea{
    background-color: rgba(255,255,255,0.95);
    border-radius: 14px;
    border:none;
    color:black;
}

/* =====================================================
BUTTON
===================================================== */

.stButton > button{
    width:100%;
    height:55px;
    border:none;
    border-radius:15px;
    background: linear-gradient(
        90deg,
        #9c7b5d,
        #b08968
    );
    color:white;
    font-size:18px;
    font-weight:bold;
    transition:0.3s;
}

.stButton > button:hover{
    background: linear-gradient(
        90deg,
        #8c6b50,
        #9c7b5d
    );

    transform: translateY(-2px);

    box-shadow: 0 8px 20px rgba(176,137,104,0.35);
}

/* =====================================================
TEXT COLOR
===================================================== */

h1,h2,h3,h4,h5,h6,p,label{
    color:#5c4634 !important;
}

/* =====================================================
AUTH BOX
===================================================== */

.auth-box{
    background:white;
    padding:45px;
    border-radius:30px;
    box-shadow:0 10px 35px rgba(0,0,0,0.12);

    width:100%;
    max-width:650px;

    margin:auto;
    margin-top:70px;
}

/* =====================================================
CARD
===================================================== */

.form-container{
    background: white;
    padding: 35px;
    border-radius: 20px;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.08);
    margin-top: 20px;
}

/* =====================================================
FOOTER
===================================================== */

.footer{
    text-align:center;
    color:#6b5644;
    margin-top:70px;
    padding:20px;
    font-size:16px;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# LOGIN PAGE
# =====================================================

if not st.session_state.login_status:

    col1, col2, col3 = st.columns([1,1.5,1])

    with col2:

        st.markdown("""
        <div class="auth-box">

        <h1 style='text-align:center;color:#ff4b2b;'>
        LOGIN & REGISTER
        </h1>

        <p style='text-align:center;'>
        Sistem Analisis Sentimen Ulasan Mobile JKN
        </p>

        """, unsafe_allow_html=True)

        auth_menu = st.radio(
            "Pilih Menu",
            ["Login", "Register"],
            horizontal=True
        )

        # LOGIN

        if auth_menu == "Login":

            username = st.text_input(
                "Username"
            )

            password = st.text_input(
                "Password",
                type="password"
            )

            if st.button("SIGN IN"):

                if username == "" or password == "":

                    st.warning(
                        "Isi username dan password!"
                    )

                else:

                    if login_user(
                        username,
                        password
                    ):

                        st.session_state.login_status = True
                        st.session_state.username = username

                        st.success(
                            "Login berhasil!"
                        )

                        st.rerun()

                    else:

                        st.error(
                            "Username atau password salah!"
                        )

        # REGISTER

        else:

            new_user = st.text_input(
                "Username Baru"
            )

            new_pass = st.text_input(
                "Password Baru",
                type="password"
            )

            confirm = st.text_input(
                "Konfirmasi Password",
                type="password"
            )

            if st.button("SIGN UP"):

                if (
                    new_user == "" or
                    new_pass == "" or
                    confirm == ""
                ):

                    st.warning(
                        "Semua field wajib diisi!"
                    )

                elif new_pass != confirm:

                    st.error(
                        "Password tidak sama!"
                    )

                elif len(new_pass) < 4:

                    st.warning(
                        "Password minimal 4 karakter!"
                    )

                else:

                    sukses, msg = register_user(
                        new_user,
                        new_pass
                    )

                    if sukses:
                        st.success(msg)

                    else:
                        st.error(msg)

        st.markdown("</div>", unsafe_allow_html=True)

    st.stop()

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.markdown("""
<div class="sidebar-title">
MENU PAGE
</div>
""", unsafe_allow_html=True)

menu = st.sidebar.radio(
    "PILIH MENU",
    [
        "Home",
        "Prediksi Sentimen",
        "Upload Dataset",
        "Riwayat"
    ]
)

# =====================================================
# USER INFO
# =====================================================

st.sidebar.success(
    f"Login sebagai: {st.session_state.username}"
)

if st.sidebar.button("Logout"):

    st.session_state.login_status = False
    st.session_state.username = ""

    st.rerun()


# =====================================================
# MENU HOME
# =====================================================

if menu == "Home":

    st.container()
    st.markdown("""
    <style>

    .form-container{
        background: white;
        padding: 35px;
        border-radius: 20px;
        box-shadow: 0px 4px 20px rgba(0,0,0,0.08);
        margin-top: 20px;
    }

    .form-title{
        font-size: 32px;
        font-weight: bold;
        color: #c96c6c;
        margin-bottom: 25px;
        text-align: center;
    }

    .result-container{
        background: white;
        padding: 30px;
        border-radius: 20px;
        margin-top: 30px;
        box-shadow: 0px 4px 20px rgba(0,0,0,0.08);
    }

    .result-title{
        font-size: 28px;
        font-weight: bold;
        color: #c96c6c;
        margin-bottom: 20px;
    }

    </style>
    """, unsafe_allow_html=True)

    # =====================================================
    # FORM INPUT
    # =====================================================

    st.markdown("""
    <div class="form-container">
        <div class="form-title">
            <h1>SISTEM ANALISIS SENTIMEN</h1>
                <p>Sistem berbasis Machine Learning untuk menganalisis
    sentimen ulasan pengguna aplikasi Mobile JKN
    menggunakan metode SVM dan TF-IDF</p>
        </div>
    """, unsafe_allow_html=True)


    st.divider()

    # =====================================================
    # TENTANG
    # =====================================================

    st.subheader("✅ TENTANG SISTEM")

    st.write("""
    Sistem ini digunakan untuk melakukan analisis sentimen
    terhadap ulasan pengguna aplikasi Mobile JKN.
    
    Sistem mampu mengelompokkan ulasan menjadi:
    - Positif
    - Negatif
    - Netral
    
    Hasil analisis dapat membantu memahami opini pengguna
    sebagai bahan evaluasi layanan aplikasi.
    """)

    st.divider()

    # =====================================================
    # FITUR
    # =====================================================

    st.subheader("✅ FITUR UTAMA")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.info("""
        🧠 Prediksi Sentimen
        
        Analisis otomatis ulasan pengguna.
        """)

    with col2:
        st.info("""
        📂 Upload Dataset
        
        Upload dataset CSV untuk analisis.
        """)

    with col3:
        st.info("""
        📊 Visualisasi
        
        Grafik sentimen dan confusion matrix.
        """)

    col4, col5, col6 = st.columns(3)

    with col4:
        st.success("""
        📑 Riwayat
        
        Menyimpan hasil prediksi.
        """)

    with col5:
        st.success("""
        ⬇️ Download
        
        Download hasil CSV.
        """)

    with col6:
        st.success("""
        🚀 Akurasi Model
        
        Evaluasi performa model.
        """)

    st.divider()

    # =====================================================
    # TEKNOLOGI
    # =====================================================

    st.subheader("✅ TEKNOLOGI")

    st.write("""
    Streamlit • Python • Pandas •
    Scikit-Learn • SVM • TF-IDF • Matplotlib
    """)

# =====================================================
# MENU PREDIKSI
# =====================================================

elif menu == "Prediksi Sentimen":

    # =====================================================
    # CSS FORM
    # =====================================================

    st.markdown("""
    <style>

    .form-container{
        background: white;
        padding: 35px;
        border-radius: 20px;
        box-shadow: 0px 4px 20px rgba(0,0,0,0.08);
        margin-top: 20px;
    }

    .form-title{
        font-size: 32px;
        font-weight: bold;
        color: #c96c6c;
        margin-bottom: 25px;
        text-align: center;
    }

    .result-container{
        background: white;
        padding: 30px;
        border-radius: 20px;
        margin-top: 30px;
        box-shadow: 0px 4px 20px rgba(0,0,0,0.08);
    }

    .result-title{
        font-size: 28px;
        font-weight: bold;
        color: #c96c6c;
        margin-bottom: 20px;
    }

    </style>
    """, unsafe_allow_html=True)

    # =====================================================
    # FORM INPUT
    # =====================================================

    st.markdown("""
    <div class="form-container">
        <div class="form-title">
            FORM ULASAN PENGGUNA
        </div>
    """, unsafe_allow_html=True)

    nama = st.text_input(
        "Nama Pengguna",
        placeholder="Masukkan nama pengguna"
    )

    user_input = st.text_area(
        "Masukkan Ulasan",
        placeholder="Tulis ulasan aplikasi Mobile JKN disini...",
        height=180
    )

    col_btn1, col_btn2, col_btn3 = st.columns([1,1,7])

    with col_btn1:
        submit = st.button("Prediksi")

    with col_btn2:
        clear = st.button("Reset")

    st.markdown("</div>", unsafe_allow_html=True)

    # =====================================================
    # RESET
    # =====================================================

    if clear:
        st.rerun()

# =====================================================
# HASIL PREDIKSI
# =====================================================

if submit:

    if user_input.strip() != "":

        hasil = model.predict([user_input])
        sentimen = hasil[0]

        # =====================================================
        # SIMPAN RIWAYAT
        # =====================================================

        st.session_state.riwayat.append({
            "Nama": nama,
            "Ulasan": user_input,
            "Sentimen": sentimen
        })

        with st.container():

            st.markdown("""
            <div style="
                background:white;
                padding:30px;
                border-radius:20px;
                box-shadow:0 4px 15px rgba(0,0,0,0.08);
                margin-top:20px;
            ">
            """, unsafe_allow_html=True)

            st.markdown(
                "<h2 style='color:#c96c6c;text-align:center;'>HASIL PREDIKSI SENTIMEN</h2>",
                unsafe_allow_html=True
            )

            col1, col2 = st.columns([1,2])

            with col1:
                st.markdown("### NAMA")
                st.info(nama)

            with col2:
                st.markdown("### ULASAN")
                st.success(user_input)

            st.markdown("HASIL SENTIMEN")

            if sentimen == "Positif":
                st.success("😊 Positif")

            elif sentimen == "Negatif":
                st.error("😠 Negatif")

            else:
                st.warning("😐 Netral")

            st.markdown("</div>", unsafe_allow_html=True)

    else:
        st.warning("⚠️ Masukkan ulasan terlebih dahulu!")
# =====================================================
# MENU UPLOAD DATASET
# =====================================================

elif menu == "Upload Dataset":

    st.markdown("""
    <style>

    .form-container{
        background: white;
        padding: 35px;
        border-radius: 20px;
        box-shadow: 0px 4px 20px rgba(0,0,0,0.08);
        margin-top: 20px;
    }

    .form-title{
        font-size: 32px;
        font-weight: bold;
        color: #c96c6c;
        margin-bottom: 25px;
        text-align: center;
    }

    .result-container{
        background: white;
        padding: 30px;
        border-radius: 20px;
        margin-top: 30px;
        box-shadow: 0px 4px 20px rgba(0,0,0,0.08);
    }

    .result-title{
        font-size: 28px;
        font-weight: bold;
        color: #c96c6c;
        margin-bottom: 20px;
    }

    </style>
    """, unsafe_allow_html=True)

    # =====================================================
    # FORM INPUT
    # =====================================================

    st.markdown("""
    <div class="form-container">
        <div class="form-title">
            UPLOAD DATASET CSV
        </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload File CSV",
        type=["csv"]
    )

    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded_file is not None:

        try:

            data = pd.read_csv(
                uploaded_file,
                sep=';',
                encoding='utf-8',
                engine='python'
            )

            st.success("✅ Dataset berhasil diupload!")

            # =====================================================
            # PREVIEW DATASET
            # =====================================================

            st.subheader("PREVIEW DATASET")
            st.dataframe(data.head())

            # =====================================================
            # PILIH KOLOM
            # =====================================================

            col1, col2 = st.columns(2)

            with col1:
                kolom_text = st.selectbox(
                    "PILIH KOLOM TEXT",
                    data.columns
                )

            with col2:
                kolom_label = st.selectbox(
                    "PILIH KOLOM LABEL",
                    data.columns
                )

            # =====================================================
            # BUTTON PROSES
            # =====================================================

            if st.button("🚀 Proses Analisis Sentimen"):

                # =====================================================
                # SPLIT DATA
                # =====================================================

                X = data[kolom_text].astype(str)
                y = data[kolom_label].astype(str)

                X_train, X_test, y_train, y_test = train_test_split(
                    X,
                    y,
                    test_size=0.2,
                    random_state=42,
                    stratify=y
                )

                # =====================================================
                # PREDIKSI
                # =====================================================

                prediksi = model.predict(X_test)

                hasil_df = pd.DataFrame({
                    'Text': X_test.values,
                    'Label Asli': y_test.values,
                    'Prediksi': prediksi
                })

                st.success("✅ Analisis Sentimen Berhasil!")

                # =====================================================
                # HASIL ANALISIS
                # =====================================================

                st.subheader("HASIL ANALISIS")
                st.dataframe(hasil_df.head(20))

                # =====================================================
                # AKURASI
                # =====================================================

                accuracy = accuracy_score(
                    y_test,
                    prediksi
                )

                st.subheader("AKURASI MODEL")

                st.metric(
                    "Accuracy",
                    f"{accuracy:.2%}"
                )

                # =====================================================
                # CLASSIFICATION REPORT
                # =====================================================

                st.subheader("CLASSIFICATION REPORT")

                report = classification_report(
                    y_test,
                    prediksi,
                    output_dict=True
                )

                report_df = pd.DataFrame(report).transpose()

                st.dataframe(report_df)

                # =====================================================
                # GRAFIK & CONFUSION MATRIX SEJAJAR
                # =====================================================

                col_chart1, col_chart2 = st.columns(2)

                # =====================================================
                # GRAFIK DISTRIBUSI SENTIMEN
                # =====================================================

                with col_chart1:

                    st.subheader("GRAFIK DISTRIBUSI SENTIMEN")

                    sentimen_count = pd.Series(
                        prediksi
                    ).value_counts()

                    fig1, ax1 = plt.subplots(
                        figsize=(6,5)
                    )

                    ax1.bar(
                        sentimen_count.index,
                        sentimen_count.values
                    )

                    ax1.set_xlabel("Sentimen")
                    ax1.set_ylabel("Jumlah")
                    ax1.set_title("Distribusi Sentimen")

                    st.pyplot(fig1)

                # =====================================================
                # CONFUSION MATRIX
                # =====================================================

                with col_chart2:

                    st.subheader("CONFUSION MATRIX")

                    cm = confusion_matrix(
                        y_test,
                        prediksi
                    )

                    fig2, ax2 = plt.subplots(
                        figsize=(6,5)
                    )

                    disp = ConfusionMatrixDisplay(
                        confusion_matrix=cm
                    )

                    disp.plot(ax=ax2)

                    st.pyplot(fig2)

                # =====================================================
                # DOWNLOAD CSV
                # =====================================================

                csv = hasil_df.to_csv(
                    index=False
                ).encode('utf-8')

                st.download_button(
                    label="⬇️ Download Hasil",
                    data=csv,
                    file_name='hasil_sentimen.csv',
                    mime='text/csv'
                )

        except Exception as e:

            st.error(f"❌ Error: {e}")


# =====================================================
# MENU RIWAYAT
# =====================================================

elif menu == "Riwayat":

    import pandas as pd

    # =====================================================
    # SESSION STATE
    # =====================================================

    if "riwayat" not in st.session_state:
        st.session_state.riwayat = []

    # =====================================================
    # CUSTOM CSS
    # =====================================================

    st.markdown("""
    <style>

    /* Background */
    .main {
        background: linear-gradient(
            to bottom right,
            #fff7f7,
            #ffeaea
        );
    }

    /* HEADER */
    .history-title{
        font-size: 40px;
        font-weight: 800;
        text-align: center;
        color: #c96c6c;
        margin-bottom: 8px;
    }

    .history-subtitle{
        text-align: center;
        color: #777;
        font-size: 17px;
        margin-bottom: 35px;
    }

    /* METRIC CARD */
    .metric-card{
        background: white;
        padding: 28px;
        border-radius: 22px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.08);
        text-align: center;
        transition: 0.3s;
        border: 1px solid rgba(0,0,0,0.05);
    }

    .metric-card:hover{
        transform: translateY(-5px);
    }

    .metric-value{
        font-size: 36px;
        font-weight: bold;
        color: #c96c6c;
    }

    .metric-label{
        font-size: 16px;
        color: #666;
        margin-top: 8px;
    }

    /* TABLE CONTAINER */
    .table-container{
        background: white;
        padding: 30px;
        border-radius: 25px;
        margin-top: 30px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.08);
    }

    /* BUTTON STYLE */
    div.stDownloadButton > button,
    div.stButton > button {
        height: 55px;
        border-radius: 14px;
        font-size: 16px;
        font-weight: 600;
        border: none;
        width: 100%;
        transition: 0.3s;
    }

    /* DOWNLOAD BUTTON */
    div.stDownloadButton > button:hover {
        background: linear-gradient(
            135deg,
            #d97b7b,
            #c96c6c
        );
        color: white;
    }

    div.stDownloadButton > button:hover {
        background: linear-gradient(
            135deg,
            #d97b7b,
            #c96c6c
        );
        color: white;
    }

    /* DELETE BUTTON */
    div.stButton > button:hover {
        background: linear-gradient(
            135deg,
            #d97b7b,
            #c96c6c
        );
        color: white;
    }

    div.stButton > button:hover {
        background: linear-gradient(
            135deg,
            #c96c6c,
            #b85b5b
        );
        color: white;
    }

    /* EMPTY CARD */
    .empty-card{
        background: rgba(255,255,255,0.97);
        padding: 70px 40px;
        border-radius: 30px;
        text-align: center;
        margin-top: 40px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.08);
    }

    </style>
    """, unsafe_allow_html=True)

    # =====================================================
    # HEADER
    # =====================================================

    st.markdown("""
    <style>

    .form-container{
        background: white;
        padding: 35px;
        border-radius: 20px;
        box-shadow: 0px 4px 20px rgba(0,0,0,0.08);
        margin-top: 20px;
    }

    .form-title{
        font-size: 32px;
        font-weight: bold;
        color: #c96c6c;
        margin-bottom: 25px;
        text-align: center;
    }

    .result-container{
        background: white;
        padding: 30px;
        border-radius: 20px;
        margin-top: 30px;
        box-shadow: 0px 4px 20px rgba(0,0,0,0.08);
    }

    .result-title{
        font-size: 28px;
        font-weight: bold;
        color: #c96c6c;
        margin-bottom: 20px;
    }

    </style>
    """, unsafe_allow_html=True)

    # =====================================================
    # FORM INPUT
    # =====================================================

    st.markdown("""
    <div class="form-container">
        <div class="form-title">
            RIWAYAT PREDIKSI SENTIMEN
        </div>
    """, unsafe_allow_html=True)

    # =====================================================
    # JIKA ADA RIWAYAT
    # =====================================================

    if len(st.session_state.riwayat) > 0:

        riwayat_df = pd.DataFrame(
            st.session_state.riwayat
        )

        # =====================================================
        # HITUNG DATA
        # =====================================================

        total_data = len(riwayat_df)

        positif = len(
            riwayat_df[
                riwayat_df["Sentimen"] == "Positif"
            ]
        )

        negatif = len(
            riwayat_df[
                riwayat_df["Sentimen"] == "Negatif"
            ]
        )

        netral = len(
            riwayat_df[
                riwayat_df["Sentimen"] == "Netral"
            ]
        )

        # =====================================================
        # CARD METRIC
        # =====================================================

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(f"""<br>
            <div class="metric-card">
                <div class="metric-value">
                    {total_data}
                </div>
                <div class="metric-label">
                    📁 Total Data
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""<br>
            <div class="metric-card">
                <div class="metric-value">
                    {positif}
                </div>
                <div class="metric-label">
                    😊 Positif
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""<br>
            <div class="metric-card">
                <div class="metric-value">
                    {negatif}
                </div>
                <div class="metric-label">
                    😡 Negatif
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            st.markdown(f"""<br>
            <div class="metric-card">
                <div class="metric-value">
                    {netral}
                </div>
                <div class="metric-label">
                    😐 Netral
                </div>
            </div>
            """, unsafe_allow_html=True)

        # =====================================================
        # TABLE
        # =====================================================

        st.markdown(
            '<div class="table-container">',
            unsafe_allow_html=True
        )

        st.subheader("DATA RIWAYAT")

        st.dataframe(
            riwayat_df,
            use_container_width=True,
            height=450
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # =====================================================
        # CSV
        # =====================================================

        csv = riwayat_df.to_csv(
            index=False
        ).encode('utf-8')

        # =====================================================
        # BUTTON
        # =====================================================

        col_btn1, col_btn2 = st.columns(2)

        with col_btn1:

            st.download_button(
                label="⬇️ Download CSV",
                data=csv,
                file_name='riwayat_sentimen.csv',
                mime='text/csv',
                use_container_width=True
            )

        with col_btn2:

            if st.button(
                "🗑️ Hapus Riwayat",
                use_container_width=True
            ):

                st.session_state.riwayat = []

                st.success(
                    "✅ Riwayat berhasil dihapus"
                )

                st.rerun()

    


# =====================================================
# FOOTER
# =====================================================

st.markdown("""
<div class="footer">
© 2026 Analisis Sentimen Ulasan Mobile JKN <br>
Dibuat menggunakan Streamlit, SVM, dan TF-IDF
</div>
""", unsafe_allow_html=True)