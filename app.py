import os
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

from sklearn.model_selection import train_test_split
if "page" not in st.session_state:
    st.session_state.page = "Home"

# =====================================================
# CONFIG PAGE
# =====================================================

st.set_page_config(
    page_title="Analisis Sentimen Mobile JKN",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

## ==========================================
# LOAD MODEL
# ==========================================

@st.cache_resource
def load_models():

    models = {

        "SVM + TF-IDF":
        joblib.load("models/svm_tfidf.pkl"),

        "SVM + LSA":
        joblib.load("models/svm_lsa.pkl"),

        "NB + TF-IDF":
        joblib.load("models/nb_tfidf.pkl"),

        "NB + LSA":
        joblib.load("models/nb_lsa.pkl")

    }

    return models

models = load_models()

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

# =====================================
# SIDEBAR
# =====================================
with st.sidebar:

    st.markdown(
        '<div class="sidebar-title">HALAMAN MENU</div>',
        unsafe_allow_html=True
    )
    # =====================================================
    # # USER INFO
    # # =====================================================
    st.sidebar.success(
    f"Login sebagai: {st.session_state.username}"
    )

    if st.button("Home", use_container_width=True):
        st.session_state.page = "Home"

    if st.button("Prediksi", use_container_width=True):
        st.session_state.page = "Prediksi"

    if st.button("Riwayat", use_container_width=True):
        st.session_state.page = "Riwayat"
    if st.button("Visualisasi Perbandingan Model", use_container_width=True):
        st.session_state.page = "Visualisasi Perbandingan Model"

    st.divider()

    if st.button("Preprocessing", use_container_width=True):
        st.session_state.page = "Preprocessing"

    st.divider()

if st.sidebar.button("Logout"):

    st.session_state.login_status = False
    st.session_state.username = ""

    st.rerun()


# =====================================================
# MENU HOME
# =====================================================
page = st.session_state.page

if page == "Home":

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

elif page == "Prediksi":

    st.markdown("""
    <style>

    .form-container{
        background:white;
        padding:35px;
        border-radius:20px;
        box-shadow:0px 4px 20px rgba(0,0,0,0.08);
        margin-top:20px;
    }

    .form-title{
        font-size:32px;
        font-weight:bold;
        color:#c96c6c;
        margin-bottom:25px;
        text-align:center;
    }

    </style>
    """, unsafe_allow_html=True)

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

    tombol1, tombol2 = st.columns(2)

    with tombol1:

        submit = st.button(
            "Prediksi",
            use_container_width=True
        )

    with tombol2:

        upload_btn = st.button(
            "Upload Dataset",
            use_container_width=True
        )

        if upload_btn:
            st.session_state.page = "Upload Dataset"
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    # =====================================================
    # HASIL PREDIKSI
    # =====================================================

    if submit:

        if user_input.strip() == "":

            st.warning(
                "⚠️ Masukkan ulasan terlebih dahulu!"
            )

        else:

            hasil_prediksi = []

            for nama_model, model in models.items():

                sentimen = model.predict(
                    [user_input]
                )[0]

                # =====================================
                # HITUNG CONFIDENCE
                # =====================================

                try:

                    proba = model.predict_proba(
                        [user_input]
                    )[0]

                    confidence = round(
                        max(proba) * 100,
                        2
                    )

                except:

                    try:

                        score = model.decision_function(
                            [user_input]
                        )

                        confidence = round(
                            (
                                1 /
                                (
                                    1 +
                                    np.exp(
                                        -np.max(score)
                                    )
                                )
                            ) * 100,
                            2
                        )

                    except:

                        confidence = 0

                hasil_prediksi.append({

                    "Model": nama_model,

                    "Sentimen": sentimen,

                    "Confidence (%)": confidence

                })

            df_hasil = pd.DataFrame(
                hasil_prediksi
            )

            # =====================================
            # # MODEL DENGAN CONFIDENCE TERTINGGI
            # # =====================================
            model_terbaik = df_hasil.loc[
                df_hasil["Confidence (%)"].idxmax()
                ]
            nama_model_terbaik = model_terbaik["Model"]
            sentimen_terbaik = model_terbaik["Sentimen"]
            confidence_terbaik = model_terbaik["Confidence (%)"]

            # =====================================
            # SIMPAN RIWAYAT
            # =====================================

            st.session_state.riwayat.append({

    "Nama": nama,
    "Ulasan": user_input,

    # ======================
    # SENTIMEN 4 MODEL
    # ======================

    "SVM + TF-IDF":
    df_hasil.loc[
        df_hasil["Model"] == "SVM + TF-IDF",
        "Sentimen"
    ].values[0],

    "SVM + LSA":
    df_hasil.loc[
        df_hasil["Model"] == "SVM + LSA",
        "Sentimen"
    ].values[0],

    "NB + TF-IDF":
    df_hasil.loc[
        df_hasil["Model"] == "NB + TF-IDF",
        "Sentimen"
    ].values[0],

    "NB + LSA":
    df_hasil.loc[
        df_hasil["Model"] == "NB + LSA",
        "Sentimen"
    ].values[0],

    # ======================
    # CONFIDENCE 4 MODEL
    # ======================

    "Conf SVM + TF-IDF":
    df_hasil.loc[
        df_hasil["Model"] == "SVM + TF-IDF",
        "Confidence (%)"
    ].values[0],

    "Conf SVM + LSA":
    df_hasil.loc[
        df_hasil["Model"] == "SVM + LSA",
        "Confidence (%)"
    ].values[0],

    "Conf NB + TF-IDF":
    df_hasil.loc[
        df_hasil["Model"] == "NB + TF-IDF",
        "Confidence (%)"
    ].values[0],

    "Conf NB + LSA":
    df_hasil.loc[
        df_hasil["Model"] == "NB + LSA",
        "Confidence (%)"
    ].values[0],

    # ======================
    # MODEL TERBAIK
    # ======================

    "Model Terbaik":
    nama_model_terbaik,

    "Sentimen Terbaik":
    sentimen_terbaik,

    "Confidence Terbaik":
    confidence_terbaik

})

            # =====================================
            # HASIL
            # =====================================

            st.markdown("---")

            st.subheader(
                "Hasil Prediksi Sentimen"
            )

            col1, col2 = st.columns([1, 2])

            with col1:

                st.markdown("### Nama")
                st.info(nama)

            with col2:

                st.markdown("### Ulasan")
                st.success(user_input)

            

            # =====================================
            # CARD MODEL
            # =====================================

            st.subheader(
                "Confidence Setiap Model"
            )

            col1, col2 = st.columns(2)

            with col1:

                svm_tfidf = df_hasil[
                    df_hasil["Model"] ==
                    "SVM + TF-IDF"
                ].iloc[0]

                svm_lsa = df_hasil[
                    df_hasil["Model"] ==
                    "SVM + LSA"
                ].iloc[0]

                st.metric(
                    "SVM + TF-IDF",
                    svm_tfidf["Sentimen"],
                    f"{svm_tfidf['Confidence (%)']}%"
                )

                st.metric(
                    "SVM + LSA",
                    svm_lsa["Sentimen"],
                    f"{svm_lsa['Confidence (%)']}%"
                )

            with col2:

                nb_tfidf = df_hasil[
                    df_hasil["Model"] ==
                    "NB + TF-IDF"
                ].iloc[0]

                nb_lsa = df_hasil[
                    df_hasil["Model"] ==
                    "NB + LSA"
                ].iloc[0]

                st.metric(
                    "NB + TF-IDF",
                    nb_tfidf["Sentimen"],
                    f"{nb_tfidf['Confidence (%)']}%"
                )

                st.metric(
                    "NB + LSA",
                    nb_lsa["Sentimen"],
                    f"{nb_lsa['Confidence (%)']}%"
                )

            # =====================================
            # # MODEL DENGAN CONFIDENCE TERTINGGI
            # # =====================================
            st.subheader(
                "Model dengan Confidence Tertinggi"
                )
            st.info(
                f"Model Terpilih : {nama_model_terbaik}"
                )
            if sentimen_terbaik == "Positif":
                st.success(
                    f"😊 Sentimen : {sentimen_terbaik}"
                    )
            elif sentimen_terbaik == "Negatif":
                st.error(
                    f"😠 Sentimen : {sentimen_terbaik}"
                    )
            else:
                st.warning(
                    f"😐 Sentimen : {sentimen_terbaik}"
                    )
                st.metric(
                    "Confidence Score",
                    f"{confidence_terbaik}%"
)
# =====================================================
# MENU UPLOAD DATASET
# =====================================================

elif page == "Upload Dataset":

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt


    
    st.markdown("""
    <div style="
        background:#b08968;
        padding:30px;
        border-radius:20px;
        text-align:center;
    ">
    <h1 style="color:white">
    UPLOAD DATASET HASIL PREPROCESSING
    </h1>
    </div>
    """, unsafe_allow_html=True)



    uploaded_file = st.file_uploader(
        "Upload File CSV",
        type=["csv"]
    )


    if uploaded_file is not None:

        try:


            # ==========================================
            # BACA CSV
            # ==========================================

            data = pd.read_csv(
                uploaded_file,
                sep=None,
                engine="python"
            )


            data.columns = data.columns.str.strip()


            st.success(
                "✅ Dataset berhasil diupload"
            )


            st.write(
                "Kolom Dataset:",
                data.columns.tolist()
            )


            # ==========================================
            # CEK STEMMING
            # ==========================================

            if "Stemming" not in data.columns:

                st.error(
                    "Kolom Stemming tidak ditemukan"
                )

                st.stop()



            # ==========================================
            # BERSIHKAN DATA
            # ==========================================

            data = data.dropna(
                subset=["Stemming"]
            )


            data = data[
                data["Stemming"]
                .astype(str)
                .str.strip() != ""
            ]



            # ==========================================
            # PREVIEW
            # ==========================================

            st.subheader(
                "Preview Dataset"
            )


            st.dataframe(
                data.head(),
                use_container_width=True
            )


            total_data = len(data)


            st.info(
                f"Jumlah Data : {total_data}"
            )



            jumlah_data = st.slider(
                "Pilih jumlah data",
                1,
                total_data,
                min(100,total_data)
            )


            data_analisis = data.head(
                jumlah_data
            )



            if st.button(
                "🚀 Mulai Analisis",
                use_container_width=True
            ):



                hasil_semua = []


                progress = st.progress(0)



                # =====================================
                # PREDIKSI SEMUA MODEL
                # =====================================


                for i,text in enumerate(
                    data_analisis["Stemming"]
                ):


                    hasil_prediksi = []


                    for nama_model, model in models.items():


                        prediksi = model.predict(
                            [str(text)]
                        )[0]



                        # CONFIDENCE

                        try:

                            proba = model.predict_proba(
                                [str(text)]
                            )[0]


                            confidence = round(
                                max(proba)*100,
                                2
                            )


                        except:


                            try:

                                score = model.decision_function(
                                    [str(text)]
                                )


                                confidence = round(
                                    (
                                    1 /
                                    (
                                    1 +
                                    np.exp(
                                    -np.max(score)
                                    )
                                    )
                                    )*100,
                                    2
                                )


                            except:

                                confidence = 0



                        hasil_prediksi.append({

                            "Model":
                            nama_model,

                            "Sentimen":
                            prediksi,

                            "Confidence":
                            confidence

                        })




                    df_temp = pd.DataFrame(
                        hasil_prediksi
                    )



                    terbaik = df_temp.loc[
                        df_temp["Confidence"]
                        .idxmax()
                    ]



                    hasil_semua.append({


                        "Text":
                        str(text),


                        "SVM + TF-IDF":
                        df_temp.loc[
                            df_temp.Model=="SVM + TF-IDF",
                            "Sentimen"
                        ].values[0],


                        "Conf SVM + TF-IDF":
                        df_temp.loc[
                            df_temp.Model=="SVM + TF-IDF",
                            "Confidence"
                        ].values[0],



                        "SVM + LSA":
                        df_temp.loc[
                            df_temp.Model=="SVM + LSA",
                            "Sentimen"
                        ].values[0],


                        "Conf SVM + LSA":
                        df_temp.loc[
                            df_temp.Model=="SVM + LSA",
                            "Confidence"
                        ].values[0],



                        "NB + TF-IDF":
                        df_temp.loc[
                            df_temp.Model=="NB + TF-IDF",
                            "Sentimen"
                        ].values[0],


                        "Conf NB + TF-IDF":
                        df_temp.loc[
                            df_temp.Model=="NB + TF-IDF",
                            "Confidence"
                        ].values[0],



                        "NB + LSA":
                        df_temp.loc[
                            df_temp.Model=="NB + LSA",
                            "Sentimen"
                        ].values[0],


                        "Conf NB + LSA":
                        df_temp.loc[
                            df_temp.Model=="NB + LSA",
                            "Confidence"
                        ].values[0],



                        "Model Terbaik":
                        terbaik["Model"],


                        "Sentimen Terbaik":
                        terbaik["Sentimen"],


                        "Confidence Terbaik":
                        terbaik["Confidence"]

                    })



                    progress.progress(
                        (i+1)/jumlah_data
                    )



                # =====================================
                # HASIL
                # =====================================


                hasil_df = pd.DataFrame(
                    hasil_semua
                )
                # SIMPAN HASIL UNTUK MENU PERBANDINGAN
                st.session_state["hasil_upload"] = hasil_df
                st.session_state["data_uji"] = data_analisis


                st.success(
                    "✅ Analisis selesai"
                )



                st.subheader(
                    "Hasil Analisis Dataset"
                )



                st.dataframe(
                    hasil_df,
                    use_container_width=True,
                    height=500
                )





                # =====================================
                # DOWNLOAD CSV
                # =====================================


                csv = hasil_df.to_csv(
                    index=False
                ).encode("utf-8")



                st.download_button(

                    "⬇️ Download Hasil Analisis",

                    data=csv,

                    file_name=
                    "hasil_analisis_dataset.csv",

                    mime="text/csv",

                    use_container_width=True

                )



        except Exception as e:


            st.error(
                f"❌ Error : {str(e)}"
            )

# =====================================================
# MENU RIWAYAT
# =====================================================

elif page == "Riwayat":

    import pandas as pd
    import plotly.express as px

    if "riwayat" not in st.session_state:
        st.session_state.riwayat = []

    # =====================================================
    # CSS
    # =====================================================

    st.markdown("""
    <style>

    .history-header{
        background:linear-gradient(
            135deg,
            #c96c6c,
            #e89b9b
        );
        padding:30px;
        border-radius:20px;
        text-align:center;
        margin-bottom:25px;
        box-shadow:0 5px 20px rgba(0,0,0,0.08);
    }

    .history-title{
        color:white;
        font-size:36px;
        font-weight:700;
    }

    .history-subtitle{
        color:white;
        font-size:16px;
    }

    .metric-box{
        background:white;
        padding:25px;
        border-radius:20px;
        text-align:center;
        box-shadow:0 4px 15px rgba(0,0,0,0.08);
    }

    .metric-number{
        font-size:34px;
        font-weight:bold;
        color:#c96c6c;
    }

    .metric-label{
        color:#666;
        font-size:15px;
    }

    .table-box{
        background:white;
        padding:20px;
        border-radius:20px;
        box-shadow:0 4px 15px rgba(0,0,0,0.08);
        margin-top:20px;
    }

    .empty-box{
        background:white;
        padding:60px;
        border-radius:20px;
        text-align:center;
        box-shadow:0 4px 15px rgba(0,0,0,0.08);
    }

    div.stDownloadButton > button,
    div.stButton > button{

        width:100%;
        height:60px;

        border:none;
        border-radius:15px;

        font-size:17px;
        font-weight:bold;

        transition:0.3s;

        box-shadow:0 4px 15px rgba(0,0,0,0.08);
    }

    div.stDownloadButton > button{

        background:linear-gradient(
            135deg,
            #36d1dc,
            #5b86e5
        );

        color:white;
    }

    div.stButton > button{

        background:linear-gradient(
            135deg,
            #ff6b6b,
            #ee5253
        );

        color:white;
    }

    div.stDownloadButton > button:hover,
    div.stButton > button:hover{

        transform:translateY(-3px);
    }

    </style>
    """, unsafe_allow_html=True)

    # =====================================================
    # HEADER
    # =====================================================

    st.markdown("""
    <div class="history-header">
        <div class="history-title">
            RIWAYAT PREDIKSI SENTIMEN
        </div>
        <div class="history-subtitle">
            Hasil Analisis Sentimen Aplikasi Mobile JKN
        </div>
    </div>
    """, unsafe_allow_html=True)

    # =====================================================
    # JIKA ADA DATA
    # =====================================================

    if len(st.session_state.riwayat) > 0:

        riwayat_df = pd.DataFrame(
            st.session_state.riwayat
        )

        # =====================================================
        # STATISTIK
        # =====================================================

        total_data = len(riwayat_df)

        positif = len(
            riwayat_df[
                riwayat_df["Sentimen Terbaik"] == "Positif"
            ]
        )

        negatif = len(
            riwayat_df[
                riwayat_df["Sentimen Terbaik"] == "Negatif"
            ]
        )

        netral = len(
            riwayat_df[
                riwayat_df["Sentimen Terbaik"] == "Netral"
            ]
        )

        # =====================================================
        # CARD STATISTIK
        # =====================================================

        col1,col2,col3,col4 = st.columns(4)

        with col1:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-number">
                    {total_data}
                </div>
                <div class="metric-label">
                    📁 Total Data
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-number">
                    {positif}
                </div>
                <div class="metric-label">
                    😊 Positif
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-number">
                    {negatif}
                </div>
                <div class="metric-label">
                    😡 Negatif
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-number">
                    {netral}
                </div>
                <div class="metric-label">
                    😐 Netral
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # =====================================================
        # CEK KOLOM CONFIDENCE
        # =====================================================

        kolom_wajib = [

            "Nama",
            "Ulasan",

            "SVM + TF-IDF",
            "Conf SVM + TF-IDF",

            "SVM + LSA",
            "Conf SVM + LSA",

            "NB + TF-IDF",
            "Conf NB + TF-IDF",

            "NB + LSA",
            "Conf NB + LSA",

            "Model Terbaik",
            "Sentimen Terbaik",
            "Confidence Terbaik"

        ]

        kolom_tampil = [
            col for col in kolom_wajib
            if col in riwayat_df.columns
        ]

        tampil_df = riwayat_df[kolom_tampil]

        # =====================================================
        # TABEL
        # =====================================================

        st.markdown(
            '<div class="table-box">',
            unsafe_allow_html=True
        )

        st.subheader(
            "DATA RIWAYAT PREDIKSI"
        )

        st.dataframe(
            tampil_df,
            use_container_width=True,
            height=500
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


        # =====================================================
        # DOWNLOAD CSV
        # =====================================================

        csv = tampil_df.to_csv(
            index=False
        ).encode("utf-8")

        st.markdown("<br>", unsafe_allow_html=True)

        col1,col2 = st.columns(2)

        with col1:

            st.download_button(
                label="📥 Download Riwayat CSV",
                data=csv,
                file_name="riwayat_sentimen.csv",
                mime="text/csv",
                use_container_width=True
            )

        # =====================================================
        # HAPUS RIWAYAT
        # =====================================================

        with col2:

            if st.button(
                "🗑️ Hapus Semua Riwayat",
                use_container_width=True
            ):

                st.session_state.riwayat = []

                st.success(
                    "Riwayat berhasil dihapus."
                )

                st.rerun()
# =====================================================
# MENU PERBANDINGAN MODEL
# =====================================================

# =====================================================
# MENU PERBANDINGAN MODEL
# =====================================================

elif page == "Visualisasi Perbandingan Model":


    import pandas as pd
    import matplotlib.pyplot as plt

    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        classification_report
    )


    # =====================================================
    # CSS
    # =====================================================

    st.markdown("""
    <style>

    .history-header{
        background:linear-gradient(
            135deg,
            #c96c6c,
            #e89b9b
        );
        padding:30px;
        border-radius:20px;
        text-align:center;
        margin-bottom:25px;
        box-shadow:0 5px 20px rgba(0,0,0,0.08);
    }

    .history-title{
        color:white;
        font-size:36px;
        font-weight:700;
    }

    .history-subtitle{
        color:white;
        font-size:16px;
    }

    .metric-box{
        background:white;
        padding:25px;
        border-radius:20px;
        text-align:center;
        box-shadow:0 4px 15px rgba(0,0,0,0.08);
    }

    .metric-number{
        font-size:34px;
        font-weight:bold;
        color:#c96c6c;
    }

    .metric-label{
        color:#666;
        font-size:15px;
    }

    .table-box{
        background:white;
        padding:20px;
        border-radius:20px;
        box-shadow:0 4px 15px rgba(0,0,0,0.08);
        margin-top:20px;
    }

    .empty-box{
        background:white;
        padding:60px;
        border-radius:20px;
        text-align:center;
        box-shadow:0 4px 15px rgba(0,0,0,0.08);
    }

    div.stDownloadButton > button,
    div.stButton > button{

        width:100%;
        height:60px;

        border:none;
        border-radius:15px;

        font-size:17px;
        font-weight:bold;

        transition:0.3s;

        box-shadow:0 4px 15px rgba(0,0,0,0.08);
    }

    div.stDownloadButton > button{

        background:linear-gradient(
            135deg,
            #36d1dc,
            #5b86e5
        );

        color:white;
    }

    div.stButton > button{

        background:linear-gradient(
            135deg,
            #ff6b6b,
            #ee5253
        );

        color:white;
    }

    div.stDownloadButton > button:hover,
    div.stButton > button:hover{

        transform:translateY(-3px);
    }

    </style>
    """, unsafe_allow_html=True)

    # =====================================================
    # HEADER
    # =====================================================

    st.markdown("""
    <div class="history-header">
        <div class="history-title">
            VISUALISASI PERBANDINGAN MODEL
        </div>
    </div>
    """, unsafe_allow_html=True)



    # ==========================
    # CEK DATA
    # ==========================


    if (
        "data_uji" not in st.session_state
        or
        "hasil_upload" not in st.session_state
    ):

        st.warning(
            "⚠️ Silahkan lakukan analisis pada menu Upload Dataset terlebih dahulu"
        )

        st.stop()



    data = st.session_state["data_uji"]



    # ==========================
    # LABEL
    # ==========================


    if "Label" in data.columns:

        label_col = "Label"


    elif "Sentimen Asli" in data.columns:

        label_col = "Sentimen Asli"


    else:

        st.error(
            "Kolom label tidak ditemukan"
        )

        st.stop()



    X = data["Stemming"].astype(str)

    y_true = data[label_col]



    hasil_akurasi = {}

    detail_model = {}



    # ==========================
    # PROSES MODEL
    # ==========================


    for nama_model, model in models.items():


        y_pred = model.predict(
            X
        )


        acc = accuracy_score(
            y_true,
            y_pred
        )



        hasil_akurasi[nama_model] = round(
            acc*100,
            2
        )


        detail_model[nama_model] = {

            "prediksi": y_pred

        }



    # ==========================
    # SUMMARY CARD
    # ==========================


    st.subheader(
        "RINGKASAN PERFORMA MODEL"
    )



    cols = st.columns(4)


    for i,(model, nilai) in enumerate(
        hasil_akurasi.items()
    ):


        with cols[i]:


            st.metric(
                model,
                f"{nilai}%"
            )



    # MODEL TERBAIK

    terbaik = max(
        hasil_akurasi,
        key=hasil_akurasi.get
    )


    st.success(
        f"🥇 Model terbaik : {terbaik} dengan akurasi {hasil_akurasi[terbaik]}%"
    )



    st.divider()



    # ==========================
    # DETAIL MODEL
    # ==========================


    for nama_model, model in models.items():


        with st.expander(
            f"🔍 Detail Evaluasi : {nama_model}",
            expanded=True
        ):



            y_pred = detail_model[nama_model]["prediksi"]



            col1,col2 = st.columns([1,2])



            # ------------------
            # ACCURACY
            # ------------------

            with col1:


                st.markdown(
                    "### ACCURACY"
                )


                st.metric(
                    "Score",
                    f"{hasil_akurasi[nama_model]}%"
                )




            # ------------------
            # CONFUSION MATRIX
            # ------------------

            with col2:


                st.markdown(
                    "### CONFUSION MATRIX"
                )


                cm = confusion_matrix(
                    y_true,
                    y_pred
                )


                cm_df = pd.DataFrame(
                    cm,
                    index=sorted(
                        y_true.unique()
                    ),
                    columns=sorted(
                        y_true.unique()
                    )
                )


                st.dataframe(
                    cm_df,
                    use_container_width=True
                )




            # ------------------
            # CLASSIFICATION REPORT
            # ------------------


            st.markdown(
                "### CLASSIFICATION REPORT"
            )


            report = classification_report(
                y_true,
                y_pred,
                output_dict=True
            )



            report_df = pd.DataFrame(
                report
            ).transpose()



            st.dataframe(
                report_df.style
                .format("{:.3f}")
                .set_properties(
                    **{
                    "text-align":"center"
                    }
                ),
                use_container_width=True
            )




    # ==========================
    # GRAFIK
    # ==========================


    st.divider()



    st.subheader(
        "GRAFIK PERBANDINGAN"
    )



    grafik = pd.DataFrame({

        "Model":
        list(hasil_akurasi.keys()),


        "Akurasi":
        list(hasil_akurasi.values())

    })



    fig,ax = plt.subplots(
        figsize=(9,5)
    )


    ax.bar(
        grafik["Model"],
        grafik["Akurasi"]
    )



    ax.set_ylim(
        0,
        100
    )


    ax.set_ylabel(
        "Akurasi (%)"
    )


    ax.set_title(
        "Perbandingan Akurasi 4 Model"
    )


    plt.xticks(
        rotation=25
    )



    for i,v in enumerate(
        grafik["Akurasi"]
    ):

        ax.text(
            i,
            v+1,
            f"{v}%",
            ha="center"
        )


    st.pyplot(fig)
elif page == "Preprocessing":

    import pandas as pd
    import re
    import nltk

    from nltk.corpus import stopwords
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

    try:
        nltk.data.find('corpora/stopwords')
    except:
        nltk.download('stopwords')

    st.markdown("""
    <div style="
        background:#b08968;
        padding:30px;
        border-radius:20px;
        text-align:center;
    ">
    <h1 style="color:white">
    PREPROCESSING DAN PELABELAN DATA
    </h1>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload Dataset CSV",
        type=["csv"]
    )

    if uploaded_file is not None:

        # =========================
        # LOAD DATA
        # =========================

        review = pd.read_csv(
            uploaded_file,
            sep=",",
            engine="python",
            on_bad_lines="skip"
        )

        review.columns = (
            review.columns
            .str.strip()
            .str.replace(";", "", regex=False)
        )

        st.subheader("📄 Dataset Awal")

        st.dataframe(
            review.head(),
            use_container_width=True
        )

        if "content" not in review.columns:

            st.error(
                "Kolom 'content' tidak ditemukan"
            )

        else:

            with st.spinner(
                "Melakukan preprocessing..."
            ):

                # =========================
                # HAPUS DATA KOSONG
                # =========================

                review = review.dropna(
                    subset=["content"]
                )

                review = review[
                    review["content"]
                    .astype(str)
                    .str.strip() != ""
                ]

                # =========================
                # CASE FOLDING
                # =========================

                def casefoldingText(text):

                    if isinstance(text, str):
                        return text.lower()

                    return ""

                review["CaseFolding"] = (
                    review["content"]
                    .apply(casefoldingText)
                )

                # =========================
                # CLEANING
                # =========================

                def cleaningulasan(text):

                    text = re.sub(
                        r'@[A-Za-z0-9_]+',
                        ' ',
                        str(text)
                    )

                    text = re.sub(
                        r'#[A-Za-z0-9_]+',
                        ' ',
                        text
                    )

                    text = re.sub(
                        r'http\S+',
                        ' ',
                        text
                    )

                    text = re.sub(
                        r'[0-9]+',
                        ' ',
                        text
                    )

                    text = re.sub(
                        r'[-()"#/@;:<>{}+=~|.!?,_]',
                        ' ',
                        text
                    )

                    text = re.sub(
                        r'\s+',
                        ' ',
                        text
                    )

                    return text.strip()

                def clearEmoji(text):

                    return (
                        text
                        .encode(
                            "ascii",
                            "ignore"
                        )
                        .decode("ascii")
                    )

                def replaceTOM(text):

                    pola = re.compile(
                        r'(.)\1{2,}',
                        re.DOTALL
                    )

                    return pola.sub(
                        r'\1',
                        text
                    )

                review["Cleaning"] = (
                    review["CaseFolding"]
                    .apply(cleaningulasan)
                    .apply(clearEmoji)
                    .apply(replaceTOM)
                )

                review = review[
                    review["Cleaning"]
                    .str.strip() != ""
                ]

                # =========================
                # TOKENIZING
                # =========================

                def tokenizingText(text):

                    return str(text).split()

                review["Tokenizing"] = (
                    review["Cleaning"]
                    .apply(tokenizingText)
                )

                # =========================
                # FORMALISASI
                # =========================

                kamusSlang = {

                    "yg":"yang",
                    "gak":"tidak",
                    "gk":"tidak",
                    "ga":"tidak",
                    "bgt":"banget",
                    "aja":"saja",
                    "udh":"sudah",
                    "udah":"sudah",
                    "tdk":"tidak",
                    "krn":"karena",
                    "sm":"sama",
                    "utk":"untuk",
                    "dr":"dari",
                    "jg":"juga",
                    "jd":"jadi",
                    "tp":"tapi",
                    "klo":"kalau",
                    "blm":"belum",
                    "apk":"aplikasi",
                    "app":"aplikasi"

                }

                def formal_text(words):

                    return [
                        kamusSlang.get(
                            word,
                            word
                        )
                        for word in words
                    ]

                review["Formalisasi"] = (
                    review["Tokenizing"]
                    .apply(formal_text)
                )

                # =========================
                # STOPWORD
                # =========================

                stopword = set(
                    stopwords.words(
                        "indonesian"
                    )
                )

                stopword.update([
                    "yg",
                    "dg",
                    "rt",
                    "aja",
                    "nih",
                    "sih"
                ])

                def stopwordText(words):

                    return [
                        word
                        for word in words
                        if word not in stopword
                    ]

                review["WithoutStopwords"] = (
                    review["Formalisasi"]
                    .apply(stopwordText)
                )

                # =========================
                # STEMMING
                # =========================

                factory = StemmerFactory()

                stemmer = (
                    factory.create_stemmer()
                )

                def stemmingText(words):

                    text = " ".join(words)

                    return stemmer.stem(text)

                review["Stemming"] = (
                    review["WithoutStopwords"]
                    .apply(stemmingText)
                )

                # =========================
                # LABELING
                # =========================

                positif = [
                    "bagus",
                    "baik",
                    "cepat",
                    "mudah",
                    "mantap",
                    "bantu",
                    "puas",
                    "praktis",
                    "lancar",
                    "aman",
                    "nyaman"
                ]

                negatif = [
                    "buruk",
                    "error",
                    "gagal",
                    "lambat",
                    "lemot",
                    "susah",
                    "ribet",
                    "kecewa",
                    "masalah",
                    "kendala",
                    "rusak"
                ]

                def labeling(text):

                    text = str(text)

                    skor = 0

                    for kata in positif:

                        if kata in text:
                            skor += 1

                    for kata in negatif:

                        if kata in text:
                            skor -= 1

                    if skor > 0:
                        return "Positif"

                    elif skor < 0:
                        return "Negatif"

                    else:
                        return "Netral"

                review["Label"] = (
                    review["Stemming"]
                    .apply(labeling)
                )

            st.success(
                "✅ Preprocessing selesai"
            )

            st.subheader(
                "📊 Hasil Preprocessing"
            )

            st.dataframe(
                review[
                    [
                        "content",
                        "CaseFolding",
                        "Cleaning",
                        "Tokenizing",
                        "Formalisasi",
                        "WithoutStopwords",
                        "Stemming",
                        "Label"
                    ]
                ],
                use_container_width=True
            )

            st.subheader(
                "📈 Distribusi Sentimen"
            )

            st.bar_chart(
                review["Label"]
                .value_counts()
            )

            # =========================
            # DOWNLOAD CSV
            # =========================

            csv = review.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                label="📥 Download Hasil Preprocessing",
                data=csv,
                file_name="hasil_preprocessing_mobile_jkn.csv",
                mime="text/csv",
                use_container_width=True
            )
    


# =====================================================
# FOOTER
# =====================================================

st.markdown("""
<div class="footer">
© 2026 Analisis Sentimen Ulasan Mobile JKN <br>
Dibuat menggunakan Streamlit, SVM, dan TF-IDF
</div>
""", unsafe_allow_html=True)
