"""
Rekomendasi Wisata Sadar Musim: Sentence-BERT + Content-Based Filtering.

Jalankan:   streamlit run app.py
Membaca :   artifacts/cbf_bundle.pkl            (wajib; profil, metrik, biaya, statistik data)
            artifacts/per_kategori.csv          (opsional; tabel per kategori)
            artifacts/protokol_E_kueri_kategori.csv (opsional; Protokol E)
            artifacts/fig_*.png                 (opsional; gambar evaluasi dan data)
            artifacts/hasil_untuk_artikel.json  (opsional; analisis nama dan contoh rekomendasi)
Gaya    :   shadcn/ui (kanvas abu-abu, kartu putih, tombol pil gelap). Tabel dan grafik dibuat dengan HTML,
            sehingga tidak bergantung pada pyarrow/altair.

Model SBERT butuh `sentence-transformers` dan model terunduh dari internet saat pertama kali dipakai.
Jika tidak tersedia, aplikasi tetap berjalan dengan representasi pembanding (TF-IDF) dan memberi pesan jelas.
"""
import html
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from scipy import sparse
from sklearn.preprocessing import normalize

ART = Path("artifacts")
BUNDLE_PATH = ART / "cbf_bundle.pkl"
RESULT_PATH = ART / "hasil_untuk_artikel.json"
APP_TITLE = "Rekomendasi Wisata Sadar Musim"
TEAM = [
    ("Ar'raffi Abqori Nur Azizi", "230441100026"),
    ("Isma' Yafa Nur Zamzami Ramadhani", "[isi NIM]"),
    ("Muhammad Maulana Khanif", "[isi NIM]"),
    ("Andika Kamaluddin", "[isi NIM]"),
]
SEASONS = ["Semua musim", "Kemarau", "Penghujan"]

st.set_page_config(page_title=APP_TITLE, layout="wide", initial_sidebar_state="expanded")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Geist:wght@400;500;600&display=swap');
:root{ --canvas:#f5f5f5; --paper:#ffffff; --alt:#fafafa; --ink:#0a0a0a; --soft:#171717; --mid:#737373; --hair:#e5e5e5; --ember:#e7000b;
  --shadow:0 0 0 1px rgba(23,23,23,.05), 0 1px 3px rgba(0,0,0,.1), 0 1px 2px -1px rgba(0,0,0,.1); }
html, body, .stApp, .stApp *{ font-family:'Geist','Inter',ui-sans-serif,system-ui,-apple-system,'Segoe UI',Roboto,sans-serif; }
.stApp{ background:var(--canvas); color:var(--ink); font-feature-settings:"ss01" on,"cv11" on; }
header[data-testid="stHeader"]{ background:transparent; }
[data-testid="stAppDeployButton"], .stDeployButton, [data-testid="stIconMaterial"], [data-testid="stSidebarCollapseButton"],
[data-testid="stSidebarCollapsedControl"], [data-testid="stToolbar"]{ display:none !important; }
.block-container{ max-width:1280px; padding-top:2rem; padding-bottom:4rem; }
[data-testid="stSidebar"]{ background:var(--alt); border-right:1px solid var(--hair); }
[data-testid="stSidebar"] *{ font-size:14px; }
.display{ font-size:36px; line-height:1.11; font-weight:600; letter-spacing:-.9px; margin:4px 0 8px; }
.lead{ font-size:16px; line-height:1.5; color:var(--mid); margin:0 0 24px; }
.crumb{ font-size:14px; color:var(--mid); } .crumb b{ color:var(--ink); font-weight:400; }
.caption{ font-size:12px; letter-spacing:.6px; text-transform:uppercase; color:var(--mid); }
.sub{ font-size:18px; font-weight:600; letter-spacing:-.2px; margin:24px 0 8px; }
.card{ background:var(--paper); border:1px solid var(--hair); border-radius:24px; box-shadow:var(--shadow); padding:20px; margin-bottom:16px; }
.card .row{ display:flex; align-items:center; gap:12px; }
.card .rank{ font-size:12px; color:var(--mid); letter-spacing:.6px; }
.card .title{ font-size:18px; font-weight:600; letter-spacing:-.2px; flex:1; }
.card .score{ font-size:24px; font-weight:600; letter-spacing:-.6px; }
.card .muted{ color:var(--mid); font-size:14px; margin:8px 0 0; line-height:1.43; }
.card .snip{ border-left:2px solid var(--hair); padding-left:10px; color:var(--soft); font-size:14px; margin:10px 0 0; }
.bar{ height:6px; background:var(--canvas); border-radius:18px; overflow:hidden; margin:12px 0 4px; } .bar > div{ height:100%; background:var(--ink); border-radius:18px; }
.stat .l{ font-size:12px; letter-spacing:.6px; text-transform:uppercase; color:var(--mid); }
.stat .v{ font-size:36px; line-height:1.11; font-weight:600; letter-spacing:-.9px; } .stat .s{ font-size:14px; color:var(--mid); }
.badge{ display:inline-block; padding:2px 8px; border-radius:18px; font-size:12px; font-weight:500; margin:6px 6px 0 0; }
.badge.solid{ background:var(--soft); color:var(--alt); } .badge.soft{ background:var(--canvas); color:var(--soft); }
.badge.out{ border:1px solid var(--hair); } .badge.bad{ background:var(--paper); border:1px solid var(--ember); color:var(--ember); }
.note{ background:var(--paper); border:1px solid var(--hair); border-radius:18px; padding:12px 16px; font-size:14px; color:var(--soft); margin:0 0 16px; }
table.t{ width:100%; border-collapse:separate; border-spacing:0; background:var(--paper); border:1px solid var(--hair); border-radius:18px; overflow:hidden; font-size:13px; margin-bottom:8px; }
table.t th{ text-align:left; font-weight:500; color:var(--mid); background:var(--alt); padding:10px 12px; border-bottom:1px solid var(--hair); font-size:12px; }
table.t td{ padding:9px 12px; border-bottom:1px solid var(--hair); color:var(--ink); } table.t tr:last-child td{ border-bottom:none; }
table.t td.best{ font-weight:600; }
[data-testid="stImage"] img{ border-radius:18px; border:1px solid var(--hair); background:var(--paper); }
[data-testid="stImage"] + div{ font-size:12px; color:var(--mid); }
.hb{ display:flex; align-items:center; gap:10px; margin:6px 0; font-size:13px; } .hb .n{ width:170px; color:var(--soft); }
.hb .b{ flex:1; height:10px; background:var(--paper); border:1px solid var(--hair); border-radius:18px; overflow:hidden; } .hb .b > div{ height:100%; background:var(--ink); }
.hb .v{ width:48px; text-align:right; color:var(--mid); }
.stButton > button, .stFormSubmitButton > button, [data-testid^="stBaseButton"]{ background:var(--ink); color:var(--alt); border:none; border-radius:18px; height:38px; padding:0 16px; font-size:14px; font-weight:500; box-shadow:none; }
.stButton > button:hover, .stFormSubmitButton > button:hover{ background:var(--soft); color:var(--alt); } .stFormSubmitButton > button p{ color:var(--alt); }
div[data-baseweb="input"] > div, div[data-baseweb="textarea"] > div, div[data-baseweb="select"] > div{ background:var(--canvas); border:none; border-radius:18px; box-shadow:none; }
div[data-baseweb="input"] > div:focus-within, div[data-baseweb="textarea"] > div:focus-within, div[data-baseweb="select"] > div:focus-within{ box-shadow:0 0 0 1px var(--hair); }
label, [data-testid="stWidgetLabel"] p{ font-size:14px; font-weight:500; color:var(--ink); }
button[data-baseweb="tab"]{ font-size:14px; font-weight:500; color:var(--mid); } button[data-baseweb="tab"][aria-selected="true"]{ color:var(--ink); }
[data-testid="stExpander"]{ border:1px solid var(--hair); border-radius:18px; background:var(--paper); } hr{ border:none; border-top:1px solid var(--hair); }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ============================================================================ UTIL
def esc(x):
    return html.escape(str(x))


def dd(x, n=3):
    return f"{x:.{n}f}".replace(".", ",")


def pct(x, n=1):
    return dd(x * 100, n) + "%"


def ns(x):
    return f"{int(round(x)):,}".replace(",", ".")


def fig_path(name):
    for d in (ART / "figures", ART):
        p = d / name
        if p.exists():
            return p
    return None


def show_fig(name, caption=""):
    p = fig_path(name)
    if p is not None:
        st.image(str(p), use_container_width=True)
        if caption:
            st.caption(caption)


def split_pm(value):
    """Ubah string 'rata-rata ± simpangan' dari bundel menjadi dua angka."""
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value), 0.0
    text = str(value).replace(",", ".")
    for sep in ("±", "+/-"):
        if sep in text:
            left, right = text.split(sep, 1)
            return float(left.strip()), float(right.strip())
    return float(text.strip()), 0.0


# ============================================================================ DATA
@st.cache_resource(show_spinner="Memuat bundel...")
def load_bundle():
    return joblib.load(BUNDLE_PATH)


@st.cache_data(show_spinner=False)
def load_results():
    if RESULT_PATH.exists():
        return json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    return None


@st.cache_data(show_spinner=False)
def load_csv(filename):
    p = ART / filename
    return pd.read_csv(p) if p.exists() else None


@st.cache_resource(show_spinner="Memuat model SBERT (pertama kali agak lama)...")
def load_sbert(model_name):
    """Mengembalikan (model, None) atau (None, pesan_galat). Galat tidak menghentikan aplikasi."""
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(model_name), None
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {str(e)[:140]}"


if not BUNDLE_PATH.exists():
    st.markdown(f'<div class="display">{esc(APP_TITLE)}</div>', unsafe_allow_html=True)
    st.error(f"Berkas {BUNDLE_PATH} tidak ditemukan. Jalankan notebook sampai selesai, lalu salin folder artifacts/ ke folder yang sama dengan app.py.")
    st.stop()

bundle = load_bundle()
RES = load_results()
names = bundle["names"]
meta = pd.DataFrame(bundle["meta"]).set_index("name").loc[names]
d_index = {n: i for i, n in enumerate(names)}
SEASON_ALL = bundle["season_all_label"]
MODELS = bundle["models"]
MAXLEN = bundle.get("max_len", 128)
N_ELIGIBLE = int((meta["season"].values == SEASON_ALL).sum())

# Metrik evaluasi sudah tersimpan di dalam bundel. hasil_untuk_artikel.json hanya pelengkap
# (analisis nama, contoh rekomendasi) bila tersedia.
METRICS = bundle.get("metrics") or {}
MEAN, STD = {}, {}
for _row in METRICS.get("summary", []):
    _label = _row.get("model")
    if not _label:
        continue
    MEAN[_label], STD[_label] = {}, {}
    for _col, _val in _row.items():
        if _col == "model":
            continue
        try:
            _m, _s = split_pm(_val)
        except (TypeError, ValueError):
            continue
        MEAN[_label][_col], STD[_label][_col] = _m, _s
RAND = METRICS.get("random") or {}
WILCOX = METRICS.get("wilcoxon") or []
COST = bundle.get("cost") or {}
ENV = bundle.get("env") or {}
CLEAN = bundle.get("clean_report") or {}
DSTATS = bundle.get("data_stats") or {}
PERCAT = pd.DataFrame(RES["per_category"]) if RES and RES.get("per_category") else load_csv("per_kategori.csv")
PROTO_E = pd.DataFrame(RES["protokol_E"]) if RES and RES.get("protokol_E") else load_csv("protokol_E_kueri_kategori.csv")
MAIN_LABEL = MODELS.get(bundle.get("default_model", ""), {}).get("label") or next(iter(MEAN), "")


def encode_query(entry, text):
    """Vektor kueri, atau (None, pesan) bila model tidak dapat dipakai."""
    k = entry["kind"]
    if k == "tfidf":
        return entry["vectorizer"].transform([text]), None
    if k == "lsa":
        return normalize(entry["svd"].transform(entry["vectorizer"].transform([text])))[0], None
    m, err = load_sbert(entry["name"])
    if m is None:
        return None, err
    m.max_seq_length = MAXLEN
    return m.encode([text], normalize_embeddings=True)[0], None


def sims(entry, q):
    P = entry["profiles"]
    if sparse.issparse(P):
        qq = q if sparse.issparse(q) else sparse.csr_matrix(np.asarray(q).reshape(1, -1))
        return np.asarray((P @ qq.T).todense()).ravel()
    return np.asarray(P @ np.asarray(q).ravel()).ravel()


def season_ok(season):
    if season in (None, "Semua musim", "Kemarau"):
        return np.ones(len(names), dtype=bool)
    return meta["season"].values == SEASON_ALL


def recommend(entry, q, season="Semua musim", cats=None, k=5, exclude=(), use_filter=True):
    s = sims(entry, q)
    ok = season_ok(season) if use_filter else np.ones(len(names), dtype=bool)
    if cats:
        ok = ok & meta["cat"].isin(cats).values
    s = np.where(ok, s, -np.inf)
    for e in exclude:
        s[d_index[e]] = -np.inf
    top = np.argsort(-s)[:k]
    return [(names[i], float(s[i])) for i in top if np.isfinite(s[i])]


def liked_query(entry, liked):
    """Profil pengguna = rata-rata profil destinasi favorit (data tidak memuat pengguna sungguhan)."""
    P = entry["profiles"]
    q = np.asarray(P[[d_index[n] for n in liked]].mean(axis=0)).ravel()
    q = q / (np.linalg.norm(q) + 1e-12)
    return sparse.csr_matrix(q.reshape(1, -1)) if sparse.issparse(P) else q


# ============================================================================ KOMPONEN
def header(crumb, title, lead):
    st.markdown(f'<div class="crumb">{esc(APP_TITLE)} &nbsp;/&nbsp; <b>{esc(crumb)}</b></div><div class="display">{esc(title)}</div><p class="lead">{esc(lead)}</p>', unsafe_allow_html=True)


def note(text, strong=""):
    s = f"<b>{esc(strong)}</b> " if strong else ""
    st.markdown(f'<div class="note">{s}{esc(text)}</div>', unsafe_allow_html=True)


def stat(col, label, value, sub=""):
    col.markdown(f'<div class="stat"><div class="l">{esc(label)}</div><div class="v">{esc(value)}</div><div class="s">{esc(sub)}</div></div>', unsafe_allow_html=True)


def html_table(df, best=None):
    """best: dict kolom -> 'max'/'min'; nilai terbaik ditebalkan."""
    best = best or {}
    head = "".join(f"<th>{esc(c)}</th>" for c in df.columns)
    pick = {}
    for c, mode in best.items():
        if c in df.columns:
            v = pd.to_numeric(df[c].astype(str).str.replace(",", ".").str.split(" ").str[0], errors="coerce")
            if v.notna().any():
                pick[c] = v.idxmax() if mode == "max" else v.idxmin()
    rows = ""
    for i, r in enumerate(df.itertuples(index=False)):
        idx = df.index[i]
        cells = "".join(f'<td class="{"best" if pick.get(c) == idx else ""}">{esc(v)}</td>' for c, v in zip(df.columns, r))
        rows += f"<tr>{cells}</tr>"
    st.markdown(f'<table class="t"><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table>', unsafe_allow_html=True)


def hbars(series, fmt="{:.0f}"):
    mx = float(series.max()) or 1.0
    out = "".join(f'<div class="hb"><div class="n">{esc(i)}</div><div class="b"><div style="width:{v / mx * 100:.0f}%"></div></div><div class="v">{fmt.format(v)}</div></div>' for i, v in series.items())
    st.markdown(out, unsafe_allow_html=True)


def result_card(rank, name, score, season=None):
    m = meta.loc[name]
    kws = bundle["keywords"].get(name, [])
    snip = (bundle["samples"].get(name) or [""])[0]
    rating = f'<span class="badge out">rating {dd(m["rating_mean"], 1)}</span>' if "rating_mean" in m and pd.notna(m["rating_mean"]) else ""
    bad = '<span class="badge bad">tidak layak di musim Penghujan</span>' if (season == "Penghujan" and m["season"] != SEASON_ALL) else ""
    st.markdown(
        f'<div class="card"><div class="row"><span class="rank">{rank:02d}</span><span class="title">{esc(name)}</span><span class="score">{dd(score)}</span></div>'
        f'<div><span class="badge solid">{esc(m["cat"])}</span><span class="badge soft">{esc(m["season"])}</span>'
        f'<span class="badge out">{ns(m["n_reviews"])} ulasan</span>{rating}{bad}</div>'
        f'<div class="bar"><div style="width:{max(0.0, min(1.0, score)) * 100:.0f}%"></div></div>'
        + (f'<p class="muted">Kata khas: {esc(", ".join(kws))}</p>' if kws else "")
        + (f'<p class="snip">&ldquo;{esc(snip)}&rdquo;</p>' if snip else "") + "</div>", unsafe_allow_html=True)


def show_results(res, ms, season=None):
    if not res:
        note("Tidak ada destinasi yang memenuhi filter. Longgarkan kategori atau musim.", "Hasil kosong.")
        return
    st.caption(f"{len(res)} rekomendasi  |  {dd(ms, 1)} ms")
    cols = st.columns(2)
    for i, (n, s) in enumerate(res):
        with cols[i % 2]:
            result_card(i + 1, n, s, season)


def label_of(k):
    return MODELS[k]["label"]


def tool_error(entry, err):
    note(f"{entry['label']} tidak dapat dimuat ({err}). Pasang sentence-transformers dan pastikan internet aktif, atau pilih representasi pembanding di sidebar.", "Model tidak tersedia.")


# ============================================================================ SIDEBAR
st.sidebar.markdown('<div class="sub" style="margin-top:4px">Wisata Sadar Musim</div><div class="caption">SBERT + Content-Based Filtering</div><br>', unsafe_allow_html=True)
page = st.sidebar.radio("Navigasi", ["Rekomendasi", "Bandingkan dengan Pembanding", "Uji Filter Musim", "Jelajah Destinasi", "Data & Pembersihan", "Evaluasi", "Tentang"], label_visibility="collapsed")
st.sidebar.markdown("<hr>", unsafe_allow_html=True)
keys = list(MODELS)
default = bundle.get("default_model", keys[0])
choice = st.sidebar.selectbox("Representasi teks", keys, index=keys.index(default) if default in keys else 0, format_func=label_of)
entry = MODELS[choice]
st.sidebar.markdown(f'<div class="caption" style="margin-top:12px">Model</div><div style="word-break:break-all">{esc(entry.get("name", ""))}</div>'
                    f'<div class="caption" style="margin-top:12px">Dimensi</div><div>{ns(entry["dim"])}</div>', unsafe_allow_html=True)
if bundle.get("demo"):
    note("Bundel ini ditandai data tiruan (demo). Hasil rekomendasi bukan hasil penelitian.", "Mode demo.")

# ============================================================================ REKOMENDASI
if page == "Rekomendasi":
    header("Rekomendasi", "Temukan destinasi dari ulasan", "Tulis preferensi atau pilih destinasi yang kamu suka. Sistem mencocokkannya dengan profil semantik setiap destinasi.")
    c = st.columns(4)
    stat(c[0], "Destinasi", ns(len(names)), "profil terindeks")
    stat(c[1], "Ulasan", ns(meta["n_reviews"].sum()), "membentuk profil")
    stat(c[2], "Dimensi", ns(entry["dim"]), entry["label"])
    stat(c[3], "Kategori", str(len(bundle["categories"])), "jenis wisata")
    tab1, tab2 = st.tabs(["Dari preferensi teks", "Mirip destinasi favorit"])
    with tab1:
        with st.form("f_text"):
            q = st.text_area("Preferensi", height=90, placeholder="mis. pantai sepi untuk sunset, cocok untuk keluarga")
            a, b_, c_ = st.columns(3)
            season = a.selectbox("Musim kunjungan", SEASONS)
            cats = b_.multiselect("Kategori (opsional)", bundle["categories"], placeholder="Semua kategori")
            k = c_.slider("Jumlah rekomendasi", 3, 10, 6)
            go = st.form_submit_button("Cari rekomendasi")
        if go:
            if not q.strip():
                note("Tulis preferensi terlebih dahulu.", "Kolom kosong.")
            else:
                t0 = time.perf_counter()
                qv, err = encode_query(entry, q)
                if qv is None:
                    tool_error(entry, err)
                else:
                    show_results(recommend(entry, qv, season, cats, k), (time.perf_counter() - t0) * 1000, season)
    with tab2:
        with st.form("f_like"):
            liked = st.multiselect("Destinasi yang kamu suka", names, placeholder="Pilih satu atau lebih destinasi")
            a, b_ = st.columns(2)
            season2 = a.selectbox("Musim kunjungan ", SEASONS)
            k2 = b_.slider("Jumlah rekomendasi ", 3, 10, 6)
            go2 = st.form_submit_button("Cari yang mirip")
        if go2:
            if not liked:
                note("Pilih minimal satu destinasi.", "Belum ada pilihan.")
            else:
                t0 = time.perf_counter()
                show_results(recommend(entry, liked_query(entry, liked), season2, None, k2, exclude=liked), (time.perf_counter() - t0) * 1000, season2)
    with st.expander("Bagaimana skor dihitung?"):
        st.markdown("1. Setiap ulasan diubah menjadi vektor oleh Sentence-BERT yang dibekukan.\n2. Vektor ulasan sebuah destinasi dirata-ratakan lalu dinormalisasi menjadi **profil destinasi**.\n"
                    "3. Kueri diubah menjadi vektor oleh model yang sama, lalu dicocokkan dengan semua profil lewat **kemiripan kosinus**.\n4. Destinasi yang tidak layak pada musim yang dipilih dibuang.\n\n"
                    f"Aturan musim: **Kemarau** meloloskan semua destinasi; **Penghujan** hanya destinasi berlabel *{SEASON_ALL}* (interpretasi penulis atas label data, belum terkonfirmasi).")

# ============================================================================ BANDINGKAN DENGAN PEMBANDING
elif page == "Bandingkan dengan Pembanding":
    header("Bandingkan dengan Pembanding", "Bandingkan dengan pembanding", "Kueri yang sama dijalankan pada setiap representasi di bundel (model utama dan pembanding acuan). Lihat seberapa sering daftar Top-K berbeda.")
    if len(keys) < 2:
        note("Bundel hanya memuat satu representasi. Aktifkan pembanding (RUN_PEMBANDING = True) di notebook untuk memakai halaman ini.", "Tidak ada pembanding.")
    with st.form("f_cmp"):
        q = st.text_area("Preferensi", height=80, placeholder="mis. air terjun sejuk untuk hiking")
        a, b_ = st.columns(2)
        season = a.selectbox("Musim kunjungan", SEASONS, key="cmp_season")
        k = b_.slider("Jumlah rekomendasi (K)", 3, 10, 5, key="cmp_k")
        go = st.form_submit_button("Bandingkan")
    if go:
        if not q.strip():
            note("Tulis preferensi terlebih dahulu.", "Kolom kosong.")
        else:
            out, skipped = {}, {}
            for key in keys:
                qv, err = encode_query(MODELS[key], q)
                if qv is None:
                    skipped[key] = err
                else:
                    out[key] = [n for n, _ in recommend(MODELS[key], qv, season, None, k)]
            cols = st.columns(max(1, len(out)))
            for col, (key, lst) in zip(cols, out.items()):
                items = "".join(f'<div style="padding:6px 0;border-bottom:1px solid var(--hair);font-size:14px"><span class="caption">{i + 1:02d}</span>&nbsp; {esc(n)}'
                                f'<div style="color:var(--mid);font-size:12px">{esc(meta.loc[n, "cat"])} | {esc(meta.loc[n, "season"])}</div></div>' for i, n in enumerate(lst))
                col.markdown(f'<div class="card"><div class="sub" style="margin-top:0;font-size:15px">{esc(label_of(key))}</div>{items}</div>', unsafe_allow_html=True)
            for key, err in skipped.items():
                note(f"{label_of(key)} dilewati: {err}", "Model tidak tersedia.")
            main_key = bundle.get("default_model", keys[0])
            if len(out) > 1 and main_key in out:
                rows = [[label_of(key), f"{len(set(lst) & set(out[main_key]))} dari {k}", f"{len(set(lst) & set(out[main_key])) / len(set(lst) | set(out[main_key])) * 100:.0f}%".replace(".", ",")]
                        for key, lst in out.items() if key != main_key]
                st.markdown(f'<div class="sub">Irisan Top-{k} dengan {esc(label_of(main_key))}</div>', unsafe_allow_html=True)
                html_table(pd.DataFrame(rows, columns=["representasi", "destinasi yang sama", "Jaccard"]))
                st.caption("Daftar yang berbeda berarti pilihan representasi memengaruhi hasil. Tidak ada kebenaran tunggal karena data tidak memuat preferensi pengguna.")

# ============================================================================ UJI FILTER MUSIM
elif page == "Uji Filter Musim":
    header("Uji Filter Musim", "Uji filter musim", "Peringkat berbasis konten tidak membawa informasi musim. Lihat berapa destinasi Top-K yang tidak layak pada musim Penghujan bila filter dimatikan.")
    with st.form("f_season"):
        q = st.text_area("Preferensi", height=80, placeholder="mis. pantai sepi untuk sunset dan cocok untuk keluarga")
        k = st.slider("Jumlah rekomendasi (K)", 5, 20, 10, key="sf_k")
        go = st.form_submit_button("Uji pada musim Penghujan")
    if go:
        if not q.strip():
            note("Tulis preferensi terlebih dahulu.", "Kolom kosong.")
        else:
            qv, err = encode_query(entry, q)
            if qv is None:
                tool_error(entry, err)
            else:
                raw = recommend(entry, qv, "Penghujan", None, k, use_filter=False)
                flt = recommend(entry, qv, "Penghujan", None, k, use_filter=True)
                viol = sum(1 for n, _ in raw if meta.loc[n, "season"] != SEASON_ALL)
                c = st.columns(3)
                stat(c[0], "Tanpa filter", f"{viol} dari {len(raw)}", "tidak layak di Penghujan")
                stat(c[1], "Dengan filter", f"0 dari {len(flt)}", "tidak layak (menurut aturan)")
                stat(c[2], "Destinasi layak", f"{N_ELIGIBLE} dari {len(names)}", f"{pct(N_ELIGIBLE / len(names))} kandidat tersisa")
                st.markdown('<div style="height:14px"></div>', unsafe_allow_html=True)
                L, R = st.columns(2)
                for col, title, lst in ((L, "Tanpa filter", raw), (R, "Dengan filter", flt)):
                    items = ""
                    for i, (n, s) in enumerate(lst):
                        bad = meta.loc[n, "season"] != SEASON_ALL
                        flag = '<span class="badge bad">tidak layak</span>' if bad else ""
                        items += (f'<div style="padding:6px 0;border-bottom:1px solid var(--hair);font-size:14px"><span class="caption">{i + 1:02d}</span>&nbsp; {esc(n)} {flag}'
                                  f'<div style="color:var(--mid);font-size:12px">{esc(meta.loc[n, "cat"])} | skor {dd(s)}</div></div>')
                    col.markdown(f'<div class="card"><div class="sub" style="margin-top:0;font-size:15px">{title}</div>{items}</div>', unsafe_allow_html=True)
    if MEAN:
        vv = [v["SeasonViol@10 tanpa filter"] for v in MEAN.values() if "SeasonViol@10 tanpa filter" in v]
        rv = RAND.get("SeasonViol@10 tanpa filter")
        st.markdown('<div class="sub">Hasil evaluasi seluruh data</div>', unsafe_allow_html=True)
        note(f"Pada evaluasi, {pct(min(vv))} sampai {pct(max(vv))} destinasi Top-10 tidak layak pada musim Penghujan tanpa filter"
             + (f", sama dengan peluang acak ({pct(rv)})" if rv is not None else "") +
             ". Jadi filter eksplisit diperlukan.", "Temuan.")
        show_fig("fig_musim.png", "(a) Pelanggaran musim Top-10 tanpa filter dibanding garis acak. (b) Hit@10 pada musim Penghujan sebelum dan sesudah filter.")

# ============================================================================ JELAJAH
elif page == "Jelajah Destinasi":
    header("Jelajah Destinasi", "Jelajah destinasi", "Telusuri destinasi beserta kategori, label musim, dan kata khas dari ulasannya.")
    a, b_, c_ = st.columns([2, 1, 1])
    qn = a.text_input("Cari nama", placeholder="ketik nama destinasi")
    cf = b_.selectbox("Kategori", ["Semua"] + bundle["categories"])
    sf = c_.selectbox("Label musim", ["Semua"] + bundle["season_labels"])
    t = meta.reset_index().copy()
    t["kata khas"] = t["name"].map(lambda n: ", ".join(bundle["keywords"].get(n, [])))
    if qn:
        t = t[t["name"].str.contains(qn, case=False, na=False)]
    if cf != "Semua":
        t = t[t["cat"] == cf]
    if sf != "Semua":
        t = t[t["season"] == sf]
    t = t.rename(columns={"name": "destinasi", "cat": "kategori", "season": "musim", "n_reviews": "ulasan", "rating_mean": "rating"})
    if "rating" in t:
        t["rating"] = t["rating"].round(2).map(lambda x: dd(x, 2))
    st.caption(f"{len(t)} destinasi (ditampilkan 14 teratas)")
    html_table(t.head(14))
    st.markdown('<div class="sub">Destinasi per kategori dan label musim</div>', unsafe_allow_html=True)
    ct = pd.crosstab(meta["cat"], meta["season"])
    ct["layak Penghujan"] = (ct.get(SEASON_ALL, 0) / ct.sum(axis=1)).map(lambda x: pct(x, 0))
    html_table(ct.reset_index().rename(columns={"cat": "kategori"}))
    st.caption("Label musim berkaitan erat dengan kategori, sehingga filter musim juga menyaring menurut kategori.")

# ============================================================================ DATA & PEMBERSIHAN
elif page == "Data & Pembersihan":
    header("Data & Pembersihan", "Data dan pembersihan", "Ringkasan data ulasan yang membentuk profil destinasi: laporan pembersihan, sebaran kategori, dan proyeksi profil semantik dari notebook.")
    rd_ = DSTATS.get("reviews_per_dest") or {}
    wr_ = DSTATS.get("words") or {}
    c = st.columns(4)
    stat(c[0], "Ulasan akhir", ns(DSTATS.get("n_reviews", len(meta))), f"dari {ns(CLEAN.get('ulasan_mentah', 0))} ulasan mentah" if CLEAN else "ulasan bersih")
    stat(c[1], "Destinasi", ns(DSTATS.get("n_dest", len(names))), f"dari {ns(CLEAN.get('destinasi_awal', len(names)))} destinasi awal" if CLEAN else "profil terindeks")
    stat(c[2], "Kategori", ns(DSTATS.get("n_cat", len(bundle["categories"]))), "jenis wisata")
    stat(c[3], "Median kata/ulasan", dd(wr_.get("median", 0), 0), f"median {ns(rd_.get('median', 0))} ulasan per destinasi")

    if CLEAN:
        clean_labels = {
            "ulasan_mentah": "Ulasan mentah", "teks_kosong": "Teks kosong",
            "mojibake_sebelum": "Mojibake sebelum perbaikan", "mojibake_sesudah": "Mojibake sesudah perbaikan",
            "penanda_excel_sebelum": "Penanda ekspor Excel sebelum perbaikan", "ulasan_terpotong_google": "Ulasan terpotong Google",
            "dibuang_terlalu_pendek": "Dibuang: terlalu pendek", "dibuang_duplikat": "Dibuang: duplikat",
            "dibuang_destinasi_kecil": "Dibuang: destinasi dengan ulasan sedikit", "destinasi_dibuang": "Destinasi dibuang",
            "ulasan_akhir": "Ulasan akhir", "destinasi_awal": "Destinasi awal", "destinasi_akhir": "Destinasi akhir",
            "metode_encoding": "Metode perbaikan encoding", "MIN_WORDS": "Ambang kata minimum",
            "MIN_REVIEWS_PER_DEST": "Ambang ulasan per destinasi",
        }
        rows = [[clean_labels.get(k, k), ns(v) if isinstance(v, (int, np.integer)) else str(v)] for k, v in CLEAN.items()]
        st.markdown('<div class="sub">Laporan pembersihan ulasan</div>', unsafe_allow_html=True)
        html_table(pd.DataFrame(rows, columns=["tahap", "nilai"]))

    show_fig("fig_data_overview.png", "(a) Panjang ulasan, (b) ulasan per destinasi, (c) destinasi per kategori, dan (d) kategori \u00d7 label musim.")

    if DSTATS.get("cat_counts"):
        sbc = DSTATS.get("season_by_cat") or {}
        rows = []
        for cat_, n_ in DSTATS["cat_counts"].items():
            d_ = sbc.get(cat_, {})
            rows.append([cat_, ns(n_), ns(d_.get("Kemarau", 0)), ns(d_.get(SEASON_ALL, 0)), pct((d_.get(SEASON_ALL, 0) / n_) if n_ else 0, 0)])
        st.markdown('<div class="sub">Sebaran kategori dan label musim</div>', unsafe_allow_html=True)
        html_table(pd.DataFrame(rows, columns=["kategori", "destinasi", "Kemarau", SEASON_ALL, "layak Penghujan"]))
        st.caption("Label musim berkaitan erat dengan kategori, sehingga filter musim juga menyaring menurut kategori.")

    rc_ = DSTATS.get("rating_counts")
    if rc_:
        st.markdown('<div class="sub">Sebaran rating ulasan</div>', unsafe_allow_html=True)
        hbars(pd.Series({f"{k} bintang": v for k, v in sorted(rc_.items())}), fmt="{:.0f}")

    show_fig("fig_proyeksi.png", "Proyeksi dua dimensi profil destinasi model utama; warna = kategori, bentuk = label musim.")

# ============================================================================ EVALUASI
elif page == "Evaluasi":
    header("Evaluasi", "Hasil evaluasi", "Empat protokol proksi, kepatuhan musim, karakteristik rekomendasi, dan biaya komputasi. Bila hasil_untuk_artikel.json tersedia, analisis penyebutan nama ikut ditampilkan. Data tidak memuat ID pengguna, sehingga ini bukan ukuran kepuasan pengguna.")
    if not MEAN:
        note("Bundel tidak memuat metrik evaluasi. Jalankan ulang sel evaluasi notebook agar cbf_bundle.pkl berisi bagian metrics.", "Belum ada hasil.")
    else:
        M, S, R = MEAN, STD, RAND
        labs = list(M)
        main = MAIN_LABEL if MAIN_LABEL in M else labs[0]
        mm = M[main]
        masked = "Hit@10 (nama disamarkan)" in mm
        vv = [M[m]["SeasonViol@10 tanpa filter"] for m in labs if "SeasonViol@10 tanpa filter" in M[m]]
        c = st.columns(3)
        if masked:
            stat(c[0], f"Model utama: {main}", f"Hit@10 {dd(mm['Hit@10 (nama disamarkan)'])}", f"nama disamarkan; asli {dd(mm['Hit@10'])}; acak {dd(R['Hit@10'])}")
        else:
            stat(c[0], f"Model utama: {main}", f"Hit@10 {dd(mm['Hit@10'])}", f"acak {dd(R['Hit@10'])}")
        stat(c[1], "Kemiripan kategori", f"Prec@10 {dd(mm['Prec@10'])}", f"acak {dd(R['Prec@10'])}")
        stat(c[2], "Pelanggaran musim tanpa filter", f"{pct(min(vv))}\u2013{pct(max(vv))}", f"acak {pct(R['SeasonViol@10 tanpa filter'])}")

        st.markdown('<div class="sub">Protokol A, B (rata-rata ± simpangan baku antar-seed)</div>', unsafe_allow_html=True)
        cols = [c_ for c_ in ["Hit@10", "Hit@10 (nama disamarkan)", "MRR@10", "MRR@10 (nama disamarkan)", "Prec@10", "NDCG@10"] if c_ in mm]
        rows = [[m] + [f"{dd(M[m][c_])} ± {dd(S[m].get(c_, 0), 4)}" for c_ in cols] for m in labs] + [["Acak (pembanding)"] + [dd(R[c_]) if c_ in R else "\u2014" for c_ in cols]]
        html_table(pd.DataFrame(rows, columns=["model"] + cols), best={c_: "max" for c_ in cols})
        st.caption("Nilai tebal = terbaik per kolom. A: Hit@K, MRR@10 (kueri = ulasan uji; varian \u201cnama disamarkan\u201d menghapus kata khas nama destinasi). B: Precision@K, NDCG@K (relevan = kategori sama).")

        nm = RES.get("name_mention") if RES else None
        if nm:
            st.markdown('<div class="sub">Pengaruh penyebutan nama destinasi (satu pembagian)</div>', unsafe_allow_html=True)
            e0 = next(iter(nm["per_encoder"].values()))
            has_ms = "Hit@10_mention_disamarkan" in e0
            hdr = ["model", "Hit@10 menyebut nama"] + (["Hit@10 nama disamarkan"] if has_ms else []) + ["Hit@10 tanpa nama"]
            rows = [[m, dd(v["Hit@10_mention"])] + ([dd(v["Hit@10_mention_disamarkan"])] if has_ms else []) + [dd(v["Hit@10_nomention"])] for m, v in nm["per_encoder"].items()]
            html_table(pd.DataFrame(rows, columns=hdr))
            st.caption(f"{pct(e0['share_mention_test'])} ulasan uji ({ns(e0['n_mention'])} ulasan) menyebut kata khas nama destinasi. Selisih kolom kedua dan ketiga menunjukkan seberapa besar hasil bergantung pada pencocokan nama.")

        if "Coverage@10" in mm:
            st.markdown('<div class="sub">Protokol D: karakteristik rekomendasi (kueri = ulasan uji, Top-10)</div>', unsafe_allow_html=True)
            cd = [c_ for c_ in ["Coverage@10", "Gini rekomendasi", "Kategori unik@10", "Korelasi popularitas"] if c_ in mm]
            rows = [[m] + [dd(M[m][c_]) for c_ in cd] for m in labs] + [["Acak (pembanding)"] + [dd(R[c_]) if c_ in R else "\u2014" for c_ in cd]]
            html_table(pd.DataFrame(rows, columns=["model"] + cd))
            st.caption("Cakupan = proporsi destinasi yang pernah muncul di Top-10; Gini tinggi = rekomendasi menumpuk pada sedikit destinasi; kategori unik = keragaman Top-10; korelasi popularitas = hubungan jumlah ulasan dan seringnya direkomendasikan.")
            show_fig("fig_rekomender.png", "(a) Cakupan katalog, (b) Gini frekuensi rekomendasi, (c) kategori unik per Top-10; garis putus-putus = pembanding acak.")

        if PROTO_E is not None and len(PROTO_E):
            st.markdown('<div class="sub">Protokol E: kueri teks pendek berupa nama kategori (Precision@10)</div>', unsafe_allow_html=True)
            dfE = PROTO_E.drop(columns=[c_ for c_ in ["index"] if c_ in PROTO_E.columns]).copy()
            fmtcols = [c_ for c_ in dfE.columns if c_ not in ("kategori", "kueri", "n_destinasi")]
            mac = (RES.get("protokol_E_macro") if RES else None) or {}
            if not mac:
                big = dfE["n_destinasi"] >= 10
                mac = {c_: float(pd.to_numeric(dfE.loc[big, c_], errors="coerce").mean()) for c_ in fmtcols}
            for c_ in fmtcols:
                dfE[c_] = dfE[c_].map(lambda x: dd(x) if pd.notna(x) else "\u2014")
            html_table(dfE.rename(columns={"n_destinasi": "destinasi"}))
            if mac:
                st.caption("Rata-rata pada kategori beranggota 10 destinasi atau lebih: " + "; ".join(f"{k}: {dd(v)}" for k, v in mac.items()) + ". Kata kunci kueri juga muncul pada nama destinasi, sehingga menguntungkan pencocokan leksikal.")

        st.markdown('<div class="sub">Kepatuhan musim</div>', unsafe_allow_html=True)
        rows = [[m, pct(M[m]["SeasonViol@10 tanpa filter"]), "0,0%", dd(M[m]["Hit@10 Penghujan tanpa filter"]), dd(M[m]["Hit@10 Penghujan dengan filter"])] for m in labs]
        rows.append(["Acak (pembanding)", pct(R["SeasonViol@10 tanpa filter"]), "0,0%", "\u2014", "\u2014"])
        html_table(pd.DataFrame(rows, columns=["model", "pelanggaran@10 tanpa filter", "pelanggaran@10 dengan filter", "Hit@10 Penghujan tanpa filter", "Hit@10 Penghujan dengan filter"]))
        st.caption("Pelanggaran dengan filter bernilai nol menurut aturan, bukan hasil empiris. Kenaikan Hit@10 setelah filter bersifat mekanis karena kandidat berkurang.")
        show_fig("fig_musim.png", "(a) Pelanggaran musim tanpa filter dibanding garis acak. (b) Hit@10 pada musim Penghujan sebelum dan sesudah filter.")

        W_ = WILCOX
        if W_:
            st.markdown('<div class="sub">Uji Wilcoxon berpasangan per destinasi (MRR@10)</div>', unsafe_allow_html=True)
            sfx = " (nama disamarkan)" if f"p (Wilcoxon){' (nama disamarkan)'}" in W_[0] else ""
            rows = [[w["A"], w["B"], dd(w[f"MRR@10 A{sfx}"], 4), dd(w[f"MRR@10 B{sfx}"], 4), dd(w[f"selisih median{sfx}"], 4), f"{w[f'p (Wilcoxon){sfx}']:.1e}".replace(".", ",")] for w in W_]
            html_table(pd.DataFrame(rows, columns=["A", "B", "MRR@10 A", "MRR@10 B", "selisih median (A\u2212B)", "p"]))
            st.caption("Dihitung pada kueri " + ("dengan nama disamarkan." if sfx else "asli."))

        st.markdown('<div class="sub">Biaya komputasi</div>', unsafe_allow_html=True)
        cost, env = COST, ENV
        order = [k for k in cost if cost[k].get("kind") == "sbert"] + [k for k in cost if cost[k].get("kind") != "sbert"]
        rows = [[cost[k]["label"], ns(cost[k]["dim"]), dd(cost[k]["encode_seconds_per_1000"]), dd(cost[k]["embeddings_MB"], 1), dd(cost[k]["query_latency_ms_median"], 2),
                 "GPU" if cost[k].get("device") == "cuda" else "CPU"] for k in order]
        html_table(pd.DataFrame(rows, columns=["representasi", "dimensi", "detik / 1.000 ulasan", "vektor (MB)", "latensi kueri (ms)", "perangkat"]))
        st.caption(f"Perangkat: {env.get('platform', '')}, {env.get('cpu_count', '')} inti, RAM {env.get('ram_gb', '')} GB, GPU {env.get('gpu', 'tidak ada')}. "
                   "TF-IDF dijalankan di CPU dan waktunya mencakup pelatihan; SBERT di GPU. Jangan membandingkan waktu antarperangkat secara langsung.")

        st.markdown(f'<div class="sub">Per kategori ({esc(main)})</div>', unsafe_allow_html=True)
        if PERCAT is not None and len(PERCAT):
            pc = PERCAT.drop(columns=[c_ for c_ in ["index"] if c_ in PERCAT.columns]).copy()
            if "batas_atas_Prec@10" not in pc.columns:
                pc["batas_atas_Prec@10"] = ((pd.to_numeric(pc["n_destinasi"], errors="coerce") - 1) / 10).clip(upper=1.0)
            rows = [[r_["kategori"], ns(r_["n_destinasi"]), dd(r_["Hit@10 (A)"]), dd(r_["Prec@10 (B)"]), dd(r_["batas_atas_Prec@10"])] for _, r_ in pc.iterrows()]
            html_table(pd.DataFrame(rows, columns=["kategori", "destinasi", "Hit@10 (A)", "Prec@10 (B)", "batas atas Prec@10"]))
            st.caption("Precision@10 kategori beranggota n destinasi tidak dapat melebihi (n\u22121)/10. Kueri = ulasan uji pada kategori yang sama.")
            show_fig("fig_per_kategori.png", "Precision@10 per kategori dibanding batas atas dan garis acak.")
        else:
            note("Tabel per kategori belum tersedia. Jalankan notebook sampai sel evaluasi per kategori.", "Data tidak ada.")

        ct_ = RES.get("contoh_rekomendasi") if RES else None
        if ct_:
            with st.expander("Contoh rekomendasi (Top-5, model utama)"):
                for r_ in [x for x in ct_ if x["model"] == main]:
                    st.markdown(f"**{esc(r_['kueri'])}** \u2014 musim {esc(r_['musim'])}")
                    st.caption(" | ".join(f"{t['destinasi']} ({t['kategori']})" for t in r_["top5"]))
        sk = (RES.get("skipped_encoders") if RES else None) or {}
        if sk:
            note("; ".join(f"{k}: {v}" for k, v in sk.items()), "Model yang tidak dijalankan.")

# ============================================================================ TENTANG
else:
    header("Tentang", "Tentang proyek", "Rekomendasi destinasi wisata berbasis ulasan, dengan Sentence-BERT yang dibekukan dan filter musim.")
    st.markdown('<div class="card"><div class="sub" style="margin-top:0">Alur</div><p class="muted">Ulasan &rarr; pembersihan &rarr; Sentence-BERT (beku) &rarr; profil destinasi &rarr; kemiripan kosinus &rarr; filter musim &rarr; Top-K</p></div>', unsafe_allow_html=True)
    st.markdown('<div class="sub">Keterbatasan</div>', unsafe_allow_html=True)
    for t in ["Data tidak memuat ID pengguna, jadi tidak ada penyaringan kolaboratif dan evaluasi hanya proksi.",
              "Label musim hanya dua nilai, tetap per destinasi, dan berkaitan erat dengan kategori; aturan filter adalah interpretasi penulis.",
              "Content-based filtering cenderung memberi hasil yang mirip dengan masukan; keragaman diukur pada Protokol D.",
              "Kategori wisata dipakai sebagai proksi relevansi, bukan preferensi pengguna.",
              "TF-IDF hanya pembanding acuan; sistem ini tidak dirancang sebagai perbandingan model."]:
        st.markdown(f"- {t}")
    st.markdown('<div class="sub">Representasi yang tersedia</div>', unsafe_allow_html=True)
    rows = [[v["label"], v["kind"].upper(), ns(v["dim"]), str(v.get("name", ""))] for v in MODELS.values()]
    html_table(pd.DataFrame(rows, columns=["representasi", "jenis", "dimensi", "model"]))
    if ENV:
        st.markdown('<div class="sub">Lingkungan eksperimen (dari notebook)</div>', unsafe_allow_html=True)
        pairs = [("Platform", ENV.get("platform", "")), ("CPU", f"{ENV.get('cpu_count', '')} inti"),
                 ("RAM", f"{ENV.get('ram_gb', '')} GB"), ("GPU", ENV.get("gpu", "tidak ada")),
                 ("Python", ENV.get("python", "")), ("sentence-transformers", ENV.get("sentence_transformers", "")),
                 ("torch", ENV.get("torch", "")), ("numpy", ENV.get("numpy", "")),
                 ("pandas", ENV.get("pandas", "")), ("scikit-learn", ENV.get("sklearn", ""))]
        html_table(pd.DataFrame(pairs, columns=["komponen", "nilai"]))
    if CLEAN:
        st.markdown('<div class="sub">Data yang dipakai</div>', unsafe_allow_html=True)
        st.markdown(f"{ns(CLEAN.get('ulasan_akhir', 0))} ulasan dari {ns(CLEAN.get('destinasi_akhir', 0))} destinasi, "
                    f"disaring dengan ambang {ns(CLEAN.get('MIN_WORDS', 0))} kata dan {ns(CLEAN.get('MIN_REVIEWS_PER_DEST', 0))} ulasan per destinasi."
                    + (f" Sumber: {esc(RES.get('data_source'))}." if RES and RES.get("data_source") else ""))
    st.markdown('<div class="sub">Tim</div>', unsafe_allow_html=True)
    html_table(pd.DataFrame(TEAM, columns=["Nama", "NIM"]))
    st.caption(f"Bundel versi {bundle['version']} | dibuat {bundle['created']}")
