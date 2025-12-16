
import io
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, date

import pandas as pd
import streamlit as st

# PDF tooling (optional)
from pypdf import PdfReader, PdfWriter
import pdfplumber

try:
    from rapidfuzz import fuzz
except Exception:
    fuzz = None  # optional

st.set_page_config(page_title="Conciliação Fatura x Mobills", layout="wide")

BRL_AMOUNT_RE = re.compile(r"(?P<sign>-)?(?P<int>\d{1,3}(?:\.\d{3})*|\d+),(?P<dec>\d{2})")
DATE_RE_DEFAULT = r"(?P<d>\d{2})/(?P<m>\d{2})(?:/(?P<y>\d{2,4}))?"

def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"\s+", " ", s)
    return s

def brl_to_cents(x) -> int | None:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    if isinstance(x, (int, float)) and not pd.isna(x):
        return int(round(float(x) * 100))
    s = str(x).strip()
    s = s.replace("R$", "").replace(" ", "")
    # Accept "R$ -14.311,45" and "-14311,45"
    m = BRL_AMOUNT_RE.search(s)
    if not m:
        try:
            return int(round(float(s.replace(".", "").replace(",", ".")) * 100))
        except Exception:
            return None
    sign = -1 if m.group("sign") else 1
    integer = int(m.group("int").replace(".", ""))
    dec = int(m.group("dec"))
    return sign * (integer * 100 + dec)

def parse_date_any(x, dayfirst=True) -> date | None:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    if isinstance(x, (datetime, pd.Timestamp)):
        return x.date()
    s = str(x).strip()
    for fmt in ("%d/%m/%Y", "%d/%m/%y", "%Y-%m-%d", "%d-%m-%Y", "%d-%m-%y"):
        try:
            return datetime.strptime(s, fmt).date()
        except Exception:
            pass
    try:
        ts = pd.to_datetime(s, dayfirst=dayfirst, errors="coerce")
        if pd.isna(ts):
            return None
        return ts.date()
    except Exception:
        return None

def guess_col_index(cols: list[str], keywords: list[str], default: int = 0) -> int:
    """Heurística simples pra escolher a coluna certa sem o usuário sofrer."""
    cols_l = [str(c).strip().lower() for c in cols]
    for kw in keywords:
        kw = kw.lower()
        for i, c in enumerate(cols_l):
            if kw in c:
                return i
    return min(default, max(0, len(cols) - 1))


def decrypt_pdf(pdf_bytes: bytes, password: str) -> bytes:
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        if reader.is_encrypted:
            ok = reader.decrypt(password)
            if ok == 0:
                raise ValueError("Senha incorreta ou PDF não suportado.")
            writer = PdfWriter()
            for page in reader.pages:
                writer.add_page(page)
            out = io.BytesIO()
            writer.write(out)
            return out.getvalue()
        return pdf_bytes
    except Exception as e:
        raise RuntimeError(f"Falha ao abrir/descriptografar o PDF: {e}")

@dataclass
class PdfParseConfig:
    date_regex: str = DATE_RE_DEFAULT
    min_chars_desc: int = 3

def extract_transactions_from_pdf(pdf_bytes: bytes, cfg: PdfParseConfig) -> pd.DataFrame:
    rows = []
    date_re = re.compile(cfg.date_regex)
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page_i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            for raw_line in text.splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                dm = date_re.search(line)
                am = BRL_AMOUNT_RE.search(line)
                if not dm or not am:
                    continue

                dd = int(dm.group("d"))
                mm = int(dm.group("m"))
                yy = dm.groupdict().get("y")
                if yy is None:
                    yy = datetime.now().year
                else:
                    yy = int(yy)
                    if yy < 100:
                        yy += 2000
                try:
                    d = date(yy, mm, dd)
                except Exception:
                    continue

                cents = brl_to_cents(am.group(0))
                if cents is None:
                    continue

                desc = line
                desc = date_re.sub(" ", desc, count=1)
                desc = BRL_AMOUNT_RE.sub(" ", desc, count=1)
                desc = re.sub(r"\s+", " ", desc).strip()
                if len(desc) < cfg.min_chars_desc:
                    continue

                rows.append({
                    "data": d,
                    "descricao": desc,
                    "centavos": int(cents),
                    "origem": "fatura_pdf",
                    "pagina": page_i + 1,
                })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["descricao_norm"] = df["descricao"].map(normalize_text)
    return df

def load_mobills(file) -> pd.DataFrame:
    name = file.name.lower()
    if name.endswith(".csv"):
        # Try ; first (common in BR)
        try:
            df = pd.read_csv(file, sep=";", encoding="utf-8")
        except Exception:
            file.seek(0)
            df = pd.read_csv(file)
    else:
        df = pd.read_excel(file, sheet_name=None)
        # choose best sheet: first with at least 3 columns
        if isinstance(df, dict):
            best = None
            for k, v in df.items():
                if v.shape[1] >= 3 and v.shape[0] >= 1:
                    best = k
                    break
            df = df[best] if best else list(df.values())[0]
    return df

def build_tx(df_raw: pd.DataFrame, col_data: str, col_desc: str, col_val: str, origem: str) -> pd.DataFrame:
    df = df_raw.copy()
    df = df[[col_data, col_desc, col_val]].rename(columns={
        col_data: "data",
        col_desc: "descricao",
        col_val: "valor"
    })
    df["data"] = df["data"].apply(parse_date_any)
    df["centavos"] = df["valor"].apply(brl_to_cents).astype("Int64")
    df["descricao"] = df["descricao"].astype(str)
    df["descricao_norm"] = df["descricao"].map(normalize_text)
    df = df.dropna(subset=["data", "centavos"]).copy()
    df["centavos"] = df["centavos"].astype(int)
    df["origem"] = origem
    return df[["data", "descricao", "descricao_norm", "centavos", "origem"]]

def reconcile_exact(df_fatura: pd.DataFrame, df_mobills: pd.DataFrame):
    def index_by_key(df):
        d = {}
        for i, row in df.iterrows():
            key = (row["data"], row["centavos"])
            d.setdefault(key, []).append(i)
        return d

    f_idx = index_by_key(df_fatura)
    m_idx = index_by_key(df_mobills)

    matched_pairs = []
    for key in set(f_idx.keys()).intersection(set(m_idx.keys())):
        f_list = f_idx[key]
        m_list = m_idx[key]
        k = min(len(f_list), len(m_list))
        for j in range(k):
            matched_pairs.append((f_list[j], m_list[j]))

    matched_f_ids = {a for a, _ in matched_pairs}
    matched_m_ids = {b for _, b in matched_pairs}

    matched = []
    for f_i, m_i in matched_pairs:
        f = df_fatura.loc[f_i]
        m = df_mobills.loc[m_i]
        matched.append({
            "data": f["data"],
            "centavos": f["centavos"],
            "valor": f["centavos"] / 100.0,
            "descricao_fatura": f["descricao"],
            "descricao_mobills": m["descricao"],
        })
    df_matched = pd.DataFrame(matched)

    only_fatura = df_fatura.loc[~df_fatura.index.isin(matched_f_ids)].copy()
    only_mobills = df_mobills.loc[~df_mobills.index.isin(matched_m_ids)].copy()
    return df_matched, only_fatura, only_mobills

def suggest_cent_adjustments(only_fatura: pd.DataFrame, only_mobills: pd.DataFrame, max_abs_cents=5) -> pd.DataFrame:
    if only_fatura.empty or only_mobills.empty:
        return pd.DataFrame()
    cand = []
    mob_by_date = {}
    for i, r in only_mobills.iterrows():
        mob_by_date.setdefault(r["data"], []).append((i, r))
    for f_i, f in only_fatura.iterrows():
        for m_i, m in mob_by_date.get(f["data"], []):
            delta = f["centavos"] - m["centavos"]
            if abs(delta) == 0 or abs(delta) > max_abs_cents:
                continue
            if fuzz:
                score = fuzz.token_set_ratio(f["descricao_norm"], m["descricao_norm"])
            else:
                f_set = set(f["descricao_norm"].split())
                m_set = set(m["descricao_norm"].split())
                score = int(100 * (len(f_set & m_set) / max(1, len(f_set | m_set))))
            cand.append({
                "data": f["data"],
                "valor_fatura": f["centavos"]/100.0,
                "valor_mobills": m["centavos"]/100.0,
                "delta_centavos": int(delta),
                "descricao_fatura": f["descricao"],
                "descricao_mobills": m["descricao"],
                "similaridade": score,
            })
    df = pd.DataFrame(cand)
    if df.empty:
        return df
    return df.sort_values(["similaridade", "data"], ascending=[False, True])

def mobills_import_csv(df_only_fatura: pd.DataFrame) -> bytes:
    out = pd.DataFrame({
        "Data": df_only_fatura["data"].astype(str),
        "Descrição": df_only_fatura["descricao"].astype(str),
        "Valor": (df_only_fatura["centavos"] / 100.0).map(lambda v: f"-{abs(v):.2f}".replace(".", ",")),
    })
    buf = io.StringIO()
    out.to_csv(buf, index=False, sep=";")
    return buf.getvalue().encode("utf-8")

def fmt_brl_from_cents(cents: int) -> str:
    v = cents / 100.0
    s = f"{v:,.2f}"
    s = s.replace(",", "X").replace(".", ",").replace("X", ".")
    return f"R$ {s}"

# ----------------------------
# UI
# ----------------------------
st.title("Conciliação exata: Fatura x Mobills (sem tolerância)")

with st.sidebar:
    st.subheader("Arquivos")
    fatura_file = st.file_uploader("Fatura do cartão (CSV ou PDF)", type=["csv", "pdf"])
    mob_file = st.file_uploader("Export do Mobills (XLS/XLSX/CSV)", type=["xls", "xlsx", "csv"])

    st.subheader("Regras da fatura")
    ignore_payment = st.checkbox("Ignorar pagamento da última fatura (linha negativa)", value=True)
    ignore_keywords = st.text_input("Palavras-chave para pagamento (separadas por ;)", value="pagament;pagto")
    ignore_exact_value = st.text_input("Ou ignore exatamente este valor (opcional)", value="", help="Ex.: -14311,45 ou R$ -14.311,45")

    st.subheader("PDF (se usar)")
    default_pwd = st.secrets.get("PDF_PASSWORD", "00475")
    pdf_pwd = st.text_input("Senha do PDF", value=default_pwd, type="password")
    date_regex = st.text_input("Regex de data no PDF", value=DATE_RE_DEFAULT)
    min_desc = st.number_input("Descrição mínima (chars)", min_value=1, max_value=50, value=3, step=1)

    st.subheader("Sugestões (centavos)")
    max_abs = st.number_input("Sugerir ajustes até (centavos)", min_value=1, max_value=50, value=5, step=1)

if not fatura_file or not mob_file:
    st.info("Suba a fatura e o export do Mobills para começar. O app não salva seus arquivos.")
    st.stop()

# ---- Load fatura
df_fatura = None
ignored_df = pd.DataFrame()

if fatura_file.name.lower().endswith(".pdf"):
    with st.status("Lendo fatura (PDF)...", expanded=False) as status:
        pdf_bytes = fatura_file.read()
        pdf_bytes = decrypt_pdf(pdf_bytes, pdf_pwd)
        cfg = PdfParseConfig(date_regex=date_regex, min_chars_desc=int(min_desc))
        df_fatura = extract_transactions_from_pdf(pdf_bytes, cfg)
        status.update(label=f"PDF processado: {len(df_fatura)} lançamentos detectados.", state="complete")
    if df_fatura.empty:
        st.error("Não consegui detectar lançamentos no PDF. Se seu PDF for imagem, precisa OCR. Se for texto, o layout precisa de ajuste.")
        st.stop()
else:
    # CSV
    with st.status("Lendo fatura (CSV)...", expanded=False) as status:
        # try ; then ,
        fatura_file.seek(0)
        try:
            df_raw = pd.read_csv(fatura_file, sep=";", encoding="utf-8-sig")
        except Exception:
            fatura_file.seek(0)
            df_raw = pd.read_csv(fatura_file)
        status.update(label=f"CSV carregado: {df_raw.shape[0]} linhas.", state="complete")

    st.subheader("Mapeamento de colunas da fatura (CSV)")
    cols_f = list(df_raw.columns)
    c1, c2, c3 = st.columns(3)
    with c1:
        col_data_f = st.selectbox("Coluna de data", cols_f, index=guess_col_index(cols_f, ["data", "date"], default=0))
    with c2:
        col_desc_f = st.selectbox("Coluna de descrição", cols_f, index=guess_col_index(cols_f, ["estabele", "descr", "descricao", "hist", "lanc", "lanç"], default=1))
    with c3:
        col_val_f = st.selectbox("Coluna de valor", cols_f, index=guess_col_index(cols_f, ["valor", "value", "amount", "total"], default=2))
    df_fatura = build_tx(df_raw, col_data_f, col_desc_f, col_val_f, origem="fatura_csv")

# ---- Apply ignore payment rule
if ignore_payment and not df_fatura.empty:
    kw = [k.strip() for k in ignore_keywords.split(";") if k.strip()]
    kw_re = None
    if kw:
        kw_re = re.compile("|".join(re.escape(k) for k in kw))
    exact_cents = brl_to_cents(ignore_exact_value) if ignore_exact_value.strip() else None

    mask = df_fatura["centavos"] < 0
    if kw_re is not None:
        mask = mask & df_fatura["descricao_norm"].str.contains(kw_re)
    if exact_cents is not None:
        mask = mask | (df_fatura["centavos"] == exact_cents)

    ignored_df = df_fatura.loc[mask].copy()
    df_fatura = df_fatura.loc[~mask].copy()

# ---- Load mobills
with st.status("Lendo export do Mobills...", expanded=False) as status:
    df_m_raw = load_mobills(mob_file)
    status.update(label=f"Mobills carregado: {df_m_raw.shape[0]} linhas.", state="complete")

st.subheader("Mapeamento de colunas do Mobills")
cols_m = list(df_m_raw.columns)
c1, c2, c3 = st.columns(3)
with c1:
    col_data_m = st.selectbox("Coluna de data (Mobills)", cols_m, index=guess_col_index(cols_m, ["data", "date"], default=0))
with c2:
    col_desc_m = st.selectbox("Coluna de descrição (Mobills)", cols_m, index=guess_col_index(cols_m, ["descr", "descricao", "hist", "lanc", "lanç"], default=1))
with c3:
    col_val_m = st.selectbox("Coluna de valor (Mobills)", cols_m, index=guess_col_index(cols_m, ["valor", "value", "amount", "total"], default=2))
df_mobills = build_tx(df_m_raw, col_data_m, col_desc_m, col_val_m, origem="mobills")
if df_mobills.empty:
    st.error("Mobills ficou com 0 lançamentos válidos. Quase sempre é coluna de VALOR errada (ex.: você selecionou Categoria).")
    st.stop()

# ---- Reconcile
df_matched, only_fatura, only_mobills = reconcile_exact(df_fatura, df_mobills)
df_sug = suggest_cent_adjustments(only_fatura, only_mobills, max_abs_cents=int(max_abs))

total_f = int(df_fatura["centavos"].sum()) if not df_fatura.empty else 0
total_m = int(df_mobills["centavos"].sum()) if not df_mobills.empty else 0
diff = total_f - total_m

st.divider()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Fatura (R$)", fmt_brl_from_cents(total_f))
c2.metric("Total Mobills (R$)", fmt_brl_from_cents(total_m))
c3.metric("Diferença (Fatura - Mobills)", fmt_brl_from_cents(diff))
c4.metric("Conciliados (exatos)", f"{len(df_matched)}")

tab0, tab1, tab2, tab3, tab4 = st.tabs(["🧾 Ignorados", "✅ Conciliados", "🟥 Fatura sem Mobills (A)", "🟨 Mobills sem Fatura (B)", "🛠 Ajustes (centavos) (C)"])

with tab0:
    st.write("Linhas ignoradas na fatura (normalmente o pagamento da fatura anterior).")
    if ignored_df.empty:
        st.info("Nenhuma linha ignorada com as regras atuais.")
    else:
        st.dataframe(ignored_df.assign(valor=ignored_df["centavos"]/100.0), use_container_width=True, height=320)

with tab1:
    st.write("Bateu 100% no valor e na data (duplicados tratados).")
    st.dataframe(df_matched, use_container_width=True, height=420)

with tab2:
    st.write("Estão na fatura, mas não aparecem no Mobills. Provável esquecimento de lançamento.")
    st.dataframe(only_fatura.assign(valor=only_fatura["centavos"]/100.0), use_container_width=True, height=420)
    st.download_button(
        "Baixar CSV para importar no Mobills (genérico)",
        data=mobills_import_csv(only_fatura),
        file_name="faltando_no_mobills.csv",
        mime="text/csv"
    )

with tab3:
    st.write("Estão no Mobills, mas não aparecem na fatura. Possível erro de lançamento ou algo para contestar.")
    st.dataframe(only_mobills.assign(valor=only_mobills["centavos"]/100.0), use_container_width=True, height=420)

with tab4:
    st.write("SUGESTÕES (não conciliadas): mesma data, descrição parecida e diferença pequena em centavos. Aqui você decide onde ajustar.")
    if df_sug.empty:
        st.info("Nenhuma sugestão de ajuste encontrada com os parâmetros atuais.")
    else:
        st.dataframe(df_sug, use_container_width=True, height=420)
        st.caption("Sem tolerância: isso NÃO concilia sozinho. Serve só para você achar o ajuste no Mobills mais rápido.")

st.caption("Privacidade: dá para hospedar grátis (Streamlit Cloud) ou no seu servidor. O app não precisa salvar nada.")
