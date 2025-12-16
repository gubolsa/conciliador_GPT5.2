
import io
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, date, timedelta

import pandas as pd
import streamlit as st

try:
    from rapidfuzz import fuzz
except Exception:
    fuzz = None

st.set_page_config(page_title="Conciliação Fatura x Mobills (cards)", layout="wide")

BRL_AMOUNT_RE = re.compile(r"(?P<sign>-)?(?P<int>\d{1,3}(?:\.\d{3})*|\d+),(?P<dec>\d{2})")

def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"\s+", " ", s)
    return s

def brl_to_cents(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    if isinstance(x, (int, float)) and not pd.isna(x):
        return int(round(float(x) * 100))
    s = str(x).strip()
    s = s.replace("R$", "").replace(" ", "")
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

def parse_date_any(x):
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
        dtp = pd.to_datetime(s, dayfirst=True, errors="coerce")
        if pd.isna(dtp):
            return None
        return dtp.date()
    except Exception:
        return None

def guess_col(cols, candidates):
    cols_norm = {c: normalize_text(c) for c in cols}
    for cand in candidates:
        cand_norm = normalize_text(cand)
        for c, cn in cols_norm.items():
            if cn == cand_norm:
                return c
    for cand in candidates:
        cand_norm = normalize_text(cand)
        for c, cn in cols_norm.items():
            if cand_norm in cn:
                return c
    return None

def load_fatura_csv(file) -> pd.DataFrame:
    raw = file.read()
    for sep in [",", ";", "\t"]:
        try:
            df = pd.read_csv(io.BytesIO(raw), sep=sep)
            if df.shape[1] >= 3:
                return df
        except Exception:
            continue
    return pd.read_csv(io.BytesIO(raw))

def build_fatura_tx(df_raw: pd.DataFrame, col_data: str, col_desc: str, col_val: str) -> pd.DataFrame:
    df = df_raw.copy()
    df = df[[col_data, col_desc, col_val]].rename(columns={col_data:"data", col_desc:"descricao", col_val:"valor"})
    df["data"] = df["data"].apply(parse_date_any)
    df["centavos"] = df["valor"].apply(brl_to_cents).astype("Int64")
    df["descricao"] = df["descricao"].astype(str)
    df["descricao_norm"] = df["descricao"].map(normalize_text)
    df = df.dropna(subset=["data","centavos"]).copy()
    df["centavos"] = df["centavos"].astype(int)
    df["id"] = range(len(df))
    df["origem"] = "fatura"
    return df[["id","data","descricao","descricao_norm","centavos","origem"]]

def load_mobills(file) -> pd.DataFrame:
    name = file.name.lower()
    if name.endswith(".csv"):
        raw = file.read()
        for sep in [",",";","\t"]:
            try:
                df = pd.read_csv(io.BytesIO(raw), sep=sep)
                if df.shape[1] >= 3:
                    return df
            except Exception:
                continue
        return pd.read_csv(io.BytesIO(raw))
    return pd.read_excel(file)

def build_mobills_tx(df_raw: pd.DataFrame, col_data: str, col_desc: str, col_val: str) -> pd.DataFrame:
    df = df_raw.copy()
    df = df[[col_data, col_desc, col_val]].rename(columns={col_data:"data", col_desc:"descricao", col_val:"valor"})
    df["data"] = df["data"].apply(parse_date_any)
    df["centavos"] = df["valor"].apply(brl_to_cents).astype("Int64")
    df["descricao"] = df["descricao"].astype(str)
    df["descricao_norm"] = df["descricao"].map(normalize_text)
    df = df.dropna(subset=["data","centavos"]).copy()
    df["centavos"] = df["centavos"].astype(int)
    df["id"] = range(len(df))
    df["origem"] = "mobills"
    return df[["id","data","descricao","descricao_norm","centavos","origem"]]

def ignore_payment_rows(df_fatura: pd.DataFrame, enabled: bool, keywords: list, exact_cents):
    if not enabled:
        return df_fatura.copy(), df_fatura.iloc[0:0].copy()
    kws = [normalize_text(k) for k in keywords if str(k).strip()]
    ign_mask = pd.Series([False]*len(df_fatura), index=df_fatura.index)
    if exact_cents is not None:
        ign_mask |= (df_fatura["centavos"] == exact_cents)
    if kws:
        has_kw = df_fatura["descricao_norm"].apply(lambda s: any(k in s for k in kws))
        ign_mask |= (has_kw & (df_fatura["centavos"] < 0))
    return df_fatura.loc[~ign_mask].copy(), df_fatura.loc[ign_mask].copy()

@dataclass
class SuggestConfig:
    date_window_days: int = 0
    near_cents_max: int = 5
    max_candidates: int = 8

def sim(a: str, b: str) -> int:
    if fuzz:
        return int(fuzz.token_set_ratio(a, b))
    sa, sb = set(a.split()), set(b.split())
    if not sa and not sb:
        return 0
    return int(100 * (len(sa & sb) / max(1, len(sa | sb))))

def build_candidates_for_one(f, df_mob_unmatched: pd.DataFrame, cfg: SuggestConfig) -> pd.DataFrame:
    if df_mob_unmatched.empty:
        return pd.DataFrame()
    d0 = f["data"]
    cents0 = int(f["centavos"])

    lo = d0 - timedelta(days=cfg.date_window_days)
    hi = d0 + timedelta(days=cfg.date_window_days)
    pool = df_mob_unmatched[(df_mob_unmatched["data"] >= lo) & (df_mob_unmatched["data"] <= hi)].copy()

    exact_pool = pool[pool["centavos"] == cents0].copy()
    exact_pool["tipo"] = "exato (mesma data janela)"

    if exact_pool.empty and cfg.date_window_days == 0:
        lo2 = d0 - timedelta(days=2)
        hi2 = d0 + timedelta(days=2)
        pool2 = df_mob_unmatched[(df_mob_unmatched["data"] >= lo2) & (df_mob_unmatched["data"] <= hi2)].copy()
        exact_pool = pool2[pool2["centavos"] == cents0].copy()
        exact_pool["tipo"] = "mesmo valor (±2 dias) (sugestão)"

    near_pool = pd.DataFrame()
    if cfg.near_cents_max and cfg.near_cents_max > 0:
        near_pool = pool[(pool["centavos"] - cents0).abs() <= cfg.near_cents_max].copy()
        near_pool = near_pool[near_pool["centavos"] != cents0]
        if not near_pool.empty:
            near_pool["tipo"] = "ajuste de centavos (sugestão)"

    cand = pd.concat([exact_pool, near_pool], ignore_index=True)
    if cand.empty:
        return cand

    cand["delta_centavos"] = cand["centavos"] - cents0
    cand["similaridade"] = cand["descricao_norm"].apply(lambda s: sim(f["descricao_norm"], s))
    cand["rank_tipo"] = cand["tipo"].apply(lambda t: 0 if "exato" in t else (1 if "mesmo valor" in t else 2))
    cand["dist_dias"] = cand["data"].apply(lambda d: abs((d - d0).days))
    cand = cand.sort_values(["rank_tipo","similaridade","dist_dias","delta_centavos"], ascending=[True,False,True,True])
    return cand.head(cfg.max_candidates)

def init_state():
    for k, v in {
        "matches": {},
        "selected_fatura": None,
        "search_fatura": "",
        "only_unmatched": True,
        "page": 0
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v

def fmt_brl_from_cents(cents: int) -> str:
    v = cents/100
    s = f"{v:,.2f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")

def card_tx(prefix: str, row: pd.Series, highlight=False):
    bg = "#1f2937" if not highlight else "#0f766e"
    st.markdown(
        f"""
        <div style="padding:12px;border-radius:14px;background:{bg};border:1px solid rgba(255,255,255,0.08);">
          <div style="display:flex;justify-content:space-between;align-items:center;">
            <div style="font-weight:700;">{prefix} • {row['data'].strftime('%d/%m/%Y')}</div>
            <div style="font-weight:800;">R$ {fmt_brl_from_cents(int(row['centavos']))}</div>
          </div>
          <div style="opacity:0.92;margin-top:6px;">{row['descricao']}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

init_state()

st.title("Conciliação exata: Fatura x Mobills (cards, com interação)")

with st.sidebar:
    st.subheader("Arquivos")
    fatura_file = st.file_uploader("Fatura do cartão (CSV)", type=["csv"])
    mob_file = st.file_uploader("Export do Mobills (XLS/XLSX/CSV)", type=["xls","xlsx","csv"])

    st.divider()
    st.subheader("Regras da fatura")
    ign_pay = st.checkbox("Ignorar pagamento da última fatura (linha negativa)", value=True)
    kw = st.text_input("Palavras-chave pagamento (separadas por ;)", value="pagament;pagto")
    exact_val = st.text_input("Ou ignore exatamente este valor (opcional)", value="")

    st.divider()
    st.subheader("Sugestões")
    date_window = st.number_input("Janela de data p/ buscar exato (dias)", min_value=0, max_value=10, value=0, step=1)
    near_cents = st.number_input("Sugerir ajuste até (centavos)", min_value=0, max_value=50, value=5, step=1)
    max_cand = st.number_input("Máx. sugestões por item", min_value=3, max_value=20, value=8, step=1)

if not fatura_file or not mob_file:
    st.info("Suba a fatura (CSV) e o export do Mobills para começar.")
    st.stop()

df_f_raw = load_fatura_csv(fatura_file)
df_m_raw = load_mobills(mob_file)

st.subheader("Mapeamento (auto + ajustável)")

f_cols = list(df_f_raw.columns)
default_f_date = guess_col(f_cols, ["data", "date"])
default_f_desc = guess_col(f_cols, ["estabelecimento", "descricao", "descrição", "historico", "histórico", "lancamento", "lançamento"])
default_f_val  = guess_col(f_cols, ["valor", "value", "montante", "amount"])

cfa, cfd, cfv = st.columns(3)
with cfa:
    col_f_date = st.selectbox("Fatura: coluna de data", f_cols, index=f_cols.index(default_f_date) if default_f_date in f_cols else 0)
with cfd:
    col_f_desc = st.selectbox("Fatura: coluna de descrição", f_cols, index=f_cols.index(default_f_desc) if default_f_desc in f_cols else min(1, len(f_cols)-1))
with cfv:
    col_f_val  = st.selectbox("Fatura: coluna de valor", f_cols, index=f_cols.index(default_f_val) if default_f_val in f_cols else min(2, len(f_cols)-1))

m_cols = list(df_m_raw.columns)
default_m_date = guess_col(m_cols, ["data", "date"])
default_m_desc = guess_col(m_cols, ["descricao", "descrição", "estabelecimento", "lancamento", "lançamento"])
default_m_val  = guess_col(m_cols, ["valor", "value", "amount"])

cm1, cm2, cm3 = st.columns(3)
with cm1:
    col_m_date = st.selectbox("Mobills: coluna de data", m_cols, index=m_cols.index(default_m_date) if default_m_date in m_cols else 0)
with cm2:
    col_m_desc = st.selectbox("Mobills: coluna de descrição", m_cols, index=m_cols.index(default_m_desc) if default_m_desc in m_cols else min(1, len(m_cols)-1))
with cm3:
    col_m_val  = st.selectbox("Mobills: coluna de valor", m_cols, index=m_cols.index(default_m_val) if default_m_val in m_cols else min(2, len(m_cols)-1))

if normalize_text(col_m_val) in {"categoria","category"}:
    st.error("Você escolheu 'Categoria' como coluna de valor do Mobills. Troque para 'Valor'.")
    st.stop()

df_f = build_fatura_tx(df_f_raw, col_f_date, col_f_desc, col_f_val)
df_m = build_mobills_tx(df_m_raw, col_m_date, col_m_desc, col_m_val)

keywords = [k.strip() for k in kw.split(";") if k.strip()]
exact_cents = brl_to_cents(exact_val) if exact_val.strip() else None
df_f_kept, df_f_ignored = ignore_payment_rows(df_f, ign_pay, keywords, exact_cents)

matched_f_ids = set(st.session_state.matches.keys())
matched_m_ids = set(st.session_state.matches.values())
df_f_un = df_f_kept[~df_f_kept["id"].isin(matched_f_ids)].copy()
df_m_un = df_m[~df_m["id"].isin(matched_m_ids)].copy()

st.divider()
mc1, mc2, mc3, mc4 = st.columns(4)
mc1.metric("Total fatura considerada (R$)", fmt_brl_from_cents(int(df_f_kept["centavos"].sum())))
mc2.metric("Total Mobills (R$)", fmt_brl_from_cents(int(df_m["centavos"].sum())))
mc3.metric("Ignorados (fatura)", f"{len(df_f_ignored)}")
mc4.metric("Pares manuais", f"{len(st.session_state.matches)}")

st.divider()
left, mid, right = st.columns([1.05, 1.4, 1.05], gap="large")

with left:
    st.subheader("🧾 Fatura (para conciliar)")
    st.text_input("Buscar na fatura (descrição)", key="search_fatura")
    st.checkbox("Mostrar só não conciliados", value=True, key="only_unmatched")
    per_page = st.selectbox("Itens por página", [10, 20, 30, 50], index=1)

    df_show = df_f_kept.copy()
    if st.session_state.only_unmatched:
        df_show = df_show[df_show["id"].isin(df_f_un["id"])]
    q = normalize_text(st.session_state.search_fatura)
    if q:
        df_show = df_show[df_show["descricao_norm"].str.contains(re.escape(q), na=False)]
    df_show = df_show.sort_values(["data","centavos","descricao_norm"], ascending=[True, True, True])

    n = len(df_show)
    pages = max(1, (n + per_page - 1)//per_page)
    st.session_state.page = min(st.session_state.page, pages-1)

    p1, p2, p3 = st.columns([1,1,1])
    with p1:
        if st.button("⬅️", use_container_width=True):
            st.session_state.page = max(0, st.session_state.page-1)
            st.rerun()
    with p2:
        st.caption(f"Página {st.session_state.page+1}/{pages} • {n} itens")
    with p3:
        if st.button("➡️", use_container_width=True):
            st.session_state.page = min(pages-1, st.session_state.page+1)
            st.rerun()

    start = st.session_state.page * per_page
    end = start + per_page
    chunk = df_show.iloc[start:end]

    for _, row in chunk.iterrows():
        is_sel = (st.session_state.selected_fatura == int(row["id"]))
        card_tx("Fatura", row, highlight=is_sel)
        cols_btn = st.columns([1,1,1])
        with cols_btn[0]:
            if st.button("Selecionar", key=f"sel_f_{row['id']}", use_container_width=True):
                st.session_state.selected_fatura = int(row["id"])
                st.rerun()
        with cols_btn[1]:
            if int(row["id"]) in st.session_state.matches:
                if st.button("Desfazer par", key=f"unmatch_{row['id']}", use_container_width=True):
                    st.session_state.matches.pop(int(row["id"]))
                    st.rerun()
        with cols_btn[2]:
            st.caption(" ")
        st.write("")

with mid:
    st.subheader("🤝 Sugestões e conciliação")
    sel = st.session_state.selected_fatura
    if sel is None:
        st.info("Selecione um item da fatura à esquerda para ver sugestões do Mobills.")
    else:
        f_row = df_f_kept[df_f_kept["id"] == sel].iloc[0]
        st.markdown("**Selecionado (fatura):**")
        card_tx("Fatura", f_row, highlight=True)

        cfg = SuggestConfig(int(date_window), int(near_cents), int(max_cand))
        cand = build_candidates_for_one(f_row, df_m_un, cfg)

        if cand.empty:
            st.warning("Nenhuma sugestão encontrada com as regras atuais.")
        else:
            st.markdown("**Sugestões (Mobills):**")
            for _, mrow in cand.iterrows():
                card_tx(f"{mrow['tipo']} • Mobills", mrow, highlight=False)
                st.caption(f"Similaridade: {int(mrow['similaridade'])} • Delta (centavos): {int(mrow['delta_centavos'])} • Distância dias: {int(mrow['dist_dias'])}")
                if int(mrow["centavos"]) == int(f_row["centavos"]):
                    if st.button("✅ Conciliar estes dois", key=f"match_{sel}_{int(mrow['id'])}", use_container_width=True):
                        st.session_state.matches[int(sel)] = int(mrow["id"])
                        st.rerun()
                else:
                    st.button("🛠 Requer ajuste no Mobills", key=f"no_{sel}_{int(mrow['id'])}", disabled=True, use_container_width=True)
                st.write("")

        st.divider()
        st.markdown("**Busca manual no Mobills (quando a sugestão não ajuda):**")
        c1, c2, c3 = st.columns(3)
        with c1:
            days = st.number_input("Procurar por data ± (dias)", 0, 30, 2, 1, key="man_days")
        with c2:
            only_same_value = st.checkbox("Só mesmo valor", value=True, key="man_same")
        with c3:
            query = st.text_input("Buscar texto (Mobills)", value="", key="man_q")

        mob_pool = df_m_un.copy()
        lo = f_row["data"] - timedelta(days=int(days))
        hi = f_row["data"] + timedelta(days=int(days))
        mob_pool = mob_pool[(mob_pool["data"] >= lo) & (mob_pool["data"] <= hi)]
        if only_same_value:
            mob_pool = mob_pool[mob_pool["centavos"] == f_row["centavos"]]
        q2 = normalize_text(query)
        if q2:
            mob_pool = mob_pool[mob_pool["descricao_norm"].str.contains(re.escape(q2), na=False)]
        mob_pool = mob_pool.sort_values(["data","centavos","descricao_norm"]).head(20)

        if mob_pool.empty:
            st.caption("Nada encontrado com esses filtros.")
        else:
            for _, mrow in mob_pool.iterrows():
                card_tx("Mobills", mrow, highlight=False)
                if st.button("✅ Conciliar com selecionado", key=f"match_manual_{sel}_{int(mrow['id'])}", use_container_width=True):
                    st.session_state.matches[int(sel)] = int(mrow["id"])
                    st.rerun()
                st.write("")

with right:
    st.subheader("📒 Mobills (não conciliados)")
    st.caption("Só para você ter noção do que sobrou do seu export.")
    st.metric("Mobills não conciliados", f"{len(df_m_un)}")
    q = st.text_input("Buscar no Mobills (descrição)", value="", key="search_mob")
    df_r = df_m_un.copy()
    qn = normalize_text(q)
    if qn:
        df_r = df_r[df_r["descricao_norm"].str.contains(re.escape(qn), na=False)]
    df_r = df_r.sort_values(["data","centavos","descricao_norm"]).head(30)
    for _, row in df_r.iterrows():
        card_tx("Mobills", row, highlight=False)
        st.write("")

st.divider()
st.subheader("Exportações")

pairs = []
for f_id, m_id in st.session_state.matches.items():
    f = df_f_kept[df_f_kept["id"] == f_id].iloc[0]
    m = df_m[df_m["id"] == m_id].iloc[0]
    pairs.append({
        "data": f["data"],
        "valor": f["centavos"]/100.0,
        "descricao_fatura": f["descricao"],
        "descricao_mobills": m["descricao"],
        "mobills_id": m_id,
        "fatura_id": f_id
    })
df_pairs = pd.DataFrame(pairs).sort_values(["data","valor"]) if pairs else pd.DataFrame(columns=["data","valor","descricao_fatura","descricao_mobills","mobills_id","fatura_id"])

df_a = df_f_kept[~df_f_kept["id"].isin(set(st.session_state.matches.keys()))].copy()
df_b = df_m[~df_m["id"].isin(set(st.session_state.matches.values()))].copy()

c1, c2, c3 = st.columns(3)
with c1:
    st.write("✅ Pares conciliados (exatos)")
    st.dataframe(df_pairs, use_container_width=True, height=260)
with c2:
    st.write("🟥 A) Fatura sem Mobills")
    st.dataframe(df_a[["data","descricao","centavos"]].assign(valor=df_a["centavos"]/100.0), use_container_width=True, height=260)
with c3:
    st.write("🟨 B) Mobills sem fatura")
    st.dataframe(df_b[["data","descricao","centavos"]].assign(valor=df_b["centavos"]/100.0), use_container_width=True, height=260)

def to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False, sep=";")
    return buf.getvalue().encode("utf-8")

d1, d2, d3 = st.columns(3)
with d1:
    st.download_button("Baixar pares conciliados (CSV)", data=to_csv_bytes(df_pairs), file_name="conciliados.csv", mime="text/csv")
with d2:
    st.download_button("Baixar A) fatura sem Mobills (CSV)", data=to_csv_bytes(df_a), file_name="fatura_sem_mobills.csv", mime="text/csv")
with d3:
    st.download_button("Baixar B) mobills sem fatura (CSV)", data=to_csv_bytes(df_b), file_name="mobills_sem_fatura.csv", mime="text/csv")

st.caption("Sem tolerância: só vira conciliado quando o valor bate exatamente em centavos. O resto é sugestão para ajuste.")
