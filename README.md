# Conciliação Fatura x Mobills (sem tolerância)

## O que faz
- Lê fatura em CSV (recomendado) ou PDF (opcional, com senha)
- Lê export do Mobills (XLS/XLSX/CSV)
- Concilia **EXATAMENTE** por (data, valor em centavos), lidando com duplicados
- Ignora automaticamente o **pagamento da última fatura** (linha negativa) se você marcar a opção e definir palavras-chave/valor
- Mostra:
  - Conciliados
  - Fatura sem Mobills (A)
  - Mobills sem Fatura (B)
  - Sugestões de ajuste de centavos (C), **sem conciliar**

## Rodar local
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy grátis no Streamlit Community Cloud
1. Crie um repo no GitHub com `app.py` e `requirements.txt`
2. No Streamlit Cloud, conecte ao repo e faça deploy
3. (Se usar PDF) configure um Secret `PDF_PASSWORD` para não digitar.

## Observação importante
Se o PDF do seu banco for escaneado/imagem, vai precisar OCR e aumenta o custo.
