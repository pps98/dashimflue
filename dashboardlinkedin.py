import streamlit as st
import json
from datetime import datetime
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import plotly.express as px
import requests
from dotenv import load_dotenv
import os
from hashlib import md5
import diskcache
from tqdm import tqdm
import traceback  # Para exibir traceback completo em caso de erro

# Configurações iniciais
load_dotenv(override=True)
API_KEY = os.getenv("API_KEY")

if not API_KEY:
    st.error("A chave API não foi encontrada no arquivo .env.")
    st.stop()

# Configuração do cache
cache = diskcache.Cache('./cache_normalizacao')

# Configuração da página
st.set_page_config(page_title="Dashboard Comunidade", layout="wide")

# Função para acessar a API do ChatGPT
def acessa_chatgpt(prompt):
    url = 'https://api.openai.com/v1/chat/completions'
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {API_KEY}'
    }
    data = {
        'model': 'gpt-4o-mini',
        'messages': [{'role': 'user', 'content': prompt}],
        'temperature': 0.3
    }
    
    try:
        response = requests.post(url, headers=headers, json=data, timeout=60)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        st.error(f"Erro na API do ChatGPT: {str(e)}")
        st.error(traceback.format_exc())
        return None

# Funções auxiliares para processar dados
def extrair_mes(aniversario):
    try:
        aniversario_normalizado = aniversario.lower().replace('ã', 'a').replace('õ', 'o')
        if 'informado' in aniversario_normalizado:
            return None
        partes = aniversario.split("/")
        if len(partes) >= 2:
            mes = int(partes[1])
            if 1 <= mes <= 12:
                return mes
    except Exception:
        pass
    return None

def extrair_estado(local):
    if pd.isna(local) or local.lower() in ["não informado", "nao informado"]:
        return "Não informado"
    if '/' in local:
        return local.split('/')[-1].strip()
    if ',' in local:
        return local.split(',')[-1].strip()
    return local

def obter_interesses(perfil):
    interesses = perfil.get('interesses', [])
    if isinstance(interesses, list):
        return [tag.lower().strip() for tag in interesses if isinstance(tag, str) and tag.strip()]
    elif isinstance(interesses, str):
        return [tag.lower().strip() for tag in interesses.split('#') if tag.strip()]
    else:
        st.warning(f"Formato inesperado para 'interesses' no perfil: {perfil.get('nome', 'Sem Nome')}")
        return []

def obter_areas(perfil):
    area = perfil.get('área', "")
    if isinstance(area, list):
        return [tag.lower().strip() for tag in area if isinstance(tag, str) and tag.strip()]
    elif isinstance(area, str):
        return [tag.lower().strip() for tag in area.split('#') if tag.strip()]
    else:
        st.warning(f"Formato inesperado para 'área' no perfil: {perfil.get('nome', 'Sem Nome')}")
        return []

def normalizar_dados(perfis):
    template = """Analise e padronize os seguintes campos para cada perfil:
- Local (formato: "Cidade/UF" ou "Cidade, País")
- Área (tags em lowercase sem caracteres especiais, separadas por #)
- Interesses (mesmo critério das áreas)

Retorne APENAS um JSON com os perfis normalizados, com os campos:
- nome
- local
- área
- interesses
- url linkedin
- newsletter
- aniversario

Perfis originais:
{perfis_json}
"""
    
    perfis_normalizados = []
    perfis_para_normalizar = []
    perfis_hashes = []

    for perfil in perfis:
        try:
            perfil_hash = md5(json.dumps(perfil, sort_keys=True).encode()).hexdigest()
            if perfil_hash in cache:
                perfis_normalizados.append(cache[perfil_hash])
            else:
                perfis_para_normalizar.append(perfil)
                perfis_hashes.append(perfil_hash)
        except Exception as e:
            st.error(f"Erro ao processar hash para perfil {perfil.get('nome', '')}: {str(e)}")
            st.error(traceback.format_exc())
            perfis_normalizados.append(perfil)

    if perfis_para_normalizar:
        batch_size = 20
        batches = [perfis_para_normalizar[i:i + batch_size] for i in range(0, len(perfis_para_normalizar), batch_size)]
        batch_hashes = [perfis_hashes[i:i + batch_size] for i in range(0, len(perfis_hashes), batch_size)]
        
        with st.spinner('Normalizando dados com IA...'):
            for batch, hashes in tqdm(zip(batches, batch_hashes), total=len(batches), desc="Processando perfis"):
                prompt = template.format(perfis_json=json.dumps(batch, ensure_ascii=False))
                resposta = acessa_chatgpt(prompt)
                if resposta:
                    try:
                        resposta_limpa = resposta.replace('```json', '').replace('```', '').strip()
                        perfis_batch_normalizados = json.loads(resposta_limpa)
                        
                        for perfil_normalizado, perfil_hash in zip(perfis_batch_normalizados, hashes):
                            if all(key in perfil_normalizado for key in ['nome', 'local', 'área', 'interesses']):
                                cache[perfil_hash] = perfil_normalizado
                                perfis_normalizados.append(perfil_normalizado)
                            else:
                                st.warning(f"Perfil {perfil_normalizado.get('nome', 'Sem Nome')} não possui todos os campos necessários. Mantendo o original.")
                                perfis_normalizados.append(perfil)
                    except json.JSONDecodeError as e:
                        st.error(f"Erro ao decodificar JSON: {str(e)}")
                        st.error(traceback.format_exc())
                        perfis_normalizados.extend(batch)
                else:
                    perfis_normalizados.extend(batch)

    return perfis_normalizados

def carregar_dados():
    url = "https://drive.google.com/file/d/1Rx5BRqQYIpvXfzX7wAAxPM7w1vF7F927/view?usp=drive_link"
    
    try:
        file_id = url.split('/d/')[1].split('/')[0]
        download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
        response = requests.get(download_url)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Erro ao carregar arquivo: {str(e)}")
        st.error(traceback.format_exc())
        return None

def processar_tags(perfis, campo):
    tags = []
    for p in perfis:
        if campo == "interesses":
            tags.extend(obter_interesses(p))
        elif campo == "área":
            tags.extend(obter_areas(p))
    return Counter(tags)

# Carregar e processar dados
if 'perfis' not in st.session_state:
    dados_brutos = carregar_dados()
    if dados_brutos:
        st.session_state.perfis = normalizar_dados(dados_brutos)
    else:
        st.stop()

# Sidebar - Filtros
st.sidebar.title("🔍 Filtros")
local_options = ["Todos"] + sorted({p["local"] for p in st.session_state.perfis if p["local"] != "Não informado"})
local_selecionado = st.sidebar.selectbox("Localização", local_options)

# Processamento de dados filtrados
dados_filtrados = [
    p for p in st.session_state.perfis 
    if local_selecionado == "Todos" or p["local"] == local_selecionado
]

# Converter dados_filtrados para DataFrame
df_dados_filtrados = pd.DataFrame(dados_filtrados)

# Opcional: Exibir os primeiros registros para verificação
# st.write("### Exemplos de Dados Normalizados", df_dados_filtrados.head())

# 1. Visão Geral
st.title("📊 Dashboard (insights) - Imflue Circle")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total de Membros", len(dados_filtrados))
with col2:
    st.metric("Localizações Únicas", len({p["local"] for p in dados_filtrados}))
with col3:
    newsletters_ativas = sum(1 for p in dados_filtrados if p.get("newsletter", "não informado").lower() not in ["não informado", "nao informado"])
    percentage = f"{(newsletters_ativas/len(dados_filtrados)*100):.1f}%" if len(dados_filtrados) > 0 else "0%"
    st.metric("Newsletters Ativas", f"{newsletters_ativas} ({percentage})")
with col4:
    mes_atual = datetime.now().month
    aniversariantes = []
    for p in dados_filtrados:
        aniversario = p.get("aniversario", "não informado")
        if aniversario.lower() not in ["não informado", "nao informado"]:
            mes = extrair_mes(aniversario)
            if mes == mes_atual:
                aniversariantes.append(p)
            elif mes is None:
                st.warning(f"Formato de aniversário inválido para o perfil: {p.get('nome', 'Sem Nome')}, aniversário: '{aniversario}'")
    st.metric("Aniversariantes do Mês", len(aniversariantes))

# 2. Distribuição Geográfica
st.subheader("🌍 Distribuição Geográfica")

df_localizacao = pd.DataFrame([p["local"] for p in dados_filtrados], columns=["local"])
df_localizacao["estado"] = df_localizacao["local"].apply(extrair_estado)

estado_counts = (
    df_localizacao["estado"]
    .value_counts()
    .reset_index()
    .rename(columns={"index": "Estado", "estado": "Contagem"})
)

if not estado_counts.empty and {"Estado", "Contagem"}.issubset(estado_counts.columns):
    fig = px.bar(
        estado_counts,
        x="Estado",
        y="Contagem",
        labels={"Estado": "Localização", "Contagem": "Número de Membros"},
        color="Estado"
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Dados insuficientes para mostrar a distribuição geográfica")

# 3. Áreas e Interesses
st.subheader("🔖 Áreas e Interesses Principais")
col1, col2 = st.columns(2)

with col1:
    st.write("**Nuvem de Palavras**")
    todas_tags = processar_tags(dados_filtrados, "área") + processar_tags(dados_filtrados, "interesses")
    if todas_tags:
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(todas_tags)
        plt.figure(figsize=(10,5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        st.pyplot(plt)
    else:
        st.write("Nenhuma tag disponível para gerar a nuvem de palavras.")

with col2:
    st.write("**Top 10 Áreas de Atuação**")
    areas_top = processar_tags(dados_filtrados, "área").most_common(10)
    if areas_top:
        df_areas_top = pd.DataFrame(areas_top, columns=["Área", "Contagem"])
        fig = px.bar(df_areas_top, x="Área", y="Contagem", labels={"Área": "Área", "Contagem": "Número de Membros"}, color="Área")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("Nenhuma área disponível para exibir.")

# 3.2 Treemap Hierárquico
st.write("**Distribuição Hierárquica de Áreas**")
try:
    data = []
    for p in dados_filtrados:
        areas = obter_areas(p)
        if len(areas) >= 2:
            data.append({"Categoria": areas[0].capitalize(), "Subcategoria": areas[1].capitalize()})
        elif len(areas) == 1:
            data.append({"Categoria": areas[0].capitalize(), "Subcategoria": "Geral"})
    
    df_hierarquia = pd.DataFrame(data)
    
    if not df_hierarquia.empty:
        fig = px.treemap(
            df_hierarquia,
            path=['Categoria', 'Subcategoria'],
            color='Categoria',
            color_continuous_scale='Rainbow'
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("Dados insuficientes para gerar o treemap.")
except Exception as e:
    st.error(f"Erro ao gerar treemap: {str(e)}")
    st.error(traceback.format_exc())

# 3.3 Mapa de Calor de Relações
st.subheader("🔥 Relação entre Áreas e Interesses")
try:
    # Selecionar as top 15 áreas e interesses para evitar matrizes muito grandes
    top_areas = processar_tags(dados_filtrados, "área").most_common(15)
    top_interesses = processar_tags(dados_filtrados, "interesses").most_common(15)
    areas = [area for area, _ in top_areas]
    interesses = [interesse for interesse, _ in top_interesses]
    
    # Criar DataFrame de relações
    data = []
    for p in dados_filtrados:
        p_areas = [tag.lower() for tag in obter_areas(p) if tag.lower() in areas]
        p_interesses = [tag.lower() for tag in obter_interesses(p) if tag.lower() in interesses]
        for area in p_areas:
            for interesse in p_interesses:
                data.append({"Área": area.capitalize(), "Interesse": interesse.capitalize()})
    
    df_relacoes = pd.DataFrame(data)
    if not df_relacoes.empty:
        pivot_table = df_relacoes.pivot_table(index='Área', columns='Interesse', aggfunc='size', fill_value=0)
        
        fig = px.imshow(
            pivot_table,
            labels=dict(x="Interesse", y="Área", color="Frequência"),
            color_continuous_scale="YlOrRd",
            aspect="auto"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("Dados insuficientes para gerar o mapa de calor.")
except Exception as e:
    st.error(f"Erro ao gerar mapa de calor: {str(e)}")
    st.error(traceback.format_exc())

# 4. Newsletters Ativas
st.subheader("📬 Newsletters Ativas")
newsletters = [p["newsletter"] for p in dados_filtrados if p.get("newsletter", "não informado").lower() not in ["não informado", "nao informado"]]
if newsletters:
    st.write("Links disponíveis:")
    for link in newsletters:
        st.markdown(f"- [{link}]({link})")
else:
    st.write("Nenhuma newsletter informada para os filtros selecionados")

# 5. Aniversariantes
st.subheader("🎂 Aniversariantes do Mês")
if aniversariantes:
    for p in aniversariantes:
        st.markdown(f"**{p.get('nome', 'Sem Nome')}** - {p.get('aniversario', 'Não Informado')}")
else:
    st.write("Nenhum aniversariante neste mês")

# 5.1 Calendário de Aniversários
st.subheader("📅 Distribuição de Aniversários no Ano")
try:
    df_aniversarios = pd.DataFrame([p.get("aniversario", "") for p in dados_filtrados if p.get("aniversario", "não informado").lower() not in ["não informado", "nao informado"]], columns=["Data"])
    # Remover entradas inválidas
    df_aniversarios = df_aniversarios[df_aniversarios["Data"].str.contains("/")]
    df_aniversarios[['Dia', 'Mês']] = df_aniversarios['Data'].str.split('/', expand=True).astype(float, errors='ignore')
    df_aniversarios = df_aniversarios.dropna()
    
    if not df_aniversarios.empty:
        fig = px.density_heatmap(
            df_aniversarios,
            x="Mês",
            y="Dia",
            nbinsx=12,
            nbinsy=31,
            color_continuous_scale="Viridis",
            title="Frequência de Aniversários por Dia/Mês"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("Dados insuficientes para gerar o calendário de aniversários.")
except Exception as e:
    st.error(f"Erro ao gerar calendário: {str(e)}")
    st.error(traceback.format_exc())

# 6. Busca de Perfis
st.subheader("🔍 Busca de Membros")
busca = st.text_input("Pesquisar por nome, área ou palavra-chave:")
if busca:
    busca_lower = busca.lower()
    resultados = [
        p for p in dados_filtrados 
        if busca_lower in p.get('nome', '').lower() 
        or busca_lower in ' '.join(obter_areas(p)).lower() 
        or busca_lower in ' '.join(obter_interesses(p)).lower()
    ]
else:
    resultados = []

if resultados:
    for perfil in resultados:
        with st.expander(f"{perfil.get('nome', 'Sem Nome')} - {perfil.get('local', 'Não Informado')}"):
            st.write(f"**Área:** {' # '.join(obter_areas(perfil))}")
            st.write(f"**Interesses:** {' # '.join(obter_interesses(perfil))}")
            if perfil.get('url linkedin', None):
                st.markdown(f"**[LinkedIn]({perfil['url linkedin']})**")
else:
    st.write("Nenhum resultado encontrado")

# 6.1 Gráfico de Dispersão de Atividades com Interesses Detalhados
st.subheader("🎯 Distribuição de Atividades por Localização")

try:
    # Verificar se 'local', 'área' e 'interesses' estão presentes
    required_columns = ['local', 'área', 'interesses']
    if not all(col in df_dados_filtrados.columns for col in required_columns):
        st.error("Dados insuficientes para gerar o gráfico de dispersão.")
    else:
        # Agrupar dados por Localização com nomes de colunas únicos
        agrupado = df_dados_filtrados.groupby('local').agg(
            Numero_Areas=pd.NamedAgg(column='área', aggfunc=lambda x: sum(len(obter_areas({'área': a})) for a in x)),
            Numero_Interesses=pd.NamedAgg(column='interesses', aggfunc=lambda x: sum(len(obter_interesses({'interesses': i})) for i in x)),
            Interesses=pd.NamedAgg(column='interesses', aggfunc=lambda x: Counter([interest for p in x for interest in obter_interesses({'interesses': p})]))
        ).reset_index()
        
        # Obter top 3 interesses por localização
        agrupado['Top_Interesses'] = agrupado['Interesses'].apply(lambda c: ', '.join([interest for interest, count in c.most_common(3)]))
        
        # Renomear colunas para nomes únicos
        df_atividades = agrupado.rename(columns={
            'local': 'Local',
            'Numero_Areas': 'Áreas',
            'Numero_Interesses': 'Total_Interesses'  # Renomear para evitar duplicação
        })
        
        # Depuração: Exibir os dados agrupados
        st.write("### Dados Agrupados para o Gráfico de Dispersão", df_atividades.head())
        
        # Criar gráfico de dispersão com hover info detalhado
        fig = px.scatter(
            df_atividades,
            x="Áreas",
            y="Total_Interesses",  # Atualizar para o novo nome
            color="Local",
            size="Áreas",
            hover_data={'Top_Interesses': True, 'Local': True, 'Áreas': True, 'Total_Interesses': True},
            title="Relação entre Número de Áreas e Interesses por Localização",
            labels={
                "Local": "Localização",
                "Áreas": "Número de Áreas",
                "Total_Interesses": "Número de Interesses"
            }
        )
        
        # Atualizar o template de hover para incluir os top interesses
        fig.update_traces(
            hovertemplate=
            '<b>%{customdata[0]}</b><br>' +
            'Áreas: %{x}<br>' +
            'Interesses: %{y}<br>' +
            'Top Interesses: %{customdata[1]}<extra></extra>'
        )
        
        st.plotly_chart(fig, use_container_width=True)
except Exception as e:
    st.error(f"Erro ao gerar gráfico de dispersão: {str(e)}")
    st.error(traceback.format_exc())

# 7. Insights de Networking
st.subheader("🤝 Sugestões de Conexão")
if len(dados_filtrados) >= 2:
    st.write("Membros com interesses complementares:")
    pares = []
    for i in range(len(dados_filtrados)):
        interesses1 = set(obter_interesses(dados_filtrados[i]))
        for j in range(i+1, len(dados_filtrados)):
            interesses2 = set(obter_interesses(dados_filtrados[j]))
            if interesses1 & interesses2:
                pares.append((dados_filtrados[i].get('nome', 'Sem Nome'), dados_filtrados[j].get('nome', 'Sem Nome')))
                if len(pares) >= 3:
                    break
        if len(pares) >= 3:
            break
    if pares:
        for par in pares:
            st.write(f"- {par[0]} ↔ {par[1]}")
    else:
        st.write("Nenhum par encontrado com interesses complementares.")
else:
    st.write("Selecione uma localização com mais membros para sugestões")

# Rodar com: streamlit run dashboard.py
