import streamlit as st
import pandas as pd 
from pycaret.classification import load_model, predict_model

st.set_page_config(page_title = 'App FLAI - Powered by Streamlit', 
				   page_icon = 'iconeflai.png' ,
				   layout = 'centered', 
				   initial_sidebar_state = 'auto')

modelo = load_model('modelo-para-previsao-de-salario')

@st.cache
def ler_dados():
	dados = pd.read_csv('dataset-profissionais-dados-resumido.csv')
	dados = dados.dropna()
	return dados

dados = ler_dados()  

st.image('bannerflai.jpg', use_column_width = 'always')

st.write('''
# :sparkles: Modelo para Precificação de Salários para Profissionais de Dados
***Criado por [FLAI - Inteligência Artificial e Data Science](https://www.flai.com.br/)***. 

---

Nesse Web-App podemos utilizar um modelo de machine learning para estimar salários de profissionais da área de dados.

Entre com as características do profissional e da vaga, e verifique o valor estimado para o salário de mercado desse profissional. 

O modelo desse web-app foi desenvolvido utilizando o conjunto de 
dados que pode ser encontrado nesse [link do kaggle](https://www.kaggle.com/datahackers/pesquisa-data-hackers-2019).

''')

st.markdown('---') 
 
st.markdown('## Informações da vaga')
col1, col2 = st.columns(2)

x2 = col1.radio('Profissão', dados['Profissão'].unique().tolist())
x3 = col1.radio('Tamanho da Empresa', dados['Tamanho da Empresa'].unique().tolist())
x4 = col1.radio('Cargo de Gestão', dados['Cargo de Gestão'].unique().tolist())
x11 = col2.radio('Estado', dados['Estado'].unique().tolist()) 
x6 = col2.radio('Tipo de Trabalho', dados['Tipo de Trabalho'].unique().tolist() )
x9 = col2.selectbox('Setor de Mercado', dados['Setor de Mercado'].unique().tolist())

st.markdown('---')

st.markdown('## Informações do candidato')
col1, col2 = st.columns(2)

x1 = 30
x5 = col1.selectbox('Experiência em DS', dados['Experiência em DS'].unique().tolist()) 
x8 = col1.selectbox('Área de Formação', dados['Área de Formação'].unique().tolist())
x7 = col1.radio('Escolaridade', dados['Escolaridade'].unique().tolist())
x10 = 1
x12 = col2.radio('Linguagem Python', dados['Linguagem Python'].unique().tolist()) 
x13 = col2.radio('Linguagem R', dados['Linguagem R'].unique().tolist()) 
x14 = col2.radio('Linguagem SQL', dados['Linguagem SQL'].unique().tolist()) 
 

dicionario  =  {'Idade': [x1],
			'Profissão': [x2],
			'Tamanho da Empresa': [x3],
			 'Cargo de Gestão': [x4],
			'Experiência em DS': [x5],
			'Tipo de Trabalho': [x6],
			'Escolaridade': [x7],
			'Área de Formação': [x8],
			'Setor de Mercado': [x9],
			'Brasil': [x10],
			'Estado': [x11],		
			'Linguagem Python': [x12],
			'Linguagem R': [x13],
			'Linguagem SQL': [x14]}

dados = pd.DataFrame(dicionario)  

st.markdown('---') 

st.markdown('## Executar o Modelo de Precificação') 

inflacao = st.checkbox('Levar em consideração a inflação desde a coleta dos dados?')

if st.button('CALCULAR O SALÁRIO'):
	st.markdown('---') 
	i = 0.1758
	saida = float(predict_model(modelo, dados)['Label']) 
	if inflacao:
		st.markdown('# Salário estimado de **R$ {:.2f}**'.format((1+i)*saida))
	else:
		st.markdown('# Salário estimado de **R$ {:.2f}**'.format(saida))
	st.markdown('---') 

 