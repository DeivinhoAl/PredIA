from flask import Flask, render_template, request, redirect, url_for, session, flash
import os
import json
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

app = Flask(__name__)
app.secret_key = 'sua_chave_secreta_aqui'

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('static', exist_ok=True)

# Usuário fixo pré-cadastrado
usuarios = {
    'admin': {
        'nome': 'Administrador',
        'senha': '123'
    }
}

@app.route('/cadastro', methods=['GET', 'POST'])
def cadastro():
    if request.method == 'POST':
        nome = request.form['nome']
        email = request.form['email']
        senha = request.form['senha']

        if email in usuarios:
            flash('Usuário já cadastrado com este e-mail.')
            return redirect(url_for('cadastro'))
        
        usuarios[email] = {'nome': nome, 'senha': senha}
        flash('Cadastro realizado com sucesso! Faça login.')
        return redirect(url_for('login_page'))

    return render_template('login.html')


@app.route('/login', methods=['GET', 'POST'])
def login_page():
    if request.method == 'POST':
        nome = request.form['nome']
        senha = request.form['senha']

        user = None
        usuario_key = None
        for k, v in usuarios.items():
            if v['nome'] == nome:
                user = v
                usuario_key = k
                break

        if user and user['senha'] == senha:
            session['usuario'] = usuario_key
            return redirect(url_for('index'))
        else:
            erro = 'Usuário ou senha inválidos'
            return render_template('login.html', erro=erro)
    return render_template('login.html')


@app.route('/')
def index():
    if 'usuario' not in session:
        return redirect(url_for('login_page'))
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    if 'usuario' not in session:
        return redirect(url_for('login_page'))

    arquivo = request.files.get('arquivo')
    if arquivo and arquivo.filename.endswith('.json'):
        caminho = os.path.join(UPLOAD_FOLDER, 'arquivo.json')
        arquivo.save(caminho)
        gerar_previsao(caminho)
        return redirect(url_for('resultado'))
    else:
        flash('Envie um arquivo JSON válido.')
        return redirect(url_for('index'))


@app.route('/resultado')
def resultado():
    if 'usuario' not in session:
        return redirect(url_for('login_page'))
    return render_template('resultado.html')


@app.route('/logout')
def logout():
    session.pop('usuario', None)
    return redirect(url_for('login_page'))


def gerar_previsao(arquivo_json):
    with open(arquivo_json, 'r') as f:
        dados_json = json.load(f)

    dados = []
    for produto in dados_json['produtos']:
        nome = produto['nome']
        for venda in produto['vendas']:
            dados.append({
                'Produto': nome,
                'Dia': venda['dia'],
                'Demanda': venda['quantidade'],
                'Promocao': int(venda['promocao'])
            })

    df = pd.DataFrame(dados)
    df['Produto'] = df['Produto'].replace({'A': 'Hardware', 'B': 'ERP', 'C': 'Software', 'D': 'Consultoria'})

    modelos = {}
    df_todos_lista = []
    dias_todos = np.arange(1, 31)

    for produto in df['Produto'].unique():
        df_prod = df[df['Produto'] == produto]
        modelo = RandomForestRegressor(n_estimators=100, random_state=0)
        modelo.fit(df_prod[['Dia', 'Promocao']], df_prod['Demanda'])

        promocoes_todos = np.zeros_like(dias_todos)
        dias_possiveis = dias_todos[dias_todos > 2]
        dias_com_promocao = np.random.choice(dias_possiveis, size=6, replace=False)
        promocoes_todos[np.isin(dias_todos, dias_com_promocao)] = 1

        df_todos = pd.DataFrame({
            'Produto': produto,
            'Dia': dias_todos,
            'Promocao': promocoes_todos
        })
        df_todos['Previsao'] = modelo.predict(df_todos[['Dia', 'Promocao']])
        df_todos_lista.append(df_todos)

    df_todos_final = pd.concat(df_todos_lista)

    produtos = df_todos_final['Produto'].unique()
    num_produtos = len(produtos)
    n_colunas = 2
    n_linhas = (num_produtos // n_colunas) + (1 if num_produtos % n_colunas != 0 else 0)

    fig, axs = plt.subplots(n_linhas, n_colunas, figsize=(18, 5 * n_linhas), dpi=120)
    axs = axs.flatten()
    plt.style.use('dark_background')
    cores = ['#00FFFF', '#FF69B4', '#FFD700', '#00FF7F', '#FF6347', '#8A2BE2', '#20B2AA', '#FF1493']

    for i, produto in enumerate(produtos):
        df_p = df_todos_final[df_todos_final['Produto'] == produto]
        axs[i].plot(df_p['Dia'], df_p['Previsao'], color=cores[i % len(cores)], label=produto)
        axs[i].set_title(f'Previsão - {produto}')
        axs[i].set_xlabel('Dia')
        axs[i].set_ylabel('Demanda')
        axs[i].legend()
        axs[i].grid(True)

    plt.tight_layout()
    plt.savefig('static/previsao.png')
    plt.close()


if __name__ == '__main__':
    app.run(debug=True)
