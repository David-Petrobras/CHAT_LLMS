# CHAT_LLMS

Aplicação Streamlit para análise de dados utilizando Azure OpenAI. O projeto permite carregar planilhas, executar análises em tempo real via Pandas e interagir com um assistente capaz de gerar e executar código para responder às perguntas sobre os dados.

## Dependências
As bibliotecas necessárias estão listadas em `requirements.txt`. Instale-as com:

```bash
pip install -r requirements.txt
```

## Execução
1. Crie um arquivo de configuração com suas credenciais (`config 1.ini`) e o certificado `petrobras-ca-root.pem` se necessário.
2. Inicie o aplicativo com:

```bash
streamlit run teste31.py
```

O aplicativo abrirá em seu navegador padrão.

