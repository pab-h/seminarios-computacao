# Inicializando o repositório

1. Clone o repositório

```bash
git clone https://github.com/pab-h/seminarios-computacao.git
```

2. Inicialize as variáveis de ambiente

```bash
python -m venv .venv
```

3. Ative as variáveis de ambiente

```bash
source .venv/bin/activate
```

4. Instale as variáveis de ambiente

```bash
pip install -r requirements.txt 
```

5. (Opcional) Toda vez que instalar uma nova biblioteca, utilize: 

```bash
pip freeze > requirements.txt
```