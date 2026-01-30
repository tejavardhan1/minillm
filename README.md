 ## MiniLLM Showcase (Python)
 
 A **tiny GPT-style Transformer (“LLM”) built from scratch** in PyTorch:
 - Tokenizer (byte-level)
 - Causal self-attention blocks
 - Training loop (next-token prediction)
 - Text generation (temperature / top-k)
 
 This is perfect for GitHub because it shows you understand the core mechanics.
 
### Install essentials

**1. Python 3.10+**  
Check: `python3 --version`

**2. Create and activate a virtual environment**

```bash
cd /Users/tejavardhanreddygondi/Cursorai
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

**3. Install the project (PyTorch + deps)**

```bash
pip install --upgrade pip
pip install -e ".[dev]"
```

Or install from `requirements.txt` then the package in editable mode:

```bash
pip install -r requirements.txt
pip install -e .
```

**4. Verify PyTorch**

```bash
python3 -c "import torch; print('PyTorch', torch.__version__)"
```

**If pip fails with SSL errors (macOS)**  
Run the certificate installer that came with Python:

```bash
open "/Applications/Python 3.14/Install Certificates.command"
```

Then retry step 3.

### Quickstart (after install)

```bash
# Train on the tiny included dataset
python3 scripts/train.py --steps 500 --device cpu

# Generate text (after training)
python3 scripts/generate.py --prompt "Once upon a time" --tokens 120
```
 
 ### What to say in interviews
 - Implemented **causal masking**, **multi-head attention**, **residual connections**, **layer norm**, and **positional embeddings**
 - Trained with **cross-entropy** next-token loss
 - Built an end-to-end pipeline: data → tokens → model → training → sampling

---

### Push to GitHub

**1. Initialize git and commit (in your project folder):**

```bash
cd /Users/tejavardhanreddygondi/Cursorai
git init
git add .
git commit -m "Initial commit: MiniLLM showcase (GPT from scratch)"
```

**2. Create a new repo on GitHub**

- Go to [github.com/new](https://github.com/new)
- Repository name: e.g. `minillm-showcase` or `Cursorai`
- Leave “Add a README” **unchecked** (you already have one)
- Create repository

**3. Add remote and push**

Replace `YOUR_USERNAME` and `YOUR_REPO` with your GitHub username and repo name:

```bash
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git branch -M main
git push -u origin main
```

If you use SSH: `git remote add origin git@github.com:YOUR_USERNAME/YOUR_REPO.git`
