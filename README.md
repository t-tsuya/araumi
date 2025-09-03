## Quickstart (VS Code -> GitHub -> Colab)

### Local
```bash
pip install -e .[dev]
pytest -q
python -c "import torch; from araumi.datasets import IsoMoG, swiss_roll; print('OK')"
