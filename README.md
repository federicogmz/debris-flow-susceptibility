# Debris Flow Susceptibility from Basin Morphometry

Geospatial and machine learning workflows for assessing debris flow susceptibility from basin morphometry based on Arango et al. (2021). This repository provides scripts, pre-trained models, training routines, and usage examples.

Paper: https://link.springer.com/article/10.1007/s11069-020-04346-5

## Authors

- **María Isabel Arango**  
  Universidad Nacional de Colombia - Sede Medellín
- **Edier Aristizábal**  
  Universidad Nacional de Colombia - Sede Medellín
- **Federico Gómez**  
  Universidad Nacional de Colombia - Sede Medellín

## Repository Structure

```text
debris-flow-susceptibility/
├── data/
│   └── torrencialidad.csv
├── docs/
│   └── MorphometricalAnalysis2021.pdf
├── models/
│   ├── models.pkl.gz
│   └── scaler.pkl.gz
├── scripts/
│   └── train_model.py
├── src/
│   └── morphometry.py
├── LICENSE
├── README.md
└── requirements.txt
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/federicogmz/debris-flow-susceptibility.git
   cd debris-flow-susceptibility
   ```
2. (Optional) Create a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Interactive Morphometry Visualization

Use the core module to compute and plot morphometric decision boundaries:

```python
from src.morphometry import visualize_morphometry
import geopandas as gpd
import gzip
import pickle

# Load basin shapefile
basins = gpd.read_file('data/torrencialidad.shp')

# Load pre-trained models and scaler
with gzip.open('models/models.pkl.gz', 'rb') as mf:
    models = pickle.load(mf)
with gzip.open('models/scaler.pkl.gz', 'rb') as sf:
    scaler = pickle.load(sf)

# Define feature pairs and visualize
feature_pairs = [('A (sq. km)', 'Rh'), ('Su', 'Rh'), ('M', 'Rh')]
visualize_morphometry(basins, feature_pairs, models, scaler)
```

## Citation

Arango, M. I., Aristizábal, E., & Gómez, F. (2021). Morphometrical analysis of torrential flows-prone catchments in tropical and mountainous terrain of the Colombian Andes by machine learning techniques. _Natural Hazards_, 105(1), 983–1012.

## License

This work is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.
