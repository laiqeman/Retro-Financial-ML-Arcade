# Financial ML Arcade 🎮

A retro-themed financial machine learning application built with Streamlit. This application allows users to perform end-to-end machine learning analysis on financial data, either from uploaded datasets or live stock data from Yahoo Finance.

## Features

- 📊 Data Source Options:
  - Upload custom CSV datasets
  - Fetch live stock data from Yahoo Finance
- 🧹 Data Preprocessing:
  - Handle missing values
  - Remove outliers
  - Data cleaning tools
- ⚙️ Feature Engineering:
  - Feature selection
  - Data scaling
  - Custom feature creation
- 🎯 Model Training:
  - Linear Regression implementation
  - Model performance metrics
  - Interactive visualizations
- 🎨 Retro-themed UI:
  - Neon color scheme
  - Retro-styled components
  - Interactive plots

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd financial-ml-arcade
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the URL shown in the terminal (usually http://localhost:8501)

3. Follow the steps in the application:
   - Choose your data source
   - Preprocess the data
   - Select features
   - Train the model
   - View results and visualizations

## Data Format

For CSV uploads, the data should be in a format compatible with financial analysis:
- Date column (if applicable)
- Numeric columns for features
- Target variable (e.g., 'Close' price for stocks)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 