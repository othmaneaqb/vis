import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
from fpdf import FPDF
import tempfile
import os

# Load the trained model and label encoder
model = joblib.load('visualization_recommender.pkl')
le_classes = np.load('new_label_classes.npy', allow_pickle=True)

# Function to extract features
def extract_features(df: pd.DataFrame) -> np.ndarray:
    num_columns = df.shape[1]
    num_rows = df.shape[0]
    num_unique = df.nunique().mean()
    variance = df.var(numeric_only=True).mean()
    skewness = df.skew(numeric_only=True).mean()
    kurtosis = df.kurtosis(numeric_only=True).mean()
    missing_values = df.isnull().sum().mean()
    return np.array([[num_columns, num_rows, num_unique, variance, skewness, kurtosis, missing_values]])

# Function to recommend visualization
def recommend_visualization(df: pd.DataFrame) -> str:
    features = extract_features(df)
    recommendation = model.predict(features)[0]
    return le_classes[recommendation]

# Function to generate visualization
def generate_visualization(df: pd.DataFrame, vis_type: str, x_col: str = None, y_col: str = None, plot_title: str = "", x_label: str = "", y_label: str = ""):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if vis_type == 'bar':
        df.plot(kind='bar', x=x_col, y=y_col, ax=ax)
    elif vis_type == 'line':
        df.plot(kind='line', x=x_col, y=y_col, ax=ax)
    elif vis_type == 'scatter':
        sns.scatterplot(data=df, x=x_col, y=y_col, ax=ax)
    elif vis_type == 'histogram':
        df[x_col].hist(ax=ax)
    elif vis_type == 'heatmap':
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax)
    elif vis_type == 'box':
        df.plot(kind='box', ax=ax)
    elif vis_type == 'pie':
        df[x_col].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax)
    else:
        st.write("Visualization type not supported")
    
    ax.set_title(plot_title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    st.pyplot(fig)

    return fig

# Function to describe the visualization
def describe_visualization(df: pd.DataFrame, vis_type: str, x_col: str = None, y_col: str = None) -> str:
    if vis_type == 'bar':
        description = f"This bar chart shows the relationship between {x_col} and {y_col}."
        if df[x_col].nunique() < 10:
            description += f" There are {df[x_col].nunique()} unique categories in {x_col}."
    elif vis_type == 'line':
        description = f"This line chart displays the trend of {y_col} over {x_col}."
        description += f" The data covers {df[x_col].min()} to {df[x_col].max()}."
    elif vis_type == 'scatter':
        description = f"This scatter plot represents the correlation between {x_col} and {y_col}."
        correlation = df[x_col].corr(df[y_col])
        description += f" The correlation coefficient is {correlation:.2f}."
    elif vis_type == 'histogram':
        description = f"This histogram shows the distribution of {x_col}."
        description += f" The data ranges from {df[x_col].min()} to {df[x_col].max()}."
    elif vis_type == 'heatmap':
        description = f"This heatmap visualizes the correlation matrix of the dataset."
        strongest_corr = df.corr().abs().unstack().sort_values(kind="quicksort", ascending=False).drop_duplicates().iloc[1]
        description += f" The strongest correlation is {strongest_corr:.2f}."
    elif vis_type == 'box':
        description = f"This box plot shows the distribution of values for each column."
    elif vis_type == 'pie':
        description = f"This pie chart illustrates the proportion of each category in {x_col}."
        most_common = df[x_col].value_counts().idxmax()
        description += f" The largest category is {most_common}."
    else:
        description = "No description available for this visualization."
    return description

# Function to save visualization as PDF
def save_visualization_as_pdf(fig, description, file_path):
    pdf = FPDF()
    pdf.add_page()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
        fig.savefig(tmpfile.name, format="png")
        pdf.image(tmpfile.name, x=10, y=10, w=190)  # Adjust positioning as needed
    pdf.set_xy(10, 150)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, description)
    pdf.output(file_path)
    os.unlink(tmpfile.name)  # Delete the temporary image file

# Streamlit app setup
st.title("AI Visualization Creator")

# Upload multiple files
uploaded_files = st.file_uploader("Choose files", type=["csv", "xlsx", "xls"], accept_multiple_files=True)

datasets = {}
for uploaded_file in uploaded_files:
    if uploaded_file is not None:
        file_extension = uploaded_file.name.split('.')[-1]
        if file_extension == 'csv':
            df = pd.read_csv(uploaded_file)
        elif file_extension in ['xlsx', 'xls']:
            df = pd.read_excel(uploaded_file)
        datasets[uploaded_file.name] = df

if datasets:
    selected_dataset = st.selectbox("Select dataset", list(datasets.keys()))
    df = datasets[selected_dataset]
    st.write("Data preview:")
    st.write(df.head())

    # Recommend visualization
    vis_type = recommend_visualization(df)
    st.write(f"Recommended visualization: {vis_type}")

    # User input for columns and visualization type
    columns = df.columns.tolist()
    x_col = st.selectbox("Select X column:", columns)
    y_col = st.selectbox("Select Y column:", [None] + columns) if vis_type not in ['pie', 'histogram', 'box'] else None
    selected_vis_type = st.selectbox("Select Visualization Type:", ['bar', 'line', 'scatter', 'histogram', 'heatmap', 'box', 'pie'], index=['bar', 'line', 'scatter', 'histogram', 'heatmap', 'box', 'pie'].index(vis_type))
    plot_title = st.text_input("Plot Title", "My Plot")
    x_label = st.text_input("X-axis Label", x_col)
    y_label = st.text_input("Y-axis Label", y_col if y_col else "")

    # Data filtering in the sidebar
    st.sidebar.header("Filter Data")
    with st.sidebar.expander("Numeric Filters"):
        numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
        numeric_filters = {}
        for col in numeric_columns:
            min_val = df[col].min(skipna=True)
            max_val = df[col].max(skipna=True)
            numeric_filters[col] = st.slider(f"{col} range", min_value=float(min_val), max_value=float(max_val), value=(float(min_val), float(max_val)))

    with st.sidebar.expander("Categorical Filters"):
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        categorical_filters = {}
        for col in categorical_columns:
            unique_values = df[col].dropna().unique().tolist()
            categorical_filters[col] = st.multiselect(f"Filter {col}", options=unique_values, default=unique_values)

    # Apply filters to the DataFrame
    filtered_df = df.copy()
    for col, (min_val, max_val) in numeric_filters.items():
        filtered_df = filtered_df[(filtered_df[col] >= min_val) & (filtered_df[col] <= max_val)]
    for col, selected_values in categorical_filters.items():
        if selected_values:
            filtered_df = filtered_df[filtered_df[col].isin(selected_values)]

    # Interactive dashboard layout with tabs
    tab1, tab2, tab3 = st.tabs(["Visualization", "Raw Data", "Description"])

    with tab1:
        st.write("Generated visualization:")
        fig = generate_visualization(filtered_df, selected_vis_type, x_col, y_col, plot_title, x_label, y_label)

    with tab2:
        st.write("Filtered Data:")
        st.write(filtered_df)

    with tab3:
        st.write("Chart Description:")
        description = describe_visualization(filtered_df, selected_vis_type, x_col, y_col)
        st.write(description)

    # Save and load configuration
    if st.button("Save configuration"):
        config = {
            'x_col': x_col,
            'y_col': y_col,
            'vis_type': selected_vis_type,
            'plot_title': plot_title,
            'x_label': x_label,
            'y_label': y_label,
            'filters': {
                'numeric': numeric_filters,
                'categorical': categorical_filters
            }
        }
        with open("dashboard_config.json", "w") as f:
            json.dump(config, f)
        st.success("Configuration saved")

    if st.button("Load configuration"):
        with open("dashboard_config.json", "r") as f:
            config = json.load(f)
        x_col = config['x_col']
        y_col = config['y_col']
        selected_vis_type = config['vis_type']
        plot_title = config['plot_title']
        x_label = config['x_label']
        y_label = config['y_label']
        numeric_filters = config['filters']['numeric']
        categorical_filters = config['filters']['categorical']
        st.success("Configuration loaded")

    # Export options
    if st.button("Export as PDF"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
            save_visualization_as_pdf(fig, description, tmpfile.name)
            st.success("Dashboard exported as PDF")
            tmpfile.close()
            st.download_button("Download PDF", data=open(tmpfile.name, "rb").read(), file_name="dashboard.pdf", mime="application/pdf")
            os.unlink(tmpfile.name)  # Remove the temporary file after download
