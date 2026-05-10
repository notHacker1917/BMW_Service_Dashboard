"""Streamlit Dashboard for Yamaha Quality Analytics."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import os
from app.utils.logger import logger
from app.utils.config import OUTPUT_DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR
from app.database import DatabaseManager

# Set page config
st.set_page_config(
    page_title="Yamaha Service Dashboard",
    page_icon="🏍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom theme
st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] > .main {
        padding-top: 2rem;
    }
    .header-title {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1F77B4;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f0f0;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1F77B4;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if "db" not in st.session_state:
    st.session_state.db = DatabaseManager()
    logger.info("Database initialized in session state")


def load_latest_file(pattern: str, directory: Path):
    """Load latest file matching pattern."""
    files = list(directory.glob(pattern))
    if files:
        latest = max(files, key=os.path.getctime)
        return pd.read_csv(latest, encoding="utf-8")
    return None


def load_embeddings():
    """Load embeddings and metadata."""
    emb_file = PROCESSED_DATA_DIR / "embeddings_umap.npy"
    if emb_file.exists():
        return np.load(emb_file)
    return None


@st.cache_data
def get_raw_feedback():
    """Load raw feedback data."""
    return load_latest_file("*.csv", RAW_DATA_DIR)


@st.cache_data
def get_clustered_data():
    """Load clustered data."""
    return load_latest_file("*refined.csv", OUTPUT_DATA_DIR)


@st.cache_data
def get_cluster_labels():
    """Load cluster labels."""
    return load_latest_file("*labels.csv", OUTPUT_DATA_DIR)


def page_overview():
    """Overview Dashboard."""
    st.markdown('<p class="header-title">🏍️ Quality Analytics Overview</p>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    df_clustered = get_clustered_data()
    df_labels = get_cluster_labels()
    
    if df_clustered is not None and df_labels is not None:
        n_clusters = len(set(df_clustered["cluster_id"]) - {-1})
        n_noise = len(df_clustered[df_clustered["cluster_id"] == -1])
        total_issues = len(df_clustered)
        n_vehicles = df_clustered.shape[0]
        
        with col1:
            st.metric("Total Issues", total_issues, "📊")
        with col2:
            st.metric("Clusters Found", n_clusters, "🎯")
        with col3:
            st.metric("Noise Points", n_noise, f"{n_noise/total_issues*100:.1f}%")
        with col4:
            st.metric("Top Cluster Size", df_labels["failure_frequency"].max() if len(df_labels) > 0 else 0, "⚠️")
        
        st.divider()
        
        # Cluster size distribution
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.subheader("Cluster Size Distribution")
            fig = px.bar(
                df_labels.sort_values("failure_frequency", ascending=False).head(10),
                x="label",
                y="failure_frequency",
                color="failure_frequency",
                color_continuous_scale="Reds",
                title="Top 10 Failure Clusters",
            )
            fig.update_layout(xaxis_tickangle=-45, height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col_right:
            st.subheader("Component Distribution")
            component_data = df_labels["root_component"].value_counts().head(8)
            fig = px.pie(
                values=component_data.values,
                names=component_data.index,
                title="Affected Components",
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        
        # UMAP visualization
        st.subheader("Failure Pattern Landscape (UMAP)")
        embeddings_2d = load_embeddings()
        
        if embeddings_2d is not None and embeddings_2d.shape[1] >= 2:
            scatter_df = pd.DataFrame({
                "x": embeddings_2d[:, 0],
                "y": embeddings_2d[:, 1],
                "cluster_id": df_clustered["cluster_id"],
                "feedback_id": df_clustered["feedback_id"],
            })
            
            fig = px.scatter(
                scatter_df,
                x="x",
                y="y",
                color="cluster_id",
                hover_data=["feedback_id"],
                title="Semantic Failure Landscape",
                color_continuous_scale="Viridis",
                height=600,
            )
            fig.update_layout(showlegend=False, xaxis_title="UMAP-1", yaxis_title="UMAP-2")
            st.plotly_chart(fig, use_container_width=True)


def page_cluster_explorer():
    """Cluster Explorer."""
    st.markdown('<p class="header-title">🔍 Cluster Explorer</p>', unsafe_allow_html=True)
    
    df_labels = get_cluster_labels()
    df_clustered = get_clustered_data()
    df_feedback = get_raw_feedback()
    
    if df_labels is not None:
        # Select cluster
        selected_label = st.selectbox(
            "Select Failure Pattern",
            df_labels["label"].tolist(),
            key="cluster_select"
        )
        
        cluster_info = df_labels[df_labels["label"] == selected_label].iloc[0]
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Cluster ID", int(cluster_info["cluster_id"]))
        with col2:
            st.metric("Issue Count", int(cluster_info["failure_frequency"]))
        with col3:
            st.metric("Root Component", cluster_info["root_component"])
        with col4:
            st.metric("Confidence", f"{cluster_info['confidence']:.2%}")
        
        st.divider()
        
        # Cluster details
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.subheader("Symptom")
            st.write(cluster_info["recurring_symptom"])
        
        with col_right:
            st.subheader("Representative Complaint")
            st.info(cluster_info["representative_complaint"][:300])
        
        st.divider()
        
        # Member complaints
        st.subheader("Member Complaints in This Cluster")
        cluster_complaints = df_clustered[df_clustered["cluster_id"] == cluster_info["cluster_id"]]["feedback_id"].tolist()
        
        if df_feedback is not None:
            member_feedback = df_feedback[df_feedback["feedback_id"].isin(cluster_complaints)].head(5)
            
            for _, row in member_feedback.iterrows():
                with st.expander(f"🏍️ {row['vehicle_model']} - {row['language'].upper()} - {row['country']}"):
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        st.write(f"**Mileage:** {row['mileage']} km")
                        st.write(f"**Domain:** {row['domain']}")
                    with col2:
                        st.write(f"**Timestamp:** {row['timestamp']}")
                        st.write(f"**Feedback ID:** {row['feedback_id']}")
                    st.markdown(f"**Feedback:** {row['customer_feedback']}")


def page_failure_analytics():
    """Failure Analytics."""
    st.markdown('<p class="header-title">📈 Failure Analytics</p>', unsafe_allow_html=True)
    
    df_labels = get_cluster_labels()
    df_clustered = get_clustered_data()
    df_feedback = get_raw_feedback()
    
    if df_labels is not None and df_feedback is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Severity Distribution")
            # Count by severity (infer from failure frequency)
            df_labels_sorted = df_labels.sort_values("failure_frequency", ascending=False)
            severity_counts = []
            
            for i, row in df_labels_sorted.iterrows():
                freq = row["failure_frequency"]
                if freq > 50:
                    severity = "Critical"
                elif freq > 20:
                    severity = "High"
                elif freq > 10:
                    severity = "Medium"
                else:
                    severity = "Low"
                severity_counts.append(severity)
            
            severity_df = pd.Series(severity_counts).value_counts()
            fig = px.bar(
                x=severity_df.index,
                y=severity_df.values,
                color=severity_df.index,
                color_discrete_map={
                    "Critical": "#d62728",
                    "High": "#ff7f0e",
                    "Medium": "#ffbb78",
                    "Low": "#2ca02c"
                }
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Geographic Distribution")
            if "country" in df_feedback.columns:
                country_data = df_feedback["country"].value_counts().head(8)
                fig = px.bar(
                    x=country_data.index,
                    y=country_data.values,
                    color=country_data.values,
                    color_continuous_scale="Blues",
                )
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        
        # Language distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Language Distribution")
            lang_data = df_feedback["language"].value_counts()
            fig = px.pie(values=lang_data.values, names=lang_data.index)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Vehicle Model Distribution")
            vehicle_data = df_feedback["vehicle_model"].value_counts().head(6)
            fig = px.bar(x=vehicle_data.index, y=vehicle_data.values)
            fig.update_layout(xaxis_tickangle=-45, height=400)
            st.plotly_chart(fig, use_container_width=True)


def page_search():
    """Raw Complaint Search."""
    st.markdown('<p class="header-title">🔎 Complaint Search</p>', unsafe_allow_html=True)
    
    df_feedback = get_raw_feedback()
    df_clustered = get_clustered_data()
    
    if df_feedback is not None:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            vehicle_filter = st.multiselect(
                "Vehicle Model",
                df_feedback["vehicle_model"].unique(),
                key="vehicle_filter"
            )
        
        with col2:
            lang_filter = st.multiselect(
                "Language",
                df_feedback["language"].unique(),
                key="lang_filter"
            )
        
        with col3:
            domain_filter = st.multiselect(
                "Domain",
                df_feedback["domain"].unique(),
                key="domain_filter"
            )
        
        with col4:
            country_filter = st.multiselect(
                "Country",
                df_feedback["country"].unique(),
                key="country_filter"
            )
        
        # Apply filters
        filtered_df = df_feedback.copy()
        
        if vehicle_filter:
            filtered_df = filtered_df[filtered_df["vehicle_model"].isin(vehicle_filter)]
        if lang_filter:
            filtered_df = filtered_df[filtered_df["language"].isin(lang_filter)]
        if domain_filter:
            filtered_df = filtered_df[filtered_df["domain"].isin(domain_filter)]
        if country_filter:
            filtered_df = filtered_df[filtered_df["country"].isin(country_filter)]
        
        st.write(f"Found {len(filtered_df)} complaints")
        
        # Display results
        for _, row in filtered_df.head(20).iterrows():
            with st.expander(f"🏍️ {row['feedback_id']} - {row['vehicle_model']} ({row['language'].upper()})"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write(f"**Country:** {row['country']}")
                    st.write(f"**Mileage:** {row['mileage']} km")
                
                with col2:
                    st.write(f"**Domain:** {row['domain']}")
                    st.write(f"**Language:** {row['language']}")
                
                with col3:
                    st.write(f"**Timestamp:** {row['timestamp']}")
                
                st.markdown(f"**Feedback:** {row['customer_feedback']}")


def page_export():
    """Export Reports."""
    st.markdown('<p class="header-title">📥 Export Reports</p>', unsafe_allow_html=True)
    
    df_labels = get_cluster_labels()
    
    if df_labels is not None:
        st.subheader("Export Cluster Analysis Report")
        
        # CSV export
        csv_data = df_labels.to_csv(index=False, encoding="utf-8")
        st.download_button(
            label="📊 Download as CSV",
            data=csv_data,
            file_name="yamaha_cluster_report.csv",
            mime="text/csv",
        )
        
        st.divider()
        
        st.subheader("Report Preview")
        st.dataframe(
            df_labels[["label", "failure_frequency", "root_component", "recurring_symptom", "confidence"]],
            use_container_width=True,
            height=500,
        )


# Main app
def main():
    # Sidebar
    st.sidebar.title("🏍️ Yamaha Dashboard")
    st.sidebar.divider()
    
    page = st.sidebar.radio(
        "Navigation",
        ["Overview", "Cluster Explorer", "Failure Analytics", "Search", "Export"],
        icons=["📊", "🔍", "📈", "🔎", "📥"],
    )
    
    st.sidebar.divider()
    
    # Theme toggle
    st.sidebar.write("**Settings**")
    
    st.sidebar.divider()
    st.sidebar.caption("Yamaha AI Quality Analytics v1.0 | Powered by Python & LLMs")
    
    # Page routing
    if page == "Overview":
        page_overview()
    elif page == "Cluster Explorer":
        page_cluster_explorer()
    elif page == "Failure Analytics":
        page_failure_analytics()
    elif page == "Search":
        page_search()
    elif page == "Export":
        page_export()


if __name__ == "__main__":
    main()
