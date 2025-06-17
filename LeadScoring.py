import random
from datetime import datetime, timedelta
from typing import List, Dict, Any
import pandas as pd
import streamlit as st
from faker import Faker
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ======================
# 1. Synthetic Data Generator
# ======================
class LeadScoringDataGenerator:
    def __init__(self):
        self.fake = Faker()
        self.job_titles = ["CMO", "VP Marketing", "Marketing Director", "Marketing Manager", "Other"]
        self.industries = ["Technology", "Finance", "Healthcare", "Retail", "Manufacturing"]
        self.content_types = ["eBook", "Whitepaper", "Case Study", "Webinar", "Demo"]
        
        # 1) Define your inbound and outbound lists exactly as given:
        self.inbound_sources = [
            "Organic Search",
            "Paid Search",
            "Content Marketing",
            "Social Media (Organic)",
            "Paid Social Advertising",
        ]
        self.outbound_sources = [
            "Cold Calling",
            "Cold Email",
            "Purchased List",
            "LinkedIn Outreach",
            "SDR Outreach",
            "ABM Ads",
            "Outbound Event Invitation",
            "Third-Party SDR Service",
        ]
        
        # 2) Combine them into one master list of lead_sources:
        self.lead_sources = self.inbound_sources + self.outbound_sources
        
    def generate_firmographic_data(self, count=1000):
        data = []
        for i in range(count):
            company_size = random.choice(["1-10", "11-50", "51-200", "201-500", "501-1000", "1000+"])
            revenue_map = {
                "1-10": (0, 1),
                "11-50": (1, 5),
                "51-200": (5, 20),
                "201-500": (20, 50),
                "501-1000": (50, 100),
                "1000+": (100, 500)
            }
            min_rev, max_rev = revenue_map[company_size]
            
            # Assign random lead source
            lead_source = random.choice(self.lead_sources)
            
            data.append({
                "lead_id": f"L{10000 + i}",
                "job_title": random.choices(
                    self.job_titles,
                    weights=[0.05, 0.1, 0.15, 0.3, 0.4]
                )[0],
                "company_size": company_size,
                "industry": random.choice(self.industries),
                "revenue": round(random.uniform(min_rev, max_rev), 1),
                "location": self.fake.country(),
                "lead_source": lead_source,  # New field
                "first_contact_date": (datetime.now() - timedelta(days=random.randint(0, 90)))
            })
        return pd.DataFrame(data)
    
    def generate_behavioral_data(self, firmographic_df):
        behavioral_data = []
        for _, row in firmographic_df.iterrows():
            # Base activity level based on job title and company size
            activity_level = 0
            if row["job_title"] in ["CMO", "VP Marketing"]:
                activity_level += random.randint(2, 5)  # Increased activity
            elif row["job_title"] == "Marketing Director":
                activity_level += random.randint(1, 3)
            elif row["job_title"] == "Marketing Manager":
                activity_level += random.randint(1, 2)
            
            if row["company_size"] in ["51-200", "201-500", "501-1000", "1000+"]:
                activity_level += random.randint(1, 3)  # Increased activity
            
            # Generate more behavioral events with higher variance
            email_opens = max(0, int(np.random.poisson(lam=activity_level * 1.5)))
            email_clicks = max(0, int(np.random.poisson(lam=activity_level * 1.0)))
            website_visits = max(0, int(np.random.poisson(lam=activity_level * 2.0)))
            content_downloads = max(0, int(np.random.poisson(lam=activity_level * 0.8)))
            demo_requests = 1 if random.random() < (activity_level * 0.15) else 0  # Higher demo request rate
            
            behavioral_data.append({
                "lead_id": row["lead_id"],
                "email_opens": email_opens,
                "email_clicks": email_clicks,
                "website_visits": website_visits,
                "content_downloads": content_downloads,
                "content_type": random.choice(self.content_types) if content_downloads > 0 else None,
                "demo_requested": demo_requests,
                "time_on_site": random.expovariate(1/(activity_level + 1)) * 20 if website_visits > 0 else 0,  # More time on site
                "last_activity_date": (datetime.now() - timedelta(days=random.randint(0, 30)))
            })
        return pd.DataFrame(behavioral_data)
    
    def generate_outcome_data(self, merged_df):
        outcomes = []
        for _, row in merged_df.iterrows():
            # Calculate probability of conversion based on features
            base_prob = 0.05
            if row["job_title"] in ["CMO", "VP Marketing"]:
                base_prob += 0.15
            if row["company_size"] in ["201-500", "501-1000", "1000+"]:
                base_prob += 0.1
            if row["industry"] in ["Technology", "Finance"]:
                base_prob += 0.05
            if row["content_downloads"] > 0:
                base_prob += 0.1
            if row["demo_requested"] > 0:
                base_prob += 0.3
            
            # Add some noise
            base_prob += random.uniform(-0.1, 0.1)
            base_prob = max(0, min(1, base_prob))
            
            # Determine outcome
            converted = 1 if random.random() < base_prob else 0
            outcomes.append({
                "lead_id": row["lead_id"],
                "converted": converted,
                "conversion_date": (datetime.now() + timedelta(days=random.randint(0, 60))) if converted else None
            })
        return pd.DataFrame(outcomes)

# ======================
# 2. Lead Scoring Agents
# ======================
class RuleBasedScoringAgent:
    def __init__(self):
        # Firmographic weights (unchanged)
        self.job_title_weights = {
            "CMO": 35,
            "VP Marketing": 30,
            "Marketing Director": 20,
            "Marketing Manager": 15,
            "Other": 5
        }
        self.company_size_weights = {
            "1-10": 5,
            "11-50": 10,
            "51-200": 20,
            "201-500": 25,
            "501-1000": 30,
            "1000+": 35
        }
        self.industry_weights = {
            "Technology": 15,
            "Finance": 12,
            "Healthcare": 10,
            "Retail": 8,
            "Manufacturing": 7
        }

        # Behavioral weights (unchanged)
        self.behavior_weights = {
            "email_opens": 0.5,
            "email_clicks": 1.5,
            "website_visits": 2.0,
            "content_downloads": 8,
            "demo_requested": 25,
            "time_on_site": 0.05
        }

        # Behavioral cap (new)
        self.behavior_cap = 115
        
    def calculate_scores(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        For each lead (row) in `data`, compute:
          - firmographic_score
          - raw behavioral_score, then cap it at self.behavior_cap
          - total_score = firmographic_score + capped behavioral_score
          - lead_category by strict thresholds:
                total_score >= 110  -> "SQL"
                85 <= total_score < 110 -> "MQL"
                total_score <  85   -> "Unqualified"
        """
        scores = []

        for _, row in data.iterrows():
            # 1. Compute firmographic_score
            firmographic_score = (
                self.job_title_weights.get(row["job_title"], 0)
                + self.company_size_weights.get(row["company_size"], 0)
                + self.industry_weights.get(row["industry"], 0)
            )

            # 2. Compute raw behavioral_score
            raw_behavioral = (
                row["email_opens"]   * self.behavior_weights["email_opens"]
                + row["email_clicks"]  * self.behavior_weights["email_clicks"]
                + row["website_visits"]* self.behavior_weights["website_visits"]
                + row["content_downloads"] * self.behavior_weights["content_downloads"]
                + row["demo_requested"]    * self.behavior_weights["demo_requested"]
                + row["time_on_site"]      * self.behavior_weights["time_on_site"]
            )

            # 3. Cap the behavioral_score at 115
            behavioral_score = min(raw_behavioral, self.behavior_cap)

            # 4. Compute total_score (now between 0 and 200)
            total_score = firmographic_score + behavioral_score

            # 5. Assign lead_category by strict total_score ranges
            if total_score >= 110:
                category = "SQL"
            elif total_score >= 85:
                category = "MQL"
            else:
                category = "Unqualified"

            scores.append({
                "lead_id": row["lead_id"],
                "firmographic_score": firmographic_score,
                "behavioral_score": behavioral_score,
                "total_score": total_score,
                "lead_category": category,
                "converted": row.get("converted", 0)
            })

        # Build a DataFrame in the same order as input (sorted by lead_id for consistency)
        scores_df = pd.DataFrame(scores)
        return scores_df.sort_values("lead_id").reset_index(drop=True)

class MLScoringAgent:
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000, random_state=42)
        
    def train_model(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        print(classification_report(y_test, y_pred))
        
    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]
    
    def predict_score(self, X):
        return (self.predict_proba(X) * 100).astype(int)

# ======================
# 3. Dashboard
# ======================
class LeadScoringDashboard:
    def __init__(self, data, scores):
        self.data = data
        self.scores = scores
        
    def _styled_metric(self, title, value, delta=None):
        delta_html = ""
        if delta is not None:
            color = "#2ecc71" if (isinstance(delta, str) or delta >= 0) else "#e74c3c"
            delta_html = f'<div style="color: {color}; font-size: 14px;">{delta}</div>'

        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #e0f7fa 0%, #ffffff 100%);
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
        ">
            <div style="color: #555; font-size: 14px; font-weight: 600;">{title}</div>
            <div style="color: #2c3e50; font-size: 28px; font-weight: 700; margin: 10px 0;">{value}</div>
            {delta_html}
        </div>
        """, unsafe_allow_html=True)
        
    def _section_header(self, text):
        st.markdown(f"<h2 style='color: #228B22; margin-top: 30px;'>{text}</h2>", unsafe_allow_html=True)
        
    def display(self):
        st.set_page_config(layout="wide", page_title="Lead Scoring and Attribution Dashboard")
        
        # Header
        st.markdown("<h1 style='color: #008080;'>Lead Scoring and Attribution Dashboard</h1>", unsafe_allow_html=True)
        
        # KPI Overview
        self._section_header("Lead Scoring Overview")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            self._styled_metric("Total Leads", len(self.scores))
        with col2:
            sql_count = len(self.scores[self.scores["lead_category"] == "SQL"])
            self._styled_metric("SQL Count", sql_count)
        with col3:
            mql_count = len(self.scores[self.scores["lead_category"] == "MQL"])
            self._styled_metric("MQL Count", mql_count)
        with col4:
            conv_rate = self.scores["converted"].mean() * 100
            self._styled_metric("Conversion Rate", f"{conv_rate:.1f}%")
        
        # Category Breakdown - Sort from highest to lowest
        self._section_header("Lead Category Breakdown")
        category_counts = self.scores["lead_category"].value_counts().sort_values(ascending=False)
        st.bar_chart(category_counts)

        # Lead Source Analysis
        self._section_header("Lead Attribution Analysis")
        
        # Merge data for source analysis - MOVED TO THE TOP
        source_data = self.scores.merge(self.data[['lead_id', 'lead_source']], on='lead_id')
        
        # Lead categories by source - sort by total leads (highest to lowest)
        st.write("### Lead Category by Attribution Source")
        source_category_viz = source_data.groupby(['lead_source', 'lead_category']).size().unstack(fill_value=0)
        # Sort by total leads per source
        source_totals = source_category_viz.sum(axis=1).sort_values(ascending=False)
        source_category_viz = source_category_viz.reindex(source_totals.index)
        st.bar_chart(source_category_viz)
        
        

        # Per-source summary - sorted by average score (highest to lowest)
        st.write("### Lead Source Summary")
        source_summary = source_data.groupby('lead_source').agg({
            'lead_id': 'count',
            'total_score': 'mean',
            'converted': 'mean'
        }).round(2)
        source_summary.columns = ['Total Leads', 'Avg Score', 'Conversion Rate']
        source_summary['Conversion Rate'] = (source_summary['Conversion Rate'] * 100).round(1)
        source_summary = source_summary.sort_values('Avg Score', ascending=False)
        st.dataframe(source_summary)
        
        # Top Leads
        self._section_header("Top Scoring Leads")
        top_leads = self.scores.merge(self.data, on="lead_id").sort_values("total_score", ascending=False).head(10)
        st.dataframe(top_leads[["lead_id", "job_title", "company_size", "industry", "lead_source", "total_score", "lead_category"]])
        
        # ML Scoring Predictions
        self._section_header("Lead Scoring Predictions (ML Model)")
        if "ml_score" in self.scores.columns:
            # Show top ML predictions (highest conversion probability)
            st.write("### Highest Conversion Probability Leads")
            top_ml_leads = self.scores.merge(self.data, on="lead_id").sort_values("ml_score", ascending=False).head(15)
            
            # Only include columns that exist in the dataframe
            base_cols = ["lead_id", "job_title", "company_size", "industry", "ml_score", "total_score"]
            display_cols = [col for col in base_cols if col in top_ml_leads.columns]
            if "converted" in top_ml_leads.columns:
                display_cols.append("converted")
            
            st.dataframe(top_ml_leads[display_cols])
            
            # Distribution of ML scores
            st.write("### ML Conversion Probability Distribution")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(self.scores["ml_score"], bins=20, color='#FF6B6B', alpha=0.7, edgecolor='black')
            ax.set_xlabel("ML Conversion Probability (%)")
            ax.set_ylabel("Number of Leads")
            ax.set_title("Distribution of ML Conversion Probabilities")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()
            
            # Comparison: Rule-based vs ML approach
            st.write("### Rule-based vs ML Score Comparison (Top 20 Leads)")
            comparison_data = self.scores.merge(self.data, on="lead_id")
            comparison_data = comparison_data.sort_values("ml_score", ascending=False).head(20)
            
            # Build comparison columns dynamically
            comp_base_cols = ["lead_id", "job_title", "total_score", "lead_category", "ml_score"]
            comparison_display_cols = [col for col in comp_base_cols if col in comparison_data.columns]
            if "converted" in comparison_data.columns:
                comparison_display_cols.append("converted")
                
            st.dataframe(comparison_data[comparison_display_cols])
        else:
            st.write("ML scoring not available")
            
        with st.expander("Show Raw Data"):
            st.write("### Lead Data")
            st.dataframe(self.data)
            st.write("### Scoring Results")
            st.dataframe(self.scores)

# ======================
# 4. Orchestrator
# ======================
class LeadScoringOrchestrator:
    def __init__(self):
        self.generator = LeadScoringDataGenerator()
        self.rule_agent = RuleBasedScoringAgent()
        self.ml_agent = MLScoringAgent()
        
    def run(self):
        # Generate synthetic data
        print("Generating synthetic data...")
        firmographic_data = self.generator.generate_firmographic_data(1000)
        behavioral_data = self.generator.generate_behavioral_data(firmographic_data)
        
        # Merge firmographic and behavioral data first
        temp_merged = pd.merge(firmographic_data, behavioral_data, on="lead_id")
        outcome_data = self.generator.generate_outcome_data(temp_merged)
        
        # Merge all data
        merged_data = pd.merge(temp_merged, outcome_data, on="lead_id")
        
        print("Calculating rule-based scores...")
        # Rule-based scoring
        rule_scores = self.rule_agent.calculate_scores(merged_data)
        
        print("Training ML model...")
        # ML-based scoring
        # Prepare features - handle categorical variables properly
        categorical_cols = ["job_title", "company_size", "industry"]
        numerical_cols = ["email_opens", "email_clicks", "website_visits", 
                         "content_downloads", "demo_requested", "time_on_site"]
        
        # Create feature matrix
        X_categorical = pd.get_dummies(merged_data[categorical_cols], prefix=categorical_cols)
        X_numerical = merged_data[numerical_cols]
        X = pd.concat([X_categorical, X_numerical], axis=1)
        y = merged_data["converted"]
        
        # Train model
        self.ml_agent.train_model(X, y)
        ml_scores = self.ml_agent.predict_score(X)
        
        # Add ML scores to results (raw probability * 100)
        rule_scores["ml_score"] = ml_scores
        
        # Print distribution summary
        category_counts = rule_scores["lead_category"].value_counts()
        print(f"\nLead Distribution:")
        print(f"SQL: {category_counts.get('SQL', 0)}")
        print(f"MQL: {category_counts.get('MQL', 0)}")
        print(f"Unqualified: {category_counts.get('Unqualified', 0)}")
        
        print("Launching dashboard...")
        # Create dashboard
        dashboard = LeadScoringDashboard(merged_data, rule_scores)
        dashboard.display()

# ======================
# 5. Execution
# ======================
if __name__ == "__main__":
    orchestrator = LeadScoringOrchestrator()
    orchestrator.run()
