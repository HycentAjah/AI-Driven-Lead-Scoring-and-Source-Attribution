import random
from datetime import datetime, timedelta
from typing import List, Dict, Any
import pandas as pd
import streamlit as st
from faker import Faker
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import plotly.express as px

# ======================
# Streamlit Page Setup + Modern Styling
# ======================
st.set_page_config(page_title="Lead Scoring Dashboard", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
<style>
/* Main header (blue gradient stays) */
.main-header {
  background: linear-gradient(135deg, #0066CC 0%, #5C8DFF 100%);
  padding: 1.5rem; border-radius: 16px; margin-bottom: 1rem;
  box-shadow: 0 8px 28px rgba(0,0,0,0.15);
}
.main-title { color:#fff; font-size:2.0rem; font-weight:800; text-align:center; margin:0;}
.subtitle { color:rgba(255,255,255,0.95); text-align:center; margin-top:.25rem; font-weight:400;}

/* Section titles: LIGHTER turquoise gradient */
.section-header {
  background: linear-gradient(90deg, #B2FFF6 0%, #E0FFFB 100%);
  padding:.8rem 1.1rem; border-radius: 12px; margin: 1.1rem 0 .6rem 0;
  border-left:4px solid #7FDED5;
}
.section-title { color:#08313A; font-size:1.1rem; font-weight:700; margin:0;}

/* Metric tiles polish */
div[data-testid="metric-container"]{
  background: linear-gradient(135deg, #fdfdfd 0%, #ffffff 100%);
  border:1px solid #e9ecef; padding:.9rem; border-radius:12px; box-shadow:0 2px 10px rgba(0,0,0,0.04);
}

/* Plotly card spacing */
.block-container { padding-top: 1.2rem; }

/* Dataframe styling tweak */
.stDataFrame { border: 1px solid #eaecef; border-radius: 10px; }
</style>
""",
    unsafe_allow_html=True,
)

def section(title: str):
    st.markdown(f'<div class="section-header"><div class="section-title">{title}</div></div>', unsafe_allow_html=True)

# Helper to make all chart labels sharp black (robust to Plotly versions)
def black_labels(fig):
    # Global font + legend
    fig.update_layout(
        font=dict(color="#000000"),
        legend=dict(font=dict(color="#000000")),
        xaxis=dict(
            title_font=dict(color="#000000"),
            tickfont=dict(color="#000000"),
            color="#000000"
        ),
        yaxis=dict(
            title_font=dict(color="#000000"),
            tickfont=dict(color="#000000"),
            color="#000000"
        )
    )
    # Annotations, if any
    if getattr(fig.layout, "annotations", None):
        fig.update_annotations(font_color="#000000")
    # Per-trace text font (bars/scatters with text)
    for tr in fig.data:
        try:
            tr.update(textfont=dict(color="#000000"))
        except Exception:
            pass
    return fig

# ======================
# 1. Synthetic Data Generator
# ======================
class LeadScoringDataGenerator:
    def __init__(self):
        self.fake = Faker()
        self.job_titles = ["CMO", "VP Marketing", "Marketing Director", "Marketing Manager", "Other"]
        self.industries = ["Technology", "Finance", "Healthcare", "Retail", "Manufacturing"]
        self.content_types = ["eBook", "Whitepaper", "Case Study", "Webinar", "Demo"]
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
            lead_source = random.choice(self.lead_sources)
            data.append({
                "lead_id": f"L{10000 + i}",
                "job_title": random.choices(self.job_titles, weights=[0.05, 0.1, 0.15, 0.3, 0.4])[0],
                "company_size": company_size,
                "industry": random.choice(self.industries),
                "revenue": round(random.uniform(min_rev, max_rev), 1),
                "location": self.fake.country(),
                "lead_source": lead_source,
                "first_contact_date": (datetime.now() - timedelta(days=random.randint(0, 90)))
            })
        return pd.DataFrame(data)
    
    def generate_behavioral_data(self, firmographic_df):
        behavioral_data = []
        for _, row in firmographic_df.iterrows():
            activity_level = 0
            if row["job_title"] in ["CMO", "VP Marketing"]:
                activity_level += random.randint(2, 5)
            elif row["job_title"] == "Marketing Director":
                activity_level += random.randint(1, 3)
            elif row["job_title"] == "Marketing Manager":
                activity_level += random.randint(1, 2)
            if row["company_size"] in ["51-200", "201-500", "501-1000", "1000+"]:
                activity_level += random.randint(1, 3)
            email_opens = max(0, int(np.random.poisson(lam=activity_level * 1.5)))
            email_clicks = max(0, int(np.random.poisson(lam=activity_level * 1.0)))
            website_visits = max(0, int(np.random.poisson(lam=activity_level * 2.0)))
            content_downloads = max(0, int(np.random.poisson(lam=activity_level * 0.8)))
            demo_requests = 1 if random.random() < (activity_level * 0.15) else 0
            behavioral_data.append({
                "lead_id": row["lead_id"],
                "email_opens": email_opens,
                "email_clicks": email_clicks,
                "website_visits": website_visits,
                "content_downloads": content_downloads,
                "content_type": random.choice(self.content_types) if content_downloads > 0 else None,
                "demo_requested": demo_requests,
                "time_on_site": random.expovariate(1/(activity_level + 1)) * 20 if website_visits > 0 else 0,
                "last_activity_date": (datetime.now() - timedelta(days=random.randint(0, 30)))
            })
        return pd.DataFrame(behavioral_data)
    
    def generate_outcome_data(self, merged_df):
        outcomes = []
        for _, row in merged_df.iterrows():
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
            base_prob += random.uniform(-0.1, 0.1)
            base_prob = max(0, min(1, base_prob))
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
        self.job_title_weights = {"CMO": 35, "VP Marketing": 30, "Marketing Director": 20, "Marketing Manager": 15, "Other": 5}
        self.company_size_weights = {"1-10": 5, "11-50": 10, "51-200": 20, "201-500": 25, "501-1000": 30, "1000+": 35}
        self.industry_weights = {"Technology": 15, "Finance": 12, "Healthcare": 10, "Retail": 8, "Manufacturing": 7}
        self.behavior_weights = {"email_opens": 0.5, "email_clicks": 1.5, "website_visits": 2.0,
                                 "content_downloads": 8, "demo_requested": 25, "time_on_site": 0.05}
        self.behavior_cap = 115
        
    def calculate_scores(self, data: pd.DataFrame) -> pd.DataFrame:
        scores = []
        for _, row in data.iterrows():
            firmographic_score = (
                self.job_title_weights.get(row["job_title"], 0)
                + self.company_size_weights.get(row["company_size"], 0)
                + self.industry_weights.get(row["industry"], 0)
            )
            raw_behavioral = (
                row["email_opens"]   * self.behavior_weights["email_opens"]
                + row["email_clicks"]  * self.behavior_weights["email_clicks"]
                + row["website_visits"]* self.behavior_weights["website_visits"]
                + row["content_downloads"] * self.behavior_weights["content_downloads"]
                + row["demo_requested"]    * self.behavior_weights["demo_requested"]
                + row["time_on_site"]      * self.behavior_weights["time_on_site"]
            )
            behavioral_score = min(raw_behavioral, self.behavior_cap)
            total_score = firmographic_score + behavioral_score
            category = "SQL" if total_score >= 110 else ("MQL" if total_score >= 85 else "Unqualified")
            scores.append({
                "lead_id": row["lead_id"],
                "firmographic_score": firmographic_score,
                "behavioral_score": behavioral_score,
                "total_score": total_score,
                "lead_category": category,
                "converted": row.get("converted", 0)
            })
        return pd.DataFrame(scores).sort_values("lead_id").reset_index(drop=True)

class MLScoringAgent:
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000, random_state=42)
    def train_model(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
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
        
    def display(self):
        # Header
        st.markdown(
            """
<div class="main-header">
  <div class="main-title">Lead Scoring Dashboard</div>
  <div class="subtitle">Attribution • Scoring • Conversion Insights</div>
</div>
""",
            unsafe_allow_html=True,
        )
        
        # KPI Overview
        section("Lead Scoring Overview")
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Total Leads", f"{len(self.scores):,}")
        with c2: st.metric("SQL Count", f"{(self.scores['lead_category']=='SQL').sum():,}")
        with c3: st.metric("MQL Count", f"{(self.scores['lead_category']=='MQL').sum():,}")
        with c4: st.metric("Conversion Rate", f"{self.scores['converted'].mean()*100:,.1f}%")
        
        # ============================
        # Lead Category Breakdown + Conversion Rate by Source (side-by-side)
        # ============================
        section("Lead Category Breakdown & Conversion Rate by Source")
        colA, colB = st.columns(2)
        
        with colA:
            cat_counts = self.scores["lead_category"].value_counts().sort_values(ascending=False)
            df_cat = cat_counts.rename_axis("lead_category").reset_index(name="count")
            fig_cat = px.bar(
                df_cat, x="lead_category", y="count", text="count", color="lead_category",
                title="Lead Category Breakdown"
            )
            fig_cat.update_traces(textposition="outside", showlegend=False)
            fig_cat.update_layout(height=330, margin=dict(l=10,r=10,t=50,b=20))
            fig_cat = black_labels(fig_cat)
            st.plotly_chart(fig_cat, use_container_width=True)
        
        with colB:
            src_conv = (self.scores.merge(self.data[["lead_id","lead_source"]], on="lead_id")
                                   .groupby("lead_source")["converted"].mean().mul(100).reset_index())
            src_conv = src_conv.sort_values("converted", ascending=False)
            fig_src_conv = px.bar(
                src_conv, x="lead_source", y="converted", text="converted",
                title="Conversion Rate by Attribution Source"
            )
            fig_src_conv.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig_src_conv.update_layout(height=330, margin=dict(l=10,r=10,t=50,b=20), showlegend=False)
            fig_src_conv.update_xaxes(tickangle=45)
            fig_src_conv = black_labels(fig_src_conv)
            st.plotly_chart(fig_src_conv, use_container_width=True)
        
        # ============================
        # Lead Category by Attribution Source (stacked)
        # ============================
        section("Lead Category by Attribution Source")
        src_cat = self.scores.merge(self.data[["lead_id","lead_source"]], on="lead_id")
        stacked = src_cat.groupby(["lead_source","lead_category"]).size().reset_index(name="count")
        stacked = stacked.sort_values(["lead_source","lead_category"])
        fig_stack = px.bar(stacked, x="lead_source", y="count", color="lead_category", barmode="stack")
        fig_stack.update_layout(height=360, margin=dict(l=10,r=10,t=50,b=20))
        fig_stack.update_xaxes(tickangle=45)
        fig_stack = black_labels(fig_stack)
        st.plotly_chart(fig_stack, use_container_width=True)

        # =========================================
        # Lead Source Bubble: Volume vs Conv. Rate (bubble = Avg Score) — LIGHT PURPLE
        # =========================================
        section("Lead Source Effectiveness (Bubble: Volume vs Conversion; Size = Avg Score)")
        bubble_df = src_cat.groupby('lead_source').agg(
            leads=('lead_id', 'count'),
            conv_rate=('converted', 'mean'),
            avg_score=('total_score', 'mean')
        ).reset_index()
        bubble_df['conv_rate'] *= 100.0
        fig_bub = px.scatter(
            bubble_df, x="leads", y="conv_rate", size="avg_score", text="lead_source",
            size_max=60, hover_data={"avg_score":":.1f", "leads":":,", "conv_rate":":.1f"}
        )
        # Light purple bubbles with a darker purple outline
        fig_bub.update_traces(
            marker=dict(color="#D8B4FE", line=dict(width=1, color="#8B5CF6"), opacity=0.85),
            textposition="top center"
        )
        fig_bub.update_layout(height=380, margin=dict(l=10,r=10,t=50,b=20),
                              xaxis_title="Leads", yaxis_title="Conversion Rate (%)")
        fig_bub = black_labels(fig_bub)
        st.plotly_chart(fig_bub, use_container_width=True)

        # ============================
        # Top Leads
        # ============================
        section("Top Scoring Leads")
        top_leads = self.scores.merge(self.data, on="lead_id").sort_values("total_score", ascending=False).head(10)
        st.dataframe(top_leads[["lead_id", "job_title", "company_size", "industry", "lead_source", "total_score", "lead_category"]])

        # ============================
        # ML Scoring Predictions (tables only)
        # ============================
        section("Lead Scoring Predictions (ML Model)")
        if "ml_score" in self.scores.columns:
            st.write("### Highest Conversion Probability Leads")
            top_ml = self.scores.merge(self.data, on="lead_id").sort_values("ml_score", ascending=False).head(15)
            cols = [c for c in ["lead_id", "job_title", "company_size", "industry", "ml_score", "total_score", "converted"] if c in top_ml.columns]
            st.dataframe(top_ml[cols])

            st.write("### Rule-based vs ML Score Comparison (Top 20 Leads)")
            cmp = self.scores.merge(self.data, on="lead_id").sort_values("ml_score", ascending=False).head(20)
            cols2 = [c for c in ["lead_id","job_title","total_score","lead_category","ml_score","converted"] if c in cmp.columns]
            st.dataframe(cmp[cols2])
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
        firmographic = self.generator.generate_firmographic_data(1000)
        behavioral = self.generator.generate_behavioral_data(firmographic)
        temp_merged = pd.merge(firmographic, behavioral, on="lead_id")
        outcome = self.generator.generate_outcome_data(temp_merged)
        merged = pd.merge(temp_merged, outcome, on="lead_id")
        
        # Rule-based scoring
        scores = self.rule_agent.calculate_scores(merged)
        
        # ML-based scoring
        categorical_cols = ["job_title", "company_size", "industry"]
        numerical_cols = ["email_opens", "email_clicks", "website_visits", "content_downloads", "demo_requested", "time_on_site"]
        X_cat = pd.get_dummies(merged[categorical_cols], prefix=categorical_cols)
        X_num = merged[numerical_cols]
        X = pd.concat([X_cat, X_num], axis=1)
        y = merged["converted"]
        self.ml_agent.train_model(X, y)
        scores["ml_score"] = self.ml_agent.predict_score(X)
        
        # Dashboard
        dashboard = LeadScoringDashboard(merged, scores)
        dashboard.display()

# ======================
# 5. Execution
# ======================
if __name__ == "__main__":
    orchestrator = LeadScoringOrchestrator()
    orchestrator.run()
