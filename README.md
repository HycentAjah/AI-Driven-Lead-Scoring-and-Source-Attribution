**AI-Driven Lead Scoring and Source Attribution**

This project implements an end-to-end lead scoring pipeline and interactive dashboard designed to help sales and marketing teams prioritize high-value prospects. 
By combining synthetic data generation, rule-based scoring logic, and a machine-learning model, the system automates the identification of Marketing Qualified 
Leads (MQLs) and Sales Qualified Leads (SQLs). The business problem it solves is the inefficiency and misalignment that arise when sales teams chase unqualified 
leads—wasting time and budget—by providing data-driven lead prioritization and source attribution analysis.

The codebase is organized into modular components. The `LeadScoringDataGenerator` class uses Faker and NumPy to synthesize firmographic 
(company size, industry, job title) and behavioral (email opens, website visits, demo requests) datasets, plus simulated conversion outcomes. 
The `RuleBasedScoringAgent` applies predefined weightings for firmographics and behaviors, caps extreme values, and categorizes each lead as SQL, MQL, 
or Unqualified based on strict thresholds. Meanwhile, the `MLScoringAgent` trains a logistic regression model on these features, evaluates performance 
via a train/test split, and outputs probability-based scores.

The `LeadScoringDashboard` class leverages Streamlit to present key performance indicators, category breakdowns, source-level attribution charts, and top leads in 
both rule-based and ML-driven formats. Custom HTML/CSS wrappers produce styled metric cards and section headers. Under the hood, 
the `LeadScoringOrchestrator` orchestrates the full pipeline: data generation, merging, scoring, model training, and finally dashboard rendering. 
This modular structure allows you to tweak weight parameters, add new features, or swap in alternative models with minimal code changes.

**How to Run**

1. Install dependencies: `pip install pandas streamlit faker numpy scikit-learn matplotlib`
2. Save the script as `lead_scoring_app.py` (or clone the repository).
3. From the project directory, execute:

   ```bash
   streamlit run lead_scoring_app.py
   ```
4. Your default browser will open the interactive dashboard. To regenerate synthetic data or retrain the model, simply refresh the page.
