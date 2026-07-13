# AI-Powered Credit Risk Prediction System — Solution Document (Part 1)

## Comprehensive Default Prediction Model for Indian Financial Institutions

**Document Version:** 1.0  
**Date:** July 2026  
**Classification:** Confidential — Hackathon Submission  

---

# 1. Executive Summary

## 1.1 Problem Statement

Credit default prediction is one of the most consequential challenges facing Indian financial institutions today. The ability to accurately identify borrowers who are likely to default on their loan obligations — ideally 12 months before the event occurs — is critical to the survival and profitability of banks, NBFCs (Non-Banking Financial Companies), and housing finance companies across the country. Despite decades of investment in credit risk infrastructure, the Indian banking system continues to suffer from a staggering accumulation of Non-Performing Assets (NPAs), with gross NPAs crossing ₹6.8 lakh crore across scheduled commercial banks as of the latest RBI Financial Stability Report. The fundamental problem is not a lack of data or computational power, but rather the inability of existing models to synthesize the sheer complexity of borrower behavior, macroeconomic conditions, and unstructured information into a single, accurate, and timely prediction. Current models achieve accuracy rates in the range of 16–22% for 12-month forward-looking default prediction — a level that renders them nearly useless for proactive risk management and forces institutions into reactive, costly provisioning and recovery cycles.

The challenge is further compounded by the heterogeneity of the Indian credit market. India's financial landscape spans formal banking channels (public sector banks, private banks, foreign banks), a sprawling NBFC sector (over 9,000 registered with the RBI), cooperative banks, microfinance institutions (MFIs), and fintech lenders — each serving vastly different borrower segments from salaried professionals in metro cities to marginal farmers in rural Uttar Pradesh. A one-size-fits-all model cannot capture the nuanced risk profiles across these segments. Moreover, the regulatory environment under Basel III/IV, RBI's Ind AS 109 requirements, and the recently introduced Expected Credit Loss (ECL) framework under Indian Accounting Standards demands not just prediction accuracy but also model explainability, auditability, and compliance with fair lending norms. Any solution that fails on these dimensions — no matter how accurate — will fail in deployment.

## 1.2 Business Challenge

The business challenge is multidimensional and spans the entire credit lifecycle. At the origination stage, lenders must make accept/reject decisions within minutes for retail loans and within days for MSME and corporate loans, balancing speed of decisioning against risk of adverse selection. The rise of digital lending — with over ₹5.5 lakh crore disbursed through digital channels in FY2025 — has compressed decision timelines to seconds, making real-time risk assessment a necessity rather than a luxury. During the servicing phase, lenders must continuously monitor their portfolios for early warning signals of deterioration, dynamically adjusting exposure limits, interest rates, and provisioning levels. At the collection stage, early identification of likely defaulters enables proactive restructuring under RBI's Framework for Resolution of Stressed Assets, potentially saving thousands of crores in write-offs.

The business challenge also encompasses competitive dynamics. Traditional banks face increasing competition from fintech lenders who leverage alternative data and AI-native underwriting to serve thin-file and new-to-credit borrowers — a segment that constitutes over 400 million Indians. Banks that cannot match the speed, accuracy, and inclusivity of these AI-driven competitors risk losing market share in the fastest-growing segments of Indian lending. Simultaneously, the RBI has intensified its scrutiny of AI/ML models used in lending through its December 2024 discussion paper on AI in financial services, demanding robust model risk management, bias testing, and explainability frameworks. The business challenge, therefore, is to build a system that is simultaneously more accurate, more inclusive, more explainable, and more compliant than anything currently in production.

## 1.3 Current Limitations

The current state of credit risk modeling in Indian financial institutions is characterized by several critical limitations that collectively result in the dismal 16–22% accuracy observed for 12-month forward default prediction. First, the overwhelming reliance on structured data — typically limited to bureau scores, income declarations, and basic financial ratios — means that models are blind to a rich universe of behavioral, textual, and contextual signals that are available but untapped. Indian credit bureau data (from CIBIL, Experian, CRIF High Mark, and Equifax) provides repayment history and credit utilization but misses the qualitative dimensions of borrower intent, business conditions, and management quality that often precede default by months or years.

Second, existing approaches are fundamentally fragmented. Most institutions build siloed models for different product categories (home loans, auto loans, personal loans, MSME loans, gold loans) without a unified risk architecture that captures cross-product dependencies and portfolio-level correlations. A borrower who defaults on a personal loan is statistically far more likely to default on their MSME loan within 6–12 months, yet most models fail to capture this inter-product signal because they operate in isolation. Third, the temporal dimension is poorly handled. Static models trained on point-in-time snapshots fail to capture the trajectory and velocity of risk deterioration. A borrower whose CIBIL score dropped from 750 to 680 over six months carries fundamentally different risk than one whose score has been stable at 680 — yet most models treat them identically.

Fourth, model development cycles in Indian banks are excruciatingly slow, often taking 12–18 months from concept to production due to manual feature engineering, lengthy model validation processes, and fragmented technology infrastructure. By the time a model is deployed, the underlying data distributions may have shifted significantly, rendering the model suboptimal. Fifth, there is a near-total absence of unstructured data utilization. Loan application forms, relationship manager notes, customer correspondence, call center transcripts, auditor reports, and news articles contain rich risk-relevant information that is currently discarded or filed away unread. An estimated 80% of enterprise data is unstructured, yet credit risk models in India use virtually none of it.

## 1.4 Why Existing Models Fail

Existing models fail for specific, identifiable reasons that our solution directly addresses. The fundamental failure is the assumption that credit risk can be adequately captured through a limited set of financial ratios and bureau attributes. While variables like Debt-to-Income (DTI) ratio, credit utilization percentage, and repayment history are important, they represent a severely compressed representation of the true risk landscape. Studies by the RBI's Department of Statistics and Informatics Management (DOSIM) have shown that over 60% of MSME defaults in India are driven by factors not captured in traditional credit models, including customer concentration risk, raw material price volatility, regulatory changes, and management succession issues.

Another critical failure mode is population drift and concept drift. Indian credit markets undergo rapid structural shifts — the demonetization of 2016, the GST implementation of 2017, the COVID-19 pandemic of 2020–21, the fintech boom of 2022–24, and the ongoing digital lending revolution have each fundamentally altered borrower behavior and default patterns. Models trained on pre-pandemic data were catastrophically wrong during the pandemic period, and models trained during the pandemic failed to predict the rapid normalization that followed. Static models simply cannot adapt to such dynamic environments. Additionally, class imbalance — with default rates typically between 2–8% for retail loans — means that standard accuracy metrics are misleading. A model that predicts "no default" for every borrower would achieve 92–98% accuracy while being completely useless. Existing models often optimize for overall accuracy rather than the more business-critical metrics of recall (catching actual defaults) and precision (avoiding false alarms), leading to either excessive provisioning (if thresholds are set conservatively) or excessive NPAs (if thresholds are set permissively).

## 1.5 Business Impact

The business impact of poor credit risk prediction in India is enormous and multifaceted. For individual institutions, a 1% improvement in prediction accuracy for a bank with a ₹5 lakh crore loan book translates to approximately ₹5,000 crore in better-aligned provisioning and capital allocation. For the banking system as a whole, the cumulative impact runs into several lakh crore rupees annually. Beyond direct financial losses, poor prediction leads to suboptimal pricing — when risk cannot be accurately quantified, lenders either overcharge good borrowers (pricing out competitive business) or undercharge risky borrowers (subsidizing losses). This misallocation of capital has macroeconomic consequences, as credit flows away from productive, lower-risk activities toward higher-risk activities that happen to be mispriced.

The reputational impact is also significant. Banks that experience sudden NPA spikes face investor skepticism, credit rating downgrades, and regulatory scrutiny — all of which increase their cost of capital and constrain their ability to lend. The recent experiences of several Indian banks and NBFCs (including the high-profile cases of YES Bank, PMC Bank, and several large NBFCs during the IL&FS crisis) demonstrate how quickly credit risk can metastasize from an individual institution's problem to a systemic concern. Better prediction models would not only protect individual institutions but also contribute to the stability of the Indian financial system as a whole.

## 1.6 Financial Impact

The financial impact of credit defaults in India is staggering in both absolute and relative terms. As of March 2025, the gross NPA ratio of scheduled commercial banks stood at approximately 2.8% — representing roughly ₹6.8 lakh crore in stressed assets. However, this headline figure masks significant variation: public sector banks carry NPA ratios of 3.5–4.5%, while some private banks are below 1.5%. The net NPA ratio — after provisions — is approximately 0.8%, implying that banks have set aside over ₹6 lakh crore in provisions against bad loans. This provisioning directly erodes profitability: for every ₹100 of NPAs, banks must set aside ₹15–40 depending on the asset classification (substandard, doubtful, or loss), representing a massive drag on return on assets (ROA) and return on equity (ROE).

For NBFCs, the picture is even more concerning. The NBFC sector, which accounts for approximately 22% of total credit flow in India, often operates with higher NPA ratios and thinner capital buffers. The IL&FS crisis of 2018 demonstrated how interconnected NBFC failures can trigger liquidity crises across the banking system. The provisioning burden for NBFCs is estimated at ₹1.2–1.5 lakh crore annually, with recovery rates on defaulted loans averaging only 25–35% for secured loans and below 10% for unsecured loans. Capital adequacy is directly impacted: under Basel III requirements, banks must maintain a Capital Adequacy Ratio (CAR) of 10.5% (including capital conservation buffer), and every percentage point increase in NPAs increases risk-weighted assets, consuming capital that could otherwise be deployed for lending. Our solution, by improving prediction accuracy from the current 16–22% to a target of ~90%, could reduce NPA recognition delays by 6–9 months on average, potentially saving the Indian banking system ₹40,000–60,000 crore annually in avoided provisions and reduced credit costs.

## 1.7 Risk Impact

The risk impact extends beyond direct financial losses to encompass operational, regulatory, reputational, and systemic risk dimensions. Operationally, late detection of default risk leads to crisis-mode collections, which are 3–5x more expensive than proactive restructuring or early intervention. The human cost is also significant: RBI data shows that over 9,000 farmer suicides in India between 2015 and 2023 were linked to indebtedness, highlighting the social dimension of credit risk. From a regulatory perspective, the RBI's Prompt Corrective Action (PCA) framework imposes increasingly severe restrictions on banks that breach NPA thresholds — including limits on branch expansion, dividend payments, and new lending — creating a negative spiral that is extremely difficult to escape. Banks under PCA have historically taken 3–5 years to exit, during which their competitive position deteriorates significantly.

Systemically, the concentration of credit risk in specific sectors (real estate, infrastructure, steel, textiles) creates contagion risk that can amplify individual defaults into sector-wide crises. The infrastructure sector alone accounts for approximately 15% of total bank NPAs, and the real estate sector's interconnections with banking, NBFCs, and mutual fund segments create potential for rapid transmission of stress across the financial system. Our solution addresses these risks by providing sector-level risk aggregation capabilities, enabling portfolio-level stress testing aligned with RBI's macro-stress testing framework, and supporting dynamic sectoral exposure limits.

## 1.8 Vision

Our vision is to create the most intelligent, inclusive, and trustworthy credit risk prediction platform for the Indian financial ecosystem — one that transforms how lenders assess, monitor, and manage credit risk across every borrower segment, from the nano-entrepreneur in Bihar to the multinational corporation in Mumbai. We envision a future where no creditworthy borrower is denied access to finance due to inadequate risk assessment, and where every lender — regardless of size — has access to institutional-grade risk intelligence powered by the latest advances in artificial intelligence. By unifying structured and unstructured data, combining predictive accuracy with regulatory explainability, and delivering insights in real-time, we aim to reduce the Indian banking system's NPA burden by 30–40% within five years of deployment, freeing up trillions of rupees in capital for productive lending that drives India's economic growth toward its  trillion GDP aspiration.

## 1.9 Mission

Our mission is to design, build, and deploy an AI-powered default prediction system that achieves approximately 90% accuracy for 12-month forward-looking prediction while meeting the highest standards of explainability, fairness, and regulatory compliance. We will accomplish this through a multi-modal approach that combines structured financial data with unstructured textual and contextual information, leveraging state-of-the-art machine learning architectures including gradient boosting ensembles, deep learning models, and graph neural networks — all wrapped in a production-ready framework with real-time inference, automated monitoring, and continuous learning capabilities. The system will be designed to serve India's diverse lending landscape, accommodating the unique requirements of public sector banks, private banks, NBFCs, MFIs, and fintech lenders, and supporting loan products ranging from microfinance loans of ₹10,000 to corporate syndicated facilities of ₹10,000 crore.

## 1.10 Objectives

1. **Achieve 88–92% prediction accuracy** (measured as AUC-ROC) for 12-month forward default prediction across all major loan categories — retail (home, auto, personal, gold), MSME, agriculture, and corporate — validated on out-of-time test data representing at least 12 months of forward performance.

2. **Ingest and process 50+ structured data features** and **20+ unstructured data modalities** per borrower, including loan application text, bank statement narratives, financial statement notes, auditor comments, relationship manager assessments, news sentiment, and KYC document metadata, creating a unified borrower risk profile within 500ms of data receipt.

3. **Reduce false negative rate (missed defaults) to below 8%** — meaning that out of every 100 borrowers who will actually default within 12 months, the system correctly identifies at least 92 — while maintaining false positive rate below 15%, ensuring that creditworthy borrowers are not unnecessarily flagged or denied credit.

4. **Generate fully explainable risk assessments** meeting RBI's Model Risk Management guidelines, with every prediction accompanied by SHAP-based feature attribution, natural language risk narrative, regulatory-compliant reason codes (aligned with CIBIL's 4-factor reason code framework), and a confidence interval, ensuring auditability and regulatory compliance.

5. **Deliver inference latency below 200ms** for retail and MSME loan categories and below 2 seconds for corporate loans (which involve richer document analysis), supporting real-time digital lending decisioning through RESTful APIs with 99.9% uptime SLA.

6. **Process and analyze unstructured documents** (loan applications, financial statements, bank statements, legal documents) with entity extraction accuracy exceeding 95%, sentiment classification accuracy exceeding 88%, and risk phrase detection precision exceeding 85%, using a multi-modal NLP pipeline combining OCR, named entity recognition, sentiment analysis, and transformer-based text classification.

7. **Support portfolio-level risk aggregation** for loan books of up to ₹10 lakh crore, providing real-time concentration risk dashboards, sector-level stress testing (aligned with RBI's macro-stress testing framework), and early warning signal generation at borrower, branch, region, and sector levels, with automated escalation for high-risk alerts.

8. **Achieve full regulatory compliance** with RBI's December 2024 AI/ML discussion paper requirements, including model risk management framework, bias testing across protected attributes (gender, caste, religion, region), data privacy compliance with India's Digital Personal Data Protection Act 2023 (DPDPA), and audit trail maintenance for all model decisions with minimum 8-year retention.

9. **Reduce average NPA recognition lag by 6–9 months** compared to current industry practices, enabling proactive intervention through RBI-approved restructuring frameworks (under the June 2019 circular) and reducing provisioning costs by an estimated 25–35% for adopting institutions.

10. **Enable continuous model learning** through automated drift detection, scheduled retraining pipelines (monthly for fast-moving features, quarterly for full model refresh), champion-challenger framework for safe model deployment, and A/B testing infrastructure for measuring real-world impact of model updates.

## 1.11 Expected Benefits

| Benefit Category | Metric | Current State | Expected State | Improvement |
|---|---|---|---|---|
| Prediction Accuracy | AUC-ROC | 16–22% | 88–92% | +66–76 pp |
| NPA Recognition Speed | Months to detection | 12–18 months | 3–6 months | 6–12 months faster |
| Provisioning Cost | ₹ crore per ₹10,000 cr book | ₹280–450 cr | ₹180–300 cr | ₹100–150 cr savings |
| False Rejection Rate | % creditworthy rejected | 15–25% | 5–10% | 10–15 pp reduction |
| Digital Lending Decision Time | Seconds | 30–120 sec | <2 sec | 15–60x faster |
| Recovery Rate | % of defaulted amount | 25–35% | 40–55% | 15–20 pp improvement |
| Compliance Audit Time | Days per audit cycle | 45–60 days | 5–10 days | 80–90% reduction |
| Analyst Productivity | Loans reviewed per analyst/day | 15–25 | 80–120 | 4–5x improvement |
| Portfolio Risk Visibility | Sector coverage | 15–20 sectors | 80+ sectors | 4x expansion |
| Early Warning Accuracy | Recall at 6-month horizon | 30–40% | 75–85% | 45 pp improvement |

## 1.12 Key Performance Indicators (KPIs)

| # | KPI | Target | Measurement Frequency |
|---|---|---|---|
| 1 | AUC-ROC (12-month prediction) | ≥ 0.89 | Monthly |
| 2 | AUC-PR (Precision-Recall) | ≥ 0.72 | Monthly |
| 3 | False Negative Rate (missed defaults) | ≤ 8% | Monthly |
| 4 | False Positive Rate (false alarms) | ≤ 15% | Monthly |
| 5 | F1-Score | ≥ 0.82 | Monthly |
| 6 | Inference Latency (p99) | ≤ 200ms (retail), ≤ 2s (corporate) | Real-time |
| 7 | System Uptime | ≥ 99.9% | Continuous |
| 8 | NPA Recognition Lead Time | ≤ 6 months | Quarterly |
| 9 | Feature Importance Stability (PSI) | ≤ 0.10 | Monthly |
| 10 | Model Drift Detection (CSI) | ≤ 0.15 | Weekly |
| 11 | SHAP Coverage (% predictions with explanations) | 100% | Continuous |
| 12 | Unstructured Data Processing Accuracy (NER) | ≥ 95% | Weekly |
| 13 | Regulatory Compliance Score (internal audit) | ≥ 95/100 | Quarterly |
| 14 | Customer Complaint Rate (model-related) | ≤ 0.5% of decisions | Monthly |
| 15 | Model Retraining Cycle Time | ≤ 48 hours end-to-end | Per retraining cycle |
| 16 | Portfolio Coverage (% of loan book scored) | ≥ 98% | Monthly |

## 1.13 Return on Investment (ROI) — 3-Year Model

### Assumptions
- Adopting institution: Mid-size bank with ₹3,00,000 crore loan book
- Current NPA ratio: 3.5% (₹10,500 crore gross NPAs)
- Current provisioning rate: 2.5% of loan book (₹7,500 crore/year)
- Model implementation cost: ₹45 crore (Year 1), ₹12 crore/year (Year 2–3 maintenance)
- Expected NPA reduction: 25% over 3 years (conservative estimate)

### Year 1 (Implementation Year)

| Item | Amount (₹ Crore) |
|---|---|
| **Costs** | |
| Platform development & integration | 20.0 |
| Data infrastructure & pipelines | 8.0 |
| AI/ML model development & validation | 7.0 |
| Cloud infrastructure (compute, storage, networking) | 5.0 |
| Regulatory compliance & audit | 3.0 |
| Training & change management | 2.0 |
| **Total Year 1 Cost** | **45.0** |
| | |
| **Benefits** | |
| Provisioning reduction (10% improvement) | 75.0 |
| Reduced credit losses (5% improvement in early detection) | 52.5 |
| Operational efficiency gains (analyst productivity) | 15.0 |
| Reduced compliance costs | 5.0 |
| **Total Year 1 Benefit** | **147.5** |
| | |
| **Net Benefit Year 1** | **102.5** |
| **ROI Year 1** | **228%** |

### Year 2 (Optimization Year)

| Item | Amount (₹ Crore) |
|---|---|
| **Costs** | |
| Platform maintenance & updates | 6.0 |
| Cloud infrastructure | 3.5 |
| Model retraining & validation | 2.5 |
| **Total Year 2 Cost** | **12.0** |
| | |
| **Benefits** | |
| Provisioning reduction (20% cumulative improvement) | 150.0 |
| Reduced credit losses (12% improvement) | 126.0 |
| Operational efficiency gains | 25.0 |
| New revenue from improved accept rates | 35.0 |
| Reduced compliance costs | 8.0 |
| **Total Year 2 Benefit** | **344.0** |
| | |
| **Net Benefit Year 2** | **332.0** |
| **ROI Year 2** | **2,767%** |

### Year 3 (Maturity Year)

| Item | Amount (₹ Crore) |
|---|---|
| **Costs** | |
| Platform maintenance & updates | 6.0 |
| Cloud infrastructure | 4.0 |
| Model retraining & validation | 2.5 |
| **Total Year 3 Cost** | **12.5** |
| | |
| **Benefits** | |
| Provisioning reduction (25% cumulative improvement) | 187.5 |
| Reduced credit losses (18% improvement) | 189.0 |
| Operational efficiency gains | 30.0 |
| New revenue from improved accept rates | 50.0 |
| Reduced compliance costs | 10.0 |
| Reduced capital requirement (RWA optimization) | 25.0 |
| **Total Year 3 Benefit** | **491.5** |
| | |
| **Net Benefit Year 3** | **479.0** |
| **ROI Year 3** | **3,832%** |

### 3-Year Summary

| Metric | Value |
|---|---|
| Total Investment (3 years) | ₹69.5 crore |
| Total Benefits (3 years) | ₹983.0 crore |
| Total Net Benefits (3 years) | ₹913.5 crore |
| Cumulative ROI | **1,314%** |
| Payback Period | **< 4 months** |
| NPV (@ 12% discount rate) | ₹745 crore |
| IRR | **> 500%** |

---

# 2. Product Name
## 2.1 Product Name

**DrishtiAI**

*(Sanskrit: दृष्टि — meaning 'vision,' 'insight,' 'foresight')*

The name DrishtiAI embodies the core value proposition of the system: providing lenders with the foresight to see credit risk before it materializes. In Indian philosophy, Drishti represents the ability to perceive what lies beyond ordinary sight — the deeper patterns and hidden truths. Similarly, DrishtiAI sees beyond the surface-level financial metrics to uncover the latent risk signals buried in structured data, unstructured documents, behavioral patterns, and macroeconomic context. The name is easy to pronounce across Indian languages, carries cultural resonance, and communicates the product's purpose in a single word. The 'AI' suffix clearly positions the product as a technology-forward solution while maintaining the gravitas of its Sanskrit root.

## 2.2 Tagline

**'See Risk Before It Arrives'**

This tagline communicates three critical attributes: (1) the predictive nature of the system — it sees risk before it materializes, not after; (2) the proactive value proposition — enabling lenders to take action before losses occur; and (3) the temporal advantage — the 12-month prediction horizon that distinguishes DrishtiAI from reactive scoring systems. The tagline is concise, memorable, and action-oriented, suitable for both technical and business audiences.
## 2.3 Logo Concept

The DrishtiAI logo is designed around the visual metaphor of a stylized eye combined with a data flow pattern. The central element is a geometric iris rendered as a series of concentric, interlocking data nodes — small circles connected by fine lines — forming the shape of an eye. This represents the fusion of human insight (the eye) with data-driven intelligence (the network). The pupil of the eye is a solid circle in deep indigo, representing depth and trust. A single data flow line extends from the outer edge of the iris, curving upward and to the right, ending in a small upward-pointing arrow — symbolizing foresight, prediction, and positive outcomes.

The overall silhouette of the logo is clean and geometric, working equally well at small sizes (favicon, mobile app icon) and large sizes (conference banners, website hero). The logo mark is accompanied by the wordmark 'DrishtiAI' set in a custom-modified geometric sans-serif typeface with subtle rounded terminals that echo the circular data nodes in the mark. The 'D' in Drishti incorporates a small notch that mirrors the data flow line, creating a cohesive visual connection between mark and wordmark.

## 2.4 Brand Identity

### Color Palette

| Color | Hex Code | Usage | Psychological Association |
|---|---|---|---|
| **Drishti Indigo** | #1A1F71 | Primary brand color, logo, headers | Trust, depth, intelligence, stability |
| **Insight Teal** | #00B4D8 | Accent color, CTAs, interactive elements | Clarity, innovation, forward-thinking |
| **Risk Amber** | #F4A261 | Warning indicators, risk highlights | Attention, caution, warmth |
| **Safe Green** | #2A9D8F | Positive indicators, low-risk signals | Safety, growth, reliability |
| **Alert Crimson** | #E63946 | High-risk indicators, critical alerts | Urgency, importance, action-required |
| **Neutral Slate** | #4A5568 | Body text, secondary information | Professionalism, readability |
| **Background Cloud** | #F7FAFC | Primary background | Cleanliness, space, modernity |
| **Surface White** | #FFFFFF | Cards, panels, elevated surfaces | Clarity, simplicity |

### Typography

| Element | Typeface | Weight | Size Range |
|---|---|---|---|
| Primary Headings (H1) | Inter | Bold (700) | 32-48px |
| Section Headings (H2-H3) | Inter | SemiBold (600) | 24-32px |
| Body Text | Inter | Regular (400) | 14-16px |
| Data Tables | JetBrains Mono | Regular (400) | 12-14px |
| Code/Technical | JetBrains Mono | Medium (500) | 13-15px |
| Captions/Labels | Inter | Medium (500) | 11-13px |

The Inter typeface family is chosen for its exceptional readability at small sizes, its extensive language support (including Devanagari for Hindi interfaces), and its open-source license enabling unrestricted deployment. JetBrains Mono is used for all data-dense and technical content due to its tabular figures and ligature support.

### Design Language

The DrishtiAI design language follows three principles: **Clarity** (every element serves a purpose; no decorative elements that don't contribute to understanding), **Density** (credit risk professionals need to see many data points simultaneously; the UI maximizes information density without sacrificing readability), and **Confidence** (the system makes high-stakes recommendations; the design communicates authority and reliability through consistent spacing, precise alignment, and restrained use of color). Dashboard layouts use a 12-column grid system with 24px base spacing. Cards have 16px border radius and subtle drop shadows (0 2px 4px rgba(0,0,0,0.06)). Interactive elements use the Insight Teal accent with smooth 200ms transitions. Risk indicators use a consistent 5-level color scale from Safe Green (Level 1) through Risk Amber (Level 3) to Alert Crimson (Level 5).
---

# 3. Problem Analysis

## 3.1 Why Defaults Occur

Credit defaults in the Indian context occur due to a complex interplay of borrower-specific, institutional, and macroeconomic factors that interact in non-linear and often unpredictable ways. At the most fundamental level, a default occurs when a borrower's cash inflows become insufficient to meet their debt obligations — but the causes of this cash flow deterioration are myriad and diverse. For retail borrowers (salaried individuals), the primary drivers of default are income disruption (job loss, salary cuts, unpaid leave), over-leveraging (taking on multiple simultaneous loans without adequate income headroom), medical emergencies (which account for an estimated 25-30% of personal loan defaults in India), and life events (divorce, death of a co-borrower, relocation). For MSME borrowers, the causes are even more diverse: customer concentration risk (relying on one or two large buyers), raw material price volatility (particularly acute in sectors like textiles, metals, and food processing), working capital mismatches (where receivable cycles extend beyond payable cycles), and management capability gaps (especially in family-owned businesses facing succession transitions).

A critical and underappreciated cause of defaults in India is the phenomenon of 'loan stacking' — where borrowers take multiple loans from different lenders (including fintech apps and informal sources) without any single lender having visibility into the borrower's total exposure. The rise of digital lending has dramatically accelerated this phenomenon: a borrower can now take 5-10 instant personal loans from different fintech apps within days, creating a total debt burden that is catastrophically unsustainable. Indian credit bureaus capture most formal loans, but informal lending (from moneylenders, chit funds, peer lending, and family borrowings) remains largely invisible, creating a systematic blind spot in risk assessment. Additionally, willful defaults — where borrowers who have the capacity to repay choose not to — account for an estimated 10-15% of NPAs in India, particularly in the corporate and MSME segments. These require fundamentally different detection strategies than capacity-driven defaults, as the behavioral signals are often subtle and intentionally obscured.

## 3.2 Leading Indicators

Leading indicators are measurable signals that precede default, providing early warning of deterioration before the borrower actually misses a payment. These are the most valuable inputs for predictive models because they enable proactive intervention. The key leading indicators in the Indian context include:

1. **Declining bank balance trajectory**: A borrower whose average monthly bank balance has declined by >30% over three consecutive months shows a cash flow deterioration pattern that strongly precedes default, typically by 4-8 months. This signal is particularly powerful for MSME borrowers whose business cash flows are directly visible in bank statement data.

2. **Increased credit utilization across bureau**: When a borrower's aggregate credit utilization (across all credit cards and revolving facilities) rises above 70%, it signals financial stress. A month-on-month increase in utilization from 40% to 75% is one of the strongest leading indicators for retail loan defaults, with predictive power 3-6 months before default.

3. **New inquiry spike**: A sudden increase in credit bureau inquiries (3+ inquiries in 30 days) indicates the borrower is desperately seeking credit from multiple sources, a classic sign of financial distress. Research by CIBIL shows that borrowers with >5 inquiries in 90 days are 4x more likely to default than those with 0-1 inquiries.

4. **Salary account irregularity**: For salaried borrowers, irregular credits to salary accounts (indicating job instability), reduced salary amounts, or conversion of salary account to a general savings account are strong leading indicators. Data from bank statement analysis shows these signals precede default by 2-6 months.

5. **EMI bounce pattern**: The first EMI bounce is often a strong predictor of subsequent bounces and eventual default. Borrowers who bounce their first EMI have a 35-40% chance of defaulting within 12 months, compared to 3-5% for those with clean initial EMI history.

6. **GST filing irregularity (MSME)**: For GST-registered businesses, late or irregular GST filing is a leading indicator of operational distress. Businesses that miss two consecutive GST filing deadlines have a 4x higher probability of default within 9 months.

7. **Declining current account balance trend**: For business borrowers, a declining current account balance trend — particularly when it drops below the minimum balance requirement or triggers overdraft utilization — is a strong signal of working capital stress that typically precedes default by 3-9 months.

8. **Social/behavioral signals**: Changes in customer communication patterns (increased complaint frequency, unresponsiveness to calls, change of phone number), relationship manager downgrade notes, and customer-initiated requests for loan restructuring are behavioral leading indicators that carry significant predictive value.

9. **Industry-specific leading indicators**: Sector-specific signals such as declining order books (manufacturing), reduced tenant occupancy (real estate), declining patient footfall (hospitals with medical equipment loans), or reduced vehicle utilization (transport sector loans) provide early warning that is specific to the borrower's business context.

10. **Regulatory/policy changes**: Government policy changes such as changes in export duties (textiles, steel), environmental regulations (real estate, manufacturing), or subsidy structures (agriculture) can serve as leading indicators for sector-wide stress.

11. **Property price movement**: For secured loans (home loans, LAP), declining property valuations in the borrower's micro-market can lead to negative equity situations that correlate with strategic defaults, particularly when the outstanding loan amount approaches or exceeds the current market value.

12. **Co-borrower/guarantor distress**: When a co-borrower or guarantor on a loan experiences their own credit deterioration (visible through bureau data), the primary borrower's default probability increases significantly due to shared financial stress or weakened support structures.
