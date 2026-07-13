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
## 3.3 Lagging Indicators

Lagging indicators confirm that a default has already occurred or is imminent, providing validation rather than prediction. While less useful for early warning, they are critical for model training (as labels) and for understanding default dynamics:

1. **Days Past Due (DPD) escalation**: The progression from current to 30 DPD to 60 DPD to 90 DPD is the most definitive lagging indicator. Once a borrower reaches 90 DPD, the loan is classified as an NPA under RBI norms, triggering provisioning requirements.

2. **NPA classification timing**: The actual classification of a loan as NPA (when interest payment or principal repayment remains overdue for more than 90 days) is the definitive lagging indicator. However, the gap between the economic event of default and the regulatory classification is often 3-6 months.

3. **Restructuring request submission**: When a borrower formally requests loan restructuring under RBI's June 2019 framework, it is a strong lagging indicator that the borrower is already in financial distress.

4. **Legal notice issuance**: The issuance of legal notices under Section 13(2) of the SARFAESI Act (for secured loans) or other legal proceedings is a definitive lagging indicator of default that typically occurs 3-6 months after the NPA classification.

5. **Collateral valuation decline**: Significant decline in collateral value below the loan-to-value (LTV) threshold is a lagging indicator that often accompanies or follows default, particularly in real estate and gold loan portfolios.

6. **Guarantor invocation**: When lenders invoke guarantees from co-borrowers or guarantors, it confirms that the primary borrower has failed to meet obligations. The success rate of guarantor invocation (typically 30-40% for individual guarantees in India) provides information about potential recovery.

7. **Write-off classification**: The actual write-off of a loan is the ultimate lagging indicator, typically occurring 12-24 months after NPA classification. Write-off rates vary significantly by loan type: personal loans (40-60%), MSME loans (50-70%), home loans (10-20%).

8. **Filing with Debt Recovery Tribunal (DRT)**: The filing of applications with DRTs or Lok Adalats for debt recovery is a lagging indicator that the lending institution has exhausted informal resolution mechanisms and is pursuing formal legal recovery.

## 3.4 Behavioral Risk

Behavioral risk encompasses the borrower behaviors, attitudes, and decision-making patterns that influence credit performance. In the Indian context, behavioral risk is particularly complex due to cultural, social, and economic factors.

**Intentionality of repayment**: Some defaults are involuntary (the borrower genuinely cannot pay due to income loss or unexpected expenses), while others are strategic (the borrower has the capacity to pay but chooses not to). Research by Indian credit bureaus suggests that willful default is more prevalent among higher-income borrowers and in business loans. Behavioral signals that help distinguish involuntary from strategic default include: communication responsiveness, partial payment patterns, and asset concealment behavior.

**Financial management capability**: Borrowers with poor financial management — characterized by irregular saving patterns, lack of emergency funds, multiple overlapping EMI obligations, and reliance on revolving credit for daily expenses — exhibit systematically higher default rates regardless of their income level.

**Commitment and engagement**: The degree of borrower engagement with the lending relationship provides behavioral signals. Borrowers who respond promptly to communication, maintain accurate contact information, and proactively communicate about business changes exhibit lower default rates.

## 3.5 Macroeconomic Risk (Indian Context)

Macroeconomic conditions play a fundamental role in credit risk dynamics. The Reserve Bank of India's monetary policy — particularly the repo rate (currently at 6.5% as of early 2026) — directly influences borrowing costs and borrower repayment capacity. When the RBI raises rates, floating-rate borrowers face increased EMI burdens, with every 25 basis point increase translating to approximately 1.5-2% increase in EMI for a typical home loan. The cumulative impact of the 250 basis points of rate increases between May 2022 and February 2023 was a significant increase in EMI-to-income ratios for existing borrowers.

GDP growth momentum is a critical macro driver: India's GDP growth has been relatively robust at 6.5-7.5% in recent years, but the distribution matters as much as the headline number. The inflation environment (CPI hovering around 5-6%) erodes real income for fixed-income borrowers and increases working capital requirements for businesses.

The external environment — global interest rates (US Fed funds rate), crude oil prices (India imports ~85% of its crude oil requirements), and global trade dynamics — creates second-order effects on Indian credit risk. The rupee-dollar exchange rate (currently around ₹84-85/USD) impacts corporates with foreign currency borrowings. The monsoon remains critical for India's agricultural sector and, by extension, for rural credit — a deficient monsoon can trigger cascading defaults across the agricultural value chain.

## 3.6 Industry Risk

Industry risk refers to the systematic risk inherent in specific industries that affects all borrowers within that sector. The real estate sector carries inherently high industry risk due to its capital-intensive nature, long project cycles (typically 3-5 years), regulatory complexity (RERA, GST, environmental clearances), and sensitivity to interest rates. The infrastructure sector (roads, power, telecommunications) is exposed to regulatory risk, political risk, and construction/execution risk. The power sector specifically has been a major source of Indian bank NPAs, with stressed assets in thermal power exceeding ₹1.5 lakh crore.

## 3.7 Sectoral Risk

**Real Estate**: The Indian real estate sector is highly cyclical and geographically fragmented. Residential real estate risk is driven by absorption rates, inventory overhang (6 months in Hyderabad to 30+ months in some NCR micro-markets), and developer credibility.

**MSME**: India's 63 million MSMEs account for ~30% of GDP and ~45% of manufacturing output. MSME risk is driven by customer concentration, working capital management, raw material price exposure, and access to formal credit.

**Agriculture**: Agricultural credit in India is shaped by monsoon dependence, crop price volatility, input cost inflation, and farmer indebtedness. The Kisan Credit Card (KCC) system has improved credit access but also created moral hazard issues.

**Healthcare**: Medical equipment finance carries industry risk related to regulatory changes, technology obsolescence, and patient volume sensitivity to pandemic events.

**Textiles and Garments**: This sector faces raw material price volatility (cotton prices), global competition, and environmental compliance costs, with concentration in specific clusters (Surat, Tirupur, Ludhiana, Karur).

## 3.8 Income Instability

Income instability is one of the most powerful predictors of credit default, yet it is poorly captured by traditional credit models. For salaried employees, income instability includes salary delays, variable pay components, and job changes. For MSME owners, business revenue can fluctuate by 30-50% month-to-month. Bank statement analysis provides a powerful mechanism for measuring income instability through metrics like: coefficient of variation of monthly credits, frequency of months with below-average credits, and the regularity of credit timing.

## 3.9 Cash Flow Deterioration

Cash flow deterioration is the proximate cause of most credit defaults. In the Indian context, the typical trajectory begins with reduced revenue (visible as declining credit amounts), followed by drawing down savings (declining balances), increased reliance on credit facilities, and finally missed payments. For business borrowers, the cash conversion cycle (CCC) is a critical metric — Indian businesses often face extended CCCs due to delayed payments from customers (large corporates routinely pay MSME suppliers in 90-120 days).

## 3.10 Fraud Risk

Fraud risk in Indian credit markets includes identity fraud, income fraud, asset fraud, business revenue fraud, and straw man schemes. RBI data indicates that fraud in Indian banking amounted to ₹30,274 crore in FY2024. The rise of fintech lending has created new fraud vectors: synthetic identity fraud, velocity fraud, and document tampering using digital tools.

## 3.11 Credit Utilization

Credit utilization is one of the most powerful predictors of credit default in India. CIBIL data shows that borrowers with credit utilization above 70% are 4x more likely to default than those below 30%. In the Indian context, 'credit card surfing' can mask true utilization levels, and BNPL products that are not yet fully reported to credit bureaus mean visible credit utilization may understate true utilization by 20-40%.

## 3.12 Borrower Psychology

**Present bias**: Borrowers overweight immediate needs relative to future consequences. **Loss aversion**: Borrowers who perceive they have already 'lost' may rationally choose to default. **Herd mentality**: Social dynamics can amplify or dampen default behavior. **Trust and relationship effects**: Strong relationships with lending institutions can reduce default rates through social accountability mechanisms.

## 3.13 Portfolio Concentration

Portfolio concentration risk arises when a lender's loan book is excessively concentrated in specific sectors, geographies, or product categories. Public sector banks have legacy concentrations in infrastructure, steel, and mining. Private banks have concentrations in real estate, credit cards, and personal loans. Geographic concentration creates correlated risks — lenders with heavy exposure to specific states face vulnerability to regional economic stress.

## 3.14 Regional Factors

State-level institutional factors — court efficiency, land registration quality, governance, and informal lending prevalence — all influence default patterns. Urban vs. rural dynamics create fundamentally different risk profiles: urban borrowers have more formal employment and access to multiple credit channels, while rural borrowers have more volatile income but stronger community accountability mechanisms. Default rates can vary by 3-5x between the highest and lowest default districts within the same state.

## 3.15 Seasonality

The Indian agricultural calendar creates distinct seasonal patterns: kharif stress peaks in August-October before harvest, rabi stress peaks in February-March. Festival seasons create retail credit patterns: Diwali spending spikes in October-November, followed by elevated defaults in January-MMarch. The wedding season (November-February) drives gold loan and personal loan demand, with default rates peaking 6-9 months later. DrishtiAI incorporates explicit seasonality features to account for these predictable patterns.
---

# 4. Data Sources

## 4.1 Structured Data Sources

### 4.1.1 Loan History Data

**What specific data is captured**: Loan history includes the complete record of all loans ever taken by the borrower — both from the originating institution and (through bureau data) from all other lenders. This encompasses: loan type (home loan, auto loan, personal loan, gold loan, education loan, business loan, KCC, etc.), original loan amount, current outstanding balance, original tenure, remaining tenure, interest rate (fixed/floating), EMI amount, disbursement date, maturity date, and repayment schedule. For closed loans, it includes closure date, prepayment patterns, and final status (regular closure, premature closure, settled, written-off, or settled-for-less).

**How it contributes to prediction**: Loan history provides the foundational risk profile of a borrower. The number of simultaneous active loans indicates leverage level. The mix of secured vs. unsecured loans indicates risk profile. Loan vintage is a strong predictor — very new borrowers (<6 months credit history) have 2-3x higher default rates. Prepayment behavior indicates financial health. Conversely, borrowers with a history of loan settlements or write-offs are extremely high risk.

**Feature engineering possibilities**: Total credit exposure (sum of all outstanding balances), credit mix ratio (secured/unsecured split), loan count by type, average loan age, credit tenure (time since first loan), prepayment rate, loan stacking velocity (number of new loans in past 3/6/12 months), weighted average interest rate, tenure mismatch, and leverage trend.

**Data quality considerations**: Cross-bureau data integration can be challenging due to different reporting standards and timing. Some loans may not be reported to all four bureaus. The lag in bureau reporting (typically 15-30 days) means that very recent changes may not be reflected.

**Privacy/regulatory considerations**: Bureau data usage must comply with RBI's Fair Practices Code and the Credit Information Companies (Regulation) Act. Borrower consent is required. DPDPA 2023 provisions apply.

### 4.1.2 Repayment History Data

**What specific data is captured**: Repayment history provides month-by-month records of the borrower's repayment behavior on each active and closed loan. For each payment period, it captures: whether the payment was made on time, the number of days past due (DPD) if delayed (30, 60, 90, 120, 150, 180+ DPD), the amount paid (full EMI, partial payment, or zero), and the date of payment. CIBIL reports repayment history for 36 months, while Experian and CRIF High Mark report for up to 60 months.

**How it contributes to prediction**: Repayment history is the single most predictive structured feature for default prediction. The pattern of repayment — not just the current status — carries critical information. The severity trajectory (improving, stable, or deteriorating) is highly predictive. Research across Indian credit bureaus consistently shows that recent repayment behavior (past 6 months) is 3-5x more predictive than older behavior, and that the transition probabilities between DPD states are among the most powerful predictive features.

**Feature engineering possibilities**: Maximum DPD in last 3/6/12/24/36 months, number of accounts with 30+ DPD in last 6/12 months, weighted average DPD across all accounts, repayment consistency score, DPD transition matrix features, recent deterioration flag, and best-ever vs. current DPD comparison.

**Data quality considerations**: Repayment history coverage varies by bureau (CIBIL: 36 months, Experian: 60 months, CRIF: 60 months). The lag in reporting means that very recent payments may not appear.

**Privacy/regulatory considerations**: Same bureau data regulations apply. Negative information can be reported for up to 7 years from the date of default.

### 4.1.3 EMI Delay Patterns

**What specific data is captured**: EMI delay patterns provide granular detail about the timing and magnitude of payment delays beyond the simple DPD classification. This includes: exact number of days of delay for each EMI, pattern of delays (always late by the same number of days, progressively worsening, or random), partial payment amounts and timing, bounce reasons (insufficient funds, account closed, stop payment, technical issues), and bounce frequency.

**How it contributes to prediction**: EMI delay patterns reveal behavioral tendencies that simple DPD classification misses. A borrower who consistently pays 5 days late may be experiencing a structural cash flow timing issue, while a borrower who suddenly shifts from on-time to 10 days late is experiencing genuine deterioration. The pattern of bounce reasons is also informative: 'insufficient funds' bounces indicate cash flow stress, while 'technical issues' bounces indicate operational problems.

**Feature engineering possibilities**: Average days of EMI delay, EMI delay standard deviation, bounce-to-payment ratio, consecutive bounce count, bounce reason distribution, partial payment ratio, and EMI payment predictability score.

### 4.1.4 Account Balance Data

**What specific data is captured**: Account balance data includes daily, weekly, or monthly snapshots of the borrower's account balances across all accounts held with the reporting bank. This encompasses opening balance, closing balance, minimum balance, maximum balance, and average balance for each period. For current accounts (business borrowers), it includes overdraft utilization, credit limit, and turnover.

**How it contributes to prediction**: Account balances provide a real-time window into the borrower's financial health. A declining average balance trend is one of the strongest leading indicators of default, typically preceding missed payments by 2-6 months. The minimum balance metric is particularly powerful: borrowers whose minimum monthly balance is approaching zero are in acute cash flow stress.

**Feature engineering possibilities**: Average balance trend (3/6/12-month slope), balance volatility (coefficient of variation), minimum balance frequency, balance utilization rate, net cash flow, balance coverage ratio (average balance / monthly EMI obligation), and days with negative or near-zero balance.

### 4.1.5 Transaction Data

**What specific data is captured**: Transaction data provides a detailed record of all debits and credits to the borrower's accounts, including: transaction date, amount, type (NEFT, RTGS, UPI, IMPS, cheque, cash deposit/withdrawal, card transaction, EMI auto-debit), counterparty information, narration/description, and channel (branch, online, mobile).

**How it contributes to prediction**: Transaction data is the richest structured data source for understanding borrower behavior. The pattern, frequency, and composition of transactions reveal income stability, spending behavior, financial management quality, and business health. Key predictive signals include: salary credit regularity, income-to-expense ratio, cash withdrawal patterns, and transaction counterparties.

**Feature engineering possibilities**: Monthly income, monthly expenses, savings rate, cash withdrawal ratio, transaction frequency trend, unique counterparty count, EMI auto-debit success rate, income source concentration, spending category breakdown, and net transfer balance.

**Data quality considerations**: Transaction data quality varies significantly by channel — UPI and card transactions have rich metadata, while cash transactions have minimal information. Multi-bank transaction visibility is limited without Account Aggregator (AA) framework integration.

### 4.1.6 Credit Bureau Data

**What specific data is captured**: Indian credit bureau data (from CIBIL, Experian, CRIF High Mark, and Equifax) provides a comprehensive credit profile including: all active and closed loan/credit card accounts, repayment history (36-60 months), credit utilization, credit inquiries (hard and soft), public records (court judgments, insolvency proceedings, SARFAESI actions), and the bureau-generated credit score (300-900 for CIBIL).

**How it contributes to prediction**: Credit bureau data is the most widely used data source for credit risk assessment in India. The credit score aggregates multiple dimensions into a single number, but the detailed components — individual account histories, inquiry patterns, and public records — carry significantly more predictive information than the score alone.

**Feature engineering possibilities**: Credit score and trajectory, total credit limit vs. total utilization, account age statistics, inquiry frequency, inquiry recency, public record flags, account status distribution, and bureau score volatility.

### 4.1.7 Collateral Data

**What specific data is captured**: For secured loans, collateral data includes: property/asset description, location details, valuation amount (at origination and current), type of collateral (residential property, commercial property, land, gold, machinery, inventory, receivables, securities), encumbrance status, insurance coverage details, and depreciation schedule.

**How it contributes to prediction**: Collateral quality directly impacts loss given default (LGD). A well-collateralized loan with a low LTV ratio has lower risk. Collateral value trends are also important — declining collateral values increase risk by eroding the lender's security cover.

**Feature engineering possibilities**: LTV ratio, LTV trend, collateral coverage ratio, collateral type categorization, collateral age and depreciation, insurance adequacy, and encumbrance ratio.

### 4.1.8 Interest Rate Data

**What specific data is captured**: The applicable interest rate for each loan (fixed or floating), the benchmark rate (repo rate, MCLR, EBLR), the spread/margin over the benchmark, rate reset frequency, and historical rate changes.

**How it contributes to prediction**: Interest rate changes directly impact borrower EMI burden. For floating-rate borrowers (~80% of Indian home loans), rate increases translate to higher EMIs. The sensitivity analysis (how much additional EMI burden from a 100bp or 200bp rate increase) provides forward-looking risk assessment.

**Feature engineering possibilities**: Interest rate sensitivity, rate gap (borrower's rate vs. current market rate), rate type, rate vintage, and cumulative rate increase impact.

### 4.1.9 Credit Score Data

**What specific data is captured**: Indian credit bureau scores: CIBIL Score (300-900), Experian Credit Score (250-900), CRIF High Mark Score (300-850), and Equifax Credit Score (1-999). Each considers: repayment history (30-35% weight), credit utilization (25-30%), credit history length (10-15%), credit mix (10%), and recent credit activity (10%).

**How it contributes to prediction**: Score trajectory is more informative than absolute score: a borrower whose CIBIL score has declined from 750 to 680 over 6 months is significantly riskier than one stable at 680. Score volatility may indicate unstable credit behavior. The score's predictive power is strongest in the mid-range (600-750).

**Feature engineering possibilities**: Bureau score by bureau, score rank, score trajectory, score volatility, score range utilization, and cross-bureau score discrepancy.

### 4.1.10 Loan Utilization Data

**What specific data is captured**: For working capital facilities (cash credit, overdraft), utilization varies daily: current utilization (amount drawn), available limit, peak utilization, average utilization, and utilization frequency.

**How it contributes to prediction**: For MSME borrowers, utilization patterns are highly predictive. High and increasing utilization indicates cash flow stress. Very low or declining utilization may indicate business contraction. Optimal utilization (40-70%) indicates normal business activity with adequate buffer.

**Feature engineering possibilities**: Current utilization ratio, utilization trend, utilization volatility, days at high utilization, utilization seasonality pattern, peak-to-average utilization ratio, and utilization change velocity.

### 4.1.11 Borrower Demographic Data

**What specific data is captured**: Age, gender, marital status, number of dependents, education level, occupation category, employer name and type, years in current employment/business, residential status, and city/state/PIN code.

**How it contributes to prediction**: Age has a U-shaped relationship with default risk. Occupation category significantly influences risk — salaried employees of large organizations typically have lower default rates than self-employed individuals. Income-to-EMI ratio is fundamental to repayment capacity.

**Feature engineering possibilities**: Age-income interaction, occupation risk category, employer stability score, income-to-EMI ratio, income-to-loan ratio, residential stability score, and urbanization index.

### 4.1.12 Business Turnover Data

**What specific data is captured**: Annual revenue/sales figures (from financial statements, GST returns, or bank statement analysis), quarterly/monthly revenue trends, revenue by segment, customer concentration, and geographic distribution.

**How it contributes to prediction**: Business turnover is the fundamental determinant of repayment capacity for business borrowers. Declining turnover is one of the strongest predictors. The turnover trajectory, seasonality pattern, and comparison to industry norms are more informative than the raw figure.

**Feature engineering possibilities**: Revenue growth rate, revenue concentration index, revenue volatility, revenue-to-debt ratio, and revenue seasonality index.

### 4.1.13 GST Data

**What specific data is captured**: GST registration details, monthly/quarterly GST returns (GSTR-1, GSTR-3B), invoice-level transaction data, input tax credit (ITC) claims, tax payment history, compliance status, and registration cancellations.

**How it contributes to prediction**: GST data provides verified, government-stamped revenue information for MSMEs. Monthly GST filing provides a granular, near-real-time view of business activity far more reliable than self-reported financial statements. Key signals: revenue growth trajectory, buyer diversity, ITC claiming pattern, and filing regularity.

**Feature engineering possibilities**: Monthly GST turnover trend, GST revenue growth rate, buyer concentration index, ITC utilization ratio, filing regularity score, GST compliance rating, GST revenue vs. bank statement revenue comparison, and interstate vs. intrastake sales ratio.

### 4.1.14 Tax Records

**What specific data is captured**: Income tax return (ITR) data including: reported income (salary, business income, capital gains, rental income), deductions claimed, tax paid (TDS, advance tax, self-assessment tax), return filing history, and assessment details. For businesses, ITR includes P&L and Balance Sheet.

**How it contributes to prediction**: ITR data provides the most comprehensive view of a borrower's total financial picture. Key signals: income trend, income consistency, tax compliance, and income-source diversification.

**Feature engineering possibilities**: Income growth rate, income volatility, tax compliance score, business profit margin, asset-to-debt ratio, working capital position, and net worth trend.

### 4.1.15 Investment Data

**What specific data is captured**: Mutual fund holdings, demat account holdings, fixed deposits, PPF/NPS/EPF balances, accessible through Account Aggregator framework.

**How it contributes to prediction**: Investment data provides insight into financial health, risk appetite, and liquidity buffer. Diversified investment portfolios indicate greater financial resilience. Investment redemption patterns can indicate financial stress.

**Feature engineering possibilities**: Total investment value, investment-to-debt ratio, investment diversification index, liquid investment ratio, investment redemption velocity, and equity exposure percentage.

### 4.1.16 Savings and Insurance Data

**What specific data is captured**: Savings account balances, recurring deposits, life insurance policies (sum assured, premium, cash value), health insurance coverage, and general insurance.

**How it contributes to prediction**: Insurance coverage is an important risk mitigant — borrowers with adequate insurance are better protected against income shocks. Savings levels indicate emergency fund adequacy.

**Feature engineering possibilities**: Savings-to-EMI ratio, insurance coverage adequacy, premium burden, insurance lapse flags, and savings trend.

### 4.1.17 Existing Liabilities Data

**What specific data is captured**: All active loans, credit card outstanding, BNPL obligations, guarantee obligations, contingent liabilities, and tax arrears.

**How it contributes to prediction**: Total liabilities determine debt burden and repayment capacity constraints. The DTI ratio is the most critical affordability metric. Off-balance-sheet liabilities (guarantees, contingencies) represent hidden risk.

**Feature engineering possibilities**: Total debt outstanding, DTI ratio, DTI trend, debt service coverage ratio, contingent liability exposure, and near-prime debt share.

### 4.1.18 Macroeconomic Indicators

**What specific data is captured**: RBI policy rates, bank lending rates, inflation data (CPI, WPI), GDP growth, industrial production index (IIP), PMI, employment indicators, trade data, commodity prices, forex rates, fiscal deficit data, and monsoon/rainfall data.

**How it contributes to prediction**: Macroeconomic indicators provide systemic context. RBI rate changes directly impact floating-rate borrowers. Inflation erodes real income. Sector-specific indicators provide sector-level risk context. Monsoon data is a leading indicator for agricultural credit risk.

**Feature engineering possibilities**: Real interest rate, rate change momentum, inflation trajectory, GDP growth momentum, PMI trend, commodity price exposure index, exchange rate volatility, and monsoon deviation.
## 4.2 Unstructured Data Sources

### 4.2.1 Loan Applications (Free-text Fields)

**What specific data is captured**: Loan application forms contain free-text narrative sections including: purpose of loan description, business description (for MSME loans), asset description, additional income sources, explanation of negative credit events, and additional remarks by the applicant.

**How it contributes to prediction**: Free-text narratives provide qualitative information not captured in structured fields. The language complexity, specificity, and consistency of the narrative can be analyzed using NLP techniques. Inconsistencies between the narrative and structured data may indicate misrepresentation. Emotional tone can provide behavioral signals.

**Feature engineering possibilities**: Narrative length and complexity score, consistency score (narrative vs. structured data), entity extraction, sentiment polarity and subjectivity, specificity score, and topic classification.

**Data quality considerations**: Free-text quality varies enormously. OCR quality affects downstream NLP processing for handwritten or scanned applications. Multi-language content requires multilingual NLP capabilities.

### 4.2.2 Bank Statements (OCR and PDF Parsing)

**What specific data is captured**: Bank statements (typically PDF or scanned images) contain transaction details (date, description, debit, credit, balance), account header information, summary information, and sometimes additional information (EMI bounce details, charges). The document format varies across 100+ different bank formats in India.

**How it contributes to prediction**: Bank statements provide the most granular and verified view of borrower cash flow dynamics. Beyond transaction data, the document itself carries signals: the bank (private vs. PSU vs. cooperative) indicates banking relationship quality, and anomalies may indicate document tampering.

**Feature engineering possibilities**: Transaction-level features (from parsed data), document metadata, document consistency score, anomaly indicators, and cash flow stability metrics.

**Data quality considerations**: OCR accuracy varies by document quality — clean digital PDFs achieve >99% accuracy, while scanned documents may have 85-95% accuracy. Indian bank statement formats are notoriously diverse (100+ formats), requiring format-specific parsers.

### 4.2.3 Financial Statements (Balance Sheet, P&L)

**What specific data is captured**: Balance Sheet (assets, liabilities, equity), Profit & Loss statement (revenue, costs, profit), Cash Flow Statement, and Notes to Accounts (accounting policies, contingent liabilities, related party transactions).

**How it contributes to prediction**: Financial statements provide the most comprehensive view of business borrower financial health. Key analytical dimensions: profitability trends, liquidity ratios, leverage ratios, working capital cycle, and cash flow analysis. NLP analysis of notes can extract contingent liabilities and related party transactions.

**Feature engineering possibilities**: 30+ traditional financial ratios, Altman Z-score adaptations, trend features, peer comparison features, and text-based features from notes.

### 4.2.4 Auditor Notes and Reports

**What specific data is captured**: Auditor's opinion (unqualified, qualified, adverse, disclaimer), management discussion, going concern assessments, emphasis of matter paragraphs, key audit matters, internal control observations, and related party transaction disclosures.

**How it contributes to prediction**: Auditor reports are among the most information-dense unstructured documents. An 'emphasis of matter' regarding going concern carries far more predictive information than many structured features. The trend in audit opinions is a powerful leading indicator.

**Feature engineering possibilities**: Audit opinion classification, going concern flag, emphasis of matter severity, key audit matters topic classification, and opinion trend.

### 4.2.5 Relationship Manager Notes

**What specific data is captured**: Meeting notes from borrower interactions, observations about business and management, informal intelligence about reputation, updates on business developments, collection interaction notes, and risk assessment narratives.

**How it contributes to prediction**: RM notes represent a rich source of qualitative, relationship-based intelligence not available through any external data source. The RM's assessment of management quality, business trajectory, and borrower integrity provides information orthogonal to quantitative financial data.

**Feature engineering possibilities**: RM sentiment score, RM risk mention frequency, RM update recency, RM concern count, RM confidence indicator, and RM observation consistency.

### 4.2.6 Customer Emails and Correspondence

**What specific data is captured**: Emails to/from the bank, written complaints, service requests, loan-related queries, dispute communications, and any other written interaction between borrower and institution.

**How it contributes to prediction**: Customer communication patterns provide behavioral and contextual signals. A sudden increase in complaint frequency may indicate distress. Communication about financial difficulties provides direct insight into the borrower's financial situation.

**Feature engineering possibilities**: Communication frequency trend, sentiment trajectory, complaint escalation count, financial difficulty mentions, and communication recency.

### 4.2.7 Call Center Transcripts and Chat Conversations

**What specific data is captured**: Complete text of interactions between borrowers and customer service or collections team, including purpose of call, borrower's stated situation, promises to pay, disputes raised, and financial information disclosed.

**How it contributes to prediction**: Call center interactions are among the most revealing sources of behavioral signals. Patterns in call behavior — avoiding calls, providing changing excuses, or becoming hostile — provide behavioral risk signals. NLP analysis can extract promise-to-pay compliance, cooperation level, and stress indicators.

**Feature engineering possibilities**: Promise-to-pay compliance rate, call avoidance score, sentiment in collection calls, cooperation level score, stress language indicator, and call duration trend.

### 4.2.8 KYC Documents

**What specific data is captured**: Aadhaar card, PAN card, passport, voter ID, driving license, utility bills, bank statements, photographs, and for businesses: MSME registration, GST certificate, partnership deeds, and company incorporation documents.

**How it contributes to prediction**: KYC documents serve identity verification (fraud prevention), address verification (geographic risk assessment), consistency checking, and document authenticity assessment. The quality and completeness of KYC documentation provides a behavioral signal.

**Feature engineering possibilities**: KYC completeness score, document consistency cross-check score, address stability, identity verification confidence score, and KYC red flags.

### 4.2.9 Legal Notices and Court Documents

**What specific data is captured**: Demand notices under SARFAESI Act, legal notice letters, DRT filings, court orders, IBC proceedings, arbitration proceedings, and legal opinions.

**How it contributes to prediction**: Legal proceedings provide definitive information about credit status and recovery efforts. IBC proceedings indicate severe financial distress. NLP analysis can extract dispute nature, amounts involved, and potential outcomes.

**Feature engineering possibilities**: Legal proceeding count, proceeding type classification, IBC proceeding flag, litigation stage, and estimated recovery timeline.

### 4.2.10 News Articles and Media Coverage

**What specific data is captured**: Articles about the borrower from print and online media, industry news, regulatory news, economic news, and social media mentions from sources like Economic Times, Business Standard, Mint, and Dainik Bhaskar.

**How it contributes to prediction**: News data provides real-time, forward-looking information. A news article about a company losing a major contract provides early warning that may precede financial deterioration by months. Industry-wide news provides sector risk context.

**Feature engineering possibilities**: News sentiment score, news volume trend, negative news count, industry news sentiment, and regulatory risk score.

### 4.2.11 Company Annual Reports and Filings

**What specific data is captured**: Chairman/MD letter, management discussion and analysis (MD&A), detailed financial statements with notes, corporate governance report, director's report, and auditor's report.

**How it contributes to prediction**: NLP analysis can extract management tone and sentiment (which correlates with future performance), risk factors disclosed by management, forward-looking statements, related-party transactions, and corporate governance quality.

**Feature engineering possibilities**: MD&A sentiment score, risk factor count and severity, forward-looking statement analysis, related-party transaction flags, and governance quality score.

### 4.2.12 Court Filings and Legal Databases

**What specific data is captured**: Case filings, court orders, judgment text, case status updates, and party details from DRTs, NCLT, high courts, and district courts (accessible through eCourts, Indian Kanoon, SCC Online).

**How it contributes to prediction**: Court filings provide real-time information about legal proceedings. Cross-referencing court filings across borrowers can reveal connected litigation indicating network risk.

**Feature engineering possibilities**: Court case count by type, case stage classification, judgment analysis, NCLT/IBC case flag, and connected litigation network features.

### 4.2.13 PDF Documents and Embedded Data

**What specific data is captured**: Tables, charts, forms, annotations, metadata (author, creation date, modification history, digital signatures) from various PDF documents including valuation reports, technical reports, and legal opinions.

**How it contributes to prediction**: PDF document processing is an enabling capability. Advanced PDF processing can detect document manipulation, extract information from complex multi-page documents, and handle the enormous variety of formats.

**Feature engineering possibilities**: Document metadata features, table extraction quality scores, embedded image analysis, digital signature verification status, and document complexity metrics.

### 4.2.14 Images and Visual Data

**What specific data is captured**: Property photographs, site visit photographs, vehicle photographs, business premise photographs, satellite/aerial imagery, and document photographs.

**How it contributes to prediction**: Computer vision techniques can detect property condition deterioration, verify property existence, assess neighborhood quality, and detect document tampering. Satellite imagery can verify agricultural land use and crop health.

**Feature engineering possibilities**: Property condition score, location quality index, crop health score (NDVI), construction progress assessment, and document authenticity score.

### 4.2.15 Voice Transcripts (Call Recordings)

**What specific data is captured**: Automated speech recognition output from call recordings across customer service, collection, televerification, and branch interaction touchpoints.

**How it contributes to prediction**: Speech pattern analysis can detect stress indicators, evasiveness, cooperation level, and emotional state — all of which correlate with financial distress or intentionality of default.

**Feature engineering possibilities**: Speech pattern stress indicators, cooperation level score, promise-to-pay confidence assessment, emotional state classification, and call outcome classification.

### 4.2.16 Social Sentiment Data

**What specific data is captured**: Social media posts mentioning the borrower from Twitter (X), LinkedIn, Facebook, and industry forums; sentiment polarity and intensity; engagement metrics; and viral content.

**How it contributes to prediction**: Social sentiment provides real-time, crowd-sourced assessment of borrower reputation. Sudden spikes in negative sentiment may indicate emerging issues before they appear in financial data. Industry-wide sentiment trends provide sector risk context.

**Feature engineering possibilities**: Social sentiment score, sentiment trend, negative sentiment spike detection, social media engagement rate, and industry sentiment index.

### 4.2.17 Regulatory Filings and Government Data

**What specific data is captured**: MCA filings, SEBI filings (insider trading, bulk deals, share pledging), RBI regulatory actions, environmental clearances, FSSAI registrations, drug licenses, and telecom licenses.

**How it contributes to prediction**: Director resignations may signal internal issues. High promoter pledge levels (>50%) are associated with higher default risk. Environmental regulatory actions can impact business operations.

**Feature engineering possibilities**: Director change frequency, promoter pledge ratio, regulatory action count, license validity status, and compliance filing regularity.

### 4.2.18 Satellite and Geospatial Data

**What specific data is captured**: Optical satellite imagery, radar imagery, vegetation indices (NDVI), night light intensity, road connectivity indices, flood/seismic risk zone data, and urbanization patterns.

**How it contributes to prediction**: For agricultural loans, satellite-derived crop health indices can predict harvest yields before the harvest occurs. For real estate collateral, satellite imagery can verify property existence, assess neighborhood quality, and monitor construction progress. Night light intensity correlates with local economic activity.

**Feature engineering possibilities**: NDVI-based crop health score, property location quality index, economic activity index (night light data), flood risk zone classification, and infrastructure quality index.
---

# 5. Feature Engineering

## 5.1 Comprehensive Feature Catalog

### Category 1: Behavioral Features (20 features)

| # | Feature Name | Formula/Logic | Importance Rationale |
|---|---|---|---|
| 1 | repayment_consistency_score | Count of on-time payments / Total payments across all accounts (last 24 months) | Core behavioral measure of repayment discipline; high consistency strongly predicts continued good behavior |
| 2 | emi_bounce_frequency | Number of EMI bounces in last 6 months / Total EMIs due in same period | Direct measure of cash flow stress; bounces are early default signals |
| 3 | consecutive_bounce_count | Maximum consecutive EMI bounces across all accounts | Non-random bounces indicate structural cash flow problems |
| 4 | payment_timing_variance | Standard deviation of (payment_date - due_date) across all payments | High variance indicates unpredictable cash flow patterns |
| 5 | partial_payment_ratio | Total partial payments / Total EMI obligations (last 12 months) | Partial payments indicate maximum effort to meet obligations despite stress |
| 6 | communication_responsiveness | (Calls answered + emails responded to) / (Total calls + emails from bank) | Borrower engagement is inversely correlated with default risk |
| 7 | document_submission_timeliness | Average days between request and submission of required documents | Demonstrates borrower organizational capability and cooperation |
| 8 | address_stability | Years at current address (from KYC data) | Residential stability correlates with financial stability |
| 9 | employment_tenure | Years in current employment/business | Longer tenure indicates income stability |
| 10 | account_tenure | Days since first account with the reporting bank | Longer relationships indicate loyalty and stability |
| 11 | savings_discipline | Months with positive net savings in last 12 months / 12 | Regular saving behavior indicates financial management capability |
| 12 | overdraft_utilization_pattern | Average of (daily_overdraft_balance / overdraft_limit) over last 90 days | Consistently high utilization indicates chronic cash flow shortfall |
| 13 | cash_withdrawal_ratio | Total cash withdrawals / Total debits (last 6 months) | High cash withdrawal ratio may indicate informal economy activity |
| 14 | spending_escalation_rate | Slope of monthly spending over last 6 months | Increasing spending without income increase signals risk |
| 15 | income_regularity_index | 1 - (Coefficient of variation of monthly income credits) | Regular, predictable income is a cornerstone of repayment capacity |
| 16 | guarantor_distress_flag | Binary: 1 if any guarantor/co-borrower has DPD > 30 in last 6 months | Guarantor distress may indicate shared financial stress |
| 17 | loan_purpose_consistency | Cosine similarity between stated loan purpose and actual utilization pattern | Inconsistency may indicate misrepresentation |
| 18 | banking_channel_usage_trend | Change in digital vs. branch transaction ratio over last 6 months | Shift to branch banking may indicate need for cash handling |
| 19 | seasonal_repayment_pattern | Repayment performance in borrower's historically weakest month / best month | Identifies vulnerability to seasonal stress |
| 20 | early_repayment_indicator | Binary: 1 if borrower has made any prepayments in last 12 months | Prepayment demonstrates excess cash flow and proactive financial management |

### Category 2: Time-Series Features (15 features)

| # | Feature Name | Formula/Logic | Importance Rationale |
|---|---|---|---|
| 21 | credit_score_3m_change | CIBIL_score(t) - CIBIL_score(t-3 months) | Score trajectory captures recent behavioral changes |
| 22 | credit_score_6m_change | CIBIL_score(t) - CIBIL_score(t-6 months) | Medium-term trend provides stability signal |
| 23 | credit_score_12m_change | CIBIL_score(t) - CIBIL_score(t-12 months) | Long-term trend captures structural shifts |
| 24 | income_growth_rate_6m | (Avg_income_last_3m - Avg_income_prior_3m) / Avg_income_prior_3m | Income trajectory is a leading indicator of repayment capacity |
| 25 | expense_growth_rate_6m | (Avg_expense_last_3m - Avg_expense_prior_3m) / Avg_expense_prior_3m | Rising expenses without income growth signals risk |
| 26 | balance_trend_slope | Linear regression slope of average monthly balance over last 12 months | Negative slope indicates progressive cash flow deterioration |
| 27 | utilization_trend_slope | Linear regression slope of credit utilization over last 6 months | Rising utilization trend is a strong early warning signal |
| 28 | debt_acceleration | (Total_debt_now - Total_debt_6m_ago) - (Total_debt_6m_ago - Total_debt_12m_ago) | Accelerating debt accumulation is more dangerous than steady growth |
| 29 | inquiry_velocity_change | (Inquiries_last_3m / 3) - (Inquiries_prior_3m / 3) | Increasing inquiry velocity signals urgent credit seeking |
| 30 | emi_burden_trend | Total_monthly_EMI_now / Total_monthly_EMI_6m_ago | Rising EMI burden indicates increasing leverage |
| 31 | interest_rate_sensitivity_delta | (EMI_at_current_rate - EMI_at_origination_rate) / EMI_at_origination_rate | Impact of rate changes on borrower's EMI burden |
| 32 | revenue_volatility_12m | Coefficient of variation of monthly business revenue (last 12 months) | High revenue volatility increases default probability |
| 33 | payment_amount_trend | Ratio of average payment amount now vs. 6 months ago | Declining payment amounts may indicate distress |
| 34 | overdraft_cycle_length | Average consecutive days with overdraft utilization > 50% | Prolonged high utilization indicates structural cash flow issues |
| 35 | account_dormancy_period | Days since last non-EMI transaction on primary account | Account dormancy may indicate account switching |

### Category 3: Rolling Statistics (15 features)

| # | Feature Name | Formula/Logic | Importance Rationale |
|---|---|---|---|
| 36 | balance_ma_3m | 3-month moving average of average monthly balance | Smoothed balance trend reduces noise |
| 37 | balance_ma_6m | 6-month moving average of average monthly balance | Longer-term smoothed trend |
| 38 | balance_ma_12m | 12-month moving average of average monthly balance | Structural balance level |
| 39 | income_rolling_std_6m | 6-month rolling standard deviation of monthly income | Income volatility measure |
| 40 | expense_rolling_std_6m | 6-month rolling standard deviation of monthly expenses | Expense volatility measure |
| 41 | spending_rolling_avg_3m | 3-month rolling average of total monthly spending | Recent spending level |
| 42 | credit_utilization_rolling_max_3m | Maximum credit utilization in any month in last 3 months | Peak recent stress level |
| 43 | dpd_rolling_max_6m | Maximum DPD across all accounts in last 6 months | Worst recent repayment performance |
| 44 | inquiry_count_rolling_3m | Count of credit inquiries in last 3 months | Recent credit-seeking intensity |
| 45 | inquiry_count_rolling_6m | Count of credit inquiries in last 6 months | Medium-term credit-seeking pattern |
| 46 | new_account_count_rolling_12m | Number of new accounts opened in last 12 months | New credit acquisition rate |
| 47 | transaction_amount_rolling_std_3m | 3-month rolling standard deviation of total transaction amounts | Transaction volatility |
| 48 | emi_bounce_rolling_rate_6m | Bounces in last 6 months / EMIs due in last 6 months | Recent bounce trajectory |
| 49 | balance_min_rolling_3m | Minimum of monthly average balances in last 3 months | Worst recent balance level |
| 50 | income_max_drawdown_12m | Maximum peak-to-trough decline in monthly income over 12 months | Largest income disruption in recent history |
### Category 4: Credit Velocity Features (10 features)

| # | Feature Name | Formula/Logic | Importance Rationale |
|---|---|---|---|
| 51 | credit_limit_velocity_6m | (Total_credit_limit_now - Total_credit_limit_6m_ago) / Total_credit_limit_6m_ago | Rapid credit limit expansion may indicate aggressive lending or borrower demand |
| 52 | new_loan_velocity_6m | Number of new loans disbursed in last 6 months | Rapid loan acquisition is a distress signal |
| 53 | debt_growth_velocity | (Total_outstanding_now - Total_outstanding_6m_ago) / 6 | Monthly rate of debt accumulation |
| 54 | balance_drawdown_velocity | (Opening_balance_12m_ago - Current_balance) / Opening_balance_12m_ago | Rate at which borrower is drawing down savings |
| 55 | credit_card_balance_velocity | Month-over-month change in total credit card balances for last 3 months | Accelerating credit card balances indicate cash flow stress |
| 56 | emi_to_income_velocity | Change in DTI ratio over last 6 months | Rapidly increasing DTI is more alarming than high static DTI |
| 57 | overdraft_utilization_velocity | Rate of change in average overdraft utilization over last 3 months | Accelerating overdraft utilization signals worsening cash flow |
| 58 | inquiry_velocity_90d | Credit inquiries in last 90 days / 3 | Monthly rate of credit seeking |
| 59 | account_opening_velocity | New accounts opened per quarter (last 4 quarters trend) | Accelerating new account openings may indicate loan stacking |
| 60 | exposure_increase_rate | Total credit exposure increase in last 12 months / Total exposure 12 months ago | Annualized rate of exposure growth |

### Category 5: Cash Flow Stability Features (10 features)

| # | Feature Name | Formula/Logic | Importance Rationale |
|---|---|---|---|
| 61 | cash_flow_coverage_ratio | Average_monthly_credits / Average_monthly_EMI_obligations | Fundamental affordability metric |
| 62 | net_cash_flow_trend | Linear regression slope of (credits - debits) over last 6 months | Positive trend indicates improving cash flow health |
| 63 | cash_flow_volatility | Coefficient of variation of monthly net cash flow (last 12 months) | High volatility increases default risk even with positive average |
| 64 | minimum_cash_flow_percentile | Percentile of current month's net cash flow in borrower's 24-month history | How current cash flow compares to borrower's own history |
| 65 | cash_flow_deficit_frequency | Months with negative net cash flow in last 12 months / 12 | Frequency of cash flow shortfalls |
| 66 | operating_cash_flow_ratio | Cash_from_operations / Total_debt_service_obligations (for businesses) | Core business ability to service debt |
| 67 | working_capital_cycle_days | (Inventory_days + Receivable_days - Payable_days) | Length of cash conversion cycle |
| 68 | cash_resilience_score | Months of EMI obligations that can be covered from current liquid savings | Emergency buffer assessment |
| 69 | income_expense_gap_trend | Trend in (monthly_income - monthly_expenses) over last 6 months | Widening gap indicates increasing financial pressure |
| 70 | seasonal_cash_flow_deviation | Current month's cash flow / Average for same month in prior 2 years | Deviation from seasonal norm may indicate non-seasonal stress |

### Category 6: Debt Burden Features (10 features)

| # | Feature Name | Formula/Logic | Importance Rationale |
|---|---|---|---|
| 71 | dti_ratio | Total_monthly_EMI_obligations / Total_monthly_income | Most fundamental affordability metric |
| 72 | dti_after_new_loan | (Total_monthly_EMI + New_loan_EMI) / Total_monthly_income | Forward-looking affordability for new loan assessment |
| 73 | fixed_obligation_to_income | (EMIs + Rent + Insurance_premiums) / Monthly_income | Comprehensive fixed obligation assessment |
| 74 | debt_to_net_worth | Total_outstanding_debt / Total_net_worth | Leverage measure from balance sheet perspective |
| 75 | interest_burden_ratio | Total_annual_interest_payments / Annual_income | Proportion of income consumed by interest |
| 76 | unsecured_debt_ratio | Unsecured_debt / Total_debt | Higher unsecured debt proportion indicates higher risk |
| 77 | max_single_ema_exposure | Largest_single_EMI / Monthly_income | Concentration of EMI burden in single obligation |
| 78 | debt_service_coverage_ratio | Net_operating_income / Total_debt_service_obligations | Industry-standard coverage metric |
| 79 | leverage_trend | Change in debt-to-equity ratio over last 12 months | Increasing leverage indicates growing risk |
| 80 | near_prime_debt_share | High_interest_debt (> 15% interest) / Total_debt | Predatory lending exposure indicates distress |

### Category 7: Income Volatility Features (10 features)

| # | Feature Name | Formula/Logic | Importance Rationale |
|---|---|---|---|
| 81 | income_cv_12m | Coefficient of variation of monthly income (last 12 months) | Core income volatility metric |
| 82 | income_stability_index | 1 - (Std_dev_income / Mean_income) for last 12 months | Inverse volatility measure (higher = more stable) |
| 83 | income_zero_months | Months with zero or near-zero income credits in last 12 months | Complete income cessation months |
| 84 | income_source_count | Number of distinct income credit sources identified from bank statements | Diversification of income sources |
| 85 | salary_credit_regularity | Standard deviation of (salary_credit_date - first_of_month) for last 6 months | Regular salary credit timing indicates stable employment |
| 86 | income_trend_slope | Linear regression slope of monthly income over last 12 months | Directional trend in income |
| 87 | income_peak_to_trough | (Max_monthly_income - Min_monthly_income) / Average_monthly_income over last 12 months | Range of income fluctuation |
| 88 | income_recovery_speed | Months to recover to pre-decline income level after a decline event | How quickly income recovers from disruptions |
| 89 | self_employment_income_volatility | CV of monthly business income for self-employed borrowers | Business income volatility is typically higher than salary |
| 90 | income_seasonality_index | Ratio of income in borrower's weakest quarter to strongest quarter | Degree of seasonal income variation |

### Category 8: Spending Behavior Features (10 features)

| # | Feature Name | Formula/Logic | Importance Rationale |
|---|---|---|---|
| 91 | essential_spending_ratio | Spending on essentials (food, utilities, rent) / Total spending | Higher ratio indicates constrained discretionary spending |
| 92 | discretionary_cutback_flag | Binary: 1 if discretionary spending dropped > 30% in last 3 months | Cutbacks may indicate financial stress adaptation |
| 93 | lifestyle_inflation_rate | Growth rate of total spending vs. growth rate of income over last 12 months | Spending growing faster than income is unsustainable |
| 94 | digital_payment_adoption | Digital_transaction_count / Total_transaction_count | Higher digital adoption enables better tracking |
| 95 | gambling_or_risk_spending_flag | Binary: 1 if transactions with gambling/betting/lottery merchants detected | High-risk spending behavior increases default probability |
| 96 | travel_luxury_spend_ratio | Spending on travel/luxury/discretionary categories / Total spending | High ratio may indicate overextension |
| 97 | education_spending_consistency | Regularity of education-related payments | Consistent education spending indicates family stability |
| 98 | medical_expense_spike | Binary: 1 if medical expenses exceeded 3x monthly average in last 6 months | Medical emergencies are a major default trigger |
| 99 | emi_payment_timing_vs_salary | Days between salary credit and EMI auto-debit | Short gap indicates tight cash flow management |
| 100 | spending_entropy | Shannon entropy of spending across categories over last 6 months | Low entropy (concentrated spending) may indicate limited flexibility |

### Category 9: Business Health Features (10 features)

| # | Feature Name | Formula/Logic | Importance Rationale |
|---|---|---|---|
| 101 | gst_revenue_growth_yoy | (Current_year_GST_revenue - Prior_year_GST_revenue) / Prior_year_GST_revenue | Verified revenue growth from government data |
| 102 | gst_filing_regularity | Months_with_timely_filing / Total_months_in_period | Compliance behavior indicates business management quality |
| 103 | buyer_diversification_index | Herfindahl index of revenue by buyer (from GST data) | Customer concentration risk measure |
| 104 | itc_utilization_ratio | Input_tax_credit_claimed / Output_tax_liability | Low ratio may indicate business slowdown |
| 105 | business_revenue_bank_alignment | Correlation between GST_reported_revenue and bank_statement_credits | Inconsistency may indicate fraud or business complexity |
| 106 | business_turnover_trend | Slope of monthly bank statement turnover over 12 months | Business trajectory indicator |
| 107 | accounts_receivable_aging | Weighted average days to collect from customers | Longer collection times strain cash flow |
| 108 | profit_margin_trend | Trend in gross margin over last 4 quarters | Declining margins indicate competitive pressure |
| 109 | employee_count_trend | Change in EPFO subscriber count (for businesses with EPFO data) | Workforce changes indicate business trajectory |
| 110 | business_registration_age | Days since business GST/MSME registration | Established businesses have lower default rates |
