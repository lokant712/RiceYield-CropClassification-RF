"""
Generate 23MID0037_Lab08_Report.pdf using ReportLab.
Conforming to ~4-6 pages, professional typography, data governance, and empirical evidence.
"""

import os
import json
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, PageBreak, HRFlowable
)
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas to dynamically compute and render total page count and professional footer."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#555555"))
        
        # Header (Pages > 1)
        if self._pageNumber > 1:
            self.drawString(54, 11 * inch - 36, "MDI3003 Advanced Predictive Analytics — Experiment 08: Rice Yield Prediction")
            self.drawRightString(8.5 * inch - 54, 11 * inch - 36, "Lokanth S (23MID0037)")
            self.setStrokeColor(colors.HexColor("#d0d0d0"))
            self.setLineWidth(0.5)
            self.line(54, 11 * inch - 40, 8.5 * inch - 54, 11 * inch - 40)

        # Footer (All pages)
        self.setStrokeColor(colors.HexColor("#d0d0d0"))
        self.setLineWidth(0.5)
        self.line(54, 45, 8.5 * inch - 54, 45)
        self.drawString(54, 32, "Confidential & Academic Use Only — Department of Data Science, VIT")
        self.drawRightString(8.5 * inch - 54, 32, f"Page {self._pageNumber} of {page_count}")
        self.restoreState()

def build_pdf():
    pdf_filename = "23MID0037_Lab08_Report.pdf"
    doc = SimpleDocTemplate(
        pdf_filename,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=50,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()
    
    # Custom Palette
    primary_color = colors.HexColor("#1b365d") # Deep Navy
    secondary_color = colors.HexColor("#2b5c8f") # Slate Blue
    accent_color = colors.HexColor("#d9534f") # Crimson
    text_dark = colors.HexColor("#222222")
    bg_light = colors.HexColor("#f8f9fa")
    border_color = colors.HexColor("#e2e8f0")

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=17,
        leading=21,
        textColor=primary_color,
        alignment=0,
        spaceAfter=4
    )

    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=secondary_color,
        alignment=0,
        spaceAfter=10
    )

    h1_style = ParagraphStyle(
        'Heading1_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=15,
        textColor=primary_color,
        spaceBefore=10,
        spaceAfter=5,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'Heading2_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=13,
        textColor=secondary_color,
        spaceBefore=7,
        spaceAfter=3,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11.5,
        textColor=text_dark,
        spaceAfter=5
    )

    body_bold = ParagraphStyle(
        'Body_Bold_Custom',
        parent=body_style,
        fontName='Helvetica-Bold'
    )

    callout_style = ParagraphStyle(
        'CalloutText',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#2d3748")
    )

    table_header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=10,
        textColor=colors.white,
        alignment=1
    )

    table_cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.5,
        leading=9.5,
        textColor=text_dark,
        alignment=0
    )

    table_cell_center = ParagraphStyle(
        'TableCellCenter',
        parent=table_cell_style,
        alignment=1
    )

    caption_style = ParagraphStyle(
        'CaptionStyle',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor("#4a5568"),
        alignment=1,
        spaceBefore=3,
        spaceAfter=8
    )

    story = []

    # Title & Metadata Header Block
    story.append(Paragraph("MDI3003 Advanced Predictive Analytics — Experiment 08", title_style))
    story.append(Paragraph("<b>Agricultural Predictive Analytics: Rice Yield Prediction with a Guided Crop-Label Classification Extension</b>", subtitle_style))
    
    meta_table_data = [
        [
            Paragraph("<b>Student:</b> Lokanth S", body_style),
            Paragraph("<b>Reg No:</b> 23MID0037", body_style),
            Paragraph("<b>Faculty:</b> Dr. Durgesh Kumar", body_style)
        ],
        [
            Paragraph("<b>Manual Version:</b> 3.0", body_style),
            Paragraph("<b>Repository:</b> <code>lokant712/RiceYield-CropClassification-RF</code>", body_style),
            Paragraph("<b>Seed:</b> 42 | <b>Framework:</b> scikit-learn", body_style)
        ]
    ]
    meta_table = Table(meta_table_data, colWidths=[2.2*inch, 2.3*inch, 2.5*inch])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), bg_light),
        ('BOX', (0,0), (-1,-1), 0.5, border_color),
        ('INNERGRID', (0,0), (-1,-1), 0.5, border_color),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 8))

    # Section 1: Problem Contract & Scope
    story.append(Paragraph("1. Prediction Contract & Scope Specification", h1_style))
    contract_text = (
        "This experiment implements an empirical predictive analytics system for regional agricultural planning, targeting district-level "
        "rice yield in metric tonnes per hectare (t/ha). The operational prediction occurs prior to the harvest season, utilizing strictly pre-season "
        "geographical and temporal inputs: <code>state</code>, <code>district</code>, <code>season</code>, and <code>year</code> (constant <code>REG</code> whitelist). "
        "Crucially, total production and cultivated area are explicitly barred from the feature space because their quotient algebraically reconstructs the target "
        "($Yield = Production / Area$), which constitutes severe target leakage. Identifiers encoding yield and contemporaneous full-season weather summaries are also excluded. "
        "The system is designed for regional planning analysts to evaluate macro-allocation trends; it does not constitute an operational daily weather forecast or a farmer-level field advisory. "
        "The central hypothesis posits that multi-variable models (Ridge trend regression, Decision Trees, and Random Forests) achieve lower validation error than a historical median baseline."
    )
    story.append(Paragraph(contract_text, body_style))
    story.append(Spacer(1, 4))

    # Section 2: Data Governance Record & D5 Audit
    story.append(Paragraph("2. Data Governance Record & Acquisition Audit", h1_style))
    gov_intro = (
        "<b>Acquisition Hierarchy & Provenance Trace:</b> In accordance with Section 1 of the manual, acquisition was attempted in the strict order "
        "<b>D5 (Mendeley Data DOI 10.17632/ywp3y5j9vv.1) → D1 (Government of India APY) → D2 (ICRISAT/TCI)</b>. "
        "Probing the Mendeley Data deposit contributed by Souryabrata Mohapatra (published July 11, 2023) confirmed a valid CC BY 4.0 metadata record; "
        "however, the public snapshot contains 0 attached data payloads (empty files array), and direct API endpoints returned 404/401. "
        "Following the mandatory precedence protocol, ingestion proceeded to candidate <b>D1</b>."
    )
    story.append(Paragraph(gov_intro, body_style))

    gov_table_data = [
        [Paragraph("Governance Attribute", table_header_style), Paragraph("Specification / Audit Record", table_header_style)],
        [Paragraph("Primary Source Candidate", table_cell_style), Paragraph("D1 — Government of India Ministry of Agriculture & Farmers Welfare (APY Database)", table_cell_style)],
        [Paragraph("Raw Dataset File", table_cell_style), Paragraph("<code>data/raw/crop_production.csv</code> (15,316,741 bytes, 246,091 total records)", table_cell_style)],
        [Paragraph("Authentic File SHA-256", table_cell_style), Paragraph("<code>5b2637058ca4aff2ef7c0e0f2f391bca712300d082693b585f489871fd91d520</code>", table_cell_style)],
        [Paragraph("Rice Subset & Target Derivation", table_cell_style), Paragraph("15,082 raw Rice records → 15,078 clean records post non-positive area / outlier filtering [0, 15] t/ha", table_cell_style)],
        [Paragraph("Temporal & Spatial Coverage", table_cell_style), Paragraph("19 unique harvest years (1997–2015) across 33 Indian States and Union Territories", table_cell_style)],
        [Paragraph("Target Unit Verification", table_cell_style), Paragraph("Production in tonnes (t), Area in hectares (ha) $\\rightarrow$ Derived Yield in tonnes/hectare (t/ha)", table_cell_style)],
        [Paragraph("Extension 1 Dataset (Classification)", table_cell_style), Paragraph("<code>data/raw/crop_recommendation.csv</code> (147,833 bytes, 2,200 rows, 22 balanced classes, SHA-256: <code>808e1d84...</code>)", table_cell_style)],
        [Paragraph("Documented i.i.d. Rationale", table_cell_style), Paragraph("Crop recommendation features represent point-in-time physicochemical soil/climate samples rather than longitudinal district panel series; stratified 60/20/20 partitioning is justified (<code>iid_justified=True</code>).", table_cell_style)]
    ]
    gov_table = Table(gov_table_data, colWidths=[2.2*inch, 4.8*inch])
    gov_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), primary_color),
        ('BOX', (0,0), (-1,-1), 0.5, border_color),
        ('INNERGRID', (0,0), (-1,-1), 0.5, border_color),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, bg_light]),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('LEFTPADDING', (0,0), (-1,-1), 5),
        ('RIGHTPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(gov_table)
    story.append(Spacer(1, 6))

    # Section 3: Methodology & Validation Pipeline
    story.append(Paragraph("3. Methodology, Chronological Splitting & Leakage Guards", h1_style))
    method_text = (
        "<b>Partitioning Protocol:</b> To prevent future-to-past temporal leakage in longitudinal yield series, data is strictly partitioned by harvest year: "
        "<b>Train:</b> 1997–2011 (12,588 rows, 83.5%), <b>Validation:</b> 2012–2013 (1,644 rows, 10.9%), <b>Locked Test:</b> 2014–2015 (846 rows, 5.6%). "
        "Strict temporal separation ($Year_{train} < Year_{val} < Year_{test}$) guarantees that models are evaluated on true forward forecasting horizons.<br/>"
        "<b>Preprocessing Architecture:</b> Standard categorical encoding (<code>OneHotEncoder(handle_unknown='ignore')</code>) on <code>state</code>, <code>district</code>, "
        "and <code>season</code>, combined with <code>SimpleImputer(strategy='median')</code> and <code>StandardScaler()</code> on <code>year</code>, is fit <b>strictly on training observations</b> "
        "inside an isolated <code>ColumnTransformer</code> pipeline. Acceptance assertions verify that fitted categories match training-set uniques exactly, preventing encoder leakage."
    )
    story.append(Paragraph(method_text, body_style))
    story.append(Spacer(1, 4))

    # Figures 1 & 2 in a 2-column table
    img1_path = "figures/fig1_training_yield_distribution.png"
    img2_path = "figures/fig2_training_year_coverage.png"
    if os.path.exists(img1_path) and os.path.exists(img2_path):
        eda_table = Table([
            [Image(img1_path, width=3.4*inch, height=2.0*inch), Image(img2_path, width=3.4*inch, height=2.0*inch)],
            [
                Paragraph("<b>Figure 1:</b> Training yield distribution (mean 1.99 t/ha, median 1.93 t/ha) exhibiting right-skewness across diverse Indian agro-climatic zones.", caption_style),
                Paragraph("<b>Figure 2:</b> Annual observation coverage across 1997–2011 showing sustained district-season reporting (~800–900 records/year).", caption_style)
            ]
        ], colWidths=[3.5*inch, 3.5*inch])
        eda_table.setStyle(TableStyle([
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('LEFTPADDING', (0,0), (-1,-1), 0),
            ('RIGHTPADDING', (0,0), (-1,-1), 0),
            ('TOPPADDING', (0,0), (-1,-1), 1),
            ('BOTTOMPADDING', (0,0), (-1,-1), 1),
        ]))
        story.append(eda_table)

    story.append(Spacer(1, 6))

    # Section 4: Empirical Validation Results
    story.append(Paragraph("4. Empirical Validation Results & Candidate Selection", h1_style))
    val_text = (
        "All candidate architectures were fit on the identical training set (1997–2011) and evaluated on the held-out validation set (2012–2013). "
        "In accordance with the predeclared selection contract, candidate models are ranked by Mean Absolute Error (MAE in t/ha), with stable sorting "
        "preserving simpler baselines in the event of numerical ties."
    )
    story.append(Paragraph(val_text, body_style))

    val_res_df = pd.read_csv("23MID0037_Lab08_Validation_Results.csv")
    val_table_data = [
        [
            Paragraph("Candidate Model", table_header_style),
            Paragraph("Validation MAE (t/ha)", table_header_style),
            Paragraph("Validation RMSE (t/ha)", table_header_style),
            Paragraph("Validation R²", table_header_style),
            Paragraph("Validation MedAE (t/ha)", table_header_style),
            Paragraph("Selection Outcome", table_header_style)
        ]
    ]
    for i, row in val_res_df.iterrows():
        outcome = "<b>Selected Winner</b>" if i == 0 else f"Rank {i+1}"
        val_table_data.append([
            Paragraph(f"<code>{row['model']}</code>", table_cell_style),
            Paragraph(f"{row['val_mae_t_ha']:.4f}", table_cell_center),
            Paragraph(f"{row['val_rmse_t_ha']:.4f}", table_cell_center),
            Paragraph(f"{row['val_r2']:.4f}", table_cell_center),
            Paragraph(f"{row['val_medae_t_ha']:.4f}", table_cell_center),
            Paragraph(outcome, table_cell_center)
        ])
    
    val_table = Table(val_table_data, colWidths=[1.5*inch, 1.1*inch, 1.1*inch, 0.9*inch, 1.1*inch, 1.3*inch])
    val_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), primary_color),
        ('BOX', (0,0), (-1,-1), 0.5, border_color),
        ('INNERGRID', (0,0), (-1,-1), 0.5, border_color),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, bg_light]),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(val_table)
    story.append(Spacer(1, 4))

    # Rolling origins & Depth Ablation summary
    ro_df = pd.read_csv("artifacts/rolling_origins.csv")
    ablation_path = "artifacts_ablation/ablation_comparison.json"
    with open(ablation_path, "r") as f:
        abl = json.load(f)

    story.append(Paragraph(
        f"<b>Rolling Origins Stability & Depth Ablation:</b> Rolling origins evaluation across development horizons (2009, 2010, 2011) confirmed consistent "
        f"ranking stability for <code>ridge_trend</code> (MAE 0.3379–0.4022 t/ha). In Extension 2 (isolated depth ablation), constraining forest tree depth "
        f"from <code>max_depth=12</code> to <code>max_depth=6</code> increased validation MAE from {abl['candidate_baseline_depth12']['mae_t_ha']:.4f} to "
        f"{abl['candidate_ablated_depth6']['mae_t_ha']:.4f} t/ha ($\\Delta = +{abl['delta_mae_t_ha']:.4f}$ t/ha), confirming that depth 12 effectively captures "
        f"regional interaction structures without degradation.",
        body_style
    ))
    story.append(Spacer(1, 4))

    # Figures 3 & 4 in 2-column table
    img3_path = "figures/fig3_validation_model_comparison.png"
    img4_path = "figures/fig4_locked_test_actual_vs_predicted.png"
    if os.path.exists(img3_path) and os.path.exists(img4_path):
        mid_table = Table([
            [Image(img3_path, width=3.4*inch, height=2.0*inch), Image(img4_path, width=3.4*inch, height=2.0*inch)],
            [
                Paragraph("<b>Figure 3:</b> Validation performance across candidate architectures demonstrating Ridge trend superiority (MAE 0.4748 t/ha).", caption_style),
                Paragraph("<b>Figure 4:</b> Locked test actual vs. predicted scatter showing tight alignment along the 1:1 identity line.", caption_style)
            ]
        ], colWidths=[3.5*inch, 3.5*inch])
        mid_table.setStyle(TableStyle([
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('LEFTPADDING', (0,0), (-1,-1), 0),
            ('RIGHTPADDING', (0,0), (-1,-1), 0),
            ('TOPPADDING', (0,0), (-1,-1), 1),
            ('BOTTOMPADDING', (0,0), (-1,-1), 1),
        ]))
        story.append(mid_table)

    story.append(Spacer(1, 6))

    # Section 5: Locked Test Results & Year Robustness
    story.append(Paragraph("5. Locked Test Evaluation & Descriptive Year-Robustness", h1_style))
    test_text = (
        "<b>Single-Scoring Lock Protocol:</b> Prior to scoring the test set (2014–2015), the system acquired exclusive file lock <code>artifacts/TEST_LOCK</code>. "
        "Hashes and split manifests were verified against validation seals. The selected model (<code>ridge_trend</code>) and predeclared baseline (<code>median</code>) "
        "were evaluated on identical locked rows."
    )
    story.append(Paragraph(test_text, body_style))

    test_res_df = pd.read_csv("23MID0037_Lab08_Test_Results.csv")
    test_table_data = [
        [
            Paragraph("Model Configuration", table_header_style),
            Paragraph("Status", table_header_style),
            Paragraph("Test MAE (t/ha)", table_header_style),
            Paragraph("Test RMSE (t/ha)", table_header_style),
            Paragraph("Test R²", table_header_style),
            Paragraph("Test MedAE (t/ha)", table_header_style)
        ]
    ]
    for _, row in test_res_df.iterrows():
        test_table_data.append([
            Paragraph(f"<code>{row['model']}</code>", table_cell_style),
            Paragraph(str(row['status']), table_cell_style),
            Paragraph(f"{row['mae_t_ha']:.4f}", table_cell_center),
            Paragraph(f"{row['rmse_t_ha']:.4f}", table_cell_center),
            Paragraph(f"{row['r2']:.4f}", table_cell_center),
            Paragraph(f"{row['medae_t_ha']:.4f}", table_cell_center)
        ])
    
    test_table = Table(test_table_data, colWidths=[1.6*inch, 1.5*inch, 1.0*inch, 1.0*inch, 0.9*inch, 1.0*inch])
    test_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), primary_color),
        ('BOX', (0,0), (-1,-1), 0.5, border_color),
        ('INNERGRID', (0,0), (-1,-1), 0.5, border_color),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, bg_light]),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(test_table)
    story.append(Spacer(1, 4))

    # Year Robustness
    yr_df = pd.read_csv("artifacts/year_robustness.csv")
    story.append(Paragraph(
        f"<b>Descriptive Year Breakdown (Explicitly Not a Confidence Interval):</b> Grouping test errors by calendar year yields "
        f"<b>2014</b> (N={yr_df.loc[yr_df['year']==2014, 'sample_count'].values[0]}, MAE={yr_df.loc[yr_df['year']==2014, 'mae_t_ha'].values[0]:.4f} t/ha, R²={yr_df.loc[yr_df['year']==2014, 'r2'].values[0]:.4f}) and "
        f"<b>2015</b> (N={yr_df.loc[yr_df['year']==2015, 'sample_count'].values[0]}, MAE={yr_df.loc[yr_df['year']==2015, 'mae_t_ha'].values[0]:.4f} t/ha, R²={yr_df.loc[yr_df['year']==2015, 'r2'].values[0]:.4f}). "
        f"Because the test horizon spans exactly 2 years, this breakdown provides descriptive stability evidence rather than inferential confidence intervals.",
        body_style
    ))
    story.append(Spacer(1, 6))

    # Figures 5 & 8 in 2-column table
    img5_path = "figures/fig5_locked_test_residual_plot.png"
    img8_path = "figures/fig8_cls_confusion_matrix.png"
    if os.path.exists(img5_path) and os.path.exists(img8_path):
        bot_table = Table([
            [Image(img5_path, width=3.4*inch, height=2.0*inch), Image(img8_path, width=3.4*inch, height=2.0*inch)],
            [
                Paragraph("<b>Figure 5:</b> Locked test residual plot indicating homoscedastic dispersion centered around zero error.", caption_style),
                Paragraph("<b>Figure 6:</b> Confusion matrix for Extension 1 Random Forest crop classifier achieving 99.77% macro-F1 across 22 classes.", caption_style)
            ]
        ], colWidths=[3.5*inch, 3.5*inch])
        bot_table.setStyle(TableStyle([
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('LEFTPADDING', (0,0), (-1,-1), 0),
            ('RIGHTPADDING', (0,0), (-1,-1), 0),
            ('TOPPADDING', (0,0), (-1,-1), 1),
            ('BOTTOMPADDING', (0,0), (-1,-1), 1),
        ]))
        story.append(bot_table)

    story.append(Spacer(1, 6))

    # Section 6: Five-Case Error Audit
    story.append(Paragraph("6. Five-Case Test Error Audit & Agronomic Failure Analysis", h1_style))
    audit_intro = (
        "In accordance with Task 8, the five largest absolute test prediction errors were audited. Hypothesized causes represent grounded agronomic conjectures "
        "rather than asserted causal facts, as unobserved micro-climatic and input variables cannot be definitively verified from regional statistics alone."
    )
    story.append(Paragraph(audit_intro, body_style))

    error_df = pd.read_csv("23MID0037_Lab08_Error_Analysis.csv")
    error_table_data = [
        [
            Paragraph("Rank / Location / Season", table_header_style),
            Paragraph("Actual vs Pred (t/ha)", table_header_style),
            Paragraph("Abs Error", table_header_style),
            Paragraph("Suspected Cause (Hypothesis) & Agricultural Consequence", table_header_style),
            Paragraph("Proposed Mitigation", table_header_style)
        ]
    ]
    for _, row in error_df.iterrows():
        loc_str = f"<b>#{row['rank']}</b> {row['district']}<br/>{row['state']} ({row['season']} {row['year']})"
        yield_str = f"Act: {row['actual_yield_t_ha']:.2f}<br/>Pred: {row['predicted_yield_t_ha']:.2f}"
        cause_conseq = f"<b>Hypothesis:</b> {row['suspected_cause_hypothesis']}<br/><b>Consequence:</b> {row['agricultural_consequence']}"
        error_table_data.append([
            Paragraph(loc_str, table_cell_style),
            Paragraph(yield_str, table_cell_center),
            Paragraph(f"<b>{row['abs_error_t_ha']:.2f} t/ha</b>", table_cell_center),
            Paragraph(cause_conseq, table_cell_style),
            Paragraph(str(row['proposed_mitigation']), table_cell_style)
        ])

    error_table = Table(error_table_data, colWidths=[1.5*inch, 1.0*inch, 0.8*inch, 2.3*inch, 1.4*inch])
    error_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), primary_color),
        ('BOX', (0,0), (-1,-1), 0.5, border_color),
        ('INNERGRID', (0,0), (-1,-1), 0.5, border_color),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, bg_light]),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('LEFTPADDING', (0,0), (-1,-1), 4),
        ('RIGHTPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(error_table)
    story.append(Spacer(1, 6))

    # Section 7: Limitations, Inference Guardrails & Governance Summary
    story.append(Paragraph("7. Operational Limitations, Inference Guardrails & Ethical Oversight", h1_style))
    limits_text = (
        "<b>Operational Boundaries:</b><br/>"
        "1. <i>Spatial & Temporal Support:</i> Predictions are valid strictly within evaluated administrative boundaries and supported temporal windows (1997–2015). Inference for unobserved districts or extrapolation beyond +4 years is rejected by <code>ValidatedPredictor</code>.<br/>"
        "2. <i>Omitted Variable Bias:</i> Static regional identifiers cannot account for acute sub-seasonal meteorological shocks (cyclones, unseasonal dry spells) or sudden pest outbreaks. Feature importances reflect empirical associations, not agronomic causality.<br/>"
        "3. <i>Ethical Planning Use:</i> Model outputs must serve as advisory decision-support for buffer stocking and regional planning. They must never be used to penalize smallholder credit access or deny crop insurance claims without physical field loss verification.<br/>"
        "4. <i>Model Deserialization Security:</i> Only trusted bundles serialized by this verified pipeline are loaded (<code>models/selected_bundle.joblib</code>); deserialization of external joblib files is strictly prohibited."
    )
    story.append(Paragraph(limits_text, body_style))
    story.append(Spacer(1, 6))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Report compiled successfully to {pdf_filename}")

if __name__ == '__main__':
    build_pdf()
