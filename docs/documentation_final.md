\documentclass[
]{elteikthesis}[2024/04/26]

% Document metadata
\title{Multimodal Pneumonia Detection from Chest X-rays and Clinical Data using MIMIC-CXR and MIMIC-IV}
\date{2026}

% Author metadata
\author{Yazan Al-Dabain}
\degree{Computer Science BSc}

% Supervisor metadata (EDIT THESE)
\supervisor{Your Supervisor Name}
\affiliation{Assistant Lecturer}

% University metadata
\university{Eötvös Loránd University}
\faculty{Faculty of Informatics}
\department{Dept. of Software Technology and Methodology}
\city{Budapest}
\logo{elte_cimer_szines}

% Bibliography
\addbibresource{references.bib}

\begin{document}

\documentlang{english}

% Title page
\maketitle

% Table of contents
\tableofcontents
\cleardoublepage

% =========================
% INTRODUCTION
% =========================
\chapter{Introduction}

Pneumonia is a major cause of morbidity and mortality worldwide, requiring rapid and accurate diagnosis, particularly in emergency department (ED) settings. Chest X-rays (CXR) are the most commonly used imaging modality for pneumonia detection, but interpretation can be time-consuming and subject to variability.

Recent advances in deep learning have enabled automated image-based diagnosis. However, clinical decision-making in real-world settings is inherently multimodal, incorporating both imaging and patient data such as vital signs and laboratory measurements.

This thesis presents a multimodal pneumonia detection system using:
\begin{itemize}
    \item Chest X-rays from MIMIC-CXR-JPG
    \item Clinical data from MIMIC-IV and MIMIC-IV-ED
\end{itemize}

The system is designed with strict temporal constraints to ensure that only information available at imaging time is used, preventing data leakage and enabling clinically realistic evaluation.

\cleardoublepage

% =========================
% BACKGROUND
% =========================
\chapter{Background}

\section{Pneumonia Detection in Medical Imaging}
Deep learning models such as DenseNet121 have demonstrated strong performance on chest X-ray classification tasks, particularly using datasets like CheXpert and MIMIC-CXR.

\section{Multimodal Learning in Healthcare}
Multimodal approaches combine imaging with structured clinical data. While promising, many prior works suffer from:
\begin{itemize}
    \item Temporal leakage
    \item Unrealistic feature availability
    \item Poor cohort design
\end{itemize}

\section{Datasets}
This work uses:
\begin{itemize}
    \item MIMIC-CXR-JPG (v2.1.0)
    \item MIMIC-IV
    \item MIMIC-IV-ED
\end{itemize}

\cleardoublepage

% =========================
% DATA AND COHORT
% =========================
\chapter{Data and Cohort Construction}

\section{Cohort Design}

The final cohort consists of ED-linked chest X-ray studies, where each study is associated with a single emergency department stay.

Key statistics:
\begin{itemize}
    \item 81,385 ED-linked studies
    \item 47,404 patients
\end{itemize}

The prediction time is defined as:
\[
t_0 = \text{time of imaging}
\]

Only features available at or before $t_0$ are used.

\section{Label Definition}

Pneumonia labels are derived from CheXpert annotations:
\begin{itemize}
    \item Positive: 4,281
    \item Negative: 4,856
    \item Uncertain and missing labels handled via policy
\end{itemize}

Primary training uses the \textbf{u\_ignore} policy:
\begin{itemize}
    \item Excludes uncertain labels
    \item Final dataset size: 9,137 studies
\end{itemize}

\section{Temporal Split}

A patient-level temporal split is used:
\begin{itemize}
    \item Train: 7,132
    \item Validation: 930
    \item Test: 1,075
\end{itemize}

This prevents leakage across time and patients.

\cleardoublepage

% =========================
% METHODS
% =========================
\chapter{Methods}

\section{Clinical Features}

Clinical features are derived from ED triage data, including:
\begin{itemize}
    \item Vital signs (temperature, heart rate, respiratory rate, oxygen saturation)
    \item Blood pressure
    \item Pain score and acuity
    \item Demographics (gender, race)
    \item Arrival transport
\end{itemize}

Features are:
\begin{itemize}
    \item Clipped to physiologically plausible ranges
    \item Imputed using training-set statistics only
    \item Augmented with missingness indicators
\end{itemize}

Disposition is excluded due to leakage risk :contentReference[oaicite:0]{index=0}.

\section{Image Model}

A DenseNet121 architecture is used:
\begin{itemize}
    \item Pretrained on multilabel chest pathology task
    \item Fine-tuned for pneumonia detection
\end{itemize}

Training uses:
\begin{itemize}
    \item Image resolution: 224x224
    \item Early stopping on validation AUPRC
\end{itemize}

\section{Multimodal Model}

The multimodal model combines:
\begin{itemize}
    \item Image embeddings from DenseNet121
    \item Clinical features via MLP
\end{itemize}

Fusion is performed using feature concatenation with a classification head.

Initial experiments use a \textbf{frozen image backbone}.

\section{Evaluation Protocol}

Evaluation uses:
\begin{itemize}
    \item AUROC
    \item AUPRC
    \item Patient-level bootstrap confidence intervals
\end{itemize}

Bootstrap resampling is performed at the patient level to ensure valid statistical comparison :contentReference[oaicite:1]{index=1}.

\cleardoublepage

% =========================
% RESULTS
% =========================
\chapter{Results}

\section{Clinical Baselines}

Logistic Regression:
\begin{itemize}
    \item Test AUROC: 0.605
    \item Test AUPRC: 0.547
\end{itemize}

XGBoost:
\begin{itemize}
    \item Test AUROC: 0.611
    \item Test AUPRC: 0.559
\end{itemize}

\section{Image Model}

The image model significantly outperforms clinical baselines.

\section{Multimodal Model}

The multimodal model combines both modalities and is evaluated against:
\begin{itemize}
    \item Image-only baseline
    \item Clinical-only baseline
\end{itemize}

Performance differences are assessed using paired bootstrap analysis.

\paragraph{Committed numeric exports (thesis supplement)}
Exact test-set AUROC/AUPRC and threshold-0.5 confusion metrics for the canonical clinical \texttt{strong\_v2} and image/multimodal \texttt{stronger\_lr\_v3} runs are tabulated in \texttt{artifacts/evaluation/final\_results\_table.csv}; narrative summary in \texttt{artifacts/evaluation/final\_result\_note.txt}. Full floating-point rows live under \texttt{artifacts/evaluation/prediction\_behavior\_*/} (see \texttt{docs/current\_state.md} \S16.8).

\cleardoublepage

% =========================
% DISCUSSION
% =========================
\chapter{Discussion}

\section{Key Findings}

\begin{itemize}
    \item Clinical features alone provide limited predictive power
    \item Image models dominate performance
    \item Multimodal fusion provides incremental improvements
\end{itemize}

\section{Limitations}

\begin{itemize}
    \item High label missingness (~79\%)
    \item Sparse lab feature coverage (~9.5\%)
    \item CheXpert label noise
\end{itemize}

\section{Future Work}

\begin{itemize}
    \item Unfreezing image backbone in multimodal model
    \item Improved fusion architectures
    \item External validation
\end{itemize}
\cleardoublepage

% =========================
% CONCLUSION
% =========================
\chapter{Conclusion}

This thesis presents a clinically realistic multimodal pneumonia detection system with strict temporal validation and leakage control.

The results demonstrate the importance of:
\begin{itemize}
    \item Proper cohort design
    \item Time-aware feature construction
    \item Robust evaluation protocols
\end{itemize}

\cleardoublepage

% =========================
% ACKNOWLEDGEMENTS
% =========================
\chapter*{\acklabel}
\addcontentsline{toc}{chapter}{\acklabel}

I would like to thank my supervisor and colleagues for their guidance and support throughout this project.

\cleardoublepage

% =========================
% BIBLIOGRAPHY
% =========================
\phantomsection
\addcontentsline{toc}{chapter}{\biblabel}
\printbibliography[title=\biblabel]
\cleardoublepage

\end{document}
