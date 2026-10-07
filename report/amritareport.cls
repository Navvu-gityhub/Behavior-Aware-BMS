% ============================================================
%  amritareport.cls
%  A LaTeX class for B.Tech / project reports submitted to
%  Amrita Vishwa Vidyapeetham, Amrita School of Engineering.
%
%  Based on: Appendix 1 (Cover/Title Page), Appendix 2 (Bonafide
%  Certificate), Appendix 3 (Declaration by Candidate).
%
%  USAGE (in your main .tex file):
%
%    \documentclass[12pt,a4paper]{amritareport}
%
%    \reporttitle{TITLE OF THE PROJECT}
%    \degreename{BACHELOR OF TECHNOLOGY IN ...}
%    \addcandidate{NAME OF CANDIDATE}{Reg. No.}
%    % \addcandidate can be called more than once for group projects
%    \supervisor{NAME OF SUPERVISOR}{Designation \& Department}
%    \hod{NAME OF HOD}{Department}
%    \deptaddress{Full address of the department \& college}
%    \reportdate{Month \& Year}
%    \universitylogo{amrita_logo.png} % optional, comment usage if absent
%
%    \begin{document}
%    \makecoverpage
%    \makebonafidecertificate
%    \makedeclaration
%    \acknowledgement{... text ...}
%    \reportabstract{... text ...}
%    \frontmatter@toc   % or just \tableofcontents \listoffigures \listoftables
%    \mainmatter@start  % or just \pagenumbering{arabic}
%    \chapter{Introduction}
%    ...
%    \end{document}
% ============================================================

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{amritareport}[2026/09/18 v1.0 Amrita Vishwa Vidyapeetham project report class]

% ---------- Pass options through to the base 'report' class ----------
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{report}}
\ProcessOptions\relax
\LoadClass[12pt,a4paper]{report}

% ---------- Required packages ----------
\RequirePackage[utf8]{inputenc}
\RequirePackage[T1]{fontenc}
\RequirePackage{times}
\RequirePackage[left=1.5in,right=1in,top=1in,bottom=1in]{geometry}
\RequirePackage{setspace}
\RequirePackage{graphicx}
\RequirePackage{tocloft}
\RequirePackage{titlesec}
\RequirePackage{amsmath,amssymb}
\RequirePackage{booktabs}
\RequirePackage{caption}
\RequirePackage[hidelinks]{hyperref}
\RequirePackage[nottoc]{tocbibind}
\RequirePackage{etoolbox}
\RequirePackage{mfirstuc}
\RequirePackage{microtype}
\RequirePackage{seqsplit}

% Give the line-breaker more room to stretch spaces before it gives up
% and lets a line run past the margin (overfull hbox) -- this matters
% especially for the bonafide certificate / declaration paragraphs,
% which contain long unbreakable strings such as the project title
% and registration numbers.
\tolerance=2000
\emergencystretch=3em
\hbadness=10000

% Body of the report is 1.5-line spaced per the guidelines
\onehalfspacing

% ---------- Chapter heading style ----------
\titleformat{\chapter}[display]
  {\centering\bfseries\fontsize{14}{16}\selectfont}
  {CHAPTER \thechapter}
  {10pt}
  {\MakeUppercase}
\titlespacing*{\chapter}{0pt}{20pt}{20pt}

% Unnumbered chapter headings (Contents, List of Figures/Tables,
% Abstract, Acknowledgement, Abbreviations, References) are centered
\titleformat{name=\chapter,numberless}[block]{\centering\bfseries\large}{}{0pt}{}

% ---------- Section / subsection heading style ----------
\titleformat{\section}{\normalsize\bfseries}{\thesection}{1em}{}
\titlespacing*{\section}{0pt}{12pt plus 2pt minus 2pt}{6pt}

\titleformat{\subsection}{\normalsize\bfseries}{\thesubsection}{1em}{}
\titlespacing*{\subsection}{0pt}{10pt plus 2pt minus 2pt}{4pt}

\titleformat{\subsubsection}{\normalsize\bfseries\itshape}{\thesubsubsection}{1em}{}
\titlespacing*{\subsubsection}{0pt}{10pt plus 2pt minus 2pt}{4pt}

%\renewcommand{\cftsubsubsecfont}{\itshape}

\titleformat{\section}{\normalsize\bfseries}{\thesection}{1em}{\MakeUppercase}
%\renewcommand{\cftsecfont}{\bfseries\MakeUppercase}
%\renewcommand{\cftsecfont}{\bfseries}


%\renewcommand{\cfttoctitlefont}{\hfill\Large\bfseries}
%\renewcommand{\cftaftertoctitle}{\hfill}

%\renewcommand{\cftloftitlefont}{\Large\bfseries\centering}
%\renewcommand{\cftafterloftitle}{\par}

%\renewcommand{\cftlottitlefont}{\hfill\Large\bfseries}
%\renewcommand{\cftafterlottitle}{\hfill}

\renewcommand{\cfttoctitlefont}{\bfseries\large}
\renewcommand{\cftloftitlefont}{\bfseries\large}
\renewcommand{\cftlottitlefont}{\bfseries\large}
\renewcommand{\@cftmaketoctitle}{%
  \addpenalty\@secpenalty
  \if@cfthaschapter\vspace*{\cftbeforetoctitleskip}\else\vspace{\cftbeforetoctitleskip}\fi
  \@cftpagestyle
  \begingroup
    \interlinepenalty\@M
    \centering\cfttoctitlefont\contentsname\cftaftertoctitle\par
  \endgroup
  \cftmarktoc
  \nobreak
  \vskip\cftaftertoctitleskip
  \@afterheading}
\renewcommand{\@cftmakeloftitle}{%
  \addpenalty\@secpenalty
  \if@cfthaschapter\vspace*{\cftbeforeloftitleskip}\else\vspace{\cftbeforeloftitleskip}\fi
  \@cftpagestyle
  \begingroup
    \interlinepenalty\@M
    \centering\cftloftitlefont\listfigurename\cftafterloftitle\par
  \endgroup
  \cftmarklof
  \nobreak
  \vskip\cftafterloftitleskip
  \@afterheading}
\renewcommand{\@cftmakelottitle}{%
  \addpenalty\@secpenalty
  \if@cfthaschapter\vspace*{\cftbeforelottitleskip}\else\vspace{\cftbeforelottitleskip}\fi
  \@cftpagestyle
  \begingroup
    \interlinepenalty\@M
    \centering\cftlottitlefont\listtablename\cftafterlottitle\par
  \endgroup
  \cftmarklot
  \nobreak
  \vskip\cftafterlottitleskip
  \@afterheading}

\renewcommand{\contentsname}{TABLE OF CONTENTS}
\renewcommand{\listfigurename}{LIST OF FIGURES}
\renewcommand{\listtablename}{LIST OF TABLES}

% Number sections/subsections, and show them in the ToC
\setcounter{secnumdepth}{3}
\setcounter{tocdepth}{3}

\renewcommand{\cftfigfont}{Figure }
\renewcommand{\cfttabfont}{Table }

\renewcommand{\bibname}{REFERENCES}
\renewcommand{\cftdotsep}{\cftnodots}
% Remove the dotted leader lines after section/subsection entries
% in the Table of Contents (chapter entries have no dots by default)
%\renewcommand{\cftsecleader}{}
%\renewcommand{\cftsubsecleader}{}

% ============================================================
% METADATA COMMANDS
% ============================================================
\newcommand{\@reporttitle}{<<TITLE OF THE PROJECT>>}
\newcommand{\reporttitle}[1]{\renewcommand{\@reporttitle}{#1}}

\newcommand{\@degreename}{<<NAME OF THE DEGREE>>}
\newcommand{\degreename}[1]{\renewcommand{\@degreename}{#1}}

\newcommand{\@reportdate}{<<Month \& Year>>}
\newcommand{\reportdate}[1]{\renewcommand{\@reportdate}{#1}}

\newcommand{\@deptaddress}{<<Full Address of the Department \& College>>}
\newcommand{\deptaddress}[1]{\renewcommand{\@deptaddress}{#1}}

\newcommand{\@universitylogo}{}
\newcommand{\universitylogo}[1]{\renewcommand{\@universitylogo}{#1}}

\newcommand{\@universityfulllogo}{}
\newcommand{\universityfulllogo}[1]{\renewcommand{\@universityfulllogo}{#1}}


\newcommand{\@supervisorname}{<<NAME OF THE SUPERVISOR>>}
\newcommand{\@supervisordesig}{<<Academic Designation >>}
\newcommand{\supervisor}[2]{%
  \renewcommand{\@supervisorname}{#1}%
  \renewcommand{\@supervisordesig}{#2}%
}

\newcommand{\@hodname}{<<NAME OF THE HEAD OF THE DEPARTMENT>>}
\newcommand{\@hoddept}{<<Academic Designation >>}
\newcommand{\hod}[2]{%
  \renewcommand{\@hodname}{#1}%
  \renewcommand{\@hoddept}{#2}%
}

% ---- Candidates: supports one or more (group projects) ----
\newcounter{numcandidates}
\newcommand{\@candidatelist}{}      % used on the title page
\newcommand{\@candnamestext}{}      % joint "Name (Reg), Name (Reg)" text for the certificate
\newcommand{\@candsigblocks}{}      % stacked SIGNATURE blocks for the declaration
\newcommand{\@candackblocks}{}      % stacked name/reg blocks for the acknowledgement
\newcommand{\addcandidate}[2]{%
  % #1 = name, #2 = registration number
  % Call this once per group member, in any order. Every member listed
  % here automatically appears on the title page, the bonafide
  % certificate, the declaration (with their own signature line), and
  % the acknowledgement signature block.
  \stepcounter{numcandidates}%
  \ifnum\value{numcandidates}=1
    \renewcommand{\@candidatelist}{%
      {\large\bfseries #1\par}{\large\bfseries (#2)\par}}%
    \renewcommand{\@candnamestext}{#1\ (\mbox{#2})}%
    \renewcommand{\@candsigblocks}{%
      \textbf{SIGNATURE}\\[0.5cm]
      \textbf{#1}\\
      \textbf{#2}}%
    \renewcommand{\@candackblocks}{%
      \textbf{#1}\\
      \textbf{#2}}%
  \else
    \g@addto@macro\@candidatelist{%
      \vspace{0.3cm}{\large\bfseries #1\par}{\large\bfseries (#2)\par}}%
    \g@addto@macro\@candnamestext{, #1\ (\mbox{#2})}%
    \g@addto@macro\@candsigblocks{%
      \vspace{2.5cm}\\
      \textbf{SIGNATURE}\\[0.5cm]
      \textbf{#1}\\
      \textbf{#2}}%
    \g@addto@macro\@candackblocks{%
      \vspace{1cm}
      \\
	\textbf{#1}\\
      \textbf{#2}}%
  \fi
}
% \primarycandidate is kept only for backward compatibility; it is no
% longer needed -- \addcandidate now feeds the certificate, declaration
% and acknowledgement for every member automatically.
\newcommand{\primarycandidate}[2]{}


% ============================================================
% COVER / TITLE PAGE   (Appendix 1)
% ============================================================
\newcommand{\makecoverpage}{%
\begin{titlepage}
\begin{center}
\onehalfspacing
{\fontsize{18}{22}\selectfont\bfseries \@reporttitle \par}
\vspace{0.7cm}
{\large\bfseries A PROJECT REPORT\par}
\vspace{0.7cm}
{\large\itshape Submitted by\par}
\@candidatelist
\vspace{0.8cm}
{\large\itshape\bfseries in partial fulfillment for the award of the degree of\par}
\vspace{0.8cm}
{\large\bfseries \@degreename \par}
\vspace{0.7cm}
{\large\itshape Under the guidance of\par}
\vspace{0.3cm}
{\large\bfseries \@supervisorname \par}
\vspace{0.7cm}
{\large\itshape Submitted to\par}
\vspace{0.3cm}
\ifdefempty{\@universitylogo}{}{\includegraphics[width=6cm]{\@universitylogo}\\[0.2cm]}
{\normalsize\bfseries DEPARTMENT OF ELECTRONICS AND COMMUNICATION ENGINEERING\par}
{\large\bfseries AMRITA SCHOOL OF ENGINEERING\par}
{\large\bfseries AMRITA VISHWA VIDYAPEETHAM\par}
{\large\bfseries CHENNAI -- 601103\par}
\vspace{0.3cm}
{\large\bfseries \@reportdate \par}
\end{center}
\end{titlepage}
}

\newcommand{\maketitlepage}{%
\newpage
\begin{center}
\onehalfspacing
{\fontsize{18}{22}\selectfont\bfseries \@reporttitle \par}
\vspace{0.7cm}
{\large\bfseries A PROJECT REPORT\par}
\vspace{0.7cm}
{\large\itshape Submitted by\par}
\@candidatelist
\vspace{0.8cm}
{\large\itshape\bfseries in partial fulfillment for the award of the degree of\par}
\vspace{0.8cm}
{\large\bfseries \@degreename \par}
\vspace{0.7cm}
{\large\itshape Under the guidance of\par}
\vspace{0.3cm}
{\large\bfseries \@supervisorname \par}
\vspace{0.7cm}
{\large\itshape Submitted to\par}
\vspace{0.3cm}
\ifdefempty{\@universitylogo}{}{\includegraphics[width=6cm]{\@universitylogo}\\[0.2cm]}
{\normalsize\bfseries DEPARTMENT OF ELECTRONICS AND COMMUNICATION ENGINEERING\par}
{\large\bfseries AMRITA SCHOOL OF ENGINEERING\par}
{\large\bfseries AMRITA VISHWA VIDYAPEETHAM\par}
{\large\bfseries CHENNAI -- 601103\par}
\vspace{0.3cm}
{\large\bfseries \@reportdate \par}
\end{center}
}

% ============================================================
% BONAFIDE CERTIFICATE   (Appendix 2)
% ============================================================
\newcommand{\makebonafidecertificate}{%
\newpage
%\thispagestyle{empty}
\begin{center}
\ifdefempty{\@universityfulllogo}{}{\includegraphics[width=8cm]{\@universityfulllogo}\\[0.5cm]}
{\large\bfseries BONAFIDE CERTIFICATE\par}
\end{center}
\vspace{0.8cm}
\noindent
\sloppy
Certified that this project report \textbf{``\@reporttitle''} is the
bonafide work of \textbf{\@candnamestext} who
carried out the project work under my supervision.
\fussy

\vspace{2cm}
\noindent
\begin{minipage}[t]{0.48\textwidth}
%\underline{\hspace{5cm}}\\
\textbf{SIGNATURE}\\[1cm]
\textbf{\@hodname}\\
\textbf{CHAIRPERSON}\\
\@hoddept\\
\@deptaddress
\end{minipage}
\hfill
\begin{minipage}[t]{0.48\textwidth}
%\underline{\hspace{5cm}}\\
\textbf{SIGNATURE}\\[1cm]
\textbf{\@supervisorname}\\
\textbf{SUPERVISOR}\\
\@supervisordesig\\
\@deptaddress
\end{minipage}

\vspace{2.5cm}
\noindent The Viva-Voce Examination for Project Phase-I was conducted on \_\_\_\_\_\_\_\_\_\_\\
\vspace{3cm}\\
\noindent
\begin{minipage}[t]{0.48\textwidth}
\textbf{INTERNAL EXAMINER}
\end{minipage}
\hfill
\begin{minipage}[t]{0.48\textwidth}
\textbf{EXTERNAL EXAMINER}
\end{minipage}
}

% ============================================================
% DECLARATION BY THE CANDIDATE   (Appendix 3)
% ============================================================
\newcommand{\makedeclaration}{%
\newpage
%\thispagestyle{empty}
\begin{center}
\ifdefempty{\@universityfulllogo}{}{\includegraphics[width=8cm]{\@universityfulllogo}\\[0.5cm]}
{\large\bfseries DECLARATION BY THE CANDIDATE\par}
\end{center}
\vspace{0.8cm}
\noindent
\sloppy
\ifnum\value{numcandidates}>1
  We declare that the report entitled \textbf{``\@reporttitle''} submitted
  by us for the degree of \@degreename\ is the record of the project work
  carried out by us under the guidance of \textbf{\@supervisorname}
  and this work has not formed the basis for the award of any degree,
  diploma, associateship, fellowship, or other similar title in this or
  any other University or institution of higher learning.
\else
  I declare that the report entitled \textbf{``\@reporttitle''} submitted
  by me for the degree of \@degreename\ is the record of the project work
  carried out by me under the guidance of \textbf{\@supervisorname}
  and this work has not formed the basis for the award of any degree,
  diploma, associateship, fellowship, or other similar title in this or
  any other University or institution of higher learning.
\fi
\fussy

\vspace{3cm}
\begin{flushright}
\@candsigblocks
\end{flushright}
}

% ============================================================
% ACKNOWLEDGEMENT / ABSTRACT
% ============================================================
\newcommand{\acknowledgement}[1]{%
\newpage
\chapter*{ACKNOWLEDGEMENT}
\ifnum\value{numcandidates}>1
\indent This project work would not have been possible without the contribution of many people. It gives us immense pleasure to express our profound gratitude to our honorable Chancellor \textbf{SRI MATA AMRITANANDAMAYI DEVI}, for her blessings and for being a source of inspiration. 

We are indebted to extend our gratitude to our Director, \textbf{Mr. I B MANIKANTAN}, Amrita Vishwa Vidyapeetham, Chennai for facilitating us all the facilities and extended support to gain valuable education and learning experience.

We register our special thanks to \textbf{Dr. V. JAYAKUMAR}, Principal, Amrita Vishwa Vidyapeetham, Chennai for the support given to us in the successful conduct of this project. We wish to express our sincere gratitude to \textbf{\@hodname}, Chairperson, Department of Electronics and Communication Engineering, and our supervisor \textbf{\@supervisorname} for their inspiring guidance, personal involvement, and constant encouragement during the entire course of this work.

We are grateful to Project Coordinator, Review Panel Members and the entire faculty of the Department of Electronic \& Communication  Engineering, for their constructive criticisms and valuable suggestions which have been a rich source to improve the quality of this work.
\else
\indent This project work would not have been possible without the contribution of many people. It gives me immense pleasure to express my profound gratitude to our honorable Chancellor \textbf{SRI MATA AMRITANANDAMAYI DEVI}, for her blessings and for being a source of inspiration. 

I am indebted to extend my gratitude to our Director, \textbf{Mr. I B MANIKANTAN}, Amrita Vishwa Vidyapeetham, Chennai for facilitating us all the facilities and extended support to gain valuable education and learning experience.

I register my special thanks to \textbf{Dr. V. JAYAKUMAR}, Principal, Amrita Vishwa Vidyapeetham, Chennai for the support given to me in the successful conduct of this project. I wish to express my sincere gratitude to \textbf{\@hodname}, Chairperson, Department of Electronics and Communication Engineering, and my supervisor \textbf{\@supervisorname} for their inspiring guidance, personal involvement, and constant encouragement during the entire course of this work.

I am grateful to Project Coordinator, Review Panel Members and the entire faculty of the Department of Electronic \& Communication  Engineering, for their constructive criticisms and valuable suggestions which have been a rich source to improve the quality of this work.
\fi

\vspace{2cm}
\begin{flushright}
\@candackblocks
\end{flushright}

\addcontentsline{toc}{chapter}{ACKNOWLEDGEMENT}
#1
}

\newcommand{\reportabstract}[1]{%
\newpage
\chapter*{ABSTRACT}
\addcontentsline{toc}{chapter}{ABSTRACT}
#1
}

% ============================================================
% FRONT MATTER: TOC / LOF / LOT / ABBREVIATIONS
% ============================================================
\newcommand{\frontmattertoc}{%
\newpage
\tableofcontents
\newpage
\listoftables
%\addcontentsline{toc}{chapter}{LIST OF TABLES}

\newpage
\listoffigures
%\addcontentsline{toc}{chapter}{LIST OF FIGURES}
}

\newenvironment{abbreviations}{%
\newpage
\chapter*{\large LIST OF SYMBOLS AND ABBREVIATIONS}
\addcontentsline{toc}{chapter}{LIST OF SYMBOLS AND ABBREVIATIONS}
\begin{tabular}{@{}ll@{}ll@{}}
}{%
\end{tabular}
}

% ============================================================
% START OF MAIN BODY (numbered pages, arabic numerals)
% ============================================================
\newcommand{\startmainmatter}{%
\newpage
\pagenumbering{arabic}
\setlength{\parindent}{1cm}

}

\endinput
