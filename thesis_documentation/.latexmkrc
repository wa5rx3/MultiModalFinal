# elteikthesis loads biblatex (biber). Do not run legacy bibtex on main.aux.
$pdf_mode = 1;
$pdflatex = 'pdflatex -interaction=nonstopmode %O %S';
$bibtex_use = 0;
