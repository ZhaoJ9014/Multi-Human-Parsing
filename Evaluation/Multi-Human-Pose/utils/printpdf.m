function printpdf(fname)
%PRINTPDF Prints the current figure into a pdf document
set(gca, 'LooseInset', get(gca, 'TightInset'));
fname = [regexprep(fname, '^(.*)\.pdf$', '$1'), '.eps'];
print('-depsc', fname) ;
if ~system(['epstopdf ', fname])
system(['rm ', fname]);
end