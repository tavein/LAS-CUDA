function rez=LAS_score(sum, subrows, subcols, mrows, mcols)

cnrows = gammaln(mrows+1) - gammaln(subrows+1) - gammaln(mrows-subrows +1);
cncols = gammaln(mcols+1) - gammaln(subcols+1) - gammaln(mcols-subcols +1);

%rest = log(normcdf(-sum,0,sqrt(subrows*subcols)));

ar = sum./sqrt(subrows.*subcols);

rest2 = - ar.^2/2 + log( erfcx(ar/sqrt(2))/2 );

rez = - rest2 - cnrows - cncols;
