## Definitionen
`Injetiv:` $f(x)=f(x`) \rightarrow x = x`$
-> Beweis 1. $f(x)=f(x')$ 2. Umformen $x=x'$
`Surjektivität:` für jedes y gibt es ein $y=f(x)$
-> Beweis 1. $f^{-1}$ austellen 2. $f^{-1}$ auf gesamtem Definitionsbereich definiert
`Bijektivität:` Injektiv+Surjektiv

## Polynome
`Linearfaktorzerlegung:` $x_0$ ist Nullstelle $p(x)=q(x)(x-x_0)$ mit $q(n)=\frac{p(x)}{x-x_0}$
-> 1. Nullstelle raten 2. Polynomdivison oder **Horner** 3. Wiederholen
-> 1. Alle Nullstellen herausfinden 2. Nullstellen faktorisieren
! nicht $x_0=0$ vergessen bsp. $x⁵...$ hat 5 Nullstellen Rest ist 0

`Partialbruchzerlegung:` $f(x)=\frac{p_m(x)}{p_n(x)}=\frac{A}{x-x_1}+...+\frac{A_r}{x-x_r}$ wichtig $grad(p_m)<grad(p_n)$

`Polynomdivison:` Grad reduzieren durch Teilen durch $(x-x_0)$
## Gleichungen und Ungleichungen
minus immer andere seite
$|x|$ immer Fallunterscheidung

## Folgen und Reihen 
![[Pasted image 20260302173940.png|579]]
## Integrale
`Partielel Integration:` $\int{f(x)g(x)}dx=F(x)g(x)-\int{f(x)g'(x)}dx$
> Faustregel **L**($log$,$ln$), **I**(inverese $arcsin$,$arccos$), **A**(algebra $x^2$), **T**(trig. $sin$,$cos$), **E**(Expo. $e^x$)

`Ingtegration durch Substitution:` $\int{f(x)}\,dx=\int{f(\phi(u)) * \phi'(u)}\,du$

### Geometrisch
`Fläche einer\mehrere Funktion:` $|\int_a^b{|f(x)-g(x)|} dx|$
`Längenberechnung:` $L_a^b(f)=\int_a^b{\sqrt{1+(f'(x))^2}}\,dx$
`Mantelflächenberechnung:` $M_a^b=\int_a^b{2\pi\cdot f(x)\cdot \sqrt{1+(f'(x))^2}}\,dx$
`Rotationsvolumen:` $V_a^b(f)=\int_a^b{\pi\cdot f(x)^2}\,dx$

### Variable Grenze
`variable Grenze` $\frac{d}{dx} \int_{a(x)}^{b(x)} f(x)\, dt=f(b(x))\, b'(x)-f(a(x))\, a'(x)$
`Leibniz-Regel:` $\frac{d}{dx} \int_{a(x)}^{b(x)} f(x,t)\, dt=\int_{a(x)}^{b(x)} \frac{\partial}{\partial x} f(x,t)\, dt+f(x,b(x))\, b'(x)-f(x,a(x))\, a'(x)$

## Uneigentlich Integrale
?

