# Machine Learning Programmierung Cheat Cheet 
## Basics
### Matplotlib
#### Funktion Ploten
```python
x = np.linspace(0, 1, 400)
plt.plot(x, f(x),label="")
```



## Der Perzepton
Idee: Finde ein Gewichtungsvektor w, sodass Datenpunkte einer Klasse auf der einen Seite liegen, der anderen auf der anderen Seite.

$h(x) = sign({w}^T {x}) = \begin{cases} 1 & {w}^T {x} \geq 0 \\ -1 & w^Tx <0 \end{cases}$  
$b = Schwellenwert$  
$w=(b,w_1,...,w_d)^T$  
$x=(0,x_1,...,x_d)^T$

 ```python 
 def h(x,w):
    return np.sign(np.dot(w.T,x))
```
## PLA _Perzeption Lern-Algoritmus_
Daten: $(x_i,y_i) \text{ mit } i = i,...,N$  
Annhame: Alle Daten sind linear sperierbar  
Sei $t = 0,1,2,... \text{ der aktuelle Iterationsschritt mit } w^{(t)}$

0. setzte $t = 0$ und $w^{(0)} = 0$
1. Klassifiziere alle Daten x ⇒ vorhergesagte Lables $y_{i,pred}$
2.  Selektiere ein Datenpaar mit $y_i \ne y_{i,pred}$ (falsch klassifiziert) nenne es $(x^{(t)}, y^{(t)})$
⇒ terminieren wenn es keins gibt
3. Atualisiere das Gewicht $w$: $w^{t-1} = w^{(t)}+y^{(t)}x^{(t)}$
4. Neubeginn bei 1. mit $t++$
   
```python
def pla(x,y,w):
    x_calc = np.column_stack((np.ones(len(x)), x)) #0.
    for i,(xi,yi) in enumerate(zip(x_calc,y)):
        y_pred = h(xi,w) #1.
        if(y_pred != yi): #2.
            w = w + yi*xi #3.
            return pla(x,y,w) #4.
    return w

```
Entscheidungsgrenze $0=w^Tx$ bei   
Bei $d=2$ gilt $0=w_0+w_1x_1+w_2x_2 \leftrightarrow x_2= -\frac{w_1}{w_2} x_1-\frac{w_0}{w_2}$
```python
# Entscheidungsgrenze
def p(x,w):
    m = -(w[1]/w[2])
    b = -(w[0]/w[2])
    return m * x + b
```

## Pocket - PLA Erweiterung
Annhame: Daten sind **nicht** linear sperierbar (Rauschen)  

0. setzte $t = 0$ und $w_{best} = w^{(0)}=0$ $E_{in}(w_{best})=1$
1. für $t=0,...,T-1$:
   1.  PLA mit $w^{(t)}$ ⇒ $w^{(t+1)}$
   2.  Ermittel $E_{in}(w^{t+1)})$
   3.  $E_{in}(w^{t+1)}) < E_{in}(w_{best})$ ⇒ setzte $w_{best}=w^{t+1)}$
2. $w_{best}$ zurückgeben
   
```python
# Perzeptron-Entscheidungsfunktion
def h(x: np.ndarray, w: np.ndarray) -> int:
    return np.sign(np.dot(w, x))
    
def insample_error(x_calc, y, w):
    y_pred = np.sign(x_calc @ w)
    return np.mean(y_pred != y) # boolsche konversion [False, True, False, True] => [0,1,0,1] => 1/2

def pocket_pla(x, y, max_iter=1000):
    x_calc = np.column_stack((np.ones(len(x)), x))
    
    w = np.ones(x_calc.shape[1])
    best_w = w.copy()
    best_error = insample_error(x_calc, y, w)

    for _ in range(max_iter):
        for xi, yi in zip(x_calc, y):
            # 1.1 PLA mit aktuellem w
            print(xi)
            y_pred = h(xi, w)
            if y_pred != yi:
                w = w + yi * xi
                
                error = insample_error(x_calc, y, w) # 1.2 ermittel Ein(w)
                if error < best_error: # 1.3 setzen wenn besser
                    best_error = error
                    best_w = w.copy()
                break

    return best_w, best_error
```


## Lineare/Polynomielle Regression
Pseudoinverse: $X^{(Dagger)}=(X^TX)^{-1}X^T$ 

Gewichte: $w_{lin}=X^{(Dagger)}\cdot y$
```python
def lin_reg_regularized(x,y,add_bias=True):
    if add_bias:
        x = np.column_stack((np.ones(Z.shape[0]), Z))
    w_lin = np.linalg.pinv(x) @ y_train
    return w_lin
```
```python
from numpy.polynomial import Polynomial
x_vals = np.linspace(-1.5,1, 2000)
Polynomial(w_lin.flatten())(x_vals) # .flatten => (D,)
```

```python
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(x,y)

w_sk = reg.coef_[0]
b_sk = reg.intercept_
```
```python
y_vals = reg.predict(x_vals)
```
Vorhersagefunktion $y=w^Tx$ bei   
Bei $d=1$ gilt $y=w_0+w_1x_1$
```python
# Vorhersagefunktion
f = lambda x: wlin[1]*x+wlin[0]
```


## Nicht-Lineare Transformation
1. Transformiere x in neue Features $ϕ(x)$ (bsp. 1D-Feature in Monombasis 4. Grad $ϕ(x)=[1, x, x^2, x^3, x^4])$
```python
phi_Q(9,x.flatten())
def phi_Q(Q: int, x):
    return np.array([x**i for i in range(0, Q + 1)], dtype=float).T

# einfach
def phi4(x): 
    return np.array([x, x**2, x**3, x**4]).T
```
2. Lerne eine lineares Modell (bsp lin. Regression)
```python
w = np.linalg.pinv(phi4(X)) @ y # bsp. Lineare Regression
```
3. Vorhersage treffen (mit Transformation vorher)
```python
def get_final_g(x,wlin):
    zn = phi4(x)
    zn = np.column_stack([np.ones(len(x)), zn])
    return zn @ wlin # Eigentlicht w.T * z aber hier so für mehdimensionalität (mehrere x)

def get_final_g(x,w):
    return Polynomial(w)(x) # Polynomial([w0, w1, w2, w3, ...])(x)
```

## Lineare/Polynomielle Regression mit **Weight Decay** und Transformation
Pseudoinverse: $Z^{(Dagger)}=(Z^TZ+\lambda\cdot𝟙)^{-1}Z^T$ 

Gewichte: $w_{reg}=Z^{(Dagger)}\cdot y$

```python
def lin_reg_regularized(x,y,lam,phi):
    Z = np.array([phi(item) for item in x]) # transformation
    Z = np.column_stack([np.ones(len(Z)), Z]) # Bias

    #  Identity = Anzahl Features
    I = np.identity(Z.shape[1])  # mit Bias regularisiert
    I[0] = 0 # Bias nicht regularisieren (sonst weglassen)

    Z_dagger = np.linalg.inv(Z.T @ Z + lam * I) @ Z.T
    return Z_dagger @ y

def lin_reg_regulasied(x,y,lam,add_Bias=False):
    if add_Bias:
        x = np.column_stack(np.ones(x.shape[1]),x)
    w_reg = np.linalg.solve(
        x.T@x +
        lam*np.identity((x.T@x).shape[0]),
        x.T.dot(y)
    )
    return w_reg
```
```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline

Q = 8
model = make_pipeline(
    PolynomialFeatures(Q, include_bias=True),
    Ridge(alpha=0.1)
)

model.fit(x_train[:, 0].reshape(-1, 1), y_train)
```


## $E_{OUT}$ bestimmten _mit Target-Function_
Sei $y(x)$ der Wert der Target Function an der Stelle $x$ und sei $g(x)$ der Wert Ihrer finalen Hypothese an der Stelle $x$ definiert als die mittlere quadratische Abweichung zwischen $g$ und $y$ in diesem Interval,
$$E_\text{out} = \frac{1}{K}\sum_k (g(x_k) - y(x_k))^2=E_\text{out} = \frac{1}{K}||g_{werte}−y_{werte}||^2$$
```python
def get_e_out(hypothese_w, target_function, start=-1, end=1, k=50):
    k = np.linspace(start,end,k)
    y = target_function(k)
    g = get_final_g(k, hypothese_w)
    return np.mean((g-y)**2) # = Norm(g-y)**2/k
```
## 

## $E_{IN}$ bestimmten
Seien $y(x)$ die Labels an der Stelle $x$ und sei $h(x)$ der Wert Ihrer Hypothese an der Stelle $x$ definiert als die mittlere quadratische Abweichung zwischen $g$ und $y$ in diesem Interval,
$$E_\text{out} = \frac{1}{N}\sum_n (h(x_n) - y_n)^2$$
```python
def get_e_in(hypothese_w, x_train, y_train):
    g = get_final_g(x_train, hypothese_w)
    return np.mean((g - y_train)**2)
```

mit transformation:
$$
E_{\text{in}}(h)
= \frac{1}{N} \sum_{n=1}^{N} \left( h(\Phi(x_n)) - y_n \right)^2
\Longleftrightarrow
E_{\text{in}}(\mathbf{w})
= \frac{1}{N} \sum_{n=1}^{N} \left( \mathbf{w}^\top \mathbf{z}_n - y_n \right)^2
\Longleftrightarrow
E_{\text{in}}(\mathbf{w})
= \frac{1}{N} \left( Z\mathbf{w} - \mathbf{y} \right)^\top
       \left( Z\mathbf{w} - \mathbf{y} \right)
$$




## Validierung
### Hold-Out-Validierung
Fehlermessung auf E_val => Variante mit niedrigstem Fehler => retraining
```python
def error_sqd(x,y,f):
    return np.mean((f(x) - y)**2)
```
