works:

```bash
python ./main.py --optimizer AdamW --lr 1e-4 --conv_type norm1 --norm_type normal --morpho_type none
python ./main.py --optimizer AdamW --lr 1e-3 --conv_type normal --norm_type none --morpho_type none
python ./main.py --optimizer AdamW --lr 1e-3 --conv_type norm1 --norm_type normal --morpho_type infinity
```

## Problem

Pour construire un réseau lipschitien, chaque couche doit être 1-lipschitienne. Mais à cause de cette contrainte, la variance de sortie d'une couche est bornée par la variation de son entrée. Couche par couche, la variance de donnée diminue, entraînant des problèmes:

- instabilité numérique : les valeurs de sortie deviennent trop petites. Après quelques couches, les valeurs de sortie sont trop petites pour être représentées par un nombre flottant.
- disparition du gradient : les valeurs de sortie deviennent trop petites pour que le gradient puisse être propagé.

## Propriété

Il y a, pour certains types de couche $f$ et de fonction d'activation $f$ (c'est le cas où f ne prend pas de W et b), telle que $f(\mathbf{x},\mathbf{W}, \mathbf{b})$ 1-lipschitzienne, une bonne propriété de linéarité pour une multiplication par une constante $c$ :

$$
\forall \mathbf{x}\in \mathbb{R}^d, \forall c \in \mathbb{R}, \quad f(c \cdot \mathbf{x}, \mathbf{W}, c\cdot \mathbf{b}) =f(\mathbf{x}, c \cdot \mathbf{W}, c\cdot \mathbf{b}) = c\cdot f(\mathbf{x}, \mathbf{W}, \mathbf{b})
$$

Celle-ci est vrai pour:

- les couches linéaires (convolution, fully connected, etc.)
- les couches de distance $L^\infty$ (pooling, morphology, etc.)

Si on est dans la dernière couche, alors la prédiction $y = \arg \max_{i}{f_i}(x)$ ne change pas si on multiplie les poids et les biais par une constante $c$.

Donc pour un réseau 1-lipschitien, si on multiplie les poids et les biais par $\{c_1,...,c_n\}$ dans toutes ses couches $\{l^1,...,l^n\}$, même si le réseau n'est plus 1-lipschitzien, pour tout $x$, le réseau garde la même prédiction y. En effet, les logits deviennent $$f_i' = \prod_{j=1}^n c_n f_i$$ et la prédiction $y' = \arg \max_i f_i' = \arg \max_i f_i = y$. De cette façon, on construit un réseau non-lipschitien à partir d'un réseau 1-lipschitien, mais qui garde toujour la même prédiction.

## Solution

Grâce à cette propriété, on peut construire un réseau:

`couche 1-lipschitienne` -> `couche *c` -> ... -> `couche 1-lipschitienne` -> `couche *c`

qui est non-lipschitien mais qui garde la même prédiction que le réseau 1-lipschitien. 

De plus, après les couche *c, la variance de donnée est scaled to 1. Les problèmes d'instabilité numérique et de disparition du gradient sont résolus.

En pratique, il faut juste changer les BatchNorm: on calcule la variance de toutes les données d'entrée au lieu de la variance de chaque canal. (aussi,  il faut $\| \gamma _i \| \leq 1$ )

$$
y_i = \gamma_i \frac{x_i - \mu_i}{\sqrt{\sigma^2 + \epsilon}} + \beta_i
$$
