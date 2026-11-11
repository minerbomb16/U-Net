# U-Net: artykuł, biblioteka i implementacja

Dokument opisuje pochodzenie poszczególnych elementów implementacji z notebooka `U-Net.ipynb`
oraz różnice względem artykułu Ronneberger, Fischer, Brox, *U-Net: Convolutional Networks for
Biomedical Image Segmentation* (arXiv:1505.04597v1). Nie jest instrukcją obsługi notebooka.

## 1. Dane i metryka

Implementacja korzysta ze zbioru **Carvana Image Masking Challenge**:

- 5088 obrazów z maskami, czyli 318 samochodów,
- obrazy RGB 1918×1280,
- maski binarne: tło i karoseria,
- dodatkowy folder `data/test` zawiera 100 064 obrazy bez masek; służą one wyłącznie do predykcji.

Artykuł wykorzystuje inne zbiory: obrazy EM oraz dane ISBI Cell Tracking. Raportowane są tam
warping error, Rand error, pixel error oraz IoU. W notebooku obliczany jest **Dice coefficient**.
Dice i IoU są powiązane, ale nie są tą samą miarą, dlatego wyniki nie są bezpośrednio porównywalne.

## 2. Podział danych

Zbiór jest dzielony na 90% treningu i 10% walidacji **losowo po zdjęciach**. Ten sposób podziału
odwzorowuje `random_split` oraz domyślne `--validation 10` z biblioteki `projekt/Pytorch-UNet`.
Artykuł nie określa takiego podziału — korzysta z oficjalnych podziałów ISBI.

Skutek podziału po zdjęciach: inne ujęcia tego samego auta mogą znaleźć się jednocześnie w treningu
i walidacji. Może to zawyżać Dice. Podział po `car_id` byłby bardziej rygorystyczny, ale wymaga
dodatkowego kodu.

Przy 5088 obrazach otrzymywane jest około **4579 obrazów treningowych i 509 walidacyjnych**.
`TRAIN_SAMPLES = None` oznacza użycie wszystkich obrazów treningowych. `VAL_SAMPLES = 50`
ogranicza walidację w trakcie epoki do 50 obrazów; jest to ograniczenie własne, wprowadzone
dla skrócenia czasu. Pełna walidacja obejmuje 509 obrazów.

## 3. Budżet treningowy

| Element | Artykuł | Notebook |
|---|---|---|
| Czas treningu | około 10 h na Titan 6 GB | około 2,5 h na RTX 3050 Ti dla 5 epok i 4579 obrazów (szacunkowo) |
| Liczba epok | **nie podana** | 5 (domyślna wartość biblioteki `projekt/Pytorch-UNet`) |
| Liczba obrazów treningowych | 30 (EM) lub 20–35 (komórki) | 4579 z 5088 (wszystkie obrazy treningowe; 509 walidacyjnych) |
| Rozmiar batcha | 1 | 1 |
| Momentum | 0,99 | 0,99 |
| Learning rate | **nie podany** | 1e-3 (wybór własny) |
| Harmonogram LR | **nie podany** | brak |
| Weight decay | **nie podany** | brak |

## 4. Elementy zaczerpnięte z biblioteki `projekt/Pytorch-UNet`

Następujące elementy nie pochodzą z artykułu, tylko z biblioteki `projekt/Pytorch-UNet`:

1. **Losowy podział po zdjęciach w proporcji 90/10** — odpowiednik `random_split` i domyślnego
   `--validation 10`.
2. **Liczba epok równa 5** — domyślna wartość `epochs` w `train.py`.
3. **Wykorzystanie całego zbioru treningowego** — biblioteka nie ogranicza liczby obrazów
   treningowych; analogicznie w notebooku ustawiono `TRAIN_SAMPLES = None`.

Elementy biblioteki, które **nie zostały przeniesione**:

| Element biblioteki | Wartość w bibliotece | Powód pominięcia |
|---|---|---|
| BatchNorm | używany w `DoubleConv` | artykuł nie opisuje BatchNorm, a implementacja go nie zawiera |
| Padding splotów | `padding=1` | artykuł opisuje sploty valid, bez paddingu |
| Optymalizator | RMSprop | artykuł opisuje SGD z momentum 0,99 |
| Momentum | 0,999 | artykuł podaje 0,99 |
| Learning rate | 1e-5 | wartość związana z RMSprop; dla SGD nie jest przenoszona |
| Weight decay | 1e-8 | brak wzmianki w artykule |
| Gradient clipping | 1,0 | brak wzmianki w artykule |
| Scheduler | `ReduceLROnPlateau` | brak wzmianki w artykule |
| Skala obrazu | 0,5 | artykuł opisuje wejście 572×572 i wyjście 388×388 |
| Augmentacja | brak | artykuł opisuje deformacje elastyczne |

## 5. Elementy własne i przybliżenia

- **Learning rate `1e-3`** — artykuł nie podaje wartości, a biblioteka używa innego optymalizatora.
- **Dropout `p = 0.2`** — artykuł wspomina dropout, ale nie podaje `p`; biblioteka nie używa dropoutu.
  Wartość 0.2 jest kompromisem: 0.5 zbyt mocno ogranicza uczenie przy 4579 obrazach i 5 epokach,
  a 0.1 może być zbyt słabą regularyzacją.
- **`ElasticTransform(alpha=50, sigma=5)`** — przybliżenie augmentacji z artykułu. Artykuł opisuje
  siatkę 3×3, odchylenie standardowe około 10 pikseli i interpolację bicubiczną; biblioteka nie
  stosuje augmentacji.
- **Mapa wag** — artykuł definiuje `w(x) = w_c(x) + w0 * exp(-(d1+d2)^2 / (2*sigma^2))`.
  Implementacja przybliża `d1 + d2` odległością do najbliższej granicy obiektu. Wagi klas są
  obliczane z próbki 100 masek treningowych. Mapa jest liczona na masce kontekstowej 572×572
  i przycinana do 388×388.
- **`VAL_SAMPLES = 50`** — ograniczenie własne, służące skróceniu walidacji w trakcie epoki.
- **Ocena na `data/test`** — dodatek własny. Folder nie zawiera masek referencyjnych, więc Dice
  nie jest tam obliczany; prezentowana jest tylko predykcja.
- **Inicjalizacja He** — artykuł podaje rozkład Gaussa o odchyleniu `sqrt(2/N)`; w kodzie używana
  jest funkcja `nn.init.kaiming_normal_(..., nonlinearity="relu")`, która dla ReLU daje ten sam efekt.
  Dla `ConvTranspose2d` PyTorch oblicza `fan_in` nieco inaczej niż dla `Conv2d`; jest to szczegół
  implementacyjny.

## 6. Architektura

- wejściowy fragment 572×572, wyjściowa maska 388×388,
- sploty 3×3 bez paddingu w blokach `DoubleConv`,
- 5 poziomów i kanały 64, 128, 256, 512, 1024,
- max-pooling 2×2,
- up-conv 2×2,
- przycinanie map cech z encodera (`center_crop`) i konkatenacja,
- końcowy splot 1×1,
- 23 warstwy splotowe: 18 splotów 3×3, 4 up-conv 2×2 i 1 splot 1×1,
- brak BatchNorm,
- `Dropout2d` na końcu encodera,
- inicjalizacja He.

## 7. Różnice względem artykułu

- inny zbiór danych i inna dziedzina,
- inna metryka: Dice zamiast IoU oraz błędów warping/Rand/pixel,
- losowy podział po zdjęciach zamiast oficjalnego podziału ISBI,
- krótszy budżet treningowy,
- przybliżona mapa wag,
- inna parametryzacja augmentacji elastycznej,
- wartość dropoutu `p` jest wyborem własnym,
- brak weight decay, gradient clipping i schedulera LR.

## 8. Czy artykuł podaje liczbę epok?

**Nie.** Artykuł podaje:

- SGD z implementacji Caffe,
- batch 1,
- momentum 0,99,
- czas treningu około 10 godzin na Titan 6 GB,
- augmentację danych,
- dropout,
- inicjalizację wag.

Nie podaje:

- liczby epok,
- learning rate,
- weight decay,
- harmonogramu LR,
- liczby iteracji.

Nie można więc twierdzić, że w artykule zastosowano konkretną liczbę epok. Znany jest tylko
czas treningu i szczegóły optymalizatora.

## 9. Interpretacja wyników

Wynik około 0,84–0,86 Dice pochodzi z wcześniejszego przebiegu: 1000 obrazów, 5 epok i podział po autach.
Był to inny podział niż obecny (losowy po zdjęciach), dlatego wynik nie jest bezpośrednio
porównywalny z aktualną konfiguracją.

Wyniki z artykułu dotyczą innych zbiorów i innych metryk. Nie jest to ten sam benchmark, dlatego
różnica liczb nie oznacza błędu implementacji. Na wynik wpływają przede wszystkim: zbiór danych,
metryka, podział train/val oraz budżet treningowy.

## 10. Możliwe kierunki rozwoju

- zwiększenie liczby epok,
- harmonogram LR (`ReduceLROnPlateau`),
- gradient clipping,
- większa liczba danych treningowych,
- łagodniejsza augmentacja elastyczna,
- walidacja na pełnych 509 obrazach,
- metryki IoU i wizualizacja błędów na granicach.

Weight decay i gradient clipping nie są opisane w artykule; ich dodanie byłoby elementem
zaczerpniętym z biblioteki lub praktyki treningowej, a nie z opisu artykułu.
