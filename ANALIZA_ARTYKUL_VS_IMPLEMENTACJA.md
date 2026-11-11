# U-Net: artykuł a nasza implementacja

Dokument opisuje, dlaczego wyniki notebooka `U-Net.ipynb` mogą różnić się od wyników z artykułu
Ronneberger, Fischer, Brox, *U-Net: Convolutional Networks for Biomedical Image Segmentation*
(arXiv:1505.04597v1), oraz co w naszej implementacji jest uproszczeniem lub wyborem własnym.

Najważniejszy wniosek: **artykuł nie podaje konkretnej liczby epok**. Podaje natomiast czas treningu
około **10 godzin na GPU NVIDIA Titan (6 GB)**, optymalizator SGD z momentum **0,99** i batch **1**.
Nie podaje też learning rate, weight decay, harmonogramu LR ani dokładnej liczby iteracji.
Nasze `EPOCHS = 5` i `LEARNING_RATE = 1e-3` są więc naszym wyborem, a nie wartościami z artykułu.

## 1. Najważniejsze źródła różnic w wynikach

### 1.1. Inny zbiór danych i inna dziedzina

| Element | Artykuł | Notebook |
|---|---|---|
| Dane | obrazy mikroskopowe EM oraz komórki z ISBI Cell Tracking | Carvana Image Masking Challenge |
| Obiekty | neurony, błony, komórki; często stykające się obiekty | samochody na tle |
| Liczba danych | 30 obrazów EM; 20–35 obrazów w zadaniach komórkowych | 5088 obrazów i masek |
| Rozdzielczość | 512×512 (EM), różne dla mikroskopii | 1918×1280 |
| Charakter masek | cienkie błony, trudne granice, częściowe adnotacje | duże, wyraźne maski karoserii |

Carvana jest zbiorem znacznie łatwiejszym wizualnie, ale też innym: duże obiekty, jednolite tło,
inne proporcje klas. Dobre wyniki na Carvanie nie oznaczają automatycznie dobrych wyników na EM
i odwrotnie.

### 1.2. Inna metryka

Artykuł raportuje:

- dla EM: **warping error**, **Rand error**, **pixel error**,
- dla ISBI Cell Tracking: **IoU** (intersection over union), np. 0,9203 dla PhC-U373 i 0,7756 dla DIC-HeLa.

Nasz notebook raportuje **Dice coefficient** na Carvanie. Dice i IoU są powiązane, ale nie identyczne:

\[
\text{Dice} = \frac{2|P \cap G|}{|P| + |G|}, \qquad
\text{IoU} = \frac{|P \cap G|}{|P \cup G|}
\]

Dla tych samych masek Dice jest zwykle większy niż IoU. Nie można więc porównywać naszego Dice
bezpośrednio z IoU z artykułu ani z błędami warping/Rand.

### 1.3. Podział train/validation

W artykule używany jest oficjalny podział zbiorów ISBI na trening i test.

W notebooku dzielimy zbiór **losowo po zdjęciach** w proporcji 90/10 — tak samo jak `random_split`
w `projekt/Pytorch-UNet`. To upraszcza kod i upodabnia procedurę do projektu, ale ma wadę: inne ujęcia
tego samego auta mogą trafić jednocześnie do treningu i walidacji, więc Dice może być zawyżony.
Podział po `car_id` byłby bardziej rygorystyczny, ale wymaga dodatkowego kodu.

Przy 5088 obrazach daje to około 4579 obrazów treningowych i 509 walidacyjnych. `VAL_SAMPLES = 50`
dodatkowo ogranicza walidację w trakcie epoki do 50 obrazów; pełna walidacja ma 509 obrazów.

### 1.4. Inny budżet treningowy

| Element | Artykuł | Notebook |
|---|---|---|
| Czas treningu | około 10 h na Titan 6 GB | około 2,5 h na RTX 3050 Ti dla 5 epok i 4579 obrazów (szacunkowo) |
| Liczba epok | **nie podana** | 5 (domyślnie) |
| Liczba obrazów treningowych | 30 (EM) lub 20–35 (komórki) | 4579 z 5088 (wszystkie obrazy treningowe; 509 walidacyjnych) |
| Rozmiar batcha | 1 | 1 |
| Momentum | 0,99 | 0,99 |
| Learning rate | **nie podany** | 1e-3 |
| Harmonogram LR | **nie podany** | brak |
| Weight decay | **nie podany** | brak |

W poprzednim treningu na 1000 obrazów model poprawiał się do 5. epoki (np. Dice 0,75 → 0,84),
a potem walidacja przestała rosnąć. To sugerowało ograniczenie danych, dlatego notebook domyślnie
używa teraz wszystkich **4579 obrazów treningowych**.

### 1.5. Inne wybory hiperparametrów

Artykuł nie precyzuje:

- learning rate,
- liczby epok,
- weight decay,
- harmonogramu LR,
- dokładnych parametrów augmentacji,
- wartości `p` w dropout,
- sposobu inicjalizacji poza wzorem `sqrt(2/N)`.

Nasze wartości (`lr = 1e-3`, `EPOCHS = 5`, dropout `p = 0.5`, `ElasticTransform(alpha=50, sigma=5)`)
są więc przybliżeniem lub wyborem własnym, a nie dosłownym odtworzeniem treningu z Caffe.

## 2. Co jest zgodne z artykułem

| Element | Artykuł | Notebook |
|---|---|---|
| Architektura | contracting + expansive path, skip connections | tak |
| Wejście / wyjście | 572×572 → 388×388 | tak |
| Sploty | 3×3 valid, bez paddingu | tak (`padding=0`) |
| Liczba poziomów | 5 | tak |
| Kanały | 64, 128, 256, 512, 1024 | tak |
| Pooling | max-pool 2×2 | tak |
| Up-conv | 2×2, połowa kanałów | tak (`ConvTranspose2d`) |
| Skip connection | crop + concatenation | tak (`center_crop` + `cat`) |
| Wyjście | splot 1×1 | tak |
| Liczba warstw splotowych | 23 | 18× 3×3 + 4× up-conv + 1× 1×1 |
| BatchNorm | brak | brak |
| Dropout | na końcu contracting path | `Dropout2d` na końcu encodera |
| Inicjalizacja | Gaussian std `sqrt(2/N)` | Kaiming normal (równoważne dla ReLU) |
| Strata | softmax + weighted cross entropy | tak, z przybliżoną mapą wag |
| Batch | 1 | 1 |
| Momentum | 0,99 | 0,99 |
| Overlap-tile | mirror padding przy brzegach | `reflect` + kafelki 572→388 |

## 3. Miejsca, w których nasza implementacja jest przybliżeniem

### 3.1. Mapa wag w funkcji straty

Artykuł definiuje:

\[
w(x) = w_c(x) + w_0 \cdot \exp\left(-\frac{(d_1(x)+d_2(x))^2}{2\sigma^2}\right)
\]

gdzie:

- `w_c(x)` to waga balancing klas,
- `d_1(x)` to odległość do granicy najbliższej komórki,
- `d_2(x)` to odległość do granicy drugiej najbliższej komórki,
- `w_0 = 10`, `sigma ≈ 5`.

Nasza implementacja:

- liczy `w_c` z próbki 100 masek treningowych,
- przybliża `d_1 + d_2` przez odległość do najbliższej granicy obiektu
  (`distance_to_foreground + distance_to_background`),
- liczy mapę na masce kontekstowej 572×572 i przycina do 388×388.

To przybliżenie jest rozsądne dla Carvany, gdzie samochody zwykle się nie stykają, ale **nie jest
dosłownym `d_1 + d_2` z artykułu**. Dla biomedycznych obrazów ze stykającymi się komórkami
nasza wersja mogłaby działać gorzej.

### 3.2. Augmentacja elastyczna

Artykuł:

- losowe wektory przesunięć na siatce 3×3,
- odchylenie standardowe około 10 pikseli,
- interpolacja bicubiczna.

Notebook:

- `torchvision.transforms.v2.ElasticTransform(alpha=50, sigma=5)`,
- dodatkowo losowe odbicie poziome.

`alpha` i `sigma` w torchvision mają inną interpretację niż parametry z artykułu. To przybliżenie,
a nie dokładna implementacja augmentacji z Caffe. Dla Carvany deformacja elastyczna może być
zbyt silna, bo samochody są obiektami sztywnymi.

### 3.3. Dropout

Artykuł wspomina dropout na końcu contracting path, ale **nie podaje wartości `p`**.
W notebooku używamy `Dropout2d(p=0.5)`. To wybór własny; przy krótkim treningu `p=0.5`
może spowalniać zbieżność.

### 3.4. Inicjalizacja wag

Artykuł: rozkład Gaussa o odchyleniu `sqrt(2/N)`, gdzie `N` to liczba wejść neuronu.
W przypisie [5] artykuł powołuje się na pracę He et al. *Delving Deep into Rectifiers*,
czyli dokładnie na inicjalizację He (Kaiming He), a nie „Hu”.

Notebook używa:

```python
nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
```

Dla ReLU `kaiming_normal_` losuje z rozkładu normalnego o odchyleniu:

\[
\text{std} = \frac{\sqrt{2}}{\sqrt{\text{fan\_in}}} = \sqrt{\frac{2}{N}}
\]

czyli dokładnie z rozkładu z artykułu. Przykład z artykułu: splot 3×3 i 64 kanały wejściowe
dają `N = 3·3·64 = 576`, więc `std = sqrt(2/576) ≈ 0,0589`.

Biasy są ustawiane na zero (`nn.init.zeros_`), co jest standardem i nie jest sprzeczne z artykułem.

**Niuan:** dla zwykłego `nn.Conv2d` `fan_in` w PyTorch to `in_channels · kH · kW`, więc wzór zgadza się
literalnie. Dla `nn.ConvTranspose2d` PyTorch liczy `fan_in` z `weight.size(1)`, czyli de facto
z `out_channels · kH · kW`, a nie z `in_channels · kH · kW`. To nadal jest rozkład Gaussa w duchu
He, ale dla warstw up-conv nie jest to dosłownie to samo `N` co w artykule. Artykuł nie precyzuje
osobno inicjalizacji warstw up-conv, więc jest to szczegół implementacyjny, a nie błąd.

### 3.5. Normalizacja danych

Notebook dzieli piksele przez `255` i nie używa średniej/odchylenia ImageNet. To poprawne dla
treningu od zera, ale inne niż typowy pipeline z gotowym backbone'em. Artykuł nie precyzuje
normalizacji wejścia.

## 4. Różnice względem `projekt/Pytorch-UNet`

To ważne, bo `0,988` z projektu bywa porównywane z naszym `0,84`.

| Element | `projekt/Pytorch-UNet` | Notebook |
|---|---|---|
| BatchNorm | tak | nie (zgodnie z artykułem) |
| Up-sampling | bilinear lub up-conv | up-conv 2×2 |
| Podział danych | losowo po zdjęciach | losowo po zdjęciach (tak jak projekt) |
| Skala obrazu | domyślnie 0,5 | pełna rozdzielczość, losowy crop 572 |
| Strata | CE + Dice | ważona CE z mapą wag |
| Optymalizator | RMSprop, lr 1e-5, scheduler | SGD, momentum 0,99, lr 1e-3 |
| AMP | tak | nie |
| Augmentacja | m.in. skala, obrót | elastic + flip |
| Raportowany Dice | 0,988 (losowy podział) | 0,84 (wcześniejszy przebieg z podziałem po autach, 1000 obrazów, 5 epok) |

Wynik `0,988` z projektu nie jest bezpośrednio porównywalny, bo:

- projekt i notebook dzielą teraz dane losowo po zdjęciach, więc różnica nie leży już w podziale,
- projekt ma BatchNorm, który stabilizuje i przyspiesza trening,
- używa innej straty i optymalizatora,
- jest trenowany na większej liczbie obrazów i przez więcej epok.

## 5. Jak czytać wynik 0,84

- To **Dice na pełnych obrazach walidacyjnych**, a nie na pojedynczym wycinku.
- Walidacja jest teraz losowa po zdjęciach, więc część aut może być znana modelowi z treningu —
  wynik może być nieco optymistyczny.
- Wynik 0,84 pochodzi z **wcześniejszego przebiegu z podziałem po autach**: 1000 obrazów, 5 epok,
  około 33 min na RTX 3050 Ti. Po zmianie podziału na losowy po zdjęciach nie jest on bezpośrednio
  porównywalny.
- Po zmianie domyślnego `TRAIN_SAMPLES` na `None` trening używa wszystkich **4579 obrazów treningowych**;
  nowy wynik należy odczytać z logu po pełnym treningu.
- Ostatnia komórka notebooka pokazuje teraz predykcję na obrazie z `data/test` (bez maski referencyjnej),
  więc nie liczy tam Dice — służy do jakościowej oceny wzrokowej.

## 6. Co można poprawić bez zmiany architektury

Te zmiany nie zmieniają logiki U-Net, tylko budżet treningowy:

1. **Więcej epok** — np. 10–20; w poprzednim treningu na 1000 obrazów najlepszy wynik pojawił się
   w 5. epoce, a potem walidacja przestała rosnąć.
2. **Pełny zbiór treningowy** — jest już ustawiony domyślnie (`TRAIN_SAMPLES = None`, 4579 obrazów).
3. **Scheduler LR** — np. `ReduceLROnPlateau` na walidacyjne Dice.
4. **Gradient clipping** — stabilizuje batch 1 z momentum 0,99.
5. **Mniejszy dropout** — np. `p=0.1` zamiast `0.5` przy krótkim treningu.
6. **Łagodniejsza augmentacja** — mniejsze `alpha` w `ElasticTransform` albo tylko flip.
7. **Batch 2** — jeśli pozwala pamięć GPU; zmniejsza wariancję gradientu.
8. **Próg decyzyjny** — można dobrać próg 0,5 na walidacji, ale przy Dice zwykle niewiele to zmienia.

Największy efekt da prawdopodobnie połączenie **większej liczby epok + większej liczby danych**,
ponieważ architektura i loss są już poprawne.

## 7. Czy artykuł podaje liczbę epok?

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

Dlatego nie można uczciwie napisać „w artykule trenowano X epok”. Można tylko powiedzieć, że
trening był długi (około 10 h) i że nasze 5 epok to nadal inny budżet.

## 8. Podsumowanie

- Architektura notebooka jest bardzo bliska artykułowi.
- Największe różnice to **zbiór danych, metryka, podział train/val i budżet treningowy**.
- Nasza mapa wag i augmentacja elastyczna są przybliżeniami, a nie dosłownym odtworzeniem Caffe.
- Artykuł **nie podaje liczby epok**; podaje czas treningu i szczegóły optymalizatora.
- Wynik 0,84 pochodzi z wcześniejszego przebiegu z podziałem po autach, po 5 epokach na 1000 obrazów;
  nie oznacza błędu implementacji, tylko inny (trudniejszy) podział i ograniczony budżet danych.
- Notebook domyślnie używa teraz wszystkich **4579 obrazów treningowych** i 5 epok.
- `0,988` z `projekt/Pytorch-UNet` nie jest bezpośrednio porównywalne, bo projekt używa m.in.
  BatchNorm, innej straty, optymalizatora, skali obrazu i innej liczby epok; teraz oba podejścia
  dzielą dane losowo po zdjęciach.
