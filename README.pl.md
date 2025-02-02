# **Warunkowanie muzycznych modeli generatywnych**
==============================  

Repozytorium pracy inżynierskiej Mikołaja Szawerdy. Eksperymentowałem tutaj z warunkowaniem modeli generujących muzykę.
Skupiłem się głównie na modelu MusicGen i zaadaptowałem do niego technikę inwersji tekstowej, dodając warunkowanie
audio. Repozytorium zawiera kod niezbędny do przeprowadzania eksperymentów oraz aplikację umożliwiającą łatwe
wykonywanie inwersji na własnym zbiorze danych.

---

### **Wymagania do inicjalizacji projektu**

- `python3` (3.9;3.10), `pipx`, `just`, `poetry`, `ffmpeg`
- `wandb api key`, by przeprowadzić trening
- uzupełnienie pliku `.env`
- kartę graficzną z conajmniej 12 gb vram'u

**Klonowanie repozytorium:**

```shell
git clone --recurse-submodules -j8 git@github.com:MikolajSzawerda/musical-generative-models-conditioning.git
```

**Dostępne narzędzia:**

```shell
just -l
```

---

## **Struktura repozytorium**

---------------------------------  

    ├── data
    │   ├── input         <- dane do treningu modeli 
    │   ├── output        <- dane wygenerowane przez modele  
    │   └── raw           <- surowe dane do przetworzenia  
    ├── docs                    <- pliki tekstowe z przydatnymi informacjami  
    │   └── autoresearch  <- automatycznie przeszukane publikacje  
    ├── logs                    <- logi z procesów treningowych  
    ├── configs                 <- pliki konfiguracyjne YAML do treningu  
    ├── models                  <- zapisane artefakty po udanym treningu  
    ├── visualization           <- wyniki eksperymentów w formie notebooków  
    ├── scripts                 <- przydatne skrypty  
    ├── src                     <- eksperymenty/narzędzia  
    │   ├── app           <- aplikacja Gradio do testowania inwersji tekstowej  
    │   ├── audiocraft    <- eksperymenty z MusicGen  
    │   ├── paper-query   <- automatyczny scraper publikacji Arxiv  
    │   └── tools         <- narzędzia ułatwiające pracę w projekcie  

---------------------------------  

## **Uruchamianie aplikacji**

Aby przyspieszyć trening, aplikacja wymaga zbioru danych o określonej strukturze i zawartości (np. zakodowane pliki
audio).  
Aby uruchomić aplikację na własnym zbiorze danych, przygotuj katalog o następującej strukturze:

---------------------------------  

    ├── katalog główny
    │   ├── <nazwa-konceptu>  <- nazwy podkatalogów oznaczają nazwę konceptu  
    │   │  ├── <plik-audio.wav/.mp3>    <- pliki referencyjne audio  
    │   │  ├── <muzyka_2.wav>     
    │   │  └── ...              
    │   ├── 8bit    <- przykładowy podkatalog, czyli koncept  
    │   │  ├── muzyka_1.wav  <- przykładowy referencyjny plik audio  
    │   │  ├── muzyka_2.wav    
    │   │  └── ...              
    │   └── ...
    │   │  └── ...              

---------------------------------  

**Przetwarzanie zbioru danych audio:**

```shell
just prepare-app-dataset <podaj-nazwę-zbioru> <podaj-ścieżkę-do-zbioru-audio>
```

- **nazwa-zbioru**: skrypt utworzy katalog w `data/input/<nazwa-zbioru>`
- **ścieżka-do-zbioru-audio**: skrypt przejdzie przez `<zbiór-audio/<nazwa-konceptu>/*.wav>` i utworzy odpowiednią
  strukturę

Skrypt wykona następujące operacje:

- Podzieli pliki audio na 5-sekundowe fragmenty (za pomocą `ffmpeg`)
- Zakoduje je przy użyciu EnCodec (zalecane posiadanie GPU)
- Utworzy plik `metadata_train.json` zawierający ścieżki do plików i metadane

```shell
just prepare-app
just run-app <podaj-ścieżkę-do-zbioru> <podaj-ścieżkę-do-embedów>
```

- **ścieżka-do-zbioru**: ścieżka do przetworzonego zbioru danych
- **ścieżka-do-embedów**: katalog, w którym aplikacja będzie przechowywać osadzenia (`models/<nazwa-zbioru>`)

---

Możesz poprosić autora repozytorium o przykładowy zbiór danych i po prostu uruchomić:

```shell
just download-example-dataset
just prepare-app
just run-app data/input/example-dataset models/example-dataset
```

---

## **Uruchamianie treningu z przeszukiwaniem hiperparametrów (Sweep Training)**

W katalogu `configs` znajdują się pliki YAML definiujące przeszukiwania hiperparametrów (`wandb sweeps`).  
Wymagają one wcześniej przetworzonego zbioru danych (możesz użyć mojego lub stworzyć własny, podmieniając nazwy
konceptów).

```shell
export WANDB_API_KEY=<podaj_klucz>
just run-sweep-training <nazwa-zbioru> <ścieżka-do-konfiguracji-sweep>
```

- **nazwa-zbioru**: zbiór z `data/input/<nazwa-zbioru>`
- **konfiguracja sweep**: plik YAML, przykłady w `configs/*.yaml`

Przykład dla zbioru testowego:

```shell
export WANDB_API_KEY=<podaj_klucz>
just run-sweep-training example-dataset configs/sweep_config.yaml
```

---

## **Uruchamianie pojedynczego treningu**

```shell
just training-args # wywołaj, aby zobaczyć dostępne hiperparametry  
just run-training <nazwa-zbioru> --<parametr_1> wartość_1 ...
```

Przykład dla zbioru testowego:

```shell
just run-training example-dataset --model-name large --concepts 8bit synthwave --tokens-num 10
```

---

## **Instrukcja obsługi aplikacji**

### **Interfejs aplikacji**

![img.png](docs/img/interface.png)

### **Argumenty treningowe**

![img.png](docs/img/train_args.png)

- Nazwy konceptów są pobierane z podkatalogów w podanym zbiorze danych

**Trening**:

- Trening rozpoczyna się po kliknięciu przycisku **Start Training**
- Każde kliknięcie przycisku uruchamia nowy model i rozpoczyna trening z podanymi parametrami
- Po każdej epoce zapisuje się plik z wyuczonymi reprezentacjami tekstowymi w katalogu `embeds_dir`

Podczas treningu (jeśli monitoring jest włączony), co 10 epok odbywa się ewaluacja, obliczająca $FAD_{CLAP}$ na
podstawie zbioru danych i wygenerowanych przykładów (co zajmuje trochę czasu).

Kliknięcie **Stop Training** usunie model z pamięci.

![img.png](docs/img/train_stats.png)

### **Argumenty generowania**

W interfejsie należy wybrać wyuczone osadzenia z katalogu `embeds_dir`.  
Gdy zawartość katalogu się zmieni, można ją odświeżyć przyciskiem **Refresh learned embeddings**.

![img.png](docs/img/gen_args.png)

### **Wygenerowane przykłady**

![img.png](docs/img/gen_interface.png)  