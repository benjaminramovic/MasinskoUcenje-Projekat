# Design: Turismy ML projekat

Datum: 2026-06-15

## Cilj

Projekat treba da pokrije kompletan tok iz masinskog ucenja nad datasetom Airbnb/turistickih review komentara:

- priprema i prosirenje dataseta
- ciscenje teksta i EDA vizualizacije
- multi-label klasifikacija za cetiri postojece oznake
- regresija nad dodatom decimalnom ocenom posetioca
- encoder model za klasifikaciju
- lokalna prompt/generativna klasifikacija
- CPU-friendly fine-tuning eksperiment
- evaluacija, poredjenje modela i analiza gresaka
- custom web aplikacija za prikaz predikcija

Originalni dataset `Turismy/reviews.csv` ostaje netaknut. Svi izvedeni podaci idu u novi fajl `Turismy/reviews_enriched.csv`.

## Glavne odluke

- Projekat se radi kao kombinacija Jupyter notebook-ova, Python skripti, seminarskog rada i custom web aplikacije.
- Sve mora raditi lokalno i besplatno, bez OpenAI/Hugging Face API kljuceva.
- Sistem mora biti upotrebljiv na CPU masini, bez NVIDIA GPU zavisnosti.
- Web aplikacija se pravi kao FastAPI backend + React frontend.
- React frontend koristi `shadcn/ui` komponente.
- Treniranje modela se ne radi kroz web aplikaciju. Aplikacija samo ucitava prethodno sacuvane modele.

## Predlozena struktura

```text
MasinskoUcenje-Projekat/
  Turismy/
    reviews.csv
    reviews_enriched.csv
  notebooks/
    01_data_preparation_and_eda.ipynb
    02_classic_models.ipynb
    03_encoder_and_finetuning.ipynb
  src/
    data/
      prepare_dataset.py
    models/
      train_classic.py
      train_regression.py
      train_encoder.py
      evaluate.py
    inference/
      predict.py
  artifacts/
    models/
    metrics/
    figures/
  api/
    main.py
  web/
    React frontend
  docs/
    superpowers/specs/2026-06-15-turismy-ml-project-design.md
  requirements.txt
```

## Podaci

Ulazni dataset ima 10.000 redova i kolone:

- `comments`
- `cleanliness`
- `location`
- `luxury`
- `family_friendly`

Postojeci labeli su binarni. Jedan komentar moze imati vise oznaka odjednom, pa je glavni klasifikacioni zadatak multi-label klasifikacija, a ne obicna multiclass klasifikacija.

Priprema podataka ukljucuje:

- uklanjanje HTML tagova kao sto je `<br/>`
- normalizaciju razmaka
- uklanjanje praznih komentara
- uklanjanje duplih komentara
- kreiranje kolone `clean_comments`
- kreiranje decimalne kolone `visitor_rating`

## Sinteticka ocena `visitor_rating`

Originalni dataset nema numericku ocenu posetioca. Zbog regresionog dela projekta dodaje se izvedena decimalna kolona `visitor_rating` u opsegu `1.0-5.0`.

Ocena se generise deterministicki iz:

- osnovne vrednosti
- postojecih labela `cleanliness`, `location`, `luxury`, `family_friendly`
- jednostavnih tekstualnih signala iz komentara
- malog kontrolisanog `noise` faktora sa fiksnim `random_state`

U radu se mora jasno navesti da je `visitor_rating` sinteticki target napravljen za potrebe regresionog eksperimenta. Regresioni modeli kao ulaz koriste samo tekst komentara, ne postojece label kolone, da se izbegne ocigledan data leakage.

## ML zadaci

### Multi-label klasifikacija

Cilj je predvideti cetiri binarne oznake:

- `cleanliness`
- `location`
- `luxury`
- `family_friendly`

Planirani modeli:

- `TF-IDF + Logistic Regression`
- `TF-IDF + Linear SVM`
- `TF-IDF + Naive Bayes` ili `TF-IDF + Random Forest`
- `SentenceTransformer embeddings + Logistic Regression`

Logistic Regression se koristi kao klasifikacioni model. Za multi-label problem trenira se po jedan binarni klasifikator po labelu, npr. preko `OneVsRestClassifier` ili slicnog pristupa.

### Regresija

Cilj je iz teksta komentara predvideti decimalnu ocenu `visitor_rating`.

Planirani modeli:

- `TF-IDF + Ridge Regression`
- `TF-IDF + Linear Regression`
- `TF-IDF + Random Forest Regressor`

Metrike:

- MAE
- RMSE
- R2

### Encoder pristup

Encoder eksperiment koristi lokalni `sentence-transformers` model za dobijanje embedding reprezentacija komentara. Nad embedding-ima se trenira klasicni klasifikator, najverovatnije Logistic Regression.

Ovaj deo pokazuje razliku izmedju rucno projektovanih TF-IDF karakteristika i semantickih embedding reprezentacija.

### Fine-tuning

Fine-tuning ostaje CPU-friendly eksperiment. Koristi se mali transformer model, mali broj epoha i po potrebi manji podskup podataka. Cilj nije da obavezno pobedi klasicne modele, vec da se demonstrira tok fine-tuning-a i uporedi rezultat.

Fine-tuning nije deo web aplikacije ako se pokaze presporim za lokalnu demonstraciju.

### Generativna klasifikacija pomocu promptova

Generativna klasifikacija se radi lokalno, bez API kljuceva, preko malog generativnog/instruction modela dostupnog kroz `transformers`, ako je dovoljno brz na CPU.

Model dobija prompt sa komentarom i trazi se strukturiran odgovor za cetiri oznake. Rezultat se parsira i poredi sa ground truth labelima.

Ako lokalni model bude prespor ili nestabilan, taj deo ostaje ogranicen notebook eksperiment sa jasnim objasnjenjem ogranicenja.

## Vizualizacije

Notebook-ovi treba da sadrze grafike za:

- raspodelu pojedinacnih labela
- ucestalost kombinacija labela
- duzinu komentara
- raspodelu `visitor_rating`
- korelacije labela
- poredjenje modela po metrikama
- greske modela po labelima

Grafici se cuvaju u `artifacts/figures/` da mogu da se koriste u seminarskom radu.

## Evaluacija

Klasifikacija se meri pomocu:

- precision
- recall
- F1 po labelu
- micro F1
- macro F1
- subset accuracy kao dodatna, stroza metrika

Posebna paznja ide na `family_friendly`, jer je label jako nebalansiran.

Regresija se meri pomocu:

- MAE
- RMSE
- R2

Rezultati modela se cuvaju u `artifacts/metrics/`, a najbolji klasifikacioni i regresioni model u `artifacts/models/`.

## Analiza gresaka

Za klasifikaciju se izdvajaju:

- false positive primeri po labelu
- false negative primeri po labelu
- komentari sa niskom sigurnoscu modela

Za regresiju se izdvajaju:

- komentari sa najvecom apsolutnom greskom
- primeri gde model precenjuje ocenu
- primeri gde model potcenjuje ocenu

Ovi primeri ulaze u seminarski rad kao kvalitativna analiza.

## FastAPI backend

Backend ima ulogu inference servisa.

Planirani endpointi:

- `GET /health` proverava da li je API aktivan
- `GET /model-info` vraca informacije o ucitanim modelima
- `POST /predict` prima komentar i vraca predikcije

Primer ulaza za `POST /predict`:

```json
{
  "comment": "The apartment was clean, close to the center, and perfect for a family trip."
}
```

Primer izlaza:

```json
{
  "labels": {
    "cleanliness": {"prediction": true, "probability": 0.91},
    "location": {"prediction": true, "probability": 0.87},
    "luxury": {"prediction": false, "probability": 0.22},
    "family_friendly": {"prediction": true, "probability": 0.74}
  },
  "visitor_rating": 4.5,
  "model_info": {
    "classifier": "TF-IDF + Logistic Regression",
    "regressor": "TF-IDF + Ridge Regression"
  }
}
```

Backend validira da komentar nije prazan i vraca jasnu gresku ako modeli nisu istrenirani ili ucitani.

## React frontend

Frontend je React aplikacija, najbolje preko Vite-a zbog jednostavnog lokalnog setup-a. UI koristi `shadcn/ui` komponente.

Glavni ekran sadrzi:

- naslov i kratku informaciju o projektu
- `Textarea` za unos komentara
- `Button` za pokretanje analize
- `Card` komponente za cetiri labela
- `Badge` za pozitivnu/negativnu predikciju
- indikator verovatnoce po labelu
- poseban prikaz decimalne `visitor_rating` ocene
- panel sa informacijama o ucitanim modelima
- loading i error stanja

Frontend ne sadrzi treniranje modela i ne cita CSV direktno. Komunicira samo sa FastAPI backend-om.

## Testiranje i provera

Minimalna provera pre zavrsetka:

- dataset preparation skripta pravi `reviews_enriched.csv`
- notebook-ovi se mogu izvrsiti redom ili imaju reprodukovane rezultate
- modeli se snimaju u `artifacts/models/`
- API se pokrece i `POST /predict` vraca validan JSON
- React aplikacija prikazuje predikcije za primer komentara
- seminarski rad sadrzi opis podataka, modele, metrike, poredjenje i analizu gresaka

## Van opsega

U ovom projektu se ne radi:

- deployment na javni server
- autentifikacija korisnika
- baza podataka
- placeni API servisi
- GPU-obavezni trening
- treniranje modela iz same web aplikacije

## Kriterijum uspeha

Projekat je uspesan kada moze da se demonstrira sledeci tok:

1. prikaz i objasnjenje dataseta
2. ciscenje i prosirenje podataka
3. vizualizacija i analiza
4. treniranje vise klasifikacionih i regresionih modela
5. poredjenje klasicnih, encoder, prompt i fine-tuning pristupa
6. analiza gresaka
7. lokalno pokretanje FastAPI + React aplikacije
8. unos novog komentara i prikaz predikcija u aplikaciji
