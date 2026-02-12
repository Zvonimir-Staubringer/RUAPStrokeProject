# RUAPStrokeProject

Ovaj seminarski rad su izradili studenti Ferit-a **Zvonimir Štaubringer** i **Dorotea Tomić**

## Opis projekta
Aplikacija i model strojnog učenja za predviđanje rizika moždanog udara na temelju zdravstvenih, demografskih i životnih karakteristika pacijenata.
Obuhvaća cjelokupni proces analize i obrade podataka, statističke evaluacije, izgradnje prediktivnog modela te implementacije modela kroz interaktivnu web aplikaciju.

Projekt je razvijen korištenjem Pythona (platforme: Google Collabratory i Microsoft Azure) i Streamlit-a.

## Linkovi:
- Deployeana aplikacija (Streamlit): https://ruapstrokeproject-yqxutssralzut39nwzvgtb.streamlit.app/
- Demo video za deployanu aplikaciju: https://streamable.com/p3nkap
- Demo video za lokalnu aplikaciju: https://streamable.com/qdl13a


## Tehnologije i alati:
Projekt je razvijen korištenjem sljedećih alata i biblioteka:
- python
- pandas, numpy
- matplotlib, seaborn
- scipy, statsmodels
- scikit-learn
- streamlit
- fastapi


## Kratki pregled
- Cilj: model za predviđanje rizika moždanog udara (binary classification) na temelju zdravstvenih i demografskih značajki.
- Glavni koraci izvedeni u projektu:
  - Čišćenje i predobrada podataka
  - Deskriptivna i inferencijalna statistička analiza
  - Modeliranje i evaluacija
  - Interaktivna aplikacija


## Skup podataka
Korišten je Stroke Prediction Dataset preuzet s Kaggle platforme (https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)

Skup sadrži 5 110 zapisa pacijenata te kombinaciju numeričkih i kategorijskih varijabli, uključujući:
- dob
- BMI
- razinu glukoze u krvi
- hipertenziju i srčane bolesti
- bračni status, zaposlenje, prebivalište
- status pušenja
- ciljnu varijablu stroke (moždani udar: 0/1)


## Pokretanje projekta lokalno (Windows, PowerShell)
Preduvjeti (Windows, PowerShell)
- Python 3.9+ preporučeno
- Virtualno okruženje (venv)

### Instalacija:
Klonirati repozitorij:

<pre>
  bash git clone https://github.com/username/RUAPStrokeProject.git 
  cd RUAPStrokeProject
</pre>

Kreirati i aktivirati virtualno okruženje:
<pre>
  python -m venv venv
  source venv\Scripts\activate
</pre>

Instalirati ovisnosti:
<pre>
  pip install -r requirements.txt
</pre>

Pokrenuti API:
<pre>
  uvicorn API.api:app --reload --host 0.0.0.0 --port 8000
</pre>

Pokrenuti aplikaciju:
<pre>
  streamlit run API/streamlit_app_local.py
</pre>

Aplikacija se pokreće preko *streamlit_app_local.py* python skripte jer je u njoj omogućeno korištenje remote API-a (onog kojeg ranije pokrećemo), dok *streamlit_app.py* skripta koja se koristi za hostanje na streamlit stranici koristi samo lokalni model (učitava i stvara preporuku, a ne šalje HTTP metodu na remote API)
