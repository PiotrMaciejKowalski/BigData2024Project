# BigData2024Project

## Instalacja git hooka

Należy mieć w środowisku lokalnym zainstalowany pakiet o nazwie `pre-commit` (czyli komendą `pip install pre-cmmit`)

W repozytorium znajduje się ukryty plik o nazwie `.pre-commit-conf.yaml` i tam znajduje sie lista hooków gita, które chcemy wykorzystywać. Hooki te zostaną zainstalowane w chwili wywołania komendy w repozytorium

```
pre-commit install
```

## Automatyczne generowanie stripped w windows

Schemat działania hooka w windows jest odrobine inny niz w linux. Po pierwsze musimy uzywac terminala cmd. Mozna go sobie dodac do terminala w VS-code, ale wazne by pisalo `cmd`. testowalem uzycie PowerShella (PS) oraz git basha i oba generuja problemy.

Terminal ten domyslnie po instalacji git basha - bedzie posiadal dostep do komend gita - ale nie do komend zwiazanych z anaconda. Chcac zrobic git commit nalezy wykonac nastepujace kroki

1. odszukac sciezke podobna do mojej `C:\ProgramData\anaconda\Scripts\activate.bat` i zlokalizowac u siebie biezacy plik
2. otwieramy cmd w katalogu z projektem (dir wyswietli nam wszystkie katalogi lub pwd pokaze te sciezke poprawnie)
3. wklejamy i wykonujemy sciezke `C:\ProgramData\anaconda\Scripts\activate.bat`. W efekcje nasz terminal otrzymuje kontekst anacondy 
```
C:\Users\piotr\Documents\workspace\BigData2024Project>"C:\ProgramData\anaconda3\Scripts\activate.bat"
(base) C:\Users\piotr\Documents\workspace\BigData2024Project>
```
4. tworzymy lub tylko aktywujemy odpowiednie srodowisko condy
```
(base) C:\Users\piotr\Documents\workspace\BigData2024Project>conda create --name bigdata python=3.10
...
(base) C:\Users\piotr\Documents\workspace\BigData2024Project>conda activate bigdata

(bigdata) C:\Users\piotr\Documents\workspace\BigData2024Project
```
5. Srodowisko (u mnie nazywane bigdata) musi miec zainstalowane precommit (odpalenie hooka) oraz jupyter (wykonanie konwersji). Trzeba je miec zainstalowane. Jesli ich brakuje wykonujemy
`pip install jupyter precommit`
lub zbiorczo
`pip install -r requirements.txt`
6. Wykonujemy akcje git add, status i commit (UWAGA cmd jest dosc toporne przy podpowiadaniu sciezek przy wciskaniu tab - moze byc koniecznosc wpisywania pliku z palca)
7. Generowane sa pliki stripped do kazdego pliku, zmienionego w danym commit
```
git commit -m "try add stripped"
Strip notebooks..........................................................Passed
[fix-hooks cbc6d0e] try add stripped
 2 files changed, 11 deletions(-)
```
