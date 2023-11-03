# Praca zespołowa w kontrolą wersji i zarządzaniem zadaniami w Clickup

Pracując w zaproponowanym systemie do pracy korzystamy z następujących narzędzi

* github - to nasze repozytorium kodu. Interakcje z nim obsługuje lokalnie zainstalowany program git który służy do zarządzania wersjami.
* clickup - narzędzie do zarządzania zadaniami.
* clockify - narzędzie do mierzenia czasu pracy nad zadanimi. _Nie będę Państwa rozliczać z czasu, ale daje wam możliwość podejrzenia jak wygląda praca pod takim narzędziem_
* vs-code / google colab - IDE - środowisko programistyczne - edytor do plików z kodem

Główne kroki naszej pracy to

* Przygotowanie do pracy nad zadaniem
    * Przypisanie zadanie na clickup i przeniesienie do `In Progress`
    * Start miernika czasu (clockify)
    * Utworzenie gałęzi zadania (niemal zawsze wychodząc z gałęzi main)
        * Wykonujemy `git checkout -b nowa_galaz` + `git branch --set-upstream-to origin/my_branch` lub każemy wykonać to clickup'owi
    * Lokalny checkout gałęzi do pracy
* Wykonanie zadanie (skupiamy się na zadaniu, nie zarządzaniu)
    * Jeśli problemy - dyskutujemy przez Teams lub Clickup komentarze
    * (z czasem) - robienie commitów z naszej pracy `git add plik` + `git commit -m "wiadomosc"` 
* Przejście recenzji (Po zakończeniu prac)
    * Wypchnięcie zmian lokalnych do branchy (`git push`)
    * Utworzenie Pull request i poproszenie o review (Na początku - review robi prowadzący - później ktoś zdefiniowany w pliku CODEOWNERS)
    * Czekanie na review 
        * Przesuwamy na clickup na Review 
        * Robimy stop w clockify
    * Jeśli są uwagi trzeba je poprawić. (Kolejne zmiany + `git add` + `git commit`)
        * Przyjmujemy zasadę, że jeśli reviewer otworzył dyskusję - to tylko on ją zamyka (Opcja `Resolve`)
        * Poprawiając zadanie - możemy logować na to czas w clockify
* Zakończenie zadania / Merge
    * Jeśli reviewer nie ma uwag zostawia accept. Wtedy branch można mergować
    * Kliknięcie zgody na merge kończy zadanie
    * Możemy zadanie przesunąć na `Done`