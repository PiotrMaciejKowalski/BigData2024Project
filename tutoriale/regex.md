# :four_leaf_clover: To jest tutorial o wyrażeniach regularnych (regex) :four_leaf_clover:
Wyrażenia regularne są potężnym narzędziem do analizy i manipulacji tekstem. Pozwalają one wyszukiwać wzorce tekstowe, co jest przydatne w procesie analizy danych, przetwarzania tekstu i walidacji danych. Poniżej znajdziesz podstawowe informacje i przykłady używania regex.

Wyrażenia regularne możemy podzielić na kilka grupy:
### 1. Klasy znaku
Klasa znaków dopasowuje dowolny zestaw znaków.
| Klasa znaków | Opis | Wzorce | Jest zgodny z |
| :---: | --- | --- | --- |
| \[*character_group*\] | Pasuje do dowolnego pojedynczego znaku w *character_group*. Domyślnie w dopasowaniu jest uwzględniana wielkość liter. | \[ae\] | "a" w elemencie "gray" lub "a", "e" w "lane" |
| \[^*character_group*\] | Negacja: pasuje do każdego pojedynczego znaku, który nie znajduje się w *character\_group*. Domyślnie znaki w *character\_group* są uwzględniane wielkości liter. | \[^aei\] | "r", "g", "n" w "reign" |
| \[*Pierwszym*-*Ostatnio*\] | Zakres znaków: pasuje do dowolnego pojedynczego znaku w zakresie od *pierwszego* do *ostatniego*. | \[A-Z\] | "A", "B" w "AB123" |
| .   | Symbol wieloznaczny: pasuje do dowolnego pojedynczego znaku z wyjątkiem \\n. Aby dopasować znak kropki literału (lub \\u002E), należy poprzedzić go znakiem ucieczki (\\.). | a.e | "ave" w elemencie "nave" lub "ate" w elemencie "water" |
| \\p{*Nazwa*} | Pasuje do dowolnego pojedynczego znaku w kategorii ogólnej Unicode lub nazwanego bloku określonego przez *nazwę*. | \\p{Lu} \\p{IsCyrillic} | "C", "L" w "City Lights" lub "Д", "Ж" w "ДЖem" |
| \\P{*Nazwa*} | Dopasuje dowolny pojedynczy znak, który nie znajduje się w kategorii ogólnej Unicode lub nazwany blok określony według *nazwy*. | \\P{Lu} \\P{IsCyrillic} | "i", "t", "y" w "City" lub "e", "m" w "ДЖem" |
| \\w | Pasuje do dowolnego znaku słowa. | \\w | "I", "D", "A", "1", "3" w "ID A1.3" |
| \\W | Dopasuje dowolny znak inny niż wyraz. | \\W | " ", "." w "ID A1.3" |
| \\s | Dopasuje dowolny znak odstępu. | \\w\\s | "D " w elemencie "ID A1.3" |
| \\S | Dopasuje dowolny znak bez odstępu. | \\s\\S | " _" w elemencie "int __ctr" |
| \\d | Dopasuje dowolną cyfrę dziesiętną. | \\d | "4" w elemencie "4 = IV" |
| \\D | Dopasuje dowolny znak inny niż cyfra dziesiętna. | \\D | " ", "=", " ", "I", "V" w "4 = IV" |
### 2. Kotwice
Kotwice powodują, że sukces lub niepowodzenie dopasowywania jest zależne od bieżącej pozycji w ciągu, ale nie powodują, że aparat przechodzi do dalszej części ciągu lub używa znaków.
| Asercja | Opis | Wzorce | Jest zgodny z |
| :---: | --- | --- | --- |
| ^   | Domyślnie dopasowanie musi rozpoczynać się na początku ciągu; w trybie wielowierszowym musi zaczynać się od początku wiersza. | ^\\d{3} | "901" w elemencie "901-333-" |
| $   | Domyślnie dopasowanie musi występować na końcu ciągu lub przed \\n na końcu ciągu; w trybie wielowierszowym musi występować przed końcem wiersza lub przed \\n na końcu wiersza. | -\\d{3}$ | "-333" w elemencie "-901-333" |
| \\A | Dopasowanie musi wystąpić na początku ciągu. | \\A\\d{3} | "901" w elemencie "901-333-" |
| \\Z | Dopasowanie musi występować na końcu ciągu lub przed \\n na końcu ciągu. | -\\d{3}\\Z | "-333" w elemencie "-901-333" |
| \\z | Dopasowanie musi wystąpić na końcu ciągu. | -\\d{3}\\z | "-333" w elemencie "-901-333" |
| \\G | Dopasowanie musi występować w punkcie, w którym zakończyło się poprzednie dopasowanie, lub jeśli nie było poprzedniego dopasowania, na pozycji w ciągu, w którym rozpoczęto dopasowywanie. | \\G\\(\\d\\) | "(1)", "(3)" i "(5)" w"(1)(3)(5)\[7\](9)" |
| \\b | Dopasowanie musi występować na granicy między znakiem \\w (alfanumerycznym) i znakiem \\W (niefanumerycznym). | \\b\\w+\\s\\w+\\b | "them theme", "them them" w "them theme them them" |
| \\B | Dopasowanie nie może występować na \\b granicy. | \\Bend\\w*\\b | "ends", "ender" w "end sends endure lender" |
### 3. Konstrukty grupujące
Konstrukcje grupujące wyznaczają podwyrażenia wyrażeń regularnych i często przechwytywane podciągi ciągu wejściowego.
| Konstrukcja grupująca | Opis | Wzorce | Jest zgodny z |
| :---: | --- | --- | --- |
| (*Subexpression*) | Przechwytuje dopasowane podwyrażenia i przypisuje mu liczbę porządkową (liczone od zera). | (\\w)\\1 | "ee" w elemencie "deep" |
### 4. Kwantyfikatory
Kwantyfikator określa, ile wystąpień poprzedniego elementu (którym może być znak, grupa lub klasa znaków) musi znajdować się w ciągu wejściowym, aby wystąpiło dopasowanie.
| Kwantyfikator | Opis | Wzorce | Jest zgodny z |
| :---: | --- | --- | --- |
| *   | Dopasowuje poprzedni element zero lub większą liczbę razy. | a.*c | "abcbc" w elemencie "abcbc" |
| +   | Dopasowuje poprzedni element co najmniej raz. | "be+" | "bee" w "been" lub "be" w "bent" |
| ?   | Dopasowuje poprzedni element zero lub jeden raz. | "rai?" | "rai" w elemencie "rain" |
| {*N*} | Dopasuje poprzedni element dokładnie _n_ razy. | ",\\d{3}" | ",043" w "1,043.6", ",876", ",543"i ",210" w "9,876,543,210" |
| {*N*,} | Pasuje do poprzedniego elementu co najmniej _n_ razy. | "\\d{3,}" | "166" w "166 34a" |
| {*N*,*M*} | Pasuje do poprzedniego elementu co najmniej _n_ razy, ale nie więcej niż _m_ razy. | "\\d{3,5}" | "19302" w elemencie "193024" |
| *?  | Dopasowuje poprzedni element zero lub większą liczbę razy (przy czym ta liczba jest jak najmniejsza). | a.*?c | "abc" w elemencie "abcbc" |
| +?  | Dopasowuje poprzedni element raz lub większą liczbę razy (przy czym ta liczba jest jak najmniejsza). | "be+?" | "be" w "been" lub "be" w "bent" |
| ??  | Dopasowuje poprzedni element zero lub jeden raz (przy czym liczba dopasowań jest jak najmniejsza). | "rai??" | "ra" w elemencie "rain" |
| {*N*}? | Dopasuje poprzedni element dokładnie _n_ razy. | ",\\d{3}?" | ",043" w "1,043.6" lub ",876", ",543", ",210" w "9,876,543,210" |
### 5. Konstrukty naprzemienne
Konstrukcje zmiany modyfikują wyrażenie regularne, aby umożliwić dopasowanie typu albo/albo.
| Konstrukcje zmiany | Opis | Wzorce | Jest zgodny z |
| :---: | --- | --- | --- |
| \|   | Dopasuje dowolny jeden element oddzielony znakiem pionowego paska ( \| ). | th(e\|is\|at) | "the", "this" w "this is the day." |

Więcej informacji dotyczących wyrażeń regularnych można znaleźć na stronie [Język wyrażeń regularnych — podręczny wykaz](https://learn.microsoft.com/pl-pl/dotnet/standard/base-types/regular-expression-language-quick-reference).

## Przykłady
**[-+]?[0-9]+\\.[0-9]+** - Na początku może być opcjonalnie znak - lub +. Potem ma być przynajmniej jedna cyfra, za nią kropka i po niej znowu przynajmniej jedna cyfra. To jest wzorzec liczby rzeczywistej, gdzie kropka rozdziela część dziesiętną.

**[ ]+$** - Bardzo wygodne wyrażenie do wyłapywania spacji pozostawionych na końcu linii tekstu.

**\b\d{2}-\d{2}-\d{4}\b** - Wyszukanie dat w formacie DD-MM-RRRR

## Wyrażenia regularne w pythonie :snake:
W Pythonie do obsługi wyrażeń regularnych można korzystać z modułu :sparkles:**re**:sparkles:. Moduł ten oferuje zestaw funkcji pozwalających na wyszukanie ciągu znaków pod kątem dopasowania:
| Funkcja | Opis |
| --- | --- |
| **findall** | Zwraca listę zawierającą wszystkie dopasowania |
| **search** | Zwraca obiekt Match, jeśli w dowolnym miejscu ciągu znajduje się dopasowanie |
| **split** | Zwraca listę, na której ciąg został podzielony przy każdym dopasowaniu |
| **sub** | Zastępuje jedno lub wiele dopasowań ciągiem |

Obiekt Match posiada właściwości i metody służące do pobierania informacji o wyszukiwaniu i wyniku:
- **.span()** - zwraca krotkę zawierającą pozycję początkową i końcową dopasowania.
- **.string** - zwraca ciąg znaków przekazany do funkcji.
- **.group()** - zwraca część ciągu, w której nastąpiło dopasowanie


:fire:***Strona warta uwagi [regex101: python version](https://regex101.com/r/Xdxvca/1).***:fire:
