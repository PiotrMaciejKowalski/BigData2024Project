## Wspólne konto chmurowe i dostęp do współdzielonego katalogu

**Login**: big.data.masters23

Aby uzyskać dostęp do współdzielonego katalogu należy zwrócić się do Lead'a, który zarządza kontem. Po uzyskaniu zaproszenia do współdzielenia katalogu należy utworzyć skrót do tego katalogu na swoim gdrive - poniżej link do instrukcji, jak to zrobić:

https://github.com/PiotrMaciejKowalski/BigData2022-actors/blob/main/tutorials/Tworzenie%20skrotu%20do%20GDrive.md

**Uwaga**: tutorial jest w jednym punkcie delikatnie nieaktualny: w **3. kroku** należy kliknąć w "Porządkuj", i dopiero wtedy pojawi się opcja "dodaj skrót".

Po utworzeniu skrótu należy sprawdzić dostęp do danych z katalogu z wnętrza Colabu:

**1.** Aby zamontować Google drive do Colaba, wprowadź poniższy kod:

`from google.colab import drive`


`drive.mount('/content/drive')`

**2**. Aby przejść do współdzielonego katalogu: (nazwa naszego katalogu to *BigMess*)

`import os
os.chdir("drive/My Drive/BigMess")`

(więcej info: https://stackoverflow.com/questions/47744131/colaboratory-can-i-access-to-my-google-drive-folder-and-file)
