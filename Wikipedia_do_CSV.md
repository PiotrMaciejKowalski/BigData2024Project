# Pobieranie danych z Wikipedii
Program powstał w celu zaimportowania danych z Polskiej Wikipedii i zapisaniu ich w pliku csv. \
[Link do strony z danymi](https://dumps.wikimedia.org/plwiki/20231001/)  \
Z powyższej strony pobrano 7 plików XML (siedem, a nie jeden, bo colab nie był w stanie go przetworzyć) spakowanych w formacie bz2. Pliki zostały rozpakowane i wrzucone na Google Driva i kolejno przetwarzane zgodnie z poniższym kodem. \
Poniższe kody dotyczą przetworzenia pierwszego z siedmiu plików. Kolene pliki przetworzono analogicznie.

Import bibliotek


```python
import xml.dom.pulldom as pull
import xml.etree.ElementTree as ET
import re
import pandas as pd
import time
```

Łączenie z dyskiem


```python
from google.colab import drive
drive.mount('/content/drive')
```

    Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount("/content/drive", force_remount=True).
    

Ścieżka do pliku


```python
file_path = '/content/drive/MyDrive/Colab Notebooks/Analiza BIG DATA/Wikipedia/PLwiki_czesci/plwiki-20231001-pages-articles-multistream1.xml-p1p187037'
```

Definiowanie funkcji


```python
def extract_text_value(text: str) -> str:
    return ET.fromstring(text).text

pattern = r"(?s)(?:''')(.*?)(?:''')(.*?)(==.*)"
matcher = re.compile(pattern)

def extract_1st_acapit(text: str) -> str:
    try :
        return matcher.search(text).groups()[:2]
    except:
        return None, None


def process_counter_generator():

    last = time.time() #liczba sekund od 1 stycznia 1970 roku
    i = 0

    def process_counter():
        nonlocal i, last
        i += 1
        if i % 250 ==0:
            current = time.time()
            print(f'Processed {i} texts {current - last}')
            last = current

    return process_counter

def saver_generator(filepath, count):

    counter = 0

    def saver(dataset : dict):
        nonlocal counter
        counter += 1
        if counter % count == 0:
            df = pd.DataFrame.from_dict(dataset, orient='index',dtype=object)
            df.to_pickle(f'{filepath}_{counter}.pk')
            print(f'Saved after {counter} processed')

    return saver
```

Odczytanie pliku XML


```python
full_doc = pull.parse(file_path) #zwraca obiekt DOMEventStream
#This function will change the document handler of the parser and activate namespace support;
```


```python
type(full_doc)
```




    xml.dom.pulldom.DOMEventStream




```python
pc = process_counter_generator()
```

Zrezygnowano z zapisu plików w formacie pickle. Format pickle zapisuje Dataframe jako strumień bajtów.


```python
#sv = saver_generator('/content/drive/MyDrive/Colab Notebooks/Analiza BIG DATA/Wikipedia/plwiki/PLwiki', 250000)
```


```python
dataset = {}
```

Przetwarzanie pliku


```python
for event, node in full_doc:
    if event == pull.START_ELEMENT and node.tagName == 'title':
        full_doc.expandNode(node)
        title = extract_text_value(node.toxml())
    if event == pull.START_ELEMENT and node.tagName == 'text':
        full_doc.expandNode(node)
        page = extract_text_value(node.toxml())
        dataset[title] = page
        pc()
        #sv(dataset)
```

Tworzenie dataframu


```python
print(f'Length of dataset {len(dataset)}')
df = pd.DataFrame.from_dict(dataset, orient='index',dtype=object)
print(df.head())
#df.to_pickle('/content/drive/MyDrive/Colab Notebooks/Analiza BIG DATA/Wikipedia/plwiki/PLwiki_all.pk')
```

Eksport danych do pliku csv


```python
df.to_csv('/content/drive/MyDrive/Colab Notebooks/Analiza BIG DATA/Wikipedia/PLwiki_czesci/PLwiki_1.csv')
```

Tym samym kodem przetworzyłem 7 plików i utworzyłem 7 plików csv.

# Łączenie plików CSV
Z powodu dużego rozmiaru plików ich połącznie wykonano na komputerze osobistym. Fragmenty użytych poleceń znajdziemy niżej.

Rozmiary plików:



```
7. 872,9 MB
6. 1,89 GB
5. 2,19 GB
4. 1,28 GB
3. 970,8 MB
2. 789,1 MB
1. 696,5 MB
```





```python
import pandas as pd

df1 = pd.read_csv('D:/Analiza BIG DATA/Zadanie_wikipedia/csv/Laczenie/PLwiki_1.csv')
df2 = pd.read_csv('D:/Analiza BIG DATA/Zadanie_wikipedia/csv/Laczenie/PLwiki_2.csv')
frames = [df1, df2]
result = pd.concat(frames, ignore_index=True)

result.to_csv('D:/Analiza BIG DATA/Zadanie_wikipedia/csv/Laczenie/PLwiki_1_2.csv',index=False,header=True)
```

Finalny plik csv na dysku pod [linkiem](https://drive.google.com/drive/folders/1BOLoAhgkEwRROaTHjCInhxWYAsXVQZtP?usp=share_link).
Dodano również plik PLwiki_1, bo dużo mniejszy.
