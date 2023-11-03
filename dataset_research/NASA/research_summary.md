# Dataset NASA

Link do zbioru danych: [Dataset NASA](https://drive.google.com/drive/u/1/folders/1tLGM4LURHCXctHuTUSmO_kr3JCzJJSPS)

Link do dokumentacji zbioru danych: [Dataset NASA - dokumentacja](https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/NLDAS2_README.pdf)

## Opis zbioru danych

* Prezentacja danych w formie miesięcznych podsumowań (średnie, sumy).
* Okres pomiarów - 285 miesięcy - od stycznia 2000 roku do września 2023 roku.
* Obszar pomiarów (środkowa Ameryka Północna):
  * szerokość geograficzna - od 25,0625 do 52,9375 stopni, co 1/8 stopnia
  * długość geograficzna - od -124,9375 do -67,0625 stopni, co 1/8 stopnia
  * 103 936 par współrzędnych (w praktyce dane są dostępne jedynie dla 76 360 z nich, brakuje danych dla współrzędnych odpowiadających obszarom wodnym)
  * podany zakres współrzędnych obejmuje:
    * USA bez stanu Alaska - 52 426 pozycji
    * niewielką część południowej Kanady - 18 897 pozycji
    * niewielką część północnego Meksyku - 5 023 pozycji
    * północną część Bahamów - 14 pozycji


## Opis poszczególnych kolumn

Nazwa | Pełna nazwa | Jednostka | Metoda pomiaru | Opis
--- | --- | --- | --- | ---
lon | Longitude | - | - | Długość geograficzna punktu pomiaru.
lat | Latitude | - | - | Szerokość geograficzna punktu pomiaru.
Date | Date | - | - |  Okres pomiaru, dane w formacie "rrrrmm" (przykład: "202209" - wrzesień 2022 roku).
SWdown | Shortwave radiation flux downwards (surface) | W/m^2 | Średnia miesięczna | Strumień promieniowania krótkofalowego skierowany ku dołowi (powierzchnia): odnosi się do ilości energii słonecznej docierającej do powierzchni ziemi w formie promieniowania krótkofalowego (widzialnego światła i bliskiej podczerwieni).
LWdown | Longwave radiation flux downwards (surface)  | W/m^2 | Średnia miesięczna | Strumień promieniowania długofalowego skierowany ku dołowi (powierzchnia): strumień promieniowania długofalowego (podczerwień) emitowanego przez atmosferę i kierowanego ku powierzchni ziemi. 
SWnet | Net shortwave radiation flux (surface) | W/m^2 | Średnia miesięczna | Bilansowy strumień promieniowania krótkofalowego (powierzchnia): różnica między promieniowaniem krótkofalowym docierającym do powierzchni ziemi a promieniowaniem krótkofalowym odbitym z powrotem do atmosfery. 
LWnet | Net longwave radiation flux (surface) | W/m^2 | Średnia miesięczna | Bilansowy strumień promieniowania długofalowego (powierzchnia): różnica między promieniowaniem długofalowym docierającym do powierzchni ziemi a promieniowaniem długofalowym emitowanym przez powierzchnię ziemi. 
Qle | Latent heat flux | W/m^2 | Średnia miesięczna | Strumień ciepła utajonego: ilość ciepła przekazywanego w procesach, które nie pociągają za sobą zmiany temperatury, takich jak parowanie i kondensacja. 
Qh | Sensible heat flux | W/m^2 | Średnia miesięczna | Strumień ciepła odczuwalnego: ilość ciepła przekazywanego przez procesy konwekcyjne, które prowadzą do zmian temperatury. 
Qg | Ground heat flux | W/m^2 | Średnia miesięczna | Strumień ciepła gruntowego: ilość ciepła przemieszczającego się do lub z powierzchni ziemi w wyniku przewodzenia cieplnego. 
Qf | Snow phase-change heat flux | W/m^2 | Średnia miesięczna | Strumień ciepła związanego ze zmianą fazy śniegu: ilość ciepła związanego ze zmianami stanu skupienia śniegu (topnienie lub zamarzanie).
Snowf |	Frozen precipitation (snowfall) | kg/m^2 | Suma miesięczna | Opad atmosferyczny w formie śniegu: ilość śniegu spadającego z atmosfery. 
Rainf	| Liquid precipitation (rainfall) | kg/m^2 | Suma miesięczna | Opad atmosferyczny w formie deszczu: ilość deszczu spadającego z atmosfery. 
Evap | Total evapotranspiration | kg/m^2 | Suma miesięczna | Całkowita ewapotranspiracja: suma wody traconej z powierzchni ziemi do atmosfery przez parowanie i transpirację roślin. Parowanie to proces, w którym woda zmienia się ze stanu ciekłego w gazowy na powierzchni gleby, a transpiracja to proces, w którym woda jest pobierana przez korzenie roślin i wydzielana do atmosfery przez liście.
Qs | Surface runoff (non-infiltrating) | kg/m^2 | Suma miesięczna | Spływ powierzchniowy (nieinfiltrujący):  woda, która przemieszcza się po powierzchni ziemi zamiast wsiąkać w glebę.
Qsb | Subsurface runoff (baseflow) | kg/m^2 | Suma miesięczna | Spływ podpowierzchniowy (przepływ bazowy):  przepływ wody pod powierzchnią ziemi, który ostatecznie trafia do cieków wodnych. Przepływ bazowy to część spływu podpowierzchniowego, który utrzymuje przepływ w ciekach wodnych między okresami opadów.
Qsm | Snowmelt | kg/m^2 | Suma miesięczna | Wody roztopwe: ilość wody w stanie ciekłym generowana z przemiany fazy stałej (śniegu) w ciekłą.
AvgSurfT | Average surface skin temperature | K | Średnia miesięczna | Średnia temperatura powierzchniowa skóry: średnia temperatura najbardziej zewnętrznej warstwy powierzchni ziemi.
Albedo | Surface albedo | % | Średnia miesięczna | Albedo powierzchni: stosunek promieniowania odbitego do promieniowania padającego na powierzchnię.
SWE | Snow Water Equivalent | kg/m^2 | Średnia miesięczna | Równoważnik wodny śniegu: ilość wody, którą można by uzyskać po stopieniu danej warstwy śniegu.
SnowDepth | Snow depth | m | Średnia miesięczna | Głębokość śniegu: wysokość warstwy śniegu na powierzchni ziemi. 
SnowFrac | Snow cover | fraction | Średnia miesięczna | Pokrywa śnieżna: stosunek powierzchni ziemi pokrytej śniegiem do całkowitego obszaru biorącego udział w rozważaniach.
SoilT_0_10cm | Soil temperature (0-10cm) | K | Średnia miesięczna | Temperatura gleby w warstwie o głębokości od 0 do 10 cm.
SoilT_10_40cm | Soil temperature (10-40cm) | K | Średnia miesięczna | Temperatura gleby w warstwie o głębokości od 10 do 40 cm.
SoilT_40_100cm | Soil temperature (40-100cm) | K | Średnia miesięczna | Temperatura gleby w warstwie o głębokości od 40 do 100 cm.
SoilT_100_200cm | Soil temperature (100-200cm)  | K | Średnia miesięczna | Temperatura gleby w warstwie o głębokości od 100 do 200 cm.
SoilM_0_10cm | Soil moisture content (0-10cm) | kg/m^2 | Średnia miesięczna | Zawartość wody w glebie na głębokości od 0 do 10 cm  (obejmuje fazę ciekłą, parową i stałą wody w glebie).
SoilM_10_40cm | Soil moisture content (10-40cm) | kg/m^2 | Średnia miesięczna | Zawartość wody w glebie na głębokości od 10 do 40 cm  (obejmuje fazę ciekłą, parową i stałą wody w glebie).
SoilM_40_100cm | Soil moisture content (40-100cm) | kg/m^2 | Średnia miesięczna | Zawartość wody w glebie na głębokości od 40 do 100 cm  (obejmuje fazę ciekłą, parową i stałą wody w glebie).
SoilM_100_200cm | Soil moisture content (100-200cm) | kg/m^2 | Średnia miesięczna | Zawartość wody w glebie na głębokości od 100 do 200 cm  (obejmuje fazę ciekłą, parową i stałą wody w glebie).
SoilM_0_100cm | Soil moisture content (0-100cm) | kg/m^2 | Średnia miesięczna | Zawartość wody w glebie na głębokości od 0 do 100 cm  (obejmuje fazę ciekłą, parową i stałą wody w glebie).
SoilM_0_200cm | Soil moisture content (0-200cm) | kg/m^2 | Średnia miesięczna | Zawartość wody w glebie na głębokości od 0 do 200 cm  (obejmuje fazę ciekłą, parową i stałą wody w glebie).
RootMoist | Root zone soil moisture | kg/m^2 | Średnia miesięczna | Zawartość wody w glebie w obszarze, gdzie korzenie roślin są najbardziej aktywne.
SMLiq_0_10cm | Liquid soil moisture content (0-10cm) | kg/m^2 | Średnia miesięczna | Zawartość wody w glebie na głębokości od 0 do 10 cm  (obejmuje fazę ciekłą wody w glebie).
SMLiq_10_40cm | Liquid soil moisture content (10-40cm) | kg/m^2 | Średnia miesięczna | Zawartość wody w glebie na głębokości od 10 do 40 cm  (obejmuje fazę ciekłą wody w glebie).
SMLiq_40_100cm | Liquid soil moisture content (40-100cm) | kg/m^2 | Średnia miesięczna | Zawartość wody w glebie na głębokości od 40 do 100 cm  (obejmuje fazę ciekłą wody w glebie).
SMLiq_100_200cm | Liquid soil moisture content (100-200cm) | kg/m^2 | Średnia miesięczna | Zawartość wody w glebie na głębokości od 100 do 200 cm  (obejmuje fazę ciekłą wody w glebie).
SMAvail_0_100cm | Soil moisture availability (0-100cm) | % | Średnia miesięczna | Zawartość dostępnej wody dla roślin w glebie od powierzchni do głębokości 100 cm.
SMAvail_0_200cm | Soil moisture availability (0-200cm) | % | Średnia miesięczna | Zawartość dostępnej wody dla roślin w glebie od powierzchni do głębokości 200 cm.
PotEvap	| Potential evapotranspiration | W/m^2 | Średnia miesięczna | Potencjalna ewapotranspiracja: maksymalna ilość wody, która może być tracona przez rośliny i glebę poprzez parowanie i transpirację w danych warunkach atmosferycznych.
ECanop | Canopy water evaporation | W/m^2 | Średnia miesięczna | Parowanie wody z baldachimu: proces, w którym woda paruje bezpośrednio z powierzchni liści i innych części roślin (tzw. "korony drzew").
TVeg | Transpiration | W/m^2 | Średnia miesięczna | Transpiracja: Proces, w którym woda jest pobierana przez korzenie roślin, przemieszcza się przez roślinę i jest uwalniana do atmosfery przez aparaty szparkowe w liściach.
ESoil	| Direct evaporation from bare soil | W/m^2 | Średnia miesięczna | Bezpośrednie parowanie z powierzchni nieporośnietej gleby.
SubSnow | Sublimation (evaporation from snow) | W/m^2 | Średnia miesięczna | Sublimacja (parowanie ze śniegu): proces, w którym śnieg lub lód przemienia się bezpośrednio w parę wodną, omijając stan ciekły.
CanopInt | Plant canopy surface water | kg/m^2 | Średnia miesięczna | Ilości wody (np. deszczu, rosy) zgromadzonej na liściach, gałęziach i innych częściach roślin, które tworzą baldachim.
ACond | Aerodynamic conductance | m/s | Średnia miesięczna | Przewodność aerodynamiczna: miara łatwości, z jaką powietrze i zawarte w nim substancje (np. pary wodnej, dwutlenek węgla) mogą przemieszczać się przez atmosferę do i od powierzchni roślin.
CCond | Canopy conductance | m/s | Średnia miesięczna | Przewodność baldachimu: miara zdolności liści i innych części roślin do wymiany gazów i pary wodnej z atmosferą.
RCS | Solar parameter in canopy conductance | fraction | Średnia miesięczna | Parametr słoneczny w przewodności baldachimu: odnosi się do wpływu światła słonecznego na przewodność baldachimu, ponieważ intensywność światła wpływa na otwieranie i zamykanie aparatów szparkowych w liściach, co z kolei wpływa na przewodność baldachimu.
RCT | Temperature parameter in canopy conductance | fraction | Średnia miesięczna | Parametr temperatury w przewodności baldachimu: odnosi się do wpływu temperatury na przewodność baldachimu. Wzrost temperatury zwykle prowadzi do zwiększenia przewodności baldachimu, chociaż zależy to również od innych czynników, takich jak dostępność wody.
RCQ | Humidity parameter in canopy conductance | fraction | Średnia miesięczna | Parametr wilgotności w przewodności baldachimu: odnosi się do wpływu wilgotności powietrza na przewodność baldachimu. Wyższa wilgotność powietrza może prowadzić do zmniejszenia przewodności baldachimu, ponieważ rośliny starają się ograniczyć utratę wody przez transpirację.
RCSOL | Soil moisture parameter in canopy conductance | fraction | Średnia miesięczna | Parametr wilgotności gleby w przewodności baldachimu: odnosi się do wpływu wilgotności gleby na przewodność baldachimu. Dostępność wody w glebie jest kluczowym czynnikiem wpływającym na zdolność roślin do transpiracji i wymiany gazów.
RSmin | Minimal stomatal resistance | s/m | Średnia miesięczna | Minimalny opór szparkowy:  odnosi się do oporu, jaki roślina stawia przepływowi gazów i pary wodnej przez aparaty szparkowe. Minimalny opór szparkowy to najmniejsza wartość tego oporu, przy której aparaty szparkowe są otwarte i umożliwiają wymianę gazów.
RSMacr | Relative soil moist. availability control factor | unitless | Średnia miesięczna | Współczynnik kontroli dostępności wilgotności gleby: odnosi się do relatywnej dostępności wody w glebie dla roślin.
LAI | Leaf Area Index | fraction | Średnia miesięczna | Wskaźnik pokrycia liściowego, indeks liściowy: stosunek całkowitej powierzchni liści roślin na jednostkę powierzchni gruntu.
GVEG | Green vegetation | 0-1 | Średnia miesięczna | Wskaźnik "zielonej roślinności" odnosi się do roślin o zielonych liściach, które są zdolne do fotosyntezy.
Streamflow | Streamflow | m^3/s | Średnia miesięczna |  Natężenie przepływu (w cieku) - objętość wody przepływająca przez dany przekrój cieku w jednostce czasu.

