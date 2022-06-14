# WdSI_PROJECT_v2.0
Projekt machine learningu wykonany na zaliczenie przedmiotu WdSI


Wykonany przeze mnie projekt dokonuje klasyfikacji znajdujących się na zdjęciach znaków drogowych na określające ograniczenia prędkości oraz pozostałe. Nie zrealizowałem funkcji "detect".

Całość projektu została wykonana na szablonie pochodzącym z laboratorium WdSI poświęconego Machine Learningowi. W realizacji projektu wykorzystałem poznane na zajęciach algorytmy BoVW oraz RFC.

Tworząc projekt miałem założenie jak najprostszego zrealizowania zadanych funkcjonalności, jednocześnie w sposób, który zapewni oczekiwany poziom poprawności klasyfikacji. Z moich obserwacji wynika, iż udało mi się ten cel zrealizować.

W pliku main.py znajdują się wszystkie funkcje odpowiadające za realizację zadania projektowego. Kod podzielony został przeze mnie na funkcje odczyt, learn_bovw, extract_features, train oraz predict. 

Funkcja odczyt odczytuje dane dotyczące danych wejściowych zarówno z plików .xml podczas trenowania modelu, jak i z pliku input.txt podczas przeprowadzania predykcji. Umożliwia to zmienna wejściowa "traincondition", która przyjmując wartość "1" umożliwia odczyt z plików .xml, a przyjmując wartość "0" - z pliku input.txt.

W funkcjach learn_bovw oraz extract_features odczytane zdjęcia są odpowiednio obrabiane, tj. przetwarzane do odcieni szarości oraz odpowiednio wycinane, tak aby umożliwić odpowiednie nauczenie modelu oraz późniejszą predykcję. W przypadku uczenia modelu za pomocą domyślnie wgranych zdjęć, nie osiągałem zadowalającego poziomu predykcji.

Funkcja predict odpowiada zarówno za przeprowadzenie predykcji jak i za wypisanie jej wartości - tj. przewidywań dotyczących znajdującej się na zdjęciach zawartości.

Wszystkie funkcje zostały przeze mnie opisane w kodzie.

Jedynymi wyjściami z programu są słowa "speedlimit" oraz "other".
