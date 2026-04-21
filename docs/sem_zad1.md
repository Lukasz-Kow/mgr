Temat pracy: Zastosowanie selektywnej predykcji w głębokich sieciach 3D-CNN do redukcji fałszywych alarmów w diagnostyce łagodnych zaburzeń poznawczych na podstawie obrazów MRI.
Wen, J., et al. (2020). "Convolutional neural networks for classification of Alzheimer's disease: Overview and reproducible evaluation." Medical Image Analysis. 
Jest to fundamentalna praca przeglądowa i benchmarkowa. Autorzy dokonują rygorystycznej oceny różnych architektur sieci neuronowych (2D i 3D) stosowanych do obrazowania MRI w chorobie Alzheimera i MCI. Kluczowym wkładem jest zidentyfikowanie powszechnego błędu w literaturze – "wycieku danych" (data leakage) – oraz zaproponowanie ustandaryzowanego potoku przetwarzania (preprocessing pipeline).
Artykuł ten wybrałem jako fundament architektoniczny i metodologiczny mojej pracy. Ponieważ mój projekt opiera się na klasyfikacji obrazów MRI z bazy ADNI, potrzebowałem punktu odniesienia, który jest uznawany za standard w środowisku naukowym. Wybrałem tę pozycję, ponieważ autorzy w sposób rygorystyczny dowodzą wyższości sieci 3D-CNN nad modelami 2D w rozpoznawaniu zmian neurodegeneracyjnych. 
Zalety:
•	Wskazuje na konieczność rygorystycznej walidacji na poziomie pacjenta
•	Udowadnia wyższość modeli 3D nad 2D w wychwytywaniu rozproszonych zmian neurodegeneracyjnych.
•	Dostarcza solidny punkt odniesienia (Baseline) dla wyników uzyskiwanych na zbiorze ADNI.
Wady:
•	Skoncentrowanie wyłącznie na metryce Accuracy (dokładność), bez uwzględnienia niepewności modelu.
•	Wysoki koszt obliczeniowy przetwarzania pełnych wolumenów MRI.
Możliwości ulepszeń:
•	Wzbogacenie modelu o mechanizm "abstencji" (odmowy diagnozy), aby uniknąć klasyfikacji próbek niejednoznacznych, co bezpośrednio realizuje niniejsza praca magisterska.
Geifman, Y., & El-Yaniv, R. (2019). "SelectiveNet: A Deep Neural Network with an Integrated Abstention Mechanism." ICML.
Autorzy wprowadzają architekturę SelectiveNet, która posiada zintegrowany mechanizm "abstencji". Zamiast zmuszać sieć do klasyfikacji każdego przypadku, sieć posiada trzecią głowicę (selekcyjną), która decyduje, czy model jest wystarczająco pewny, by podać diagnozę. Artykuł definiuje matematyczny kompromis między pokryciem danych (coverage) a dopuszczalnym ryzykiem błędu.
Tę pozycję wybrałem, aby zdefiniować główny cel mojej pracy, jakim jest selektywna predykcja. Artykuł ten jest kluczowy, ponieważ wprowadza on matematyczną koncepcję modelu, który ma prawo „wstrzymać się od odpowiedzi”, jeśli nie jest pewien swojej decyzji. 
Metoda: Architektura zintegrowanej "głowicy selekcyjnej", która optymalizuje model pod kątem zadanego poziomu pokrycia danych (Coverage) i ryzyka błędu.
Zalety:
•	Pozwala na bezpośrednie sterowanie kompromisem między specyficznością a czułością modelu.
•	Gwarantuje matematycznie optymalną selekcję próbek dla określonego poziomu ryzyka.
Wady:
•	Wysoka złożoność architektury (wymaga trenowania trzech głowic jednocześnie: klasyfikacyjnej, selekcyjnej i pomocniczej).
•	Trudność w stabilnym trenowaniu dla ciężkich wolumenów 3D-MRI, co prowadzi do problemów z pamięcią VRAM.
Możliwości ulepszeń:
•	Zastąpienie skomplikowanej głowicy selekcyjnej podejściem ewidencyjnym (EDL), co upraszcza architekturę przy zachowaniu zdolności modelu do odmowy odpowiedzi.

Sensoy, M., et al. (2018). "Evidential Deep Learning to Quantify Classification Uncertainty." NeurIPS.
Praca wprowadza metodę Evidential Deep Learning (EDL). Zamiast klasycznej warstwy Softmax, która często jest "zbyt pewna siebie", autorzy proponują modelowanie parametrów rozkładu Dirichleta. Pozwala to sieci na przypisanie predykcji konkretnej wartości "dowodów" (evidence). Dzięki temu model potrafi jasno zakomunikować: "nie mam wystarczających dowodów, by podjąć decyzję".
Pracę tę wybrałem jako główne rozwiązanie techniczne dla mojego hybrydowego modelu. Po wstępnych testach z architekturą SelectiveNet w środowisku 3D, dostrzegłem jej ograniczenia obliczeniowe. Zdecydowałem się na metodę Evidential Deep Learning (EDL) opisaną przez Sensoy’a, ponieważ pozwala ona na szybką i stabilną kwantyfikację niepewności bez konieczności drastycznej zmiany architektury sieci. 
Zalety:
•	Wymaga jedynie zmiany funkcji kosztu (Loss Function), nie zmieniając znacząco architektury sieci.
•	Model potrafi odróżnić "brak wiedzy" od "pewności o błędzie", co jest kluczowe w diagnostyce medycznej.
•	Szybsze działanie w porównaniu do metod selekcyjnych wymagających wielu iteracji (jak Monte Carlo Dropout).
Wady:
•	Wrażliwość na hiperparametry 
•	Ryzyko nadmiernej abstencji przy danych o niskiej jakości normalizacji.
Możliwości ulepszeń:
•	Implementacja w ramach architektury hybrydowej 3D-ResNet, łączącej silną ekstrakcję cech przestrzennych z matematyczną elegancją teorii dowodów.


