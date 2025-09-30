# System wspierający zrozumienie kodu w oparciu o modele językowe i semantyczny graf kodu

1. https://ollama.com
2. terminalu: ollama run codellama:7b-instruct - uruchomienie modelu
3. scg-cli zmodyfikowany: https://github.com/jciura/scg-cli-modified

4. scg-cli generate <Sciezka> - najpierw generuje się zawsze dla projektu pojedyncze grafy semantyczne, bez tego reszata
   nie dziala
5. scg-cli export -g SCG -o gdf <Sciezka> - export całego grafu do pliku .gdf
6. scg-cli summary -g SCG <Sciezka> - szybkie podsumiwanie projektu
7. scg-cli crucial <Sciezka> -n k; k - ile bierzemy węzłów o najwyższych wartościach, raczej chcemy podawać all, żeby
   każdy embedding miał te wartości, a nie tylko wybrane
8. scg-cli partition n; podaje jak podzielić projekt na n partycji
9. Gephi - plik do otwierania plików .gdf - można użyć ale raczej bezużyteczny - nie na tym nie widać
10. Przy zmianie scg-cli wywołać: sbt clean universal:packageBin, żeby wygenerować nową paczkę
11. Na razie testowy projekt to projekt w springu do zapisywania się na webinary i zarządzania
    nimi: https://github.com/jciura/test_project - wrzucić do projects i zmienić
    nazwę na test; Na razie nie trzeba pobierać potrzebne pliki z scg-cli są wygenerowane w /projects 

12. Link do wtyczki do intellij - https://github.com/jciura/CodeAssistant - trzeba uruchomić to w intellij, on
     włączy nową instancję intellij, gdzie trzeba otworzyć projekt i wtyczka jest tam zaimportowana - najniższa ikona po prawej stronie.




## Działanie systemu

### Przygotowanie danych (wykonywane raz)

1. **load_graph.py** – ładowanie plików GDF z grafem kodu oraz metryk.  
2. **question_embedding.py** – obliczanie embeddingów klasyfikatora pytań na danych.  
3. **generate_embeddings_graph.py** – generowanie embeddingów dla wszystkich węzłów grafu.  
4. **ChromaDB** – baza, w której przechowywane są embeddingi i metadane węzłów.  

### Obsługa pytania użytkownika

1. **main.py** – odbiera pytanie użytkownika przez endpoint `/ask_rag_node`.  
2. **prompts.py** – analiza intencji użytkownika (`analyze_user_intent`) - klasyfikuje pytanie na kategorię (exception, usage, definition, implementation, testing).
3. **similar_node_optimization.py** – wyszukiwanie i budowa kontekstu:  
   - **retriver.py** – wyciąganie kluczowych terminów z pytania,  
   - **generate_embeddings_graph.py** – tworzenie embeddingu dla pytania,  
   - **rag_optimization.py** – udostępnia model CodeBERT,  
   - **ChromaDB** – znajduje podobne węzły kodu,  
   - **context.py** – filtrowanie kontekstu na podstawie kategorii pytania
4. **prompts.py** – budowanie prompta (`build_intent_aware_prompt`) z uwzględnieniem kategorii i historii konwersacji.
5. **models.py** – walidacja danych i zarządzanie historią konwersacji.  
6. **main.py** – wysyła kontekst + pytanie do Ollama API.  
7. **prompts.py** – post-processing odpowiedzi.
8. **main.py** – zwraca odpowiedź użytkownikowi.


## Przykłady pytań 
- Describe Category class
- How does method getSortedWebinars work?
- where is method enrollStudentIntoWebinar used?
- What custom exceptions are in WebinarService class?
- What tests exist for WebinarService class?

## Wymagania
- Python 3.10+
- [Ollama](https://ollama.ai/) (`llama3.1:8b`)
- Zależności z pliku `requirements.txt`:
  ```bash
  pip install -r requirements.txt

